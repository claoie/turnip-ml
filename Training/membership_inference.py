#!/usr/bin/env python3
"""Membership-inference check on the holdout report (TRAINING_DESIGN.md §6.2).

Every device receives the published model, so treating it as anonymous
needs evidence that its outputs do not give away which sources it was
trained on. This script runs a simple membership-inference attack: can
the model's outputs tell its training sources from holdout sources
better than chance? The result goes into the holdout report, and the
promotion gate (Training/promote.py) refuses to promote a model that
fails it (§8).

Inputs
------
--model        model.npz from Training/train.py: the model under test.
--split        split.json the model was trained on. Train samples are the
               members, holdout samples the non-members. Checked against
               the report's inputs.split_sha256, so the check always
               describes the run the report scores.
--samples-dir  sample JSON dir (keypoints and ground-truth tricks).
--report       holdout_report.json from Training/evaluate.py; rewritten
               in place (atomically) with a "membership_inference" block.

The attack
----------
A confidence attack (Yeom et al., 2018, "Privacy Risk in Machine
Learning"): a model that memorized its training data is more confident
on it. Each window gets the model's softmax probability for its true
label, with the windowing, labels and standardization train.py used
(leakage_gap.py computes holdout window accuracy the same way). A
window whose label is not one of the model's classes has no such
probability and is left out; the block counts them. A sample's score is
the mean over its windows.

A sample is one segment in the farm's dataset (fetch_dataset.py writes
one sample per segment), and the splitter keeps every segment of a split
bucket in one split, so a session is on one side. Members and
non-members are therefore segments; the code and the report call them
sources, as they were before the switch to segments.

The attack's power is the ROC AUC of those scores, members against
non-members: the probability that a random training source scores above
a random holdout source, ties counting one half. 0.5 is chance. Scores
are rounded to 6 decimals before they are compared, so the AUC does not
depend on float noise.

Pass iff auc <= MAX_AUC and each side has at least MIN_SOURCES
scored sources. *(Judgment calls, TRAINING_DESIGN.md §10 Q6: 0.60 means
the attacker ranks a training source above a holdout source at most 60%
of the time, an advantage of at most 0.2 over chance. 20 sources keeps
the AUC's chance spread small enough to read: with 20 holdout sources
and many more training ones, a model that leaks nothing still scores
above 0.60 in about 7% of runs; such a run archives its challenger and
the next run tries again. Fewer sources give no evidence either way, so the check fails
rather than pass on noise.)*

The holdout's people are outside the contributor pool (§2.6), so the
non-members also differ from the members in who is in them: a model
that learned individuals fails this check as well as one that memorized
sources. That errs on the side of failing.

Outputs
-------
The report gains "membership_inference": attack, auc, max_auc,
min_sources, passed, the source and window counts on each side, and
the mean score on each side (same format evaluate.py writes: indent 2,
sorted keys, trailing newline). Stdout carries aggregates only; no sample or source id.

Exit codes: 0 = passed, 3 = failed (not an error: the report records it
and the promotion gate reads it), 2 = input error (fail closed):
unreadable or malformed inputs, a report from another split, a model
whose window is longer than every sample on one side, or a side with no
scorable sample.

Determinism: same inputs -> byte-identical report (sorted keys, floats
rounded to 6 decimals, samples in sorted order).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NoReturn

import evaluate
import leakage_gap
import numpy as np
import train

ATTACK = "mean-true-class-confidence"
MAX_AUC = 0.60
MIN_SOURCES = 20
REPORT_KEY = "membership_inference"
EXIT_PASS = 0
EXIT_ERROR = 2
EXIT_FAIL = 3


class CheckError(Exception):
    """Input error: the check refuses to run rather than guess."""


def _fail(msg: str) -> NoReturn:
    raise CheckError(msg)


def sample_scores(
    ckpt: dict[str, Any], samples: list[dict[str, Any]]
) -> tuple[list[float], int, int]:
    """Per-sample mean true-label confidence: (scores, windows scored,
    windows left out because their label is not a model class).

    Samples shorter than the model's window, or whose every window has
    an unknown label, have no score and are left out of the list.
    """
    window_frames = int(ckpt["window_frames"])
    window_stride = int(ckpt["window_stride"])
    margin_frames = (
        int(ckpt["background_margin_frames"])
        if "background_margin_frames" in ckpt
        else 0
    )
    classes = [str(c) for c in ckpt["classes"]]
    index = {c: i for i, c in enumerate(classes)}
    windowed = [s for s in samples if s["keypoints"].shape[0] >= window_frames]
    if not windowed:
        return [], 0, 0
    X_raw, labels, prov = train.build_dataset(
        np, windowed, window_frames, window_stride, margin_frames
    )
    X = (np.asarray(X_raw) - ckpt["feat_mean"]) / ckpt["feat_std"]
    probs = train.predict_proba(np, X, ckpt["W"], ckpt["b"])
    per_sample: dict[str, list[float]] = {}
    unseen = 0
    for row, (label, (sid, _, _)) in enumerate(zip(labels, prov, strict=True)):
        if label not in index:
            unseen += 1
            continue
        per_sample.setdefault(sid, []).append(float(probs[row, index[label]]))
    scores = [
        evaluate.r6(float(np.mean(per_sample[sid]))) for sid in sorted(per_sample)
    ]
    return scores, len(labels) - unseen, unseen


def auc(members: list[float], non_members: list[float]) -> float:
    """P(member score > non-member score), ties counting one half."""
    wins = 0.0
    for m in members:
        for n in non_members:
            if m > n:
                wins += 1.0
            elif m == n:
                wins += 0.5
    return wins / (len(members) * len(non_members))


def passes(auc_value: float, n_train: int, n_holdout: int) -> bool:
    """The pass rule (module docstring, "The attack")."""
    return auc_value <= MAX_AUC and min(n_train, n_holdout) >= MIN_SOURCES


def check(
    ckpt: dict[str, Any], split: dict[str, Any], samples_dir: Path
) -> dict[str, Any]:
    """The report block for this model and split."""
    sides: dict[str, tuple[list[float], int, int]] = {}
    for part in ("train", "holdout"):
        try:
            samples = leakage_gap.load_split_samples(split, samples_dir, part)
        except leakage_gap.GapError as e:
            _fail(str(e))
        scores, windows, unseen = sample_scores(ckpt, samples)
        if not scores:
            _fail(f"no {part} sample has a scorable window: the attack is undefined")
        sides[part] = (scores, windows, unseen)
    members, n_train_windows, train_unseen = sides["train"]
    non_members, n_holdout_windows, holdout_unseen = sides["holdout"]
    value = evaluate.r6(auc(members, non_members))
    return {
        "attack": ATTACK,
        "auc": value,
        "max_auc": MAX_AUC,
        "min_sources": MIN_SOURCES,
        "passed": passes(value, len(members), len(non_members)),
        "n_train_sources": len(members),
        "n_holdout_sources": len(non_members),
        "n_train_windows": n_train_windows,
        "n_holdout_windows": n_holdout_windows,
        "n_windows_unseen_label": train_unseen + holdout_unseen,
        "train_mean_score": evaluate.r6(float(np.mean(members))),
        "holdout_mean_score": evaluate.r6(float(np.mean(non_members))),
    }


def run(
    model_path: Path, split_path: Path, samples_dir: Path, report_path: Path
) -> dict[str, Any]:
    """Compute the block and write it into the report; returns the block."""
    split = evaluate.load_json(split_path)
    if not isinstance(split, dict):
        _fail("split must be a JSON object of sample_id -> split")
    for part in split.values():
        if part not in evaluate.SPLITS:
            _fail(f"split has unknown part {part!r}")
    report = evaluate.load_json(report_path)
    inputs = report.get("inputs") if isinstance(report, dict) else None
    if not isinstance(inputs, dict) or inputs.get(
        "split_sha256"
    ) != evaluate.sha256_file(split_path):
        _fail("report inputs.split_sha256 does not match split.json")
    ckpt = train.load_checkpoint(model_path)

    block = check(ckpt, split, samples_dir)
    report[REPORT_KEY] = block
    tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(report_path)
    return block


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Membership-inference check.")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--samples-dir", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args(argv)
    try:
        block = run(args.model, args.split, args.samples_dir, args.report)
    except (CheckError, ValueError, TypeError) as e:
        print(f"membership_inference.py: error: {e}", file=sys.stderr)
        return EXIT_ERROR
    verdict = "pass" if block["passed"] else "FAIL"
    print(
        f"membership inference: {verdict}, auc {block['auc']:.4f} "
        f"(max {block['max_auc']}) over {block['n_train_sources']} train and "
        f"{block['n_holdout_sources']} holdout sources (min {block['min_sources']})"
    )
    return EXIT_PASS if block["passed"] else EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
