#!/usr/bin/env python3
"""Per-name impact metrics for the model manifest (turnip-ml#41).

Writes name_metrics.json, the `name_metrics` object that POST /api/models
publishes and GET /api/models/current serves (docs/MODEL_CONTRACT.md §1,
docs/TRAINING_DESIGN.md §6.1). It is the app's impact feedback: how well
the model knows each trick, and which tricks it is still learning.

    {"cork": {"precision": 0.88, "recall": 0.79,
              "holdout_count": 24, "train_count": 310},
     "kick": {"precision": null, "recall": null,
              "holdout_count": 3, "train_count": 12}}

Inputs
------
--report       holdout_report.json from Training/evaluate.py (its per_name).
--split        split.json the report was computed on. Checked against the
               report's inputs.split_sha256, so the counts and the metrics
               always describe the same run.
--samples-dir  per-sample JSON files; the train split's "tricks" give
               train_count. Holdout and val samples are not read.
--out          name_metrics.json path.

Definitions
-----------
- holdout_count: holdout ground-truth segments carrying the name (the
  report's per_name `support`).
- train_count: train-split ground-truth segments carrying the name,
  counted the same way. fetch_dataset.py writes one trick entry per
  (clip, canonical name), so this is the name's training clips: the
  per-name train counts of TRAINING_DESIGN.md §2.7.
- precision, recall: the report's per-name values (a name match is
  required, see evaluate.py); null when holdout_count < 5
  (MIN_HOLDOUT_COUNT), a sample too small to support a number.
- Names: every name in the train split or in the report (holdout ground
  truth or predictions). "background" is the absence of a segment, never
  a name. A name the model trained on but the holdout lacks gets
  holdout_count 0 and null metrics: a trick it is still learning.

Exit codes: 0 = written, 2 = input error (fail closed): unreadable or
malformed inputs, a report from a different split, or a train sample with
no file.

Determinism: same inputs -> byte-identical file (sorted keys, floats
rounded to 6 decimals, no timestamps). Stdout carries counts only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import evaluate

MIN_HOLDOUT_COUNT = 5


def _count(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where} must be a non-negative integer")
    return value


def _unit(value: Any, where: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{where} must be a number in [0, 1]")
    return float(value)


def holdout_per_name(report: Any) -> dict[str, dict[str, Any]]:
    """The report's per_name section, validated."""
    if not isinstance(report, dict) or not isinstance(report.get("per_name"), dict):
        raise TypeError("holdout report needs a per_name object")
    per_name: dict[str, dict[str, Any]] = {}
    for name, entry in report["per_name"].items():
        where = f"holdout report per_name[{name!r}]"
        if not isinstance(name, str) or not name or not isinstance(entry, dict):
            raise ValueError(f"{where} must be an object under a non-empty name")
        per_name[name] = {
            "support": _count(entry.get("support"), f"{where}.support"),
            "precision": _unit(entry.get("precision"), f"{where}.precision"),
            "recall": _unit(entry.get("recall"), f"{where}.recall"),
        }
    return per_name


def train_name_counts(split: dict[str, str], samples_dir: Path) -> Counter[str]:
    """Ground-truth segments per name over the train split's samples."""
    counts: Counter[str] = Counter()
    for sid in sorted(s for s, part in split.items() if part == "train"):
        path = samples_dir / f"{sid}.json"
        if not path.is_file():
            raise ValueError(f"train sample {sid!r} has no file at {path}")
        sample = evaluate.load_json(path)
        tricks = sample.get("tricks") if isinstance(sample, dict) else None
        if not isinstance(tricks, list):
            raise TypeError(f"{path}: missing 'tricks' list")
        for i, trick in enumerate(tricks):
            _, _, names = evaluate.parse_segment(trick, f"{path} trick #{i}")
            counts.update(names)
    return counts


def build_name_metrics(
    per_name: dict[str, dict[str, Any]], train_counts: Counter[str]
) -> dict[str, dict[str, Any]]:
    """Pure: one manifest entry per name, sorted."""
    out: dict[str, dict[str, Any]] = {}
    for name in sorted(set(per_name) | set(train_counts)):
        if name == "background":
            raise ValueError('"background" is not a trick name')
        holdout = per_name.get(name, {"support": 0, "precision": 0.0, "recall": 0.0})
        scored = holdout["support"] >= MIN_HOLDOUT_COUNT
        out[name] = {
            "precision": evaluate.r6(holdout["precision"]) if scored else None,
            "recall": evaluate.r6(holdout["recall"]) if scored else None,
            "holdout_count": holdout["support"],
            "train_count": train_counts.get(name, 0),
        }
    return out


def run(report_path: Path, split_path: Path, samples_dir: Path) -> dict[str, Any]:
    report = evaluate.load_json(report_path)
    split = evaluate.load_split(split_path)
    inputs = report.get("inputs") if isinstance(report, dict) else None
    if not isinstance(inputs, dict) or inputs.get("split_sha256") != (
        evaluate.sha256_file(split_path)
    ):
        raise ValueError(
            f"{report_path} was not computed on {split_path} (split_sha256 differs)"
        )
    return build_name_metrics(
        holdout_per_name(report), train_name_counts(split, samples_dir)
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--report", required=True, type=Path, help="holdout_report.json")
    p.add_argument("--split", required=True, type=Path, help="split.json")
    p.add_argument("--samples-dir", required=True, type=Path, help="sample JSON dir")
    p.add_argument("--out", required=True, type=Path, help="name_metrics.json path")
    args = p.parse_args(argv)
    try:
        metrics = run(args.report, args.split, args.samples_dir)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, sort_keys=True)
            fh.write("\n")
        tmp.replace(args.out)
    except (ValueError, TypeError, OSError) as e:
        print(f"name_metrics.py: error: {e}", file=sys.stderr)
        return 2
    scored = sum(1 for m in metrics.values() if m["precision"] is not None)
    print(
        f"name metrics for {len(metrics)} names ({scored} with >= "
        f"{MIN_HOLDOUT_COUNT} holdout examples)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
