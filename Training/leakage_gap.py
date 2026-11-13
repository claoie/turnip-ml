"""Val-minus-holdout person-leakage gap (turnip-ml#42, TRAINING_DESIGN.md §2.6).

Person leakage is a measured risk: the export carries no account
identifier, so one person's videos can land in both train and val, and
val is what the promotion gate reads. The holdout is built from footage
of people outside the contributor pool, so a model that learned
individuals rather than tricks shows up as a gap between val and
holdout. Every run reports that gap; a gap above 0.05 absolute on three
consecutive promoted runs flags the split-grouping question.

Inputs
------
--metrics      metrics.json from Training/train.py (val_accuracy,
               val_segments, classes, split_sha256).
--model        model.npz from the same train.py run: the holdout side of
               the promotion-metric gap is computed with it.
--report       holdout_report.json from Training/evaluate.py (f1,
               iou_threshold, inputs.split_sha256 and
               inputs.predictions_sha256).
--split        split.json all three were computed on. Checked against
               both SHAs, like name_metrics.py, so the gap always
               describes one run.
--samples-dir  sample JSON dir (val and holdout ground truth, holdout
               keypoints).
--registry     decision log (JSON array; default Training/registry.json).
               Gap records are appended; the log is what the nightly
               trigger reads, so the streak survives fresh checkouts
               via R2.
--promotion    this run's promotion decision: promote | archive |
               bootstrap | not-run.
TRAINING_RUN_ID (environment, optional) names the run in its registry
               record, as publish_model.py names it to the farm; see
               "Run id" below.

What it computes
----------------
Each gap compares one metric, computed the same way on both splits.

- promotion_metric gap: val_accuracy − holdout window accuracy.
  val_accuracy is what the promotion gate reads (promote.py's default
  --metric): train.py's window-level argmax accuracy over every val
  window, "background" windows included. The holdout side is that same
  computation over every holdout window: the model's windowing
  (train.build_dataset at its window_frames/window_stride and the
  background margin the checkpoint trained with), labels from
  train.label_for_window over evaluate.ground_truth_segments, the
  checkpoint's feature standardization, and train.predict's argmax.
  The holdout report's name_accuracy_given_detection is a segment-level
  metric (names matched among IoU-matched detections), so it is not
  this gap's holdout side. A holdout window whose label train never saw
  can't be predicted and counts as a miss (train.py refuses such labels
  on val, so val has none); the record counts those windows. No holdout
  window at all fails closed: the accuracy would be undefined.
- segment_f1 gap: val segment F1 − holdout f1. Val F1 is computed on the
  same footing as holdout F1: metrics.json's val_segments decoded by
  evaluate.parse_segment (a combo label is its names, not one name),
  scored by evaluate.compute_metrics against val ground truth at the
  report's IoU threshold.
- streak: consecutive promoted runs with either gap above 0.05 absolute.
  A promoted run at or under threshold, or a non-promoted run (archive,
  not-run), resets the streak to 0. A bootstrap run leaves it unchanged:
  it passed no gate, so it is neither a promoted run nor a break in the
  pattern. Runs where this script never ran leave no record and do not
  move the streak either way.
- flagged: streak >= 3.

Run id
------
Every registry record carries a run_id, and the streak counts runs, not
records. The run id is TRAINING_RUN_ID when set (the workflow passes
"<run_id>-<run_attempt>"), else "sha256:" + a hash of the run's inputs
and results (split, metrics.json, holdout predictions, promotion, gaps),
so a local re-run of the same run gets the same id.

The report write and the registry append are separately atomic, not
jointly so, so a step can be re-run after its registry append. A re-run
is idempotent: when the registry already holds a record for this run id
with the same promotion and gaps, the report is written with that
record's streak and the registry is left as it is. A record for this run
id with different results fails closed: one run id names one run. A gap
record with no run_id, or two with the same one, is a malformed registry.

Outputs
-------
- The report gains "val_minus_holdout": both gaps, their inputs, the
  holdout window counts, the streak, and the flag (same format
  evaluate.py writes: indent 2, sorted keys, trailing newline). The run
  id stays out of the report, so its bytes depend on the data alone.
- A {"kind": "leakage_gap", "run_id": ..., ...} record is appended to the
  registry (atomic write, like promote.py). Other record kinds in the
  log are ignored when replaying the streak.
- Stdout carries aggregates only (gaps, streak, flag). Per-sample
  outputs stay in data/run/; no sample ids, no account-shaped data.

Exit codes: 0 = written, 2 = input error (fail closed): unreadable or
malformed inputs, split mismatch, a model whose classes differ from the
metrics', no holdout window, a val_segments entry outside the val
split, a malformed existing registry, or a run id already recorded with
different results.

Determinism: same inputs -> byte-identical report and registry payload
(sorted keys, floats rounded to 6 decimals), except the registry
record's wall-clock `decided_at` (a registry is a log, not a
reproducible artifact).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

import evaluate
import numpy as np
import train

GAP_THRESHOLD = 0.05
FLAG_STREAK = 3
PROMOTIONS = ("promote", "archive", "bootstrap", "not-run")
RUN_ID_ENV = "TRAINING_RUN_ID"
EXIT_OK = 0
EXIT_ERROR = 2


class GapError(Exception):
    """Input error: the script refuses to compute rather than guess."""


def _fail(msg: str) -> NoReturn:
    raise GapError(msg)


def _load_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        _fail(f"cannot read JSON from {path}: {e}")


def _unit(value: Any, where: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        _fail(f"{where} must be a number in [0, 1]")
    return float(value)


def _iou_threshold(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 < value <= 1.0
    ):
        _fail("iou_threshold must be a number in (0, 1]")
    return float(value)


def _val_predictions(
    val_segments: Any, val_ids: list[str]
) -> dict[str, list[tuple[int, int, frozenset[str]]]]:
    """metrics.json val_segments as evaluate.py predictions.

    Decoded by evaluate.parse_segment, the same decoder the holdout
    path uses: a segment's `name` splits on "+" into names (a combo
    label is its names, not one name), empty name components fail, and
    a contradictory is_combo flag fails — instead of the loose local
    decoding this used to carry.
    """
    if not isinstance(val_segments, list):
        _fail("metrics val_segments must be a list")
    val_set = set(val_ids)
    pred: dict[str, list[tuple[int, int, frozenset[str]]]] = {
        sid: [] for sid in val_ids
    }
    for i, seg in enumerate(val_segments):
        where = f"metrics val_segments[{i}]"
        if not isinstance(seg, dict):
            _fail(f"{where} must be an object")
        sid = seg.get("sample_id")
        if sid not in val_set:
            _fail(f"{where} sample_id {sid!r} is not in the val split")
        pred[sid].append(evaluate.parse_segment(seg, where))
    return pred


def load_split_samples(
    split: dict[str, Any], samples_dir: Path, split_name: str
) -> list[dict[str, Any]]:
    """The named split's samples: keypoints and ground-truth tricks.

    train.load_samples never returns holdout samples (its poisoning
    defense) and predict.load_split_samples drops the tricks, so this
    reads them itself, with the same checks: each file holds its sample,
    keypoints pass train.check_keypoints, tricks is a list
    (train.build_dataset validates each entry when it labels windows), and
    held_back_windows pass train.held_back_windows, so a held-back clip is
    left out of the holdout windows as it is out of val's.
    """
    ids = sorted(sid for sid, part in split.items() if part == split_name)
    if not ids:
        _fail(f"no {split_name} samples in the split: window accuracy is undefined")
    samples = []
    for sid in ids:
        path = samples_dir / f"{sid}.json"
        rec = _load_json(path)
        if not isinstance(rec, dict) or rec.get("sample_id") != sid:
            _fail(f"sample file {path} does not hold sample {sid!r}")
        tricks = rec.get("tricks")
        if not isinstance(tricks, list):
            _fail(f"{path}: missing 'tricks' list")
        keypoints = train.check_keypoints(rec.get("keypoints"), sid)
        samples.append(
            {
                "sample_id": sid,
                "keypoints": keypoints,
                "tricks": tricks,
                "held_back_windows": train.held_back_windows(rec),
            }
        )
    return samples


def window_accuracy(
    ckpt: dict[str, Any], samples: list[dict[str, Any]]
) -> tuple[float, int, int]:
    """The checkpoint's window-level accuracy over `samples`, computed as
    train.py computes val_accuracy: (accuracy, windows, windows whose
    label is not one of the model's classes).

    Same windowing and labels (train.build_dataset, label_for_window),
    same standardization (the checkpoint's feature stats, which train.py
    applied to val before saving them), same argmax (train.predict), and
    the fraction of windows whose predicted label equals the true one.
    The background margin is the one the checkpoint trained with
    (`background_margin_frames`, absent on older checkpoints meaning 0),
    so the window set matches the run's exactly.
    """
    window_frames = int(ckpt["window_frames"])
    window_stride = int(ckpt["window_stride"])
    margin_frames = (
        int(ckpt["background_margin_frames"])
        if "background_margin_frames" in ckpt
        else 0
    )
    classes = [str(c) for c in ckpt["classes"]]
    windowed = [s for s in samples if s["keypoints"].shape[0] >= window_frames]
    if not windowed:
        _fail(
            f"no windows: every sample is shorter than the model's "
            f"window_frames ({window_frames}), so window accuracy is undefined"
        )
    X_raw, labels, _ = train.build_dataset(
        np, windowed, window_frames, window_stride, margin_frames
    )
    X = (np.asarray(X_raw) - ckpt["feat_mean"]) / ckpt["feat_std"]
    pred = [classes[i] for i in train.predict(np, X, ckpt["W"], ckpt["b"])]
    hits = np.array([p == y for p, y in zip(pred, labels, strict=True)])
    known = set(classes)
    unseen = sum(1 for y in labels if y not in known)
    return float(np.mean(hits)), len(labels), unseen


def compute_gaps(
    metrics: Any, report: Any, split: Any, samples_dir: Path, ckpt: dict[str, Any]
) -> dict[str, Any]:
    """Both val-minus-holdout gaps plus their inputs, rounded to 6."""
    if not isinstance(metrics, dict):
        _fail("metrics must be a JSON object")
    if not isinstance(report, dict):
        _fail("report must be a JSON object")
    if not isinstance(split, dict):
        _fail("split must be a JSON object of sample_id -> split")
    val_accuracy = _unit(metrics.get("val_accuracy"), "metrics val_accuracy")
    holdout_f1 = _unit(report.get("f1"), "report f1")
    iou = _iou_threshold(report.get("iou_threshold"))
    if metrics.get("classes") != [str(c) for c in ckpt["classes"]]:
        _fail("model classes do not match metrics classes: not the same train.py run")

    holdout = load_split_samples(split, samples_dir, "holdout")
    holdout_acc, n_holdout_windows, n_unseen = window_accuracy(ckpt, holdout)

    val_ids = sorted(sid for sid, part in split.items() if part == "val")
    gt = evaluate.load_ground_truth(samples_dir, val_ids)
    pred = _val_predictions(metrics.get("val_segments"), val_ids)
    val_f1 = evaluate.compute_metrics(gt, pred, iou)["f1"]
    val_f1 = _unit(val_f1, "computed val f1")

    return {
        "promotion_metric": evaluate.r6(val_accuracy - holdout_acc),
        "segment_f1": evaluate.r6(val_f1 - holdout_f1),
        "val_accuracy": evaluate.r6(val_accuracy),
        "val_f1": evaluate.r6(val_f1),
        "holdout_window_accuracy": evaluate.r6(holdout_acc),
        "holdout_f1": evaluate.r6(holdout_f1),
        "n_holdout_windows": n_holdout_windows,
        "n_holdout_windows_unseen_label": n_unseen,
    }


def _over_threshold(gaps: dict[str, Any]) -> bool:
    return (
        abs(gaps["promotion_metric"]) > GAP_THRESHOLD
        or abs(gaps["segment_f1"]) > GAP_THRESHOLD
    )


def _gap_value(gaps: dict[str, Any], key: str, where: str) -> float:
    """A recorded gap value, validated — never silently defaulted.

    Gaps are differences of [0, 1] metrics, so they live in [-1, 1].
    A missing key or a non-numeric value is a malformed registry, not a
    zero gap: failing closed (exit 2) beats silently resetting the
    streak on a corrupt log.
    """
    value = gaps.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not -1.0 <= value <= 1.0
    ):
        _fail(f"{where} gap {key!r} must be a number in [-1, 1], got {value!r}")
    return float(value)


def _record_run_id(rec: dict[str, Any], where: str) -> str:
    run_id = rec.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        _fail(f"{where} has no run_id: the streak counts runs, not records")
    return run_id


def replay_streak(records: list[Any]) -> int:
    """Streak over prior leakage_gap records, in log order.

    A promoted record's gaps are validated on replay: a non-numeric or
    missing gap fails closed (GapError -> exit 2) instead of raising
    TypeError or silently reading as 0. Every record needs a run_id, and
    no run may have two records: either would let one run count twice.
    """
    streak = 0
    seen: set[str] = set()
    for i, rec in enumerate(records):
        if not isinstance(rec, dict) or rec.get("kind") != "leakage_gap":
            continue
        where = f"registry record #{i}"
        run_id = _record_run_id(rec, where)
        if run_id in seen:
            _fail(f"{where} repeats run_id {run_id!r}: one record per run")
        seen.add(run_id)
        promotion = rec.get("promotion")
        gaps = rec.get("val_minus_holdout")
        if promotion == "bootstrap":
            continue
        if promotion == "promote":
            if not isinstance(gaps, dict):
                _fail(f"registry record #{i} has no val_minus_holdout dict")
            # Validate both before comparing: `or` would short-circuit
            # past the second validation when the first gap is over.
            promotion_gap = _gap_value(gaps, "promotion_metric", where)
            segment_gap = _gap_value(gaps, "segment_f1", where)
            over = (
                abs(promotion_gap) > GAP_THRESHOLD or abs(segment_gap) > GAP_THRESHOLD
            )
            streak = streak + 1 if over else 0
        else:
            streak = 0
    return streak


def run_id_for(fingerprint: dict[str, Any]) -> str:
    """TRAINING_RUN_ID when set, else a hash of the run's inputs and
    results (module docstring, "Run id")."""
    if os.environ.get(RUN_ID_ENV):
        return os.environ[RUN_ID_ENV]
    blob = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _recorded(registry: list[Any], run_id: str) -> int | None:
    """Index of this run's leakage_gap record in the registry, if any."""
    for i, rec in enumerate(registry):
        if (
            isinstance(rec, dict)
            and rec.get("kind") == "leakage_gap"
            and rec.get("run_id") == run_id
        ):
            return i
    return None


def run(
    metrics_path: Path,
    model_path: Path,
    report_path: Path,
    split_path: Path,
    samples_dir: Path,
    registry_path: Path,
    promotion: str,
) -> tuple[dict[str, Any], bool]:
    """Write the gaps; returns (this run's record, whether it was appended)."""
    metrics = _load_json(metrics_path)
    report = _load_json(report_path)
    split = _load_json(split_path)
    if not isinstance(split, dict):
        _fail("split must be a JSON object of sample_id -> split")
    for part in split.values():
        if part not in evaluate.SPLITS:
            _fail(f"split has unknown part {part!r}")
    split_sha = evaluate.sha256_file(split_path)
    if not isinstance(metrics, dict) or metrics.get("split_sha256") != split_sha:
        _fail("metrics split_sha256 does not match split.json")
    inputs = report.get("inputs") if isinstance(report, dict) else None
    if not isinstance(inputs, dict) or inputs.get("split_sha256") != split_sha:
        _fail("report inputs.split_sha256 does not match split.json")
    ckpt = train.load_checkpoint(model_path)

    gaps = compute_gaps(metrics, report, split, samples_dir, ckpt)
    run_id = run_id_for(
        {
            "split_sha256": split_sha,
            "metrics_sha256": evaluate.sha256_file(metrics_path),
            "predictions_sha256": inputs.get("predictions_sha256"),
            "promotion": promotion,
            "val_minus_holdout": gaps,
        }
    )

    registry = load_registry(registry_path)
    streak = replay_streak(registry)  # validates the whole log, run ids included
    existing = _recorded(registry, run_id)
    if existing is not None:
        # A re-run takes the streak as of its own record, not after it.
        streak = replay_streak(registry[:existing])
    if promotion == "promote":
        streak = streak + 1 if _over_threshold(gaps) else 0
    elif promotion == "bootstrap":
        pass  # neither a promoted run nor a break
    else:
        streak = 0
    flagged = streak >= FLAG_STREAK

    record: dict[str, Any] = {
        "kind": "leakage_gap",
        "decided_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "run_id": run_id,
        "promotion": promotion,
        "val_minus_holdout": gaps,
        "streak_promoted_runs_over_threshold": streak,
        "flagged": flagged,
    }
    if existing is not None:
        old = registry[existing]
        compared = (
            "promotion",
            "val_minus_holdout",
            "streak_promoted_runs_over_threshold",
            "flagged",
        )
        if any(old.get(key) != record[key] for key in compared):
            _fail(
                f"registry already records run {run_id!r} with different "
                "results: one run id names one run"
            )

    report["val_minus_holdout"] = {
        **gaps,
        "streak_promoted_runs_over_threshold": streak,
        "flagged": flagged,
    }
    tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(report_path)

    if existing is not None:
        return registry[existing], False
    registry.append(record)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = registry_path.with_suffix(registry_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(registry_path)
    return record, True


def load_registry(path: Path) -> list[Any]:
    """promote.py's registry read, failing with GapError (exit 2)."""
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        _fail(f"cannot read registry file {path}: {e}")
    if not isinstance(data, list):
        _fail(f"registry file {path} must contain a JSON array")
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Val-minus-holdout leakage gap.")
    ap.add_argument("--metrics", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--samples-dir", required=True, type=Path)
    ap.add_argument("--registry", required=True, type=Path)
    ap.add_argument("--promotion", required=True, choices=PROMOTIONS)
    args = ap.parse_args(argv)
    try:
        record, appended = run(
            args.metrics,
            args.model,
            args.report,
            args.split,
            args.samples_dir,
            args.registry,
            args.promotion,
        )
    except (GapError, ValueError, TypeError) as e:
        print(f"leakage_gap.py: error: {e}", file=sys.stderr)
        return EXIT_ERROR
    gaps = record["val_minus_holdout"]
    print(
        f"val-minus-holdout: promotion {gaps['promotion_metric']:+.4f}, "
        f"segment_f1 {gaps['segment_f1']:+.4f}, "
        f"streak {record['streak_promoted_runs_over_threshold']}, "
        f"flagged {record['flagged']}"
        + ("" if appended else " (run already recorded; registry unchanged)")
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
