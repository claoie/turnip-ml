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
--metrics      metrics.json from Training/train.py (its train_name_counts).
--split        split.json both were computed on. Checked against the
               report's inputs.split_sha256 and the metrics' split_sha256,
               so the counts and the metrics always describe the same run.
--out          name_metrics.json path.

Definitions
-----------
- holdout_count: holdout ground-truth segments carrying the name (the
  report's per_name `support`).
- train_count: train-split labeled segments carrying the name, as
  train.py counted them for training (TRAINING_DESIGN.md §2.7).
  fetch_dataset.py writes one trick entry per (clip, canonical name), so
  this is the name's training clips.
- precision, recall: the report's per-name values (a name match is
  required, see evaluate.py); both null when holdout_count < 5
  (MIN_HOLDOUT_COUNT), a sample too small to support a number. precision
  is also null when the model never predicted the name: 0 of 0 is
  undefined, not 0%.
- Names: every name in the train split or in the report (holdout ground
  truth or predictions). "background" is the absence of a segment, never
  a name. A name the model trained on but the holdout lacks gets
  holdout_count 0 and null metrics: a trick it is still learning.

Exit codes: 0 = written, 2 = input error (fail closed): unreadable or
malformed inputs, or a report or metrics file from a different split.

Determinism: same inputs -> byte-identical file (sorted keys, floats
rounded to 6 decimals, no timestamps). Stdout and stderr carry counts
only: no trick names (unmapped ones are a contributor's free text) and no
sample ids.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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
    for i, (name, entry) in enumerate(sorted(report["per_name"].items())):
        where = f"holdout report per_name entry #{i}"
        if not isinstance(name, str) or not name or not isinstance(entry, dict):
            raise ValueError(f"{where} must be an object under a non-empty name")
        per_name[name] = {
            "support": _count(entry.get("support"), f"{where} support"),
            "predicted": _count(entry.get("predicted"), f"{where} predicted"),
            "precision": _unit(entry.get("precision"), f"{where} precision"),
            "recall": _unit(entry.get("recall"), f"{where} recall"),
        }
    return per_name


def train_counts(metrics: Any) -> dict[str, int]:
    """train.py's per-name train counts, validated."""
    counts = metrics.get("train_name_counts") if isinstance(metrics, dict) else None
    if not isinstance(counts, dict):
        raise TypeError("metrics.json needs train_name_counts (train.py)")
    for i, (name, count) in enumerate(sorted(counts.items())):
        if not isinstance(name, str) or not name:
            raise ValueError(f"metrics.json train_name_counts entry #{i}: no name")
        _count(count, f"metrics.json train_name_counts entry #{i}")
    return counts


def build_name_metrics(
    per_name: dict[str, dict[str, Any]], train_counts: dict[str, int]
) -> dict[str, dict[str, Any]]:
    """Pure: one manifest entry per name, sorted."""
    out: dict[str, dict[str, Any]] = {}
    empty = {"support": 0, "predicted": 0, "precision": 0.0, "recall": 0.0}
    for name in sorted(set(per_name) | set(train_counts)):
        if name == "background":
            raise ValueError('"background" is not a trick name')
        holdout = per_name.get(name, empty)
        scored = holdout["support"] >= MIN_HOLDOUT_COUNT
        out[name] = {
            "precision": (
                evaluate.r6(holdout["precision"])
                if scored and holdout["predicted"] > 0
                else None
            ),
            "recall": evaluate.r6(holdout["recall"]) if scored else None,
            "holdout_count": holdout["support"],
            "train_count": train_counts.get(name, 0),
        }
    return out


ENTRY_KEYS = {"precision", "recall", "holdout_count", "train_count"}


def validate(data: Any) -> dict[str, dict[str, Any]]:
    """A name_metrics.json document as the manifest ships it, checked."""
    if not isinstance(data, dict) or not data:
        raise ValueError("name_metrics must be a non-empty object of name -> metrics")
    for i, (name, entry) in enumerate(sorted(data.items())):
        where = f"name_metrics entry #{i}"
        if not name or not isinstance(entry, dict) or set(entry) != ENTRY_KEYS:
            raise ValueError(f"{where} needs exactly {sorted(ENTRY_KEYS)}")
        _count(entry["holdout_count"], f"{where} holdout_count")
        _count(entry["train_count"], f"{where} train_count")
        for key in ("precision", "recall"):
            if entry[key] is not None:
                _unit(entry[key], f"{where} {key}")
    return data


def run(report_path: Path, metrics_path: Path, split_path: Path) -> dict[str, Any]:
    report = evaluate.load_json(report_path)
    metrics = evaluate.load_json(metrics_path)
    split_sha = evaluate.sha256_file(split_path)
    inputs = report.get("inputs") if isinstance(report, dict) else None
    if not isinstance(inputs, dict) or inputs.get("split_sha256") != split_sha:
        raise ValueError(
            f"{report_path} was not computed on {split_path} (split_sha256 differs)"
        )
    if not isinstance(metrics, dict) or metrics.get("split_sha256") != split_sha:
        raise ValueError(
            f"{metrics_path} was not trained on {split_path} (split_sha256 differs)"
        )
    return build_name_metrics(holdout_per_name(report), train_counts(metrics))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--report", required=True, type=Path, help="holdout_report.json")
    p.add_argument("--metrics", required=True, type=Path, help="train metrics.json")
    p.add_argument("--split", required=True, type=Path, help="split.json")
    p.add_argument("--out", required=True, type=Path, help="name_metrics.json path")
    args = p.parse_args(argv)
    try:
        metrics = run(args.report, args.metrics, args.split)
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
