#!/usr/bin/env python3
"""Champion/challenger promotion gate.

ML pipeline step 6: promote the challenger only if it beats the current
champion by >=1% on validation; otherwise archive it and try next cycle.
The decision (metrics, timestamps) is recorded in the model registry.

Inputs
------
--champion    metrics.json from Training/train.py for the current champion.
--challenger  metrics.json from Training/train.py for the challenger.
--metric      validation metric key present in both files.
              Default: val_accuracy.
--threshold   minimum *relative* improvement required to promote, as a
              fraction. Default: 0.01 (">=1%").
--refresh     a scheduled from-scratch retrain (Training/retrain_schedule.py):
              the threshold is -REFRESH_TOLERANCE instead (see "Refresh").
              Not combinable with --threshold.
--registry    model-registry stand-in: a JSON array file each decision
              record is appended to. Default: Training/registry.json.
              This file is the audit trail the scheduled nightly trigger
              reads.

Decision rule
-------------
    improvement = (challenger - champion) / champion      (champion != 0)
Promote iff improvement >= threshold. When the champion's metric is 0,
promote iff the challenger's metric is > 0; 0-vs-0 is "no improvement"
and archives, except on a refresh (negative threshold), which needs no
improvement.

Refresh
-------
A scheduled retrain exists to replace the champion, whose weights still
carry data deleted since it was trained (TRAINING_DESIGN.md §2.1), so it
need not beat the champion: it promotes unless it is more than
REFRESH_TOLERANCE (2%) worse, relative. *(Judgment call on 2%.)* The
fixture gate runs before it as for every challenger. The record carries
"refresh": true.

Exit codes: 0 = promote, 3 = archive, 2 = input/usage error (fail closed).

Determinism
-----------
The decision is a pure function of the two metric values and the
threshold. The registry record carries a wall-clock `decided_at` timestamp
by design (a registry is a log, not a reproducible artifact); everything
else is derived from the inputs. Floats are rounded to 6 decimals in the
record.

Fail closed
-----------
Missing/unreadable files, invalid JSON, a missing or non-numeric
(non-finite) metric, a non-finite or negative threshold, or a malformed
existing registry file are errors, never silently treated as "archive"
or "promote".

Concurrency note: two gate runs appending to the same registry file at
once can interleave; the nightly scheduled trigger runs one gate at a
time, so this is not a concern for the current pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXIT_PROMOTE = 0
EXIT_ARCHIVE = 3
EXIT_ERROR = 2

DEFAULT_METRIC = "val_accuracy"
DEFAULT_THRESHOLD = 0.01
REFRESH_TOLERANCE = 0.02
DEFAULT_REGISTRY = "Training/registry.json"


class GateError(Exception):
    """Input/usage error: the gate refuses to decide rather than guess."""


def load_metrics(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        raise GateError(f"cannot read metrics JSON from {path}: {e}")
    if not isinstance(data, dict):
        raise GateError(f"metrics file {path} must contain a JSON object")
    return data


def metric_value(metrics: dict[str, Any], name: str, path: Path) -> float:
    if name not in metrics:
        raise GateError(f"metric {name!r} missing from {path}")
    value = metrics[name]
    # bool is a subclass of int: True/False are not metrics.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"metric {name!r} in {path} is not a number: {value!r}")
    if not math.isfinite(value):
        raise GateError(f"metric {name!r} in {path} is not finite: {value!r}")
    return float(value)


def decide(
    champion: float, challenger: float, threshold: float, metric: str
) -> tuple[str, float, str]:
    """Pure decision function.

    Returns (decision, improvement, reason) with decision "promote" or
    "archive". `improvement` is the relative improvement; for the
    0-vs-0 case it is reported as 0.0. `metric` is the validation metric
    name, used in the zero-champion reason strings so the registry record
    names the metric actually compared.
    """
    if champion == 0.0:
        if challenger > 0.0:
            return (
                "promote",
                float("inf"),
                f"champion {metric}=0, challenger={challenger:.6f} > 0",
            )
        if threshold < 0.0 and challenger == 0.0:
            # A refresh needs no improvement, only no loss.
            return (
                "promote",
                0.0,
                f"champion {metric}=0, challenger=0: no loss, threshold {threshold:.4%}",
            )
        return (
            "archive",
            0.0,
            f"champion {metric}=0, challenger={challenger:.6f}: no improvement",
        )
    improvement = (challenger - champion) / abs(champion)
    if improvement >= threshold:
        return (
            "promote",
            improvement,
            f"relative improvement {improvement:.4%} >= threshold {threshold:.4%}",
        )
    return (
        "archive",
        improvement,
        f"relative improvement {improvement:.4%} < threshold {threshold:.4%}",
    )


def load_registry(path: Path) -> list[Any]:
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        raise GateError(f"cannot read registry file {path}: {e}")
    if not isinstance(data, list):
        raise GateError(f"registry file {path} must contain a JSON array")
    return data


def run(
    champion_path: Path,
    challenger_path: Path,
    metric: str,
    threshold: float,
    registry_path: Path,
    refresh: bool = False,
) -> tuple[dict[str, Any], int]:
    """Evaluate the gate. Returns (decision_record, exit_code)."""
    champion_metrics = load_metrics(champion_path)
    challenger_metrics = load_metrics(challenger_path)
    champion_v = metric_value(champion_metrics, metric, champion_path)
    challenger_v = metric_value(challenger_metrics, metric, challenger_path)

    decision, improvement, reason = decide(champion_v, challenger_v, threshold, metric)
    record = {
        "decided_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "metric": metric,
        "threshold": round(threshold, 6),
        "champion_metric": round(champion_v, 6),
        "challenger_metric": round(challenger_v, 6),
        "improvement": (round(improvement, 6) if math.isfinite(improvement) else "inf"),
        "decision": decision,
        "reason": reason,
        "champion_metrics": champion_metrics,
        "challenger_metrics": challenger_metrics,
        "refresh": refresh,
    }

    registry = load_registry(registry_path)
    registry.append(record)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: a crash mid-write must not leave a malformed registry
    # behind (the next run fail-closes on it and the audit history is lost).
    tmp = registry_path.with_suffix(registry_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(registry_path)

    return record, EXIT_PROMOTE if decision == "promote" else EXIT_ARCHIVE


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Champion/challenger promotion gate.")
    p.add_argument(
        "--champion",
        required=True,
        type=Path,
        help="metrics.json for the current champion",
    )
    p.add_argument(
        "--challenger", required=True, type=Path, help="metrics.json for the challenger"
    )
    p.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=f"validation metric key (default: {DEFAULT_METRIC})",
    )
    p.add_argument(
        "--threshold",
        default=None,
        type=float,
        help=f"minimum relative improvement to promote (default: {DEFAULT_THRESHOLD})",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help=f"scheduled retrain: promote unless {REFRESH_TOLERANCE:.0%} worse",
    )
    p.add_argument(
        "--registry",
        default=Path(DEFAULT_REGISTRY),
        type=Path,
        help=f"registry file to append the decision to (default: {DEFAULT_REGISTRY})",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        if args.refresh:
            if args.threshold is not None:
                raise GateError("--refresh sets the threshold; drop --threshold")
            threshold = -REFRESH_TOLERANCE
        else:
            threshold = DEFAULT_THRESHOLD if args.threshold is None else args.threshold
            if not math.isfinite(threshold) or threshold < 0:
                raise GateError(
                    f"threshold must be a finite number >= 0, got {threshold}"
                )
        record, code = run(
            args.champion,
            args.challenger,
            args.metric,
            threshold,
            args.registry,
            args.refresh,
        )
    except GateError as e:
        print(f"promote: error: {e}", file=sys.stderr)
        return EXIT_ERROR
    print(json.dumps(record, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
