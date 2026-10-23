#!/usr/bin/env python3
"""Fixture regression gate (anti-poisoning).

Runs after a challenger is trained and before the holdout evaluation and
the champion/challenger promotion gate. It compares the challenger's
fixture scores against the champion's recorded fixture scores; a
regression means the night's new data is suspect, so the gate alerts,
quarantines that night's ingested sources, and skips promotion.

Inputs
------
--champion    fixture scores recorded for the current champion.
--challenger  fixture scores for the challenger, from the same fixture set.
--ingested    JSON array of the source ids ingested since the last
              watermark: the quarantine candidates. Plain id strings, so
              nothing account-shaped can ride along (training data carries
              no account identifier).
--out         fresh per-run directory; quarantine.json is written here.
--tolerance   absolute slack on the composite score. Default: 0.01.
--registry    model-registry stand-in: a JSON array file each decision
              record is appended to. Default: Training/registry.json.
--override    maintainer-only escape hatch with a mandatory reason: a
              regression is recorded but neither alerts, quarantines, nor
              blocks promotion. CLI-only by design: the nightly config has
              no way to set it.

Fixture scores file:
    {"fixture_set": "<id>", "segment_iou": 0.82, "name_accuracy": 0.91}
`fixture_set` identifies the fixture set the scores were computed on
(e.g. a digest of the fixtures directory); both files must match, since
scores from different fixture sets are not comparable. Both components
are in [0, 1]. Extra keys are kept in the record but not compared.

`fixture_set` is shown in the alert, so it must be printable with no
whitespace or backticks: it renders inside a Discord code span and cannot
inject lines, formatting or links.

Decision rule
-------------
    composite  = mean(segment_iou, name_accuracy)
    regression iff challenger_composite < champion_composite - tolerance
On regression:
  - quarantine.json: one row per ingested source, shaped like a
    `data_quarantine` insert (source_id, reason, metrics); the farm
    assigns id and quarantined_at. No account id: who uploaded a source
    is read farm-side from source_attribution while it exists.
    report_quarantine.py sends the rows (the workflow's farm job only).
  - Discord alert (webhook from TURNIP_DISCORD_WEBHOOK_URL) with the
    metrics delta and the affected source ids. Never account ids: a
    Discord message outlives the farm's 30-day attribution window.
  - Exit 3 so the pipeline skips promotion.
quarantine.json and the registry record (with `alert_status: "pending"`)
are written before the alert is attempted; the record is then updated
with the outcome. The alert is best-effort: an alert that cannot be
delivered (webhook unset or invalid, request failed) never changes the
decision, promotion is still skipped, and the failure is recorded as
`alert_status`. A record left at "pending" means the run died during or
right after alerting.

Exit codes: 0 = pass (or overridden), 3 = regression, 2 = input/usage
error (fail closed).

Determinism
-----------
The decision, the alert text and quarantine.json are pure functions of the
inputs. quarantine.json is written on every decided run (an empty array
when nothing is quarantined) with sorted keys and rows sorted by
source_id; a run that exits 2 writes nothing, so --out must be a fresh
directory per run for its quarantine.json to be trusted. The registry
record carries a wall-clock `decided_at` (a registry is a log, not a
reproducible artifact). Computed floats (components, composites, delta,
tolerance in `metrics`) are rounded to 6 decimals; `champion_scores` and
`challenger_scores` are recorded verbatim as read.

Fail closed
-----------
Missing/unreadable files, invalid JSON (including NaN, Infinity and numbers
that overflow to infinity), a missing, non-numeric or out-of-range score,
mismatched fixture sets, a malformed ingested-source list, a non-finite or
negative tolerance, an empty override reason, or a malformed existing
registry are errors, never silently a pass. All inputs are validated and
every output is serialized before any side effect (file write, alert).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXIT_PASS = 0
EXIT_REGRESSION = 3
EXIT_ERROR = 2

DEFAULT_TOLERANCE = 0.01
DEFAULT_REGISTRY = "Training/registry.json"
QUARANTINE_FILE = "quarantine.json"
WEBHOOK_ENV = "TURNIP_DISCORD_WEBHOOK_URL"

COMPONENTS = ("segment_iou", "name_accuracy")
SOURCE_ID_RE = re.compile(r"[0-9a-f]{64}")
# Safe inside a Discord code span: no whitespace (no injected lines), no
# backticks (no breaking out of the span).
IDENTIFIER_RE = re.compile(r"[^\s`]+")

# Discord rejects message content over 2000 characters; the full lists
# live in quarantine.json, the alert only previews them.
DISCORD_MAX_CONTENT = 2000
ALERT_LIST_PREVIEW = 10
WEBHOOK_TIMEOUT_S = 10.0

AlertSender = Callable[[str, dict[str, Any]], None]


class GateError(Exception):
    """Input/usage error: the gate refuses to decide rather than guess."""


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name} is not valid JSON")


def _finite_float(text: str) -> float:
    # json parses an overflowing literal such as 1e400 to inf without
    # going through parse_constant.
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"number {text} overflows to {value}")
    return value


def load_json(path: Path, what: str) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(
                fh, parse_constant=_reject_constant, parse_float=_finite_float
            )
    except (OSError, ValueError, RecursionError) as e:
        # ValueError covers JSONDecodeError and UnicodeDecodeError;
        # RecursionError is pathologically deep nesting.
        raise GateError(f"cannot read {what} JSON from {path}: {e}") from e


def is_identifier(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.isprintable()
        and IDENTIFIER_RE.fullmatch(value) is not None
    )


def load_scores(path: Path) -> dict[str, Any]:
    data = load_json(path, "fixture scores")
    if not isinstance(data, dict):
        raise GateError(f"fixture scores file {path} must contain a JSON object")
    fixture_set = data.get("fixture_set")
    if not is_identifier(fixture_set):
        raise GateError(
            f"fixture_set in {path} must be a non-empty printable string without "
            f"whitespace or backticks: {fixture_set!r}"
        )
    for name in COMPONENTS:
        if name not in data:
            raise GateError(f"score {name!r} missing from {path}")
        value = data[name]
        # bool is a subclass of int: True/False are not scores.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GateError(f"score {name!r} in {path} is not a number: {value!r}")
        # Compared without float() so a huge int cannot overflow.
        if not 0 <= value <= 1:
            raise GateError(f"score {name!r} in {path} is not in [0, 1]: {value!r}")
    return data


def composite(scores: dict[str, Any]) -> float:
    return sum(float(scores[name]) for name in COMPONENTS) / len(COMPONENTS)


def load_ingested(path: Path) -> list[str]:
    """Validated ingested source ids, deduplicated and sorted."""
    data = load_json(path, "ingested sources")
    if not isinstance(data, list):
        raise GateError(f"ingested sources file {path} must contain a JSON array")
    for i, source_id in enumerate(data):
        # Only bare ids: an object entry could carry an account id along.
        if not isinstance(source_id, str) or not SOURCE_ID_RE.fullmatch(source_id):
            raise GateError(
                f"ingested entry {i} in {path} must be a source id of 64 "
                "lowercase hex chars"
            )
    return sorted(set(data))


def load_registry(path: Path) -> list[Any]:
    try:
        if not path.exists():
            return []
    except OSError as e:  # e.g. an unsearchable parent directory
        raise GateError(f"cannot access registry file {path}: {e}") from e
    data = load_json(path, "registry")
    if not isinstance(data, list):
        raise GateError(f"registry file {path} must contain a JSON array")
    return data


def decide(champion: float, challenger: float, tolerance: float) -> tuple[bool, str]:
    """Pure decision function: (is_regression, reason)."""
    bound = f"champion {champion:.6f} - tolerance {tolerance:.6f}"
    if challenger < champion - tolerance:
        return (
            True,
            f"fixture regression: challenger composite {challenger:.6f} < {bound}",
        )
    return (
        False,
        f"no fixture regression: challenger composite {challenger:.6f} >= {bound}",
    )


def metrics_snapshot(
    champion: dict[str, Any], challenger: dict[str, Any], tolerance: float
) -> dict[str, Any]:
    """The regression metrics stored on each quarantine row and alerted."""
    champion_c, challenger_c = composite(champion), composite(challenger)
    return {
        "fixture_set": challenger["fixture_set"],
        "tolerance": round(tolerance, 6),
        "champion": {
            **{n: round(float(champion[n]), 6) for n in COMPONENTS},
            "composite": round(champion_c, 6),
        },
        "challenger": {
            **{n: round(float(challenger[n]), 6) for n in COMPONENTS},
            "composite": round(challenger_c, 6),
        },
        "delta": round(challenger_c - champion_c, 6),
    }


def quarantine_rows(
    ingested: list[str], reason: str, metrics: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {"source_id": source_id, "reason": reason, "metrics": metrics}
        for source_id in ingested
    ]


def _preview(items: list[str]) -> str:
    if not items:
        return "(none)"
    shown = ", ".join(f"`{item}`" for item in items[:ALERT_LIST_PREVIEW])
    hidden = len(items) - ALERT_LIST_PREVIEW
    return f"{shown}, ... +{hidden} more" if hidden > 0 else shown


def alert_content(metrics: dict[str, Any], ingested: list[str]) -> str:
    champ, chall = metrics["champion"], metrics["challenger"]
    lines = [
        "**turnip-ml: fixture regression, promotion skipped**",
        f"fixture set: `{metrics['fixture_set']}`",
        (
            f"composite: champion {champ['composite']:.6f} -> challenger "
            f"{chall['composite']:.6f} (delta {metrics['delta']:+.6f}, "
            f"tolerance {metrics['tolerance']:.6f})"
        ),
        *(
            f"{n}: {champ[n]:.6f} -> {chall[n]:.6f} (delta {chall[n] - champ[n]:+.6f})"
            for n in COMPONENTS
        ),
        f"quarantined {len(ingested)} source(s); full list in {QUARANTINE_FILE}",
        f"sources: {_preview(ingested)}",
    ]
    content = "\n".join(lines)
    if len(content) > DISCORD_MAX_CONTENT:
        content = content[: DISCORD_MAX_CONTENT - 3] + "..."
    return content


def send_discord(url: str, payload: dict[str, Any]) -> None:
    """POST a webhook message. Raises on any non-2xx or transport error."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, sort_keys=True).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            # Discord refuses urllib's default User-Agent.
            "User-Agent": "turnip-ml-fixture-gate",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=WEBHOOK_TIMEOUT_S):
        pass  # urlopen raises HTTPError on non-2xx; Discord answers 204.


def deliver_alert(
    content: str, webhook_url: str | None, send: AlertSender | None
) -> str:
    """Send the alert; returns the `alert_status` recorded in the registry."""
    if not webhook_url:
        status = "not-configured"
    elif not webhook_url.startswith("https://"):
        status = "failed: webhook URL is not https"
    else:
        # No mentions: the alert informs, it never pings anyone.
        payload = {"content": content, "allowed_mentions": {"parse": []}}
        try:
            (send or send_discord)(webhook_url, payload)
            return "sent"
        except Exception as e:  # noqa: BLE001
            # Deliberately blind. The alert is best-effort: the decision must
            # never depend on Discord, and an escaping error (e.g. the
            # http.client.InvalidURL a malformed URL raises) would print the
            # webhook URL, token included, in its traceback.
            status = f"failed: {_describe(e)}"
    print(f"fixture_gate: alert not delivered ({status})", file=sys.stderr)
    return status


def _describe(e: Exception) -> str:
    # Never str(e) wholesale: some urllib errors echo the URL, and the
    # webhook URL embeds its secret token.
    if isinstance(e, urllib.error.HTTPError):
        return f"HTTP {e.code}"
    if isinstance(e, urllib.error.URLError) and isinstance(e.reason, str):
        return f"URLError: {e.reason}"
    return type(e).__name__


def dump_json(data: Any, path: Path) -> str:
    try:
        return json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (ValueError, RecursionError) as e:
        # ValueError: allow_nan=False refusing a non-finite float;
        # RecursionError: nesting the encoder cannot handle.
        raise GateError(f"cannot serialize {path}: {e}") from e


def write_text_atomic(path: Path, text: str) -> None:
    # A crash mid-write must not leave a malformed file behind.
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise GateError(f"cannot write {path}: {e}") from e


def run(
    champion_path: Path,
    challenger_path: Path,
    ingested_path: Path,
    out_dir: Path,
    tolerance: float,
    registry_path: Path,
    override: str | None = None,
    webhook_url: str | None = None,
    send: AlertSender | None = None,
) -> tuple[dict[str, Any], int]:
    """Evaluate the gate. Returns (decision_record, exit_code)."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise GateError(f"tolerance must be a finite number >= 0, got {tolerance}")
    if override is not None and not override.strip():
        raise GateError("--override needs a non-empty reason")
    champion = load_scores(champion_path)
    challenger = load_scores(challenger_path)
    if champion["fixture_set"] != challenger["fixture_set"]:
        raise GateError(
            f"fixture sets differ: champion {champion['fixture_set']!r}, "
            f"challenger {challenger['fixture_set']!r}; re-score the champion "
            "on the current fixture set"
        )
    ingested = load_ingested(ingested_path)
    registry = load_registry(registry_path)

    regressed, reason = decide(composite(champion), composite(challenger), tolerance)
    metrics = metrics_snapshot(champion, challenger, tolerance)

    if regressed and override is None:
        decision = "regression"
        rows = quarantine_rows(ingested, reason, metrics)
    elif regressed:
        decision = "override"
        rows = []
    else:
        decision = "pass"
        rows = []

    record = {
        "gate": "fixture-regression",
        "decided_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "decision": decision,
        "reason": reason,
        "override_reason": override,
        "metrics": metrics,
        "champion_scores": champion,
        "challenger_scores": challenger,
        "quarantined_sources": len(rows),
        "alert_status": "pending" if decision == "regression" else "not-needed",
    }
    # Serialize everything up front: from here on only I/O can fail.
    quarantine_path = out_dir / QUARANTINE_FILE
    quarantine_text = dump_json(rows, quarantine_path)
    registry_text = dump_json([*registry, record], registry_path)

    if decision == "override":
        print(f"fixture_gate: regression overridden: {override}", file=sys.stderr)
    # Durable state before the alert: if either write fails, nothing was sent.
    write_text_atomic(quarantine_path, quarantine_text)
    write_text_atomic(registry_path, registry_text)
    if decision == "regression":
        record["alert_status"] = deliver_alert(
            alert_content(metrics, ingested), webhook_url, send
        )
        write_text_atomic(registry_path, dump_json([*registry, record], registry_path))

    return record, EXIT_REGRESSION if decision == "regression" else EXIT_PASS


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fixture regression gate.")
    p.add_argument(
        "--champion", required=True, type=Path, help="champion fixture scores JSON"
    )
    p.add_argument(
        "--challenger",
        required=True,
        type=Path,
        help="challenger fixture scores JSON",
    )
    p.add_argument(
        "--ingested",
        required=True,
        type=Path,
        help="JSON array of source ids ingested since the watermark",
    )
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        help=f"run directory to write {QUARANTINE_FILE} into",
    )
    p.add_argument(
        "--tolerance",
        default=DEFAULT_TOLERANCE,
        type=float,
        help="absolute composite-score slack (default: %(default)s)",
    )
    p.add_argument(
        "--registry",
        default=Path(DEFAULT_REGISTRY),
        type=Path,
        help="registry file to append the decision to (default: %(default)s)",
    )
    p.add_argument(
        "--override",
        metavar="REASON",
        help="maintainer override: let a regression through, recording REASON",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None, send: AlertSender | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        record, code = run(
            args.champion,
            args.challenger,
            args.ingested,
            args.out,
            args.tolerance,
            args.registry,
            override=args.override,
            webhook_url=os.environ.get(WEBHOOK_ENV),
            send=send,
        )
    except GateError as e:
        print(f"fixture_gate: error: {e}", file=sys.stderr)
        return EXIT_ERROR
    print(json.dumps(record, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
