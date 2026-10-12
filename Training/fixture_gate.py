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
--ingested    JSON array of the sources ingested since the last watermark,
              one {"source_id", "user_id"} object per source. These are the
              quarantine candidates; attribution is the source's owner.
--out         run directory; quarantine.json is written here.
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

Decision rule
-------------
    composite  = mean(segment_iou, name_accuracy)
    regression iff challenger_composite < champion_composite - tolerance
On regression:
  - quarantine.json: one row per ingested source, shaped like a
    `data_quarantine` insert (source_id, user_id, reason, metrics); the
    farm assigns id and quarantined_at.
  - Discord alert (webhook from TURNIP_DISCORD_WEBHOOK_URL) with the
    metrics delta and the affected source and user ids.
  - Exit 3 so the pipeline skips promotion.
The quarantine rows are written before the alert is attempted, and an
alert that cannot be delivered (webhook unset or request failed) does not
change the decision: promotion is still skipped and the failure is
recorded as `alert_status`.

Exit codes: 0 = pass (or overridden), 3 = regression, 2 = input/usage
error (fail closed).

Determinism
-----------
The decision, the alert text and quarantine.json are pure functions of the
inputs. quarantine.json is always written (an empty array when nothing is
quarantined) with sorted keys and rows sorted by source_id, so a stale
file from an earlier run is never left behind. The registry record carries
a wall-clock `decided_at` (a registry is a log, not a reproducible
artifact). Floats are rounded to 6 decimals in every output.

Fail closed
-----------
Missing/unreadable files, invalid JSON (including NaN/Infinity), a missing,
non-numeric or out-of-range score, mismatched fixture sets, a malformed
ingested-source list, a non-finite or negative tolerance, an empty override
reason, or a malformed existing registry are errors, never silently a pass.
All inputs are validated before any side effect (file write, alert).
"""

from __future__ import annotations

import argparse
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


def load_json(path: Path, what: str) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh, parse_constant=_reject_constant)
    except (OSError, ValueError) as e:
        # ValueError covers JSONDecodeError and UnicodeDecodeError.
        raise GateError(f"cannot read {what} JSON from {path}: {e}") from e


def load_scores(path: Path) -> dict[str, Any]:
    data = load_json(path, "fixture scores")
    if not isinstance(data, dict):
        raise GateError(f"fixture scores file {path} must contain a JSON object")
    fixture_set = data.get("fixture_set")
    if not isinstance(fixture_set, str) or not fixture_set:
        raise GateError(f"fixture_set in {path} must be a non-empty string")
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


def load_ingested(path: Path) -> list[dict[str, str]]:
    """Validated ingested sources, deduplicated and sorted by source_id."""
    data = load_json(path, "ingested sources")
    if not isinstance(data, list):
        raise GateError(f"ingested sources file {path} must contain a JSON array")
    owners: dict[str, str] = {}
    for i, entry in enumerate(data):
        if not isinstance(entry, dict) or set(entry) != {"source_id", "user_id"}:
            raise GateError(
                f"ingested entry {i} in {path} must be an object with exactly "
                f"source_id and user_id: {entry!r}"
            )
        source_id, user_id = entry["source_id"], entry["user_id"]
        if not isinstance(source_id, str) or not SOURCE_ID_RE.fullmatch(source_id):
            raise GateError(
                f"ingested entry {i} in {path}: source_id must be 64 lowercase "
                f"hex chars: {source_id!r}"
            )
        if not isinstance(user_id, str) or not user_id:
            raise GateError(
                f"ingested entry {i} in {path}: user_id must be a non-empty "
                f"string: {user_id!r}"
            )
        # A source has exactly one owner; two owners means a corrupt list.
        if owners.setdefault(source_id, user_id) != user_id:
            raise GateError(
                f"source {source_id} listed with two owners in {path}: "
                f"{owners[source_id]!r} and {user_id!r}"
            )
    return [{"source_id": s, "user_id": owners[s]} for s in sorted(owners)]


def load_registry(path: Path) -> list[Any]:
    if not path.exists():
        return []
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
    ingested: list[dict[str, str]], reason: str, metrics: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "source_id": src["source_id"],
            "user_id": src["user_id"],
            "reason": reason,
            "metrics": metrics,
        }
        for src in ingested
    ]


def _preview(items: list[str]) -> str:
    if not items:
        return "(none)"
    shown = ", ".join(items[:ALERT_LIST_PREVIEW])
    hidden = len(items) - ALERT_LIST_PREVIEW
    return f"{shown}, ... +{hidden} more" if hidden > 0 else shown


def alert_content(metrics: dict[str, Any], ingested: list[dict[str, str]]) -> str:
    champ, chall = metrics["champion"], metrics["challenger"]
    sources = [s["source_id"] for s in ingested]
    users = sorted({s["user_id"] for s in ingested})
    lines = [
        "**turnip-ml: fixture regression, promotion skipped**",
        f"fixture set: {metrics['fixture_set']}",
        (
            f"composite: champion {champ['composite']:.6f} -> challenger "
            f"{chall['composite']:.6f} (delta {metrics['delta']:+.6f}, "
            f"tolerance {metrics['tolerance']:.6f})"
        ),
        *(
            f"{n}: {champ[n]:.6f} -> {chall[n]:.6f} (delta {chall[n] - champ[n]:+.6f})"
            for n in COMPONENTS
        ),
        (
            f"quarantined {len(sources)} source(s) from {len(users)} user(s); "
            f"full list in {QUARANTINE_FILE}"
        ),
        f"sources: {_preview(sources)}",
        f"users: {_preview(users)}",
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
        # No mentions: the content carries user-supplied ids, never pings.
        payload = {"content": content, "allowed_mentions": {"parse": []}}
        try:
            (send or send_discord)(webhook_url, payload)
            return "sent"
        except (OSError, ValueError) as e:
            # URLError/HTTPError/timeouts are OSError. The decision stands
            # regardless: quarantine and the skipped promotion must not
            # depend on Discord being reachable.
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


def write_json_atomic(path: Path, data: Any) -> None:
    # A crash mid-write must not leave a malformed file behind.
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True, allow_nan=False)
            fh.write("\n")
        tmp.replace(path)
    except OSError as e:
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
        print(f"fixture_gate: regression overridden: {override}", file=sys.stderr)
    else:
        decision = "pass"
        rows = []
    write_json_atomic(out_dir / QUARANTINE_FILE, rows)

    alert_status = "not-needed"
    if decision == "regression":
        alert_status = deliver_alert(
            alert_content(metrics, ingested), webhook_url, send
        )

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
        "alert_status": alert_status,
    }
    registry.append(record)
    write_json_atomic(registry_path, registry)

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
        help="JSON array of {source_id, user_id} ingested since the watermark",
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
