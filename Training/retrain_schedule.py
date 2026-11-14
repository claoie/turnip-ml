#!/usr/bin/env python3
"""Fixed retrain schedule: deleted data leaves the model within one cycle.

TRAINING_DESIGN.md §2.1, §8. Every dataset is a full pull, so data
deleted farm-side never reaches a new training run. But the nightly run
fine-tunes the champion, whose weights carry everything it was ever
trained on, and it only runs when more than 50 labels on new segments
come in. So
the farm job also retrains from scratch on a fixed schedule: once the
last full retrain is RETRAIN_INTERVAL_DAYS old, the next nightly run
trains from scratch whatever the label count, and promote.py --refresh
lets that challenger replace the champion unless it is clearly worse
(promote.py, "Refresh"). Until a from-scratch champion lands, every
nightly run tries again.

*(Judgment call on 30 days, TRAINING_DESIGN.md §10 Q12.)* Data deleted
right after a full retrain leaves the published model within about 30
days, plus the nights a refresh is refused by a gate.

Subcommands
-----------
due --state FILE --today YYYY-MM-DD
    Prints {"due": bool, "days_since_full_retrain": int | null}. Due when
    the state file is absent (no full retrain recorded: the champion's
    history is unknown) or the last full retrain is at least
    RETRAIN_INTERVAL_DAYS before today.
record --state FILE --today YYYY-MM-DD --outcome outcome.json
    Records today as the last full retrain when pipeline.sh's outcome
    says this run trained from scratch and its challenger became the
    champion (promotion "promote" or "bootstrap"); otherwise leaves the
    state as it is. Prints {"recorded": bool}.

The state file is {"last_full_retrain": "YYYY-MM-DD"}, written
atomically. The workflow passes today's UTC date, so the script reads no
clock.

Exit codes: 0 = done, 2 = input error (fail closed): a malformed date,
state or outcome, or a last full retrain after today.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

RETRAIN_INTERVAL_DAYS = 30
EXIT_OK = 0
EXIT_ERROR = 2


class ScheduleError(Exception):
    """Input error: the schedule refuses to guess."""


def parse_date(value: Any, where: str) -> dt.date:
    if not isinstance(value, str):
        raise ScheduleError(f"{where} must be a YYYY-MM-DD date")
    try:
        date = dt.date.fromisoformat(value)
    except ValueError as e:
        raise ScheduleError(f"{where} must be a YYYY-MM-DD date") from e
    if date.isoformat() != value:
        raise ScheduleError(f"{where} must be a YYYY-MM-DD date")
    return date


def _load_json(path: Path, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ScheduleError(f"cannot read {what} {path}: {e}") from e


def last_full_retrain(state_path: Path) -> dt.date | None:
    if not state_path.exists():
        return None
    state = _load_json(state_path, "state")
    if not isinstance(state, dict) or set(state) != {"last_full_retrain"}:
        raise ScheduleError(f'state {state_path} must be {{"last_full_retrain": ...}}')
    return parse_date(state["last_full_retrain"], "last_full_retrain")


def due(state_path: Path, today: dt.date) -> dict[str, Any]:
    last = last_full_retrain(state_path)
    if last is None:
        return {"due": True, "days_since_full_retrain": None}
    days = (today - last).days
    if days < 0:
        raise ScheduleError("the last full retrain is after today")
    return {"due": days >= RETRAIN_INTERVAL_DAYS, "days_since_full_retrain": days}


def should_record(outcome: Any) -> bool:
    """A from-scratch challenger that became the champion."""
    if not isinstance(outcome, dict):
        raise ScheduleError("outcome must be a JSON object")
    from_scratch = outcome.get("from_scratch")
    promotion = outcome.get("promotion")
    if not isinstance(from_scratch, bool) or not isinstance(promotion, str):
        raise ScheduleError("outcome needs from_scratch (bool) and promotion")
    return from_scratch and promotion in ("promote", "bootstrap")


def record(state_path: Path, today: dt.date, outcome_path: Path) -> bool:
    if not should_record(_load_json(outcome_path, "outcome")):
        return False
    last = last_full_retrain(state_path)
    if last is not None and last > today:
        raise ScheduleError("the last full retrain is after today")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"last_full_retrain": today.isoformat()}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(state_path)
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("due", "record"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--state", required=True, type=Path)
        cmd.add_argument("--today", required=True)
        if name == "record":
            cmd.add_argument("--outcome", required=True, type=Path)
    args = p.parse_args(argv)
    try:
        today = parse_date(args.today, "--today")
        if args.command == "due":
            result: dict[str, Any] = due(args.state, today)
        else:
            result = {"recorded": record(args.state, today, args.outcome)}
    except (ScheduleError, OSError) as e:
        print(f"retrain_schedule.py: error: {e}", file=sys.stderr)
        return EXIT_ERROR
    print(json.dumps(result, sort_keys=True))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
