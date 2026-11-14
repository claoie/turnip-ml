"""Unit tests for Training/retrain_schedule.py (TRAINING_DESIGN.md §2.1).

Run: python -m unittest discover -s Training/tests -t .
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import retrain_schedule as rs

TODAY = "2026-11-13"


class RetrainScheduleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="retrain-schedule-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.state = self.tmp / "retrain_state.json"
        self.outcome = self.tmp / "outcome.json"
        self.stdout = self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.stderr = self.enterContext(contextlib.redirect_stderr(io.StringIO()))

    def last(self, date: str) -> None:
        self.state.write_text(json.dumps({"last_full_retrain": date}))

    def due(self, today: str = TODAY) -> tuple[int, dict]:
        argv = ["due", "--state", str(self.state), "--today", today]
        code = rs.main(argv)
        out = self.stdout.getvalue()
        self.stdout.seek(0)
        self.stdout.truncate()
        return code, json.loads(out) if code == 0 else {}

    def record(self, outcome: dict, today: str = TODAY) -> int:
        self.outcome.write_text(json.dumps(outcome))
        argv = ["record", "--state", str(self.state), "--today", today]
        return rs.main([*argv, "--outcome", str(self.outcome)])

    def test_no_recorded_retrain_is_due(self) -> None:
        self.assertEqual(
            self.due(), (0, {"due": True, "days_since_full_retrain": None})
        )

    def test_due_at_the_interval(self) -> None:
        # 2026-10-14 is 30 days before 2026-11-13; 2026-10-15 is 29.
        self.assertEqual(rs.RETRAIN_INTERVAL_DAYS, 30)
        self.last("2026-10-15")
        self.assertEqual(self.due(), (0, {"due": False, "days_since_full_retrain": 29}))
        self.last("2026-10-14")
        self.assertEqual(self.due(), (0, {"due": True, "days_since_full_retrain": 30}))

    def test_a_retrain_after_today_fails_closed(self) -> None:
        self.last("2026-11-14")
        self.assertEqual(self.due()[0], rs.EXIT_ERROR)

    def test_malformed_inputs_fail_closed(self) -> None:
        for bad_state in ("{}", '{"last_full_retrain": "2026-1-5"}', "[]", "nope"):
            with self.subTest(state=bad_state):
                self.state.write_text(bad_state)
                self.assertEqual(self.due()[0], rs.EXIT_ERROR)
        self.state.unlink()
        for bad_today in ("2026-13-01", "13/11/2026", "2026-11-13T00:00"):
            with self.subTest(today=bad_today):
                self.assertEqual(self.due(bad_today)[0], rs.EXIT_ERROR)

    def test_a_from_scratch_champion_is_recorded(self) -> None:
        for promotion in ("promote", "bootstrap"):
            with self.subTest(promotion=promotion):
                self.last("2026-01-01")
                code = self.record({"from_scratch": True, "promotion": promotion})
                self.assertEqual(code, rs.EXIT_OK)
                self.assertEqual(
                    json.loads(self.state.read_text()), {"last_full_retrain": TODAY}
                )
                self.assertFalse(self.state.with_suffix(".json.tmp").exists())
        self.assertIn('"recorded": true', self.stdout.getvalue())

    def test_nothing_else_is_recorded(self) -> None:
        for outcome in (
            {"from_scratch": True, "promotion": "archive"},
            {"from_scratch": True, "promotion": "not-run"},
            {"from_scratch": False, "promotion": "promote"},
        ):
            with self.subTest(**outcome):
                self.last("2026-01-01")
                self.assertEqual(self.record(outcome), rs.EXIT_OK)
                self.assertEqual(
                    json.loads(self.state.read_text()),
                    {"last_full_retrain": "2026-01-01"},
                )

    def test_a_malformed_outcome_fails_closed(self) -> None:
        for outcome in (
            {"promotion": "promote"},
            {"from_scratch": "true", "promotion": "promote"},
        ):
            with self.subTest(**outcome):
                self.assertEqual(self.record(outcome), rs.EXIT_ERROR)
        self.assertFalse(self.state.exists())


if __name__ == "__main__":
    unittest.main()
