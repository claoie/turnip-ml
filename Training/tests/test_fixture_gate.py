"""Unit tests for Training/fixture_gate.py.

Fast and hermetic: synthetic score fixtures and a fake alert sender only,
no network, no model.
Run with:  python -m unittest discover -s Training/tests -t .
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fixture_gate
from fixture_gate import (
    EXIT_ERROR,
    EXIT_PASS,
    EXIT_REGRESSION,
    WEBHOOK_ENV,
    GateError,
    composite,
    decide,
)

WEBHOOK = "https://discord.example/api/webhooks/123/SECRET-TOKEN"
SRC_A = "a" * 64
SRC_B = "b" * 64
SRC_C = "c" * 64


class FakeSender:
    """Records webhook calls; optionally raises to simulate an outage."""

    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def __call__(self, url, payload):
        self.calls.append((url, payload))
        if self.error is not None:
            raise self.error


class DecideTest(unittest.TestCase):
    def test_regression_below_floor(self):
        regressed, reason = decide(0.80, 0.70, 0.01)
        self.assertTrue(regressed)
        self.assertIn("fixture regression", reason)

    def test_drop_within_tolerance_passes(self):
        regressed, _ = decide(0.80, 0.795, 0.01)
        self.assertFalse(regressed)

    def test_exact_boundary_is_not_regression(self):
        # challenger == champion - tolerance is NOT a regression (strict <).
        # 0.75 - 0.25 == 0.5 exactly in binary floating point, so a `<=`
        # mutant flips this test red.
        regressed, _ = decide(0.75, 0.5, 0.25)
        self.assertFalse(regressed)

    def test_just_below_boundary_is_regression(self):
        regressed, _ = decide(0.75, 0.4375, 0.25)
        self.assertTrue(regressed)

    def test_improvement_passes(self):
        regressed, _ = decide(0.50, 0.90, 0.01)
        self.assertFalse(regressed)

    def test_zero_tolerance(self):
        self.assertFalse(decide(0.5, 0.5, 0.0)[0])
        self.assertTrue(decide(0.5, 0.25, 0.0)[0])

    def test_composite_is_component_mean(self):
        self.assertEqual(composite({"segment_iou": 0.5, "name_accuracy": 1.0}), 0.75)


class GateTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out = self.root / "run"
        self.registry = self.root / "registry.json"
        self.quarantine = self.out / "quarantine.json"
        self.sender = FakeSender()
        # The gate reports on stdout/stderr; keep the test output clean.
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))

    def tearDown(self):
        self.tmp.cleanup()

    def write_json(self, name, data):
        path = self.root / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def scores(self, name, segment_iou, name_accuracy, fixture_set="fx-1", **extra):
        return self.write_json(
            name,
            {
                "fixture_set": fixture_set,
                "segment_iou": segment_iou,
                "name_accuracy": name_accuracy,
                **extra,
            },
        )

    def ingested(self, entries=None, name="ingested.json"):
        if entries is None:
            entries = [
                {"source_id": SRC_C, "user_id": "user-2"},
                {"source_id": SRC_A, "user_id": "user-1"},
                {"source_id": SRC_B, "user_id": "user-1"},
                {"source_id": SRC_A, "user_id": "user-1"},  # dup, same owner
            ]
        return self.write_json(name, entries)

    def run_gate(self, champion, challenger, ingested=None, **kwargs):
        kwargs.setdefault("webhook_url", WEBHOOK)
        kwargs.setdefault("send", self.sender)
        return fixture_gate.run(
            champion,
            challenger,
            ingested if ingested is not None else self.ingested(),
            self.out,
            kwargs.pop("tolerance", 0.01),
            self.registry,
            **kwargs,
        )

    def registry_entries(self):
        return json.loads(self.registry.read_text(encoding="utf-8"))

    def quarantine_rows(self):
        return json.loads(self.quarantine.read_text(encoding="utf-8"))


class RegressionTest(GateTestCase):
    def test_regression_alerts_quarantines_and_skips_promotion(self):
        champion = self.scores("champion.json", 0.80, 0.90)
        challenger = self.scores("challenger.json", 0.60, 0.70)
        record, code = self.run_gate(champion, challenger)

        self.assertEqual(code, EXIT_REGRESSION)
        self.assertEqual(record["decision"], "regression")
        self.assertEqual(record["metrics"]["delta"], -0.2)

        # Alert fired once, at the configured webhook, with the delta and
        # the affected source and user ids; mentions disabled.
        self.assertEqual(len(self.sender.calls), 1)
        url, payload = self.sender.calls[0]
        self.assertEqual(url, WEBHOOK)
        self.assertEqual(payload["allowed_mentions"], {"parse": []})
        content = payload["content"]
        self.assertIn("promotion skipped", content)
        self.assertIn("delta -0.200000", content)
        for src in (SRC_A, SRC_B, SRC_C):
            self.assertIn(src, content)
        self.assertIn("users: user-1, user-2", content)
        self.assertEqual(record["alert_status"], "sent")

        # One quarantine row per ingested source (deduped, sorted), shaped
        # like a data_quarantine insert.
        rows = self.quarantine_rows()
        self.assertEqual([r["source_id"] for r in rows], [SRC_A, SRC_B, SRC_C])
        self.assertEqual([r["user_id"] for r in rows], ["user-1", "user-1", "user-2"])
        for row in rows:
            self.assertEqual(set(row), {"source_id", "user_id", "reason", "metrics"})
            self.assertEqual(row["reason"], record["reason"])
            self.assertEqual(row["metrics"], record["metrics"])
        self.assertEqual(record["quarantined_sources"], 3)

        entries = self.registry_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["gate"], "fixture-regression")
        self.assertEqual(entries[0]["decision"], "regression")
        self.assertTrue(entries[0]["decided_at"].endswith("Z"))

    def test_regression_with_no_ingested_sources(self):
        champion = self.scores("champion.json", 0.80, 0.90)
        challenger = self.scores("challenger.json", 0.60, 0.70)
        _, code = self.run_gate(champion, challenger, self.ingested([]))
        self.assertEqual(code, EXIT_REGRESSION)
        self.assertEqual(self.quarantine_rows(), [])
        self.assertIn("quarantined 0 source(s)", self.sender.calls[0][1]["content"])

    def test_alert_failure_still_quarantines_and_skips(self):
        self.sender = FakeSender(urllib.error.URLError("timed out"))
        champion = self.scores("champion.json", 0.80, 0.90)
        challenger = self.scores("challenger.json", 0.60, 0.70)
        record, code = self.run_gate(champion, challenger)
        self.assertEqual(code, EXIT_REGRESSION)
        self.assertEqual(record["alert_status"], "failed: URLError: timed out")
        self.assertEqual(len(self.quarantine_rows()), 3)

    def test_alert_failure_never_records_the_webhook_secret(self):
        champion = self.scores("champion.json", 0.80, 0.90)
        challenger = self.scores("challenger.json", 0.60, 0.70)
        errors = [
            ValueError(f"URL can't contain control characters. {WEBHOOK!r}"),
            urllib.error.HTTPError(WEBHOOK, 404, "Not Found", None, None),
        ]
        for error, expected in zip(errors, ["failed: ValueError", "failed: HTTP 404"]):
            with self.subTest(error=type(error).__name__):
                self.sender = FakeSender(error)
                record, _ = self.run_gate(champion, challenger)
                self.assertEqual(record["alert_status"], expected)
        self.assertNotIn("SECRET-TOKEN", self.registry.read_text(encoding="utf-8"))

    def test_webhook_unset_still_quarantines_and_skips(self):
        champion = self.scores("champion.json", 0.80, 0.90)
        challenger = self.scores("challenger.json", 0.60, 0.70)
        record, code = self.run_gate(champion, challenger, webhook_url=None)
        self.assertEqual(code, EXIT_REGRESSION)
        self.assertEqual(record["alert_status"], "not-configured")
        self.assertEqual(self.sender.calls, [])
        self.assertEqual(len(self.quarantine_rows()), 3)

    def test_non_https_webhook_is_not_called(self):
        champion = self.scores("champion.json", 0.80, 0.90)
        challenger = self.scores("challenger.json", 0.60, 0.70)
        record, code = self.run_gate(
            champion, challenger, webhook_url="http://discord.example/hook"
        )
        self.assertEqual(code, EXIT_REGRESSION)
        self.assertEqual(record["alert_status"], "failed: webhook URL is not https")
        self.assertEqual(self.sender.calls, [])

    def test_alert_content_fits_discord_limit(self):
        ingested = [
            {"source_id": f"{i:064x}", "user_id": f"user-{i:04d}-" + "x" * 200}
            for i in range(500)
        ]
        metrics = {
            "fixture_set": "fx-1",
            "tolerance": 0.01,
            "delta": -0.2,
            "champion": {"segment_iou": 0.8, "name_accuracy": 0.9, "composite": 0.85},
            "challenger": {"segment_iou": 0.6, "name_accuracy": 0.7, "composite": 0.65},
        }
        content = fixture_gate.alert_content(metrics, ingested)
        self.assertLessEqual(len(content), fixture_gate.DISCORD_MAX_CONTENT)
        self.assertIn("quarantined 500 source(s) from 500 user(s)", content)
        self.assertIn("+490 more", content)


class PassTest(GateTestCase):
    def test_no_regression_proceeds(self):
        champion = self.scores("champion.json", 0.80, 0.90)
        challenger = self.scores("challenger.json", 0.80, 0.895)
        record, code = self.run_gate(champion, challenger)
        self.assertEqual(code, EXIT_PASS)
        self.assertEqual(record["decision"], "pass")
        self.assertEqual(record["alert_status"], "not-needed")
        self.assertEqual(record["quarantined_sources"], 0)
        self.assertEqual(self.sender.calls, [])
        self.assertEqual(self.quarantine_rows(), [])
        self.assertEqual(self.registry_entries()[0]["decision"], "pass")

    def test_pass_overwrites_stale_quarantine_file(self):
        self.out.mkdir()
        self.quarantine.write_text('[{"source_id": "stale"}]', encoding="utf-8")
        champion = self.scores("champion.json", 0.80, 0.90)
        challenger = self.scores("challenger.json", 0.85, 0.90)
        _, code = self.run_gate(champion, challenger)
        self.assertEqual(code, EXIT_PASS)
        self.assertEqual(self.quarantine_rows(), [])

    def test_override_lets_regression_through_and_is_recorded(self):
        champion = self.scores("champion.json", 0.80, 0.90)
        challenger = self.scores("challenger.json", 0.60, 0.70)
        record, code = self.run_gate(
            champion, challenger, override="fixture labels corrected"
        )
        self.assertEqual(code, EXIT_PASS)
        self.assertEqual(record["decision"], "override")
        self.assertEqual(record["override_reason"], "fixture labels corrected")
        self.assertIn("fixture regression", record["reason"])
        self.assertEqual(self.sender.calls, [])
        self.assertEqual(self.quarantine_rows(), [])
        self.assertEqual(self.registry_entries()[0]["decision"], "override")

    def test_registry_appends_after_existing_entries(self):
        # The registry is shared with the promotion gate.
        self.write_json("registry.json", [{"decision": "promote"}])
        champion = self.scores("champion.json", 0.80, 0.90)
        challenger = self.scores("challenger.json", 0.80, 0.90)
        self.run_gate(champion, challenger)
        entries = self.registry_entries()
        self.assertEqual([e["decision"] for e in entries], ["promote", "pass"])
        self.assertFalse((self.root / "registry.json.tmp").exists())

    def test_extra_score_keys_are_kept_not_compared(self):
        champion = self.scores("champion.json", 0.80, 0.90, model="trick-v1.3")
        challenger = self.scores("challenger.json", 0.80, 0.90, model="cand")
        record, code = self.run_gate(champion, challenger)
        self.assertEqual(code, EXIT_PASS)
        self.assertEqual(record["champion_scores"]["model"], "trick-v1.3")

    def test_outputs_are_deterministic(self):
        champion = self.scores("champion.json", 0.80, 0.90)
        challenger = self.scores("challenger.json", 0.60, 0.70)
        r1, _ = self.run_gate(champion, challenger)
        q1 = self.quarantine.read_bytes()
        content1 = self.sender.calls[0][1]["content"]
        # Same inputs, different listing order of the ingested sources.
        reordered = self.ingested(
            [
                {"source_id": SRC_B, "user_id": "user-1"},
                {"source_id": SRC_A, "user_id": "user-1"},
                {"source_id": SRC_C, "user_id": "user-2"},
            ],
            name="ingested2.json",
        )
        r2, _ = self.run_gate(champion, challenger, reordered)
        self.assertEqual(self.quarantine.read_bytes(), q1)
        self.assertEqual(self.sender.calls[1][1]["content"], content1)
        r1.pop("decided_at")
        r2.pop("decided_at")
        self.assertEqual(r1, r2)


class FailClosedTest(GateTestCase):
    def assert_refused(self, champion, challenger, ingested=None, **kwargs):
        with self.assertRaises(GateError):
            self.run_gate(champion, challenger, ingested, **kwargs)
        # Validation happens before any side effect.
        self.assertEqual(self.sender.calls, [])
        self.assertFalse(self.quarantine.exists())
        self.assertFalse(self.registry.exists())

    def test_fixture_set_mismatch(self):
        champion = self.scores("champion.json", 0.80, 0.90, fixture_set="fx-1")
        challenger = self.scores("challenger.json", 0.10, 0.10, fixture_set="fx-2")
        self.assert_refused(champion, challenger)

    def test_missing_fixture_set(self):
        champion = self.write_json(
            "champion.json", {"segment_iou": 0.8, "name_accuracy": 0.9}
        )
        challenger = self.scores("challenger.json", 0.80, 0.90)
        self.assert_refused(champion, challenger)

    def test_missing_component(self):
        champion = self.scores("champion.json", 0.80, 0.90)
        challenger = self.write_json(
            "challenger.json", {"fixture_set": "fx-1", "segment_iou": 0.8}
        )
        self.assert_refused(champion, challenger)

    def test_invalid_score_values(self):
        champion = self.scores("champion.json", 0.80, 0.90)
        for bad in (True, "0.9", None, 1.5, -0.1, 10**400):
            with self.subTest(value=bad):
                challenger = self.scores("challenger.json", 0.80, bad)
                self.assert_refused(champion, challenger)

    def test_non_finite_json_rejected(self):
        champion = self.scores("champion.json", 0.80, 0.90)
        for literal in ("NaN", "Infinity", "-Infinity", "1e400", "-1e400"):
            with self.subTest(literal=literal):
                path = self.root / "challenger.json"
                path.write_text(
                    '{"fixture_set": "fx-1", "segment_iou": 0.8, '
                    f'"name_accuracy": 0.9, "note": {literal}}}',
                    encoding="utf-8",
                )
                self.assert_refused(champion, path)

    def test_unreadable_inputs(self):
        champion = self.scores("champion.json", 0.80, 0.90)
        bad_utf8 = self.root / "bad.json"
        bad_utf8.write_bytes(b'{"fixture_set": "\xff\xfe"}')
        not_json = self.root / "not.json"
        not_json.write_text("not json", encoding="utf-8")
        for path in (self.root / "missing.json", bad_utf8, not_json):
            with self.subTest(path=path.name):
                self.assert_refused(champion, path)

    def test_malformed_ingested(self):
        champion = self.scores("champion.json", 0.80, 0.90)
        challenger = self.scores("challenger.json", 0.60, 0.70)
        cases = {
            "not-array": {"source_id": SRC_A, "user_id": "u"},
            "extra-key": [{"source_id": SRC_A, "user_id": "u", "x": 1}],
            "missing-user": [{"source_id": SRC_A}],
            "uppercase-id": [{"source_id": "A" * 64, "user_id": "u"}],
            "short-id": [{"source_id": "a" * 63, "user_id": "u"}],
            "trailing-newline": [{"source_id": SRC_A + "\n", "user_id": "u"}],
            "empty-user": [{"source_id": SRC_A, "user_id": ""}],
            "two-owners": [
                {"source_id": SRC_A, "user_id": "u1"},
                {"source_id": SRC_A, "user_id": "u2"},
            ],
        }
        for name, data in cases.items():
            with self.subTest(case=name):
                ingested = self.write_json(f"{name}.json", data)
                self.assert_refused(champion, challenger, ingested)

    def test_malformed_registry(self):
        self.registry.write_text("not json", encoding="utf-8")
        champion = self.scores("champion.json", 0.80, 0.90)
        challenger = self.scores("challenger.json", 0.60, 0.70)
        with self.assertRaises(GateError):
            self.run_gate(champion, challenger)
        self.assertEqual(self.sender.calls, [])
        self.assertFalse(self.quarantine.exists())

    def test_bad_tolerance(self):
        champion = self.scores("champion.json", 0.80, 0.90)
        challenger = self.scores("challenger.json", 0.60, 0.70)
        for bad in (float("nan"), float("inf"), -0.01):
            with self.subTest(tolerance=bad):
                self.assert_refused(champion, challenger, tolerance=bad)

    def test_empty_override_reason(self):
        champion = self.scores("champion.json", 0.80, 0.90)
        challenger = self.scores("challenger.json", 0.60, 0.70)
        self.assert_refused(champion, challenger, override="   ")

    def test_unwritable_out_dir(self):
        blocker = self.root / "blocker"
        blocker.write_text("", encoding="utf-8")
        self.out = blocker / "run"  # parent is a file
        champion = self.scores("champion.json", 0.80, 0.90)
        challenger = self.scores("challenger.json", 0.60, 0.70)
        with self.assertRaises(GateError):
            self.run_gate(champion, challenger)
        self.assertEqual(self.sender.calls, [])


class MainTest(GateTestCase):
    def argv(self, champion, challenger, *extra):
        return [
            "--champion",
            str(champion),
            "--challenger",
            str(challenger),
            "--ingested",
            str(self.ingested()),
            "--out",
            str(self.out),
            "--registry",
            str(self.registry),
            *extra,
        ]

    def test_exit_codes_and_webhook_from_env(self):
        champion = self.scores("champion.json", 0.80, 0.90)
        good = self.scores("good.json", 0.80, 0.90)
        bad = self.scores("bad.json", 0.60, 0.70)
        broken = self.write_json("broken.json", {"fixture_set": "fx-1"})
        with mock.patch.dict(os.environ, {WEBHOOK_ENV: WEBHOOK}):
            self.assertEqual(
                fixture_gate.main(self.argv(champion, good), send=self.sender),
                EXIT_PASS,
            )
            self.assertEqual(
                fixture_gate.main(self.argv(champion, bad), send=self.sender),
                EXIT_REGRESSION,
            )
            self.assertEqual(
                fixture_gate.main(self.argv(champion, broken), send=self.sender),
                EXIT_ERROR,
            )
        self.assertEqual([url for url, _ in self.sender.calls], [WEBHOOK])

    def test_override_flag(self):
        champion = self.scores("champion.json", 0.80, 0.90)
        bad = self.scores("bad.json", 0.60, 0.70)
        code = fixture_gate.main(
            self.argv(champion, bad, "--override", "rebaselined"), send=self.sender
        )
        self.assertEqual(code, EXIT_PASS)
        self.assertEqual(self.registry_entries()[0]["override_reason"], "rebaselined")

    def test_non_finite_tolerance_via_cli(self):
        champion = self.scores("champion.json", 0.80, 0.90)
        challenger = self.scores("challenger.json", 0.60, 0.70)
        for bad in ("nan", "inf", "-1"):
            with self.subTest(tolerance=bad):
                code = fixture_gate.main(
                    self.argv(champion, challenger, "--tolerance", bad),
                    send=self.sender,
                )
                self.assertEqual(code, EXIT_ERROR)
        self.assertFalse(self.registry.exists())


class SendDiscordTest(unittest.TestCase):
    def test_posts_json_with_explicit_user_agent(self):
        with mock.patch("urllib.request.urlopen") as urlopen:
            fixture_gate.send_discord(WEBHOOK, {"content": "hi"})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, WEBHOOK)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(request.get_header("User-agent"), "turnip-ml-fixture-gate")
        self.assertEqual(json.loads(request.data), {"content": "hi"})
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 10.0)


if __name__ == "__main__":
    unittest.main()
