"""Unit tests for Training/promote.py.

Fast and hermetic: synthetic metric fixtures only, no network, no model.
Run with:  python -m unittest discover -s Training/tests -t .
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import promote
from promote import EXIT_ARCHIVE, EXIT_ERROR, EXIT_PROMOTE, decide


class DecideTest(unittest.TestCase):
    def test_promote_clear_win(self):
        # 0.80 -> 0.82 is +2.5% relative, above the 1% threshold.
        decision, improvement, _ = decide(0.80, 0.82, 0.01, "val_accuracy")
        self.assertEqual(decision, "promote")
        self.assertAlmostEqual(improvement, 0.025)

    def test_promote_exact_boundary(self):
        # improvement == threshold promotes (>=). 4.0 -> 5.0 is exactly
        # 0.25 relative, and 0.25 is exactly representable in binary
        # floating point, so a strict-> mutant (>) flips this test red.
        decision, improvement, _ = decide(4.0, 5.0, 0.25, "val_accuracy")
        self.assertEqual(decision, "promote")
        self.assertEqual(improvement, 0.25)

    def test_archive_below_threshold(self):
        # 0.80 -> 0.805 is +0.625% relative: real gain, not enough.
        decision, improvement, _ = decide(0.80, 0.805, 0.01, "val_accuracy")
        self.assertEqual(decision, "archive")
        self.assertLess(improvement, 0.01)

    def test_archive_regression(self):
        decision, improvement, _ = decide(0.80, 0.75, 0.01, "val_accuracy")
        self.assertEqual(decision, "archive")
        self.assertLess(improvement, 0.0)

    def test_zero_champion_positive_challenger_promotes(self):
        decision, _, _ = decide(0.0, 0.30, 0.01, "val_accuracy")
        self.assertEqual(decision, "promote")

    def test_zero_vs_zero_archives(self):
        decision, improvement, _ = decide(0.0, 0.0, 0.01, "val_accuracy")
        self.assertEqual(decision, "archive")
        self.assertEqual(improvement, 0.0)

    def test_zero_champion_reason_uses_actual_metric(self):
        # The registry is the audit trail: the reason must name the metric
        # actually compared, not the default metric name.
        _, _, reason = decide(0.0, 0.78, 0.01, "val_f1")
        self.assertIn("val_f1", reason)
        self.assertNotIn("val_accuracy", reason)
        _, _, reason = decide(0.0, 0.0, 0.01, "val_f1")
        self.assertIn("val_f1", reason)
        self.assertNotIn("val_accuracy", reason)

    def test_custom_threshold(self):
        # +2.5% passes 1% but not 5%.
        self.assertEqual(decide(0.80, 0.82, 0.05, "val_accuracy")[0], "archive")
        self.assertEqual(decide(0.80, 0.85, 0.05, "val_accuracy")[0], "promote")


class RefreshDecideTest(unittest.TestCase):
    def test_refresh_promotes_down_to_the_tolerance(self) -> None:
        threshold = -promote.REFRESH_TOLERANCE
        # 50 -> 49 is exactly -1/50, the same double as -0.02: promotes.
        self.assertEqual(decide(50.0, 49.0, threshold, "val_accuracy")[0], "promote")
        self.assertEqual(decide(50.0, 48.9, threshold, "val_accuracy")[0], "archive")
        # No change at all promotes a refresh, where a normal run archives.
        self.assertEqual(decide(0.8, 0.8, threshold, "val_accuracy")[0], "promote")
        self.assertEqual(decide(0.8, 0.8, 0.01, "val_accuracy")[0], "archive")

    def test_refresh_promotes_zero_over_zero(self) -> None:
        threshold = -promote.REFRESH_TOLERANCE
        self.assertEqual(decide(0.0, 0.0, threshold, "val_accuracy")[0], "promote")
        self.assertEqual(decide(0.0, 0.0, 0.01, "val_accuracy")[0], "archive")


class RunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.registry = self.root / "registry.json"

    def tearDown(self):
        self.tmp.cleanup()

    def write_metrics(self, name, **metrics):
        path = self.root / name
        path.write_text(json.dumps(metrics), encoding="utf-8")
        return path

    def test_promote_end_to_end(self):
        champion = self.write_metrics(
            "champion.json", val_accuracy=0.80, train_loss=0.4
        )
        challenger = self.write_metrics(
            "challenger.json", val_accuracy=0.82, train_loss=0.3
        )
        record, code = promote.run(
            champion, challenger, "val_accuracy", 0.01, self.registry
        )
        self.assertEqual(code, EXIT_PROMOTE)
        self.assertEqual(record["decision"], "promote")
        self.assertEqual(record["metric"], "val_accuracy")
        self.assertAlmostEqual(record["improvement"], 0.025)
        self.assertIn("decided_at", record)  # timestamps recorded
        self.assertTrue(record["decided_at"].endswith("Z"))
        self.assertEqual(record["champion_metrics"]["val_accuracy"], 0.80)
        self.assertEqual(record["challenger_metrics"]["val_accuracy"], 0.82)
        # Registry got the entry.
        entries = json.loads(self.registry.read_text(encoding="utf-8"))
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["decision"], "promote")

    def test_archive_end_to_end(self):
        champion = self.write_metrics("champion.json", val_accuracy=0.80)
        challenger = self.write_metrics("challenger.json", val_accuracy=0.805)
        record, code = promote.run(
            champion, challenger, "val_accuracy", 0.01, self.registry
        )
        self.assertEqual(code, EXIT_ARCHIVE)
        self.assertEqual(record["decision"], "archive")
        entries = json.loads(self.registry.read_text(encoding="utf-8"))
        self.assertEqual(entries[0]["decision"], "archive")

    def test_registry_appends(self):
        champion = self.write_metrics("champion.json", val_accuracy=0.80)
        c1 = self.write_metrics("c1.json", val_accuracy=0.82)
        c2 = self.write_metrics("c2.json", val_accuracy=0.79)
        promote.run(champion, c1, "val_accuracy", 0.01, self.registry)
        promote.run(champion, c2, "val_accuracy", 0.01, self.registry)
        entries = json.loads(self.registry.read_text(encoding="utf-8"))
        self.assertEqual([e["decision"] for e in entries], ["promote", "archive"])

    def test_registry_write_is_atomic(self):
        # run() must not leave the tmp file behind: the registry is
        # written to registry.json.tmp then atomically replaced.
        champion = self.write_metrics("champion.json", val_accuracy=0.80)
        c1 = self.write_metrics("c1.json", val_accuracy=0.82)
        promote.run(champion, c1, "val_accuracy", 0.01, self.registry)
        self.assertFalse((self.root / "registry.json.tmp").exists())
        entries = json.loads(self.registry.read_text(encoding="utf-8"))
        self.assertEqual(len(entries), 1)

    def test_custom_metric_key(self):
        champion = self.write_metrics("champion.json", val_f1=0.70)
        challenger = self.write_metrics("challenger.json", val_f1=0.78)
        record, code = promote.run(champion, challenger, "val_f1", 0.05, self.registry)
        self.assertEqual(code, EXIT_PROMOTE)
        self.assertEqual(record["metric"], "val_f1")

    def test_decision_is_deterministic(self):
        # Same inputs -> same decision; only the timestamp may differ.
        champion = self.write_metrics("champion.json", val_accuracy=0.80)
        challenger = self.write_metrics("challenger.json", val_accuracy=0.82)
        r1, _ = promote.run(
            champion, challenger, "val_accuracy", 0.01, self.root / "r1.json"
        )
        r2, _ = promote.run(
            champion, challenger, "val_accuracy", 0.01, self.root / "r2.json"
        )
        for key in (
            "metric",
            "threshold",
            "champion_metric",
            "challenger_metric",
            "improvement",
            "decision",
            "reason",
            "champion_metrics",
            "challenger_metrics",
        ):
            self.assertEqual(r1[key], r2[key], key)

    # -- fail-closed paths -------------------------------------------------
    def test_missing_metric_key(self):
        champion = self.write_metrics("champion.json", val_accuracy=0.80)
        challenger = self.write_metrics("challenger.json", train_loss=0.3)
        with self.assertRaises(promote.GateError):
            promote.run(champion, challenger, "val_accuracy", 0.01, self.registry)
        self.assertFalse(self.registry.exists())  # nothing recorded

    def test_non_numeric_metric(self):
        champion = self.write_metrics("champion.json", val_accuracy="high")
        challenger = self.write_metrics("challenger.json", val_accuracy=0.82)
        with self.assertRaises(promote.GateError):
            promote.run(champion, challenger, "val_accuracy", 0.01, self.registry)

    def test_bool_metric_rejected(self):
        champion = self.write_metrics("champion.json", val_accuracy=True)
        challenger = self.write_metrics("challenger.json", val_accuracy=0.82)
        with self.assertRaises(promote.GateError):
            promote.run(champion, challenger, "val_accuracy", 0.01, self.registry)

    def test_nan_metric_rejected(self):
        champion = self.write_metrics("champion.json", val_accuracy=0.80)
        # json.dump writes NaN by default; the gate must refuse it.
        nan_path = self.root / "nan.json"
        nan_path.write_text('{"val_accuracy": NaN}', encoding="utf-8")
        with self.assertRaises(promote.GateError):
            promote.run(champion, nan_path, "val_accuracy", 0.01, self.registry)

    def test_missing_file(self):
        with self.assertRaises(promote.GateError):
            promote.run(
                self.root / "nope.json",
                self.root / "nope2.json",
                "val_accuracy",
                0.01,
                self.registry,
            )

    def test_malformed_registry_refused(self):
        self.registry.write_text("not json", encoding="utf-8")
        champion = self.write_metrics("champion.json", val_accuracy=0.80)
        challenger = self.write_metrics("challenger.json", val_accuracy=0.82)
        with self.assertRaises(promote.GateError):
            promote.run(champion, challenger, "val_accuracy", 0.01, self.registry)

    def test_non_finite_threshold_rejected(self):
        # A nan/inf threshold must fail closed (exit 2), never silently
        # archive or corrupt the registry with an Infinity threshold.
        champion = self.write_metrics("champion.json", val_accuracy=0.80)
        challenger = self.write_metrics("challenger.json", val_accuracy=0.95)
        for bad in ("nan", "inf"):
            with self.subTest(threshold=bad):
                code = promote.main(
                    [
                        "--champion",
                        str(champion),
                        "--challenger",
                        str(challenger),
                        "--threshold",
                        bad,
                        "--registry",
                        str(self.registry),
                    ]
                )
                self.assertEqual(code, EXIT_ERROR)
        self.assertFalse(self.registry.exists())  # nothing recorded

    def test_main_refresh(self):
        champion = self.write_metrics("champion.json", val_accuracy=0.80)
        same = self.write_metrics("same.json", val_accuracy=0.80)
        base = [
            "--champion",
            str(champion),
            "--challenger",
            str(same),
            "--registry",
            str(self.registry),
        ]
        self.assertEqual(promote.main(base), EXIT_ARCHIVE)
        self.assertEqual(promote.main([*base, "--refresh"]), EXIT_PROMOTE)
        entries = json.loads(self.registry.read_text(encoding="utf-8"))
        self.assertEqual([e["refresh"] for e in entries], [False, True])
        self.assertEqual(entries[1]["threshold"], -promote.REFRESH_TOLERANCE)
        # The refresh sets the threshold: both is a usage error.
        self.assertEqual(
            promote.main([*base, "--refresh", "--threshold", "0.01"]), EXIT_ERROR
        )
        self.assertEqual(len(json.loads(self.registry.read_text())), 2)

    def test_main_exit_codes(self):
        champion = self.write_metrics("champion.json", val_accuracy=0.80)
        challenger = self.write_metrics("challenger.json", val_accuracy=0.82)
        base = [
            "--champion",
            str(champion),
            "--challenger",
            str(challenger),
            "--registry",
            str(self.registry),
        ]
        self.assertEqual(promote.main(base), EXIT_PROMOTE)
        weak = self.write_metrics("weak.json", val_accuracy=0.801)
        self.assertEqual(
            promote.main(
                [
                    "--champion",
                    str(champion),
                    "--challenger",
                    str(weak),
                    "--registry",
                    str(self.registry),
                ]
            ),
            EXIT_ARCHIVE,
        )
        missing = self.write_metrics("missing.json", train_loss=1.0)
        self.assertEqual(
            promote.main(
                [
                    "--champion",
                    str(champion),
                    "--challenger",
                    str(missing),
                    "--registry",
                    str(self.registry),
                ]
            ),
            EXIT_ERROR,
        )


if __name__ == "__main__":
    unittest.main()
