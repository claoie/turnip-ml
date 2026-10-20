"""Unit tests for Training/name_metrics.py (turnip-ml#41).

Run: python -m unittest discover -s Training/tests -t .
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import evaluate
import name_metrics


def write_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def seg(name: str, start: int) -> dict:
    return {"name": name, "start_frame": start, "end_frame": start + 8}


class BuildNameMetricsTest(unittest.TestCase):
    def per_name(self, support: int, predicted: int = 4) -> dict:
        return {
            "gainer": {
                "support": support,
                "predicted": predicted,
                "precision": 0.75,
                "recall": 0.5,
            }
        }

    def test_fewer_than_five_holdout_examples_publish_null_metrics(self) -> None:
        out = name_metrics.build_name_metrics(self.per_name(4), {"gainer": 9})
        self.assertEqual(
            out,
            {
                "gainer": {
                    "precision": None,
                    "recall": None,
                    "holdout_count": 4,
                    "train_count": 9,
                }
            },
        )

    def test_five_holdout_examples_publish_the_metrics(self) -> None:
        out = name_metrics.build_name_metrics(self.per_name(5), {"gainer": 9})
        self.assertEqual(
            out["gainer"],
            {"precision": 0.75, "recall": 0.5, "holdout_count": 5, "train_count": 9},
        )

    def test_never_predicted_name_has_no_precision(self) -> None:
        # 0 of 0 predictions is undefined, not 0% precise; recall stands.
        per_name = self.per_name(12, predicted=0)
        per_name["gainer"].update(precision=0.0, recall=0.0)
        out = name_metrics.build_name_metrics(per_name, {"gainer": 9})
        self.assertEqual(
            out["gainer"],
            {"precision": None, "recall": 0.0, "holdout_count": 12, "train_count": 9},
        )

    def test_entries_carry_exactly_the_manifest_keys(self) -> None:
        out = name_metrics.build_name_metrics(self.per_name(7), {})
        self.assertEqual(
            set(out["gainer"]), {"precision", "recall", "holdout_count", "train_count"}
        )

    def test_train_only_name_is_still_learning(self) -> None:
        out = name_metrics.build_name_metrics({}, {"cork": 3})
        self.assertEqual(
            out["cork"],
            {"precision": None, "recall": None, "holdout_count": 0, "train_count": 3},
        )

    def test_background_is_never_a_name(self) -> None:
        with self.assertRaises(ValueError):
            name_metrics.build_name_metrics({}, {"background": 3})

    def test_validate_accepts_only_the_manifest_shape(self) -> None:
        good = {
            "gainer": {
                "precision": None,
                "recall": 0.5,
                "holdout_count": 6,
                "train_count": 2,
            }
        }
        self.assertEqual(name_metrics.validate(good), good)
        bad = [
            {},
            {"gainer": {**good["gainer"], "support": 6}},
            {"gainer": {**good["gainer"], "recall": 1.5}},
            {"gainer": {**good["gainer"], "train_count": -1}},
        ]
        for data in bad:
            with self.subTest(data), self.assertRaises(ValueError):
                name_metrics.validate(data)


class NameMetricsCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="name-metrics-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.stderr = self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        self.samples = self.tmp / "samples"
        self.samples.mkdir()
        self.split = self.tmp / "split.json"
        self.report = self.tmp / "holdout_report.json"
        self.metrics = self.tmp / "metrics.json"
        self.out = self.tmp / "name_metrics.json"

    def sample(self, sid: str, tricks: list[dict]) -> None:
        write_json(self.samples / f"{sid}.json", {"sample_id": sid, "tricks": tricks})

    def evaluate(self, split: dict[str, str], predictions: dict) -> None:
        write_json(self.split, split)
        pred = write_json(self.tmp / "predictions.json", predictions)
        rc = evaluate.main(
            [
                "--predictions",
                str(pred),
                "--split",
                str(self.split),
                "--samples-dir",
                str(self.samples),
                "--out",
                str(self.report),
            ]
        )
        self.assertEqual(rc, 0)

    def run_cli(self) -> int:
        return name_metrics.main(
            [
                "--report",
                str(self.report),
                "--metrics",
                str(self.metrics),
                "--split",
                str(self.split),
                "--out",
                str(self.out),
            ]
        )

    def write_metrics(self, counts: dict[str, int]) -> None:
        sha = hashlib.sha256(self.split.read_bytes()).hexdigest()
        write_json(self.metrics, {"split_sha256": sha, "train_name_counts": counts})

    def dataset(self) -> None:
        # Holdout: five gainers (four found), two corks (both found). The
        # train counts are train.py's (test_train covers how it counts).
        self.sample("h1", [seg("gainer", 0), seg("gainer", 20), seg("cork", 40)])
        self.sample("h2", [seg("gainer", 0), seg("gainer", 20), seg("gainer", 40)])
        self.sample("h3", [seg("cork", 0)])
        self.evaluate(
            {"h1": "holdout", "h2": "holdout", "h3": "holdout", "t1": "train"},
            {
                "h1": [seg("gainer", 0), seg("gainer", 20), seg("cork", 40)],
                "h2": [seg("gainer", 0), seg("gainer", 20)],
                "h3": [seg("cork", 0)],
            },
        )
        self.write_metrics({"cork": 1, "gainer": 3, "kick": 1})

    def test_writes_manifest_shaped_name_metrics(self) -> None:
        self.dataset()
        self.assertEqual(self.run_cli(), 0)
        self.assertEqual(
            json.loads(self.out.read_text(encoding="utf-8")),
            {
                "cork": {
                    "precision": None,
                    "recall": None,
                    "holdout_count": 2,
                    "train_count": 1,
                },
                "gainer": {
                    "precision": 1.0,
                    "recall": 0.8,
                    "holdout_count": 5,
                    "train_count": 3,
                },
                "kick": {
                    "precision": None,
                    "recall": None,
                    "holdout_count": 0,
                    "train_count": 1,
                },
            },
        )

    def test_byte_identical_reruns(self) -> None:
        self.dataset()
        self.assertEqual(self.run_cli(), 0)
        first = self.out.read_bytes()
        self.assertEqual(self.run_cli(), 0)
        self.assertEqual(self.out.read_bytes(), first)

    def test_report_from_another_split_is_refused(self) -> None:
        self.dataset()
        write_json(self.split, {"h1": "holdout", "t1": "train"})
        self.assertEqual(self.run_cli(), 2)
        self.assertIn("split_sha256", self.stderr.getvalue())
        self.assertFalse(self.out.exists())

    def test_metrics_from_another_split_are_refused(self) -> None:
        self.dataset()
        write_json(self.metrics, {"split_sha256": "0" * 64, "train_name_counts": {}})
        self.assertEqual(self.run_cli(), 2)
        self.assertIn("split_sha256", self.stderr.getvalue())
        self.assertFalse(self.out.exists())

    def test_metrics_without_train_counts_are_refused(self) -> None:
        self.dataset()
        sha = hashlib.sha256(self.split.read_bytes()).hexdigest()
        write_json(self.metrics, {"split_sha256": sha})
        self.assertEqual(self.run_cli(), 2)
        self.assertIn("train_name_counts", self.stderr.getvalue())

    def test_errors_name_no_trick(self) -> None:
        # Unmapped names are a contributor's free text: never in the log.
        self.dataset()
        report = json.loads(self.report.read_text(encoding="utf-8"))
        report["per_name"]["gainer"]["recall"] = 2.0
        write_json(self.report, report)
        self.assertEqual(self.run_cli(), 2)
        self.assertNotIn("gainer", self.stderr.getvalue())

    def test_malformed_report_is_refused(self) -> None:
        self.dataset()
        report = json.loads(self.report.read_text(encoding="utf-8"))
        report["per_name"]["gainer"]["precision"] = 1.5
        write_json(self.report, report)
        self.assertEqual(self.run_cli(), 2)


if __name__ == "__main__":
    unittest.main()
