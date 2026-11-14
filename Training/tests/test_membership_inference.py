"""Unit tests for Training/membership_inference.py (TRAINING_DESIGN.md §6.2).

Run: python -m unittest discover -s Training/tests -t .
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import membership_inference as mi

# The hand-built test model: two classes, 2-frame windows at stride 2.
# The "cork" logit is the window's mean x of joint 0 (feature 0), so a
# sample whose joint 0 sits at x has cork confidence sigmoid(x) on every
# window: tests place each sample's score by its x.
CLASSES = ["background", "cork"]
WINDOW = 2
N_FEATURES = 52  # train.featurize_window: 13 joints x (x, y) x (mean, speed)
FRAMES = 4  # two windows per sample


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class AucTest(unittest.TestCase):
    def test_separated_scores(self) -> None:
        self.assertEqual(mi.auc([0.9, 0.8], [0.1, 0.2]), 1.0)
        self.assertEqual(mi.auc([0.1, 0.2], [0.9, 0.8]), 0.0)

    def test_ties_count_one_half(self) -> None:
        self.assertEqual(mi.auc([0.5], [0.5]), 0.5)
        # 0.5 vs [0.5, 0.9]: one tie (1/2), one loss (0) -> 1/4.
        self.assertEqual(mi.auc([0.5], [0.5, 0.9]), 0.25)

    def test_partial_overlap(self) -> None:
        # 0.3 beats neither, 0.7 beats 0.5, 0.9 beats both: 3 of 6 pairs.
        self.assertEqual(mi.auc([0.3, 0.7, 0.9], [0.5, 0.8]), 0.5)


class PassesTest(unittest.TestCase):
    def test_auc_boundary(self) -> None:
        n = mi.MIN_SOURCES
        # 3/5 is the same double as the literal 0.6: the boundary passes.
        self.assertTrue(mi.passes(3 / 5, n, n))
        self.assertFalse(mi.passes(0.600001, n, n))

    def test_source_minimum_on_each_side(self) -> None:
        n = mi.MIN_SOURCES
        self.assertTrue(mi.passes(0.5, n, n))
        self.assertFalse(mi.passes(0.5, n - 1, 1000))
        self.assertFalse(mi.passes(0.5, 1000, n - 1))


class RunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="membership-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.samples_dir = self.tmp / "samples"
        self.samples_dir.mkdir()
        self.split_path = self.tmp / "split.json"
        self.model_path = self.tmp / "model.npz"
        self.report_path = self.tmp / "report.json"
        self.stdout = self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        W = np.zeros((N_FEATURES, len(CLASSES)))
        W[0, 1] = 1.0
        np.savez(
            self.model_path,
            W=W,
            b=np.zeros(len(CLASSES)),
            classes=np.array(CLASSES),
            feat_mean=np.zeros(N_FEATURES),
            feat_std=np.ones(N_FEATURES),
            window_frames=np.int64(WINDOW),
            window_stride=np.int64(WINDOW),
        )

    def write_dataset(
        self, train_x: list[float], holdout_x: list[float], name: str = "cork"
    ) -> None:
        """One sample per x, labeled `name` on its whole length."""
        split = {}
        for part, xs in (("train", train_x), ("holdout", holdout_x)):
            for i, x in enumerate(xs):
                sid = f"{part}-{i:03d}"
                split[sid] = part
                frame = [[0.0, 0.0, 1.0] for _ in range(13)]
                frame[0] = [x, 0.0, 1.0]
                sample = {
                    "sample_id": sid,
                    "keypoints": [frame] * FRAMES,
                    "tricks": [{"name": name, "start_frame": 0, "end_frame": FRAMES}],
                }
                (self.samples_dir / f"{sid}.json").write_text(json.dumps(sample))
        self.split_path.write_text(json.dumps(split, sort_keys=True))
        digest = hashlib.sha256(self.split_path.read_bytes()).hexdigest()
        self.report_path.write_text(
            json.dumps({"f1": 0.5, "inputs": {"split_sha256": digest}})
        )

    def run_check(self) -> int:
        return mi.main(
            [
                "--model",
                str(self.model_path),
                "--split",
                str(self.split_path),
                "--samples-dir",
                str(self.samples_dir),
                "--report",
                str(self.report_path),
            ]
        )

    def block(self) -> dict:
        return json.loads(self.report_path.read_text())["membership_inference"]

    def test_memorized_training_sources_fail(self) -> None:
        n = mi.MIN_SOURCES
        self.write_dataset([2.0] * n, [0.0] * n)
        self.assertEqual(self.run_check(), mi.EXIT_FAIL)
        block = self.block()
        self.assertEqual(block["auc"], 1.0)
        self.assertIs(block["passed"], False)
        self.assertEqual(block["train_mean_score"], round(sigmoid(2.0), 6))
        self.assertEqual(block["holdout_mean_score"], 0.5)
        self.assertEqual(block["n_train_windows"], 2 * n)
        # The report's other keys survive.
        self.assertEqual(json.loads(self.report_path.read_text())["f1"], 0.5)

    def test_indistinguishable_sources_pass(self) -> None:
        n = mi.MIN_SOURCES
        self.write_dataset([1.0] * n, [1.0] * n)
        self.assertEqual(self.run_check(), mi.EXIT_PASS)
        block = self.block()
        self.assertEqual(block["auc"], 0.5)
        self.assertIs(block["passed"], True)
        self.assertEqual(block["attack"], mi.ATTACK)
        self.assertEqual(block["max_auc"], mi.MAX_AUC)
        self.assertEqual(block["min_sources"], mi.MIN_SOURCES)
        self.assertEqual(block["n_train_sources"], n)
        self.assertEqual(block["n_holdout_sources"], n)

    def test_too_few_sources_fail_even_at_chance(self) -> None:
        n = mi.MIN_SOURCES
        self.write_dataset([1.0] * n, [1.0] * (n - 1))
        self.assertEqual(self.run_check(), mi.EXIT_FAIL)
        self.assertEqual(self.block()["auc"], 0.5)
        self.assertIs(self.block()["passed"], False)

    def test_unseen_label_windows_are_left_out_and_counted(self) -> None:
        with mock.patch.object(mi, "MIN_SOURCES", 1):
            self.write_dataset([1.0], [1.0])
            unseen = self.samples_dir / "holdout-001.json"
            sample = json.loads((self.samples_dir / "holdout-000.json").read_text())
            sample["sample_id"] = "holdout-001"
            sample["tricks"][0]["name"] = "gainer"
            unseen.write_text(json.dumps(sample))
            split = json.loads(self.split_path.read_text())
            split["holdout-001"] = "holdout"
            self.split_path.write_text(json.dumps(split, sort_keys=True))
            digest = hashlib.sha256(self.split_path.read_bytes()).hexdigest()
            self.report_path.write_text(
                json.dumps({"inputs": {"split_sha256": digest}})
            )
            self.assertEqual(self.run_check(), mi.EXIT_PASS)
        block = self.block()
        self.assertEqual(block["n_windows_unseen_label"], 2)
        self.assertEqual(block["n_holdout_sources"], 1)
        self.assertEqual(block["n_holdout_windows"], 2)

    def test_stdout_carries_no_sample_id(self) -> None:
        n = mi.MIN_SOURCES
        self.write_dataset([2.0] * n, [0.0] * n)
        self.run_check()
        out = self.stdout.getvalue()
        self.assertIn("auc 1.0000", out)
        self.assertNotIn("train-0", out)
        self.assertNotIn("holdout-0", out)

    def test_output_is_deterministic(self) -> None:
        n = mi.MIN_SOURCES
        self.write_dataset([0.5 + i / 10 for i in range(n)], [1.0] * n)
        original = self.report_path.read_bytes()
        self.run_check()
        first = self.report_path.read_bytes()
        self.report_path.write_bytes(original)
        self.run_check()
        self.assertEqual(self.report_path.read_bytes(), first)

    def test_report_from_another_split_fails_closed(self) -> None:
        self.write_dataset([1.0], [1.0])
        self.report_path.write_text(json.dumps({"inputs": {"split_sha256": "0" * 64}}))
        self.assertEqual(self.run_check(), mi.EXIT_ERROR)
        self.assertNotIn("membership_inference", self.report_path.read_text())

    def test_no_holdout_sample_fails_closed(self) -> None:
        self.write_dataset([1.0], [])
        self.assertEqual(self.run_check(), mi.EXIT_ERROR)

    def test_side_with_only_unseen_labels_fails_closed(self) -> None:
        self.write_dataset([1.0], [1.0], name="gainer")
        self.assertEqual(self.run_check(), mi.EXIT_ERROR)


if __name__ == "__main__":
    unittest.main()
