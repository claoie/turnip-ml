"""Unit tests for Training/leakage_gap.py (turnip-ml#42).

Run: python -m unittest discover -s Training/tests -t .
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import evaluate
import leakage_gap
import predict
import train

REPO = Path(__file__).resolve().parent.parent.parent
SMOKE = REPO / "Training" / "fixtures" / "smoke"
SMOKE_CONFIG = REPO / "Training" / "config" / "smoke.toml"

# The hand-built test model: two classes, 2-frame windows at stride 2,
# zero weights and a bias toward "cork", so every window predicts "cork"
# and a sample's window accuracy is exactly its share of cork windows.
CLASSES = ["background", "cork"]
WINDOW = 2
N_FEATURES = 68  # train.featurize_window: 17 joints x (x, y) x (mean, speed)
HOLDOUT_WINDOWS = 100


def write_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seg(name: str, start: int, end: int) -> dict:
    return {"name": name, "start_frame": start, "end_frame": end}


def write_model(path: Path) -> None:
    np.savez(
        path,
        W=np.zeros((N_FEATURES, len(CLASSES))),
        b=np.array([0.0, 1.0]),
        classes=np.array(CLASSES),
        feat_mean=np.zeros(N_FEATURES),
        feat_std=np.ones(N_FEATURES),
        window_frames=np.int64(WINDOW),
        window_stride=np.int64(WINDOW),
    )


def gap_record(
    run_id: str,
    promotion: str = "promote",
    promotion_metric: float = 0.2,
    segment_f1: float = 0.1,
) -> dict:
    """A prior leakage_gap registry record (over threshold by default)."""
    return {
        "kind": "leakage_gap",
        "run_id": run_id,
        "promotion": promotion,
        "val_minus_holdout": {
            "promotion_metric": promotion_metric,
            "segment_f1": segment_f1,
        },
    }


class LeakageGapTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="leakage-gap-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.samples_dir = self.tmp / "samples"
        self.samples_dir.mkdir()
        self.split_path = self.tmp / "split.json"
        self.metrics_path = self.tmp / "metrics.json"
        self.model_path = self.tmp / "model.npz"
        self.report_path = self.tmp / "report.json"
        self.registry_path = self.tmp / "registry.json"
        # Every test picks its run id; none inherits one from the shell.
        env = {k: v for k, v in os.environ.items() if k != leakage_gap.RUN_ID_ENV}
        self.enterContext(mock.patch.dict(os.environ, env, clear=True))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))

    def _run(self, promotion: str = "promote", run_id: str | None = None) -> int:
        env = {leakage_gap.RUN_ID_ENV: run_id} if run_id else {}
        with mock.patch.dict(os.environ, env):
            return leakage_gap.main(
                [
                    "--metrics",
                    str(self.metrics_path),
                    "--model",
                    str(self.model_path),
                    "--report",
                    str(self.report_path),
                    "--split",
                    str(self.split_path),
                    "--samples-dir",
                    str(self.samples_dir),
                    "--registry",
                    str(self.registry_path),
                    "--promotion",
                    promotion,
                ]
            )

    def _report(self) -> dict:
        return json.loads(self.report_path.read_text())

    def _registry(self) -> list:
        return json.loads(self.registry_path.read_text())


class LeakageGapTest(LeakageGapTestCase):
    def _sample(self, sid: str, tricks: list[dict], frames: int = 8) -> None:
        write_json(
            self.samples_dir / f"{sid}.json",
            {
                "sample_id": sid,
                "fps": 10,
                "keypoints": [[[0.0, 0.0, 1.0]] * 17] * frames,
                "tricks": tricks,
            },
        )

    def _write_inputs(
        self,
        val_accuracy: float = 0.9,
        val_segments: list[dict] | None = None,
        holdout_f1: float = 0.9,
        holdout_acc: float = 0.7,
    ) -> None:
        # One val sample, one GT segment; the default val_segments match
        # it exactly, so val_f1 == 1.0. The holdout sample has 100 windows
        # and the model predicts "cork" on all of them, so labeling the
        # first holdout_acc of them "cork" sets the holdout window accuracy.
        self._sample("v1", [seg("cork", 0, 8)])
        n_cork = round(holdout_acc * HOLDOUT_WINDOWS)
        self._sample(
            "h1",
            [seg("cork", 0, n_cork * WINDOW)] if n_cork else [],
            frames=HOLDOUT_WINDOWS * WINDOW,
        )
        write_json(self.split_path, {"v1": "val", "h1": "holdout", "t1": "train"})
        split_sha = sha256_file(self.split_path)
        if val_segments is None:
            val_segments = [
                {
                    "sample_id": "v1",
                    "name": "cork",
                    "start_frame": 0,
                    "end_frame": 8,
                    "is_combo": False,
                }
            ]
        write_json(
            self.metrics_path,
            {
                "classes": CLASSES,
                "val_accuracy": val_accuracy,
                "val_segments": val_segments,
                "split_sha256": split_sha,
            },
        )
        write_model(self.model_path)
        write_json(
            self.report_path,
            {
                "f1": holdout_f1,
                # Segment-level: not the promotion metric's holdout side.
                "name_accuracy_given_detection": 0.123,
                "iou_threshold": 0.5,
                "inputs": {"predictions_sha256": "p" * 64, "split_sha256": split_sha},
            },
        )

    def test_gaps_written_to_report(self) -> None:
        self._write_inputs()
        self.assertEqual(self._run("archive"), 0)
        gap = self._report()["val_minus_holdout"]
        # promotion: 0.9 - 0.7 = 0.2; segment_f1: 1.0 - 0.9 = 0.1.
        self.assertEqual(gap["promotion_metric"], 0.2)
        self.assertEqual(gap["segment_f1"], 0.1)
        self.assertEqual(gap["val_f1"], 1.0)
        self.assertEqual(gap["streak_promoted_runs_over_threshold"], 0)
        self.assertFalse(gap["flagged"])

    def test_promotion_gap_uses_holdout_window_accuracy(self) -> None:
        # The holdout side is window accuracy over every holdout window
        # (background windows included), like val_accuracy — not the
        # report's segment-level name_accuracy_given_detection.
        self._write_inputs(holdout_acc=0.7)
        self.assertEqual(self._run("archive"), 0)
        gap = self._report()["val_minus_holdout"]
        self.assertEqual(gap["holdout_window_accuracy"], 0.7)
        self.assertEqual(gap["n_holdout_windows"], HOLDOUT_WINDOWS)
        self.assertEqual(gap["n_holdout_windows_unseen_label"], 0)
        self.assertEqual(gap["val_accuracy"], 0.9)
        self.assertNotIn("holdout_name_accuracy", gap)

    def test_unseen_holdout_label_counts_as_a_miss(self) -> None:
        # 70 cork windows (hits), 10 "gainer" windows (a label the model
        # never trained on: misses, counted), 20 background windows (misses).
        self._write_inputs()
        self._sample(
            "h1",
            [seg("cork", 0, 140), seg("gainer", 140, 160)],
            frames=HOLDOUT_WINDOWS * WINDOW,
        )
        self.assertEqual(self._run("archive"), 0)
        gap = self._report()["val_minus_holdout"]
        self.assertEqual(gap["holdout_window_accuracy"], 0.7)
        self.assertEqual(gap["n_holdout_windows_unseen_label"], 10)

    def test_held_back_windows_reach_the_window_count(self) -> None:
        split = {"h1": "holdout"}
        self.split_path.write_text(json.dumps(split))
        self._sample("h1", [seg("cork", 0, 8)], frames=8)
        rec = json.loads((self.samples_dir / "h1.json").read_text())
        rec["held_back_windows"] = [[0, 4]]
        write_json(self.samples_dir / "h1.json", rec)
        [sample] = leakage_gap.load_split_samples(split, self.samples_dir, "holdout")
        self.assertEqual(sample["held_back_windows"], [(0, 4)])
        rec["held_back_windows"] = [[4, 4]]
        write_json(self.samples_dir / "h1.json", rec)
        with self.assertRaises(ValueError):
            leakage_gap.load_split_samples(split, self.samples_dir, "holdout")

    def test_no_holdout_window_fails_closed(self) -> None:
        # Every holdout sample shorter than window_frames: no window, so
        # the accuracy is undefined rather than vacuously 1.0.
        self._write_inputs()
        self._sample("h1", [], frames=WINDOW - 1)
        before = self.report_path.read_text()
        self.assertEqual(self._run("promote"), 2)
        self.assertEqual(self.report_path.read_text(), before)
        self.assertFalse(self.registry_path.exists())

    def test_model_from_another_run_fails_closed(self) -> None:
        self._write_inputs()
        metrics = json.loads(self.metrics_path.read_text())
        metrics["classes"] = ["background", "gainer"]
        write_json(self.metrics_path, metrics)
        self.assertEqual(self._run("promote"), 2)
        self.assertFalse(self.registry_path.exists())

    def test_streak_advances_and_flags_on_third(self) -> None:
        self._write_inputs()
        write_json(self.registry_path, [gap_record("r1"), gap_record("r2")])
        self.assertEqual(self._run("promote"), 0)
        gap = self._report()["val_minus_holdout"]
        self.assertEqual(gap["streak_promoted_runs_over_threshold"], 3)
        self.assertTrue(gap["flagged"])
        self.assertEqual(len(self._registry()), 3)

    def test_streak_two_is_unflagged(self) -> None:
        # Pins FLAG_STREAK: one prior over-threshold promoted run plus
        # this one is streak 2, not flagged. A FLAG_STREAK = 2 mutant
        # would flag here and fail this test.
        self._write_inputs()
        write_json(self.registry_path, [gap_record("r1")])
        self.assertEqual(self._run("promote"), 0)
        gap = self._report()["val_minus_holdout"]
        self.assertEqual(gap["streak_promoted_runs_over_threshold"], 2)
        self.assertFalse(gap["flagged"])

    def test_combo_segment_decodes_to_both_names(self) -> None:
        # A combo val segment must score as its names, not as one name.
        self._write_inputs(
            val_segments=[
                {
                    "sample_id": "v1",
                    "name": "cork+backflip",
                    "start_frame": 0,
                    "end_frame": 8,
                    "is_combo": True,
                }
            ]
        )
        # Ground truth: two tricks on the same window merge to one
        # segment carrying both names.
        self._sample("v1", [seg("cork", 0, 8), seg("backflip", 0, 8)])
        self.assertEqual(self._run("archive"), 0)
        gap = self._report()["val_minus_holdout"]
        self.assertEqual(gap["val_f1"], 1.0)
        self.assertEqual(gap["segment_f1"], 0.1)

    def test_combo_decoding_rejects_malformed_names(self) -> None:
        # parse_segment, not loose split: "cork+" has an empty component,
        # and is_combo=True contradicts a single name. Both fail closed.
        self._write_inputs()
        for bad in [
            {
                "sample_id": "v1",
                "name": "cork+",
                "start_frame": 0,
                "end_frame": 8,
                "is_combo": True,
            },
            {
                "sample_id": "v1",
                "name": "cork",
                "start_frame": 0,
                "end_frame": 8,
                "is_combo": True,
            },
        ]:
            metrics = json.loads(self.metrics_path.read_text())
            metrics["val_segments"] = [bad]
            write_json(self.metrics_path, metrics)
            self.assertEqual(self._run("promote"), 2)

    def test_streak_resets_on_archive(self) -> None:
        self._write_inputs()
        write_json(self.registry_path, [gap_record("r1")])
        self.assertEqual(self._run("archive"), 0)
        gap = self._report()["val_minus_holdout"]
        self.assertEqual(gap["streak_promoted_runs_over_threshold"], 0)
        self.assertFalse(gap["flagged"])

    def test_streak_resets_when_under_threshold(self) -> None:
        self._write_inputs(val_accuracy=0.72, holdout_f1=0.99)
        write_json(self.registry_path, [gap_record("r1")])
        self.assertEqual(self._run("promote"), 0)
        gap = self._report()["val_minus_holdout"]
        # promotion 0.02, segment_f1 0.01: both under 0.05.
        self.assertEqual(gap["promotion_metric"], 0.02)
        self.assertEqual(gap["segment_f1"], 0.01)
        self.assertEqual(gap["streak_promoted_runs_over_threshold"], 0)

    def test_bootstrap_leaves_streak_unchanged(self) -> None:
        self._write_inputs()
        write_json(self.registry_path, [gap_record("r1")])
        self.assertEqual(self._run("bootstrap"), 0)
        gap = self._report()["val_minus_holdout"]
        self.assertEqual(gap["streak_promoted_runs_over_threshold"], 1)
        self.assertFalse(gap["flagged"])

    def test_other_registry_kinds_ignored(self) -> None:
        self._write_inputs()
        write_json(
            self.registry_path,
            [
                {
                    "decision": "promote",
                    "metric": "val_accuracy",
                }
            ],
        )
        self.assertEqual(self._run("promote"), 0)
        gap = self._report()["val_minus_holdout"]
        self.assertEqual(gap["streak_promoted_runs_over_threshold"], 1)

    def test_split_mismatch_fails_closed(self) -> None:
        self._write_inputs()
        metrics = json.loads(self.metrics_path.read_text())
        metrics["split_sha256"] = "0" * 64
        write_json(self.metrics_path, metrics)
        before = self.report_path.read_text()
        self.assertEqual(self._run("promote"), 2)
        self.assertEqual(self.report_path.read_text(), before)
        self.assertFalse(self.registry_path.exists())

    def test_report_split_mismatch_fails_closed(self) -> None:
        # The metrics-side tamper is covered above; the report side
        # (inputs.split_sha256) must fail closed the same way.
        self._write_inputs()
        report = json.loads(self.report_path.read_text())
        report["inputs"]["split_sha256"] = "0" * 64
        write_json(self.report_path, report)
        before = self.report_path.read_text()
        self.assertEqual(self._run("promote"), 2)
        self.assertEqual(self.report_path.read_text(), before)
        self.assertFalse(self.registry_path.exists())

    def test_exact_threshold_promotion_is_not_over(self) -> None:
        # Strictly above 0.05: a promotion gap of exactly 0.05 is not
        # over threshold, so the streak stays 0. A >= mutant fails here.
        self._write_inputs(val_accuracy=0.75, holdout_f1=1.0, holdout_acc=0.7)
        self.assertEqual(self._run("promote"), 0)
        gap = self._report()["val_minus_holdout"]
        self.assertEqual(gap["promotion_metric"], 0.05)
        self.assertEqual(gap["segment_f1"], 0.0)
        self.assertEqual(gap["streak_promoted_runs_over_threshold"], 0)
        self.assertFalse(gap["flagged"])

    def test_exact_threshold_segment_is_not_over(self) -> None:
        # Same boundary on the segment-F1 side.
        self._write_inputs(val_accuracy=0.7, holdout_f1=0.95, holdout_acc=0.7)
        self.assertEqual(self._run("promote"), 0)
        gap = self._report()["val_minus_holdout"]
        self.assertEqual(gap["promotion_metric"], 0.0)
        self.assertEqual(gap["segment_f1"], 0.05)
        self.assertEqual(gap["streak_promoted_runs_over_threshold"], 0)
        self.assertFalse(gap["flagged"])

    def test_malformed_prior_gap_fails_closed(self) -> None:
        # Replay validates recorded gaps: a non-numeric gap raises
        # GapError (exit 2), and a missing gap key fails instead of
        # silently reading as 0.
        self._write_inputs()
        for bad_gaps in [
            {"promotion_metric": "0.2", "segment_f1": 0.1},
            {"promotion_metric": 0.2},
        ]:
            prior = gap_record("r1")
            prior["val_minus_holdout"] = bad_gaps
            write_json(self.registry_path, [prior])
            self.assertEqual(self._run("promote"), 2)

    def test_val_segment_outside_val_fails(self) -> None:
        self._write_inputs(
            val_segments=[
                {
                    "sample_id": "h1",
                    "name": "cork",
                    "start_frame": 0,
                    "end_frame": 8,
                    "is_combo": False,
                }
            ]
        )
        self.assertEqual(self._run("promote"), 2)

    def test_missing_report_key_fails(self) -> None:
        self._write_inputs()
        report = json.loads(self.report_path.read_text())
        del report["f1"]
        write_json(self.report_path, report)
        self.assertEqual(self._run("promote"), 2)

    # -- run id --------------------------------------------------------------

    def test_record_carries_training_run_id(self) -> None:
        self._write_inputs()
        self.assertEqual(self._run("promote", run_id="40296249195-1"), 0)
        (record,) = self._registry()
        self.assertEqual(record["run_id"], "40296249195-1")
        self.assertNotIn("run_id", self._report()["val_minus_holdout"])

    def test_run_id_falls_back_to_an_input_hash(self) -> None:
        self._write_inputs()
        self.assertEqual(self._run("promote"), 0)
        (record,) = self._registry()
        self.assertRegex(record["run_id"], r"^sha256:[0-9a-f]{64}$")

    def test_rerun_of_the_same_run_does_not_inflate_the_streak(self) -> None:
        # The R2 retry case: the step ran, appended its record, and runs
        # again for the same run. Streak 3 stays 3, the log stays as is.
        for run_id in ("40296249195-1", None):  # TRAINING_RUN_ID, input hash
            with self.subTest(run_id=run_id):
                self._write_inputs()
                write_json(self.registry_path, [gap_record("r1"), gap_record("r2")])
                self.assertEqual(self._run("promote", run_id=run_id), 0)
                report = self.report_path.read_text()
                registry = self.registry_path.read_text()
                self.assertEqual(self._run("promote", run_id=run_id), 0)
                self.assertEqual(self.registry_path.read_text(), registry)
                self.assertEqual(self.report_path.read_text(), report)
                gap = self._report()["val_minus_holdout"]
                self.assertEqual(gap["streak_promoted_runs_over_threshold"], 3)
                self.assertTrue(gap["flagged"])

    def test_rerun_reports_the_streak_as_of_its_own_record(self) -> None:
        # A re-run of an earlier run keeps that run's streak (2), not the
        # streak after a later archive reset it.
        self._write_inputs()
        write_json(self.registry_path, [gap_record("r1")])
        self.assertEqual(self._run("promote", run_id="r2"), 0)
        log = self._registry() + [gap_record("r3", promotion="archive")]
        write_json(self.registry_path, log)
        before = self.registry_path.read_text()
        self.assertEqual(self._run("promote", run_id="r2"), 0)
        self.assertEqual(self.registry_path.read_text(), before)
        gap = self._report()["val_minus_holdout"]
        self.assertEqual(gap["streak_promoted_runs_over_threshold"], 2)

    def test_run_id_with_different_results_fails_closed(self) -> None:
        self._write_inputs()
        self.assertEqual(self._run("promote", run_id="r1"), 0)
        report = self.report_path.read_text()
        registry = self.registry_path.read_text()
        self._write_inputs(val_accuracy=0.8)
        write_json(self.report_path, json.loads(report))
        self.assertEqual(self._run("promote", run_id="r1"), 2)
        self.assertEqual(self.registry_path.read_text(), registry)
        self.assertEqual(self.report_path.read_text(), report)

    def test_registry_record_without_run_id_fails_closed(self) -> None:
        self._write_inputs()
        prior = gap_record("r1")
        del prior["run_id"]
        write_json(self.registry_path, [prior])
        self.assertEqual(self._run("promote"), 2)

    def test_registry_with_a_run_recorded_twice_fails_closed(self) -> None:
        self._write_inputs()
        write_json(self.registry_path, [gap_record("r1"), gap_record("r1")])
        self.assertEqual(self._run("promote"), 2)


class LikeForLikeTest(LeakageGapTestCase):
    """The promotion-metric gap compares one metric on both splits: on a
    real train.py run, window_accuracy over val reproduces train.py's
    val_accuracy exactly, so its holdout value is the same definition."""

    def setUp(self) -> None:
        super().setUp()
        # Pin source-c (smoke-007/008) to holdout, as test_predict.py does:
        # every val label is still seen in training.
        self.split = json.loads((SMOKE / "split.json").read_text())
        self.split.update({"smoke-007": "holdout", "smoke-008": "holdout"})
        write_json(self.split_path, self.split)
        config = self.tmp / "config.toml"
        lines = SMOKE_CONFIG.read_text().splitlines()
        lines = [
            ln
            for ln in lines
            if not ln.startswith(("split_path", "samples_dir", "epochs"))
        ]
        # One epoch keeps val accuracy strictly between 0 and 1, so the
        # equality below can't hold vacuously.
        lines += [
            f'split_path = "{self.split_path}"',
            f'samples_dir = "{SMOKE / "samples"}"',
            "epochs = 1",
        ]
        config.write_text("\n".join(lines) + "\n")
        run = self.tmp / "run"
        self.metrics = train.run(config, str(run))
        self.model_path = run / "model.npz"
        self.metrics_path = run / "metrics.json"
        self.samples_dir = SMOKE / "samples"
        predictions = self.tmp / "holdout_predictions.json"
        argv = ["--model", str(self.model_path), "--split", str(self.split_path)]
        argv += ["--samples-dir", str(self.samples_dir), "--out", str(predictions)]
        self.assertEqual(predict.main(argv), 0)
        argv = ["--predictions", str(predictions), "--split", str(self.split_path)]
        argv += ["--samples-dir", str(self.samples_dir), "--out", str(self.report_path)]
        self.assertEqual(evaluate.main(argv), 0)

    def accuracy(self, split_name: str) -> tuple[float, int, int]:
        ckpt = train.load_checkpoint(self.model_path)
        samples = leakage_gap.load_split_samples(
            self.split, self.samples_dir, split_name
        )
        return leakage_gap.window_accuracy(ckpt, samples)

    def test_val_window_accuracy_reproduces_train_val_accuracy(self) -> None:
        val_accuracy, n_windows, unseen = self.accuracy("val")
        self.assertEqual(val_accuracy, self.metrics["val_accuracy"])
        self.assertEqual(n_windows, self.metrics["n_val_windows"])
        self.assertEqual(unseen, 0)
        self.assertGreater(val_accuracy, 0.0)
        self.assertLess(val_accuracy, 1.0)

    def test_gap_is_val_accuracy_minus_holdout_window_accuracy(self) -> None:
        holdout_accuracy, n_windows, _ = self.accuracy("holdout")
        self.assertEqual(self._run("archive", run_id="r1"), 0)
        gap = self._report()["val_minus_holdout"]
        self.assertEqual(gap["val_accuracy"], evaluate.r6(self.metrics["val_accuracy"]))
        self.assertEqual(gap["holdout_window_accuracy"], evaluate.r6(holdout_accuracy))
        self.assertEqual(
            gap["promotion_metric"],
            evaluate.r6(self.metrics["val_accuracy"] - holdout_accuracy),
        )
        self.assertEqual(gap["n_holdout_windows"], n_windows)

    def test_output_is_deterministic(self) -> None:
        # CONTRIBUTING rule 2: the same inputs twice -> byte-identical
        # report and the same registry record, decided_at aside.
        original = self.report_path.read_text()
        outputs = []
        for _ in range(2):
            self.report_path.write_text(original)
            self.registry_path.unlink(missing_ok=True)
            self.assertEqual(self._run("promote"), 0)
            (record,) = self._registry()
            del record["decided_at"]
            outputs.append((self.report_path.read_bytes(), record))
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
