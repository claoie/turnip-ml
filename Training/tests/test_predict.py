"""Unit tests for Training/predict.py.

Hermetic: trains the smoke model on the committed synthetic fixture.
Run with:  python -m unittest discover -s Training/tests -t .
"""

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import evaluate
import predict
import train

REPO = Path(__file__).resolve().parent.parent.parent
SMOKE = REPO / "Training" / "fixtures" / "smoke"
SMOKE_CONFIG = REPO / "Training" / "config" / "smoke.toml"


class PredictTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.stderr = self.enterContext(contextlib.redirect_stderr(io.StringIO()))

    def tearDown(self):
        self.tmp.cleanup()

    def train_on(self, split):
        """Train the smoke config on `split` (sample_id -> split name)."""
        split_path = self.root / "split.json"
        split_path.write_text(json.dumps(split, sort_keys=True))
        config = self.root / "config.toml"
        lines = SMOKE_CONFIG.read_text().splitlines()
        lines = [ln for ln in lines if not ln.startswith(("split_path", "samples_dir"))]
        lines += [
            f'split_path = "{split_path}"',
            f'samples_dir = "{SMOKE / "samples"}"',
        ]
        config.write_text("\n".join(lines) + "\n")
        out = self.root / "run"
        metrics = train.run(config, str(out))
        return out / "model.npz", split_path, metrics

    def predict(self, model, split_path, split_name="holdout", samples=None):
        out = self.root / f"pred-{split_name}.json"
        argv = [
            "--model",
            str(model),
            "--split",
            str(split_path),
            "--samples-dir",
            str(samples or SMOKE / "samples"),
            "--split-name",
            split_name,
            "--out",
            str(out),
        ]
        return predict.main(argv), out


def smoke_split_with_holdout():
    # Pin source-c (smoke-007/008) to holdout: train keeps source-a, val
    # stays source-b, and every val label is still seen in training.
    split = json.loads((SMOKE / "split.json").read_text())
    split.update({"smoke-007": "holdout", "smoke-008": "holdout"})
    return split


class PredictTest(PredictTestCase):
    def test_val_predictions_match_train_decoded_segments(self):
        model, split_path, metrics = self.train_on(
            json.loads((SMOKE / "split.json").read_text())
        )
        code, out = self.predict(model, split_path, "val")
        self.assertEqual(code, 0)
        predictions = json.loads(out.read_text())
        expected = {"smoke-005": [], "smoke-006": []}
        for seg in metrics["val_segments"]:
            expected[seg["sample_id"]].append(
                {
                    "start_frame": seg["start_frame"],
                    "end_frame": seg["end_frame"],
                    "names": sorted(seg["name"].split("+")),
                    "is_combo": seg["is_combo"],
                }
            )
        self.assertEqual(predictions, expected)
        self.assertTrue(metrics["val_segments"])  # not vacuously equal

    def test_holdout_predictions_feed_evaluate(self):
        model, split_path, _ = self.train_on(smoke_split_with_holdout())
        code, out = self.predict(model, split_path)
        self.assertEqual(code, 0)
        predictions = json.loads(out.read_text())
        self.assertEqual(sorted(predictions), ["smoke-007", "smoke-008"])
        report = self.root / "holdout_report.json"
        code = evaluate.main(
            [
                "--predictions",
                str(out),
                "--split",
                str(split_path),
                "--samples-dir",
                str(SMOKE / "samples"),
                "--out",
                str(report),
            ]
        )
        self.assertEqual(code, 0)
        metrics = json.loads(report.read_text())
        # smoke-008's single gainer segment is found and named.
        self.assertEqual(metrics["detection_rate"], 1.0)

    def test_ground_truth_never_reaches_predictions(self):
        model, split_path, _ = self.train_on(smoke_split_with_holdout())
        _, with_labels = self.predict(model, split_path)
        stripped = self.root / "stripped"
        shutil.copytree(SMOKE / "samples", stripped)
        for sid in ("smoke-007", "smoke-008"):
            path = stripped / f"{sid}.json"
            rec = json.loads(path.read_text())
            rec["tricks"] = [{"name": "decoy", "start_frame": 0, "end_frame": 32}]
            path.write_text(json.dumps(rec))
        first = with_labels.read_bytes()
        _, without = self.predict(model, split_path, samples=stripped)
        self.assertEqual(without.read_bytes(), first)

    def test_output_is_deterministic(self):
        model, split_path, _ = self.train_on(smoke_split_with_holdout())
        _, out = self.predict(model, split_path)
        first = out.read_bytes()
        _, again = self.predict(model, split_path)
        self.assertEqual(again.read_bytes(), first)


class FailClosedTest(PredictTestCase):
    def setUp(self):
        super().setUp()
        self.model, self.split_path, _ = self.train_on(smoke_split_with_holdout())

    def test_missing_or_malformed_model(self):
        bad = self.root / "bad.npz"
        bad.write_text("not a model")
        for model in (self.root / "missing.npz", bad):
            with self.subTest(model=model.name):
                code, out = self.predict(model, self.split_path)
                self.assertEqual(code, 2)
                self.assertFalse(out.exists())

    def test_empty_split(self):
        split = json.loads((SMOKE / "split.json").read_text())  # no holdout
        path = self.root / "no-holdout.json"
        path.write_text(json.dumps(split))
        code, _ = self.predict(self.model, path)
        self.assertEqual(code, 2)
        self.assertIn("no holdout samples", self.stderr.getvalue())

    def test_missing_or_mismatched_sample_file(self):
        samples = self.root / "samples"
        shutil.copytree(SMOKE / "samples", samples)
        (samples / "smoke-008.json").unlink()
        code, _ = self.predict(self.model, self.split_path, samples=samples)
        self.assertEqual(code, 2)
        shutil.copy(SMOKE / "samples" / "smoke-001.json", samples / "smoke-008.json")
        code, _ = self.predict(self.model, self.split_path, samples=samples)
        self.assertEqual(code, 2)

    def test_unknown_split_name(self):
        with self.assertRaises(SystemExit):
            self.predict(self.model, self.split_path, "test")


if __name__ == "__main__":
    unittest.main()
