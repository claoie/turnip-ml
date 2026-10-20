"""Unit tests for Training/train.py (turnip-ml#13).

Run: python -m unittest discover -s Training/tests -t .
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# train.py imports its sibling evaluate.py by module name, as when run as a
# script from Training/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Training import evaluate, train

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SMOKE_CONFIG = REPO_ROOT / "Training" / "config" / "smoke.toml"


class TrainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="train-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.out = self.tmp / "run"

    def _run(self, config: Path = SMOKE_CONFIG, out: Path | None = None) -> int:
        return train.main(["--config", str(config), "--out", str(out or self.out)])

    def _metrics(self, out: Path | None = None) -> dict:
        with open((out or self.out) / "metrics.json", encoding="utf-8") as fh:
            return json.load(fh)

    def test_smoke_end_to_end(self) -> None:
        self.assertEqual(self._run(), 0)
        for name in ("config.toml", "model.npz", "metrics.json"):
            self.assertTrue((self.out / name).exists(), name)
        # The fixture is separable by construction: anything less than
        # perfect val accuracy is a training-pipeline bug, not data noise.
        m = self._metrics()
        self.assertEqual(m["val_accuracy"], 1.0)
        self.assertEqual(
            (self.out / "config.toml").read_bytes(), SMOKE_CONFIG.read_bytes()
        )
        # Decoded segments honor the output contract: source-frame coords,
        # one segment with N names, derived is_combo.
        self.assertTrue(m["val_segments"])
        for seg in m["val_segments"]:
            self.assertLess(seg["start_frame"], seg["end_frame"])
            self.assertEqual(seg["is_combo"], "+" in seg["name"])

    def test_loss_decreases(self) -> None:
        self._run()
        m = self._metrics()
        self.assertLess(m["train_loss_final"], m["train_loss_initial"])

    def test_deterministic(self) -> None:
        """Same config + data => byte-identical metrics (Training contract)."""
        out2 = self.tmp / "run2"
        self.assertEqual(self._run(out=self.out), 0)
        self.assertEqual(self._run(out=out2), 0)
        self.assertEqual(
            (self.out / "metrics.json").read_bytes(),
            (out2 / "metrics.json").read_bytes(),
        )
        # np.savez embeds zip timestamps, so compare arrays, not bytes.
        a, b = np.load(self.out / "model.npz"), np.load(out2 / "model.npz")
        self.assertEqual(set(a.files), set(b.files))
        for k in a.files:
            self.assertTrue(np.array_equal(a[k], b[k]), k)

    def test_finetune_from_checkpoint(self) -> None:
        self.assertEqual(self._run(), 0)
        ckpt = self.out / "model.npz"
        ft_config = self.tmp / "ft.toml"
        ft_config.write_text(
            SMOKE_CONFIG.read_text().replace(
                'init_checkpoint = ""', f'init_checkpoint = "{ckpt}"'
            ),
            encoding="utf-8",
        )
        out2 = self.tmp / "finetuned"
        self.assertEqual(self._run(config=ft_config, out=out2), 0)
        m = self._metrics(out2)
        self.assertEqual(m["val_accuracy"], 1.0)
        self.assertEqual(m["init_checkpoint"], str(ckpt))
        self.assertLessEqual(m["train_loss_final"], self._metrics()["train_loss_final"])
        # Discrimination: the fine-tune run must *start* from the checkpoint,
        # not silently retrain from scratch. `init_checkpoint` in the metrics
        # only echoes the config, so a mutant that forces `init` to None in
        # `run()` still passes every assertion above -- but its initial loss
        # equals the from-scratch run's instead of the checkpoint's.
        scratch = self._metrics()
        self.assertLess(m["train_loss_initial"], scratch["train_loss_initial"])

    def _rewrite_config(self, old: str, new: str) -> Path:
        p = self.tmp / "t.toml"
        p.write_text(SMOKE_CONFIG.read_text().replace(old, new), encoding="utf-8")
        return p

    def test_unknown_config_key_fails_closed(self) -> None:
        p = self.tmp / "bad.toml"
        p.write_text("seed = 0\nbogus_key = 1\n", encoding="utf-8")
        self.assertEqual(self._run(config=p), 2)

    def test_gpu_compute_fails_closed(self) -> None:
        p = self._rewrite_config('compute = "cpu"', 'compute = "gpu"')
        self.assertEqual(self._run(config=p), 2)

    def test_holdout_never_trained(self) -> None:
        """A split with no train samples refuses to run (poisoning defense)."""
        split = {
            k: "holdout"
            for k in json.loads(
                (
                    REPO_ROOT / "Training" / "fixtures" / "smoke" / "split.json"
                ).read_text(encoding="utf-8")
            )
        }
        split_path = self.tmp / "split.json"
        split_path.write_text(json.dumps(split), encoding="utf-8")
        p = self._rewrite_config(
            'split_path = "Training/fixtures/smoke/split.json"',
            f'split_path = "{split_path}"',
        )
        self.assertEqual(self._run(config=p), 2)

    def test_sample_missing_from_split_fails_closed(self) -> None:
        split = json.loads(
            (REPO_ROOT / "Training" / "fixtures" / "smoke" / "split.json").read_text(
                encoding="utf-8"
            )
        )
        del split["smoke-001"]
        split_path = self.tmp / "split.json"
        split_path.write_text(json.dumps(split), encoding="utf-8")
        p = self._rewrite_config(
            'split_path = "Training/fixtures/smoke/split.json"',
            f'split_path = "{split_path}"',
        )
        self.assertEqual(self._run(config=p), 2)

    def test_combo_label_and_decode(self) -> None:
        segments = evaluate.ground_truth_segments(
            [
                {"name": "gainer", "start_frame": 0, "end_frame": 16},
                {"name": "corkscrew", "start_frame": 0, "end_frame": 16},
            ],
            "test",
        )
        self.assertEqual(train.label_for_window(segments, 0, 8), "corkscrew+gainer")
        self.assertEqual(train.label_for_window(segments, 100, 108), "background")
        prov = [
            ("s", 0, "corkscrew+gainer"),
            ("s", 8, "corkscrew+gainer"),
            ("s", 16, "background"),
        ]
        segs = train.decode_segments(
            prov, ["corkscrew+gainer", "corkscrew+gainer", "background"], 8
        )
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0]["name"], "corkscrew+gainer")
        self.assertTrue(segs[0]["is_combo"])
        self.assertEqual((segs[0]["start_frame"], segs[0]["end_frame"]), (0, 16))

    def test_window_label_counts_each_name_once(self) -> None:
        # Two clips labeling overlapping spans "gainer": the window is
        # "gainer", not "gainer+gainer".
        segments = evaluate.ground_truth_segments(
            [
                {"clip_id": "c1", "name": "gainer", "start_frame": 0, "end_frame": 16},
                {"clip_id": "c2", "name": "gainer", "start_frame": 2, "end_frame": 16},
            ],
            "test",
        )
        self.assertEqual(train.label_for_window(segments, 0, 8), "gainer")

    def test_window_label_matches_evaluation_ground_truth(self) -> None:
        # fetch_dataset's combo clip (one entry per name): the label train
        # learns inside it is exactly the segment evaluate.py scores against.
        tricks = [
            {"clip_id": "c1", "name": n, "start_frame": 0, "end_frame": 16}
            for n in ("gainer", "corkscrew")
        ]
        [(_, _, names)] = evaluate.ground_truth_segments(tricks, "test")
        sample = {
            "sample_id": "s",
            "keypoints": np.zeros((16, 17, 3)),
            "tricks": tricks,
        }
        _, labels, _ = train.build_dataset(np, [sample], 8, 8)
        self.assertEqual(labels, ["+".join(sorted(names))] * 2)

    def test_build_dataset_refuses_a_clip_split_across_windows(self) -> None:
        tricks = [
            {"clip_id": "c1", "name": "gainer", "start_frame": 0, "end_frame": 16},
            {"clip_id": "c1", "name": "corkscrew", "start_frame": 0, "end_frame": 12},
        ]
        sample = {
            "sample_id": "s",
            "keypoints": np.zeros((16, 17, 3)),
            "tricks": tricks,
        }
        with self.assertRaises(ValueError):
            train.build_dataset(np, [sample], 8, 8)


if __name__ == "__main__":
    unittest.main()
