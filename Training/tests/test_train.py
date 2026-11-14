"""Unit tests for Training/train.py (turnip-ml#13).

Run: python -m unittest discover -s Training/tests -t .
"""

from __future__ import annotations

import hashlib
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
SMOKE = REPO_ROOT / "Training" / "fixtures" / "smoke"


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

    def test_train_name_counts_cover_the_train_split_only(self) -> None:
        self.assertEqual(self._run(), 0)
        m = self._metrics()
        split = json.loads((SMOKE / "split.json").read_text())
        want: dict[str, int] = {}
        for sid, part in split.items():
            if part != "train":
                continue
            sample = json.loads((SMOKE / "samples" / f"{sid}.json").read_text())
            for trick in sample["tricks"]:
                for name in set(trick["name"].split("+")):
                    want[name] = want.get(name, 0) + 1
        self.assertEqual(m["train_name_counts"], want)
        self.assertEqual(
            m["split_sha256"],
            hashlib.sha256((SMOKE / "split.json").read_bytes()).hexdigest(),
        )

    def test_combo_segments_count_for_each_name(self) -> None:
        samples = [
            {"tricks": [{"name": "cork+gainer"}, {"name": "gainer"}]},
            {"tricks": [{"name": "gainer+gainer"}]},
            {},
        ]
        self.assertEqual(train.train_name_counts(samples), {"cork": 1, "gainer": 3})

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

    def test_background_margin_excludes_boundary_windows(self) -> None:
        # TRAINING_DESIGN.md §2.5: windows within ±2 s of a labeled segment
        # are excluded from the background pool, not labeled background.
        # Fixture is 10 Hz, so the 2 s margin is 20 frames.
        segments = evaluate.ground_truth_segments(
            [{"name": "gainer", "start_frame": 100, "end_frame": 116}],
            "test",
        )
        margin = 20
        # Far from the segment: background, with or without the margin.
        self.assertEqual(train.label_for_window(segments, 0, 8), "background")
        self.assertEqual(train.label_for_window(segments, 0, 8, margin), "background")
        # Ending 12 frames before the segment starts: inside the margin.
        self.assertEqual(train.label_for_window(segments, 80, 88, margin), None)
        # Touching the segment with under-half overlap: gap 0, excluded.
        self.assertEqual(train.label_for_window(segments, 92, 100, margin), None)
        self.assertEqual(train.label_for_window(segments, 94, 102, margin), None)
        # Exactly margin_frames away: background, not excluded.
        self.assertEqual(train.label_for_window(segments, 72, 80, margin), "background")
        # Without the margin the same window is background (default 0).
        self.assertEqual(train.label_for_window(segments, 80, 88), "background")

    def test_background_margin_never_demotes_a_trick_window(self) -> None:
        # >= half overlap labels the window even inside another segment's
        # margin: the margin excludes background candidates only.
        segments = evaluate.ground_truth_segments(
            [
                {"name": "gainer", "start_frame": 100, "end_frame": 116},
                {"name": "corkscrew", "start_frame": 120, "end_frame": 136},
            ],
            "test",
        )
        self.assertEqual(train.label_for_window(segments, 108, 116, 20), "gainer")

    def test_build_dataset_drops_margin_windows(self) -> None:
        # One 16-frame segment at [100, 116) in a 200-frame sample, 8-frame
        # windows at stride 8: 25 windows, of which the 4 inside the 20-frame
        # margin (starts 80, 88, 120, 128) drop out entirely — no row, so
        # they never enter the background pool.
        sample = {
            "sample_id": "s",
            "keypoints": np.zeros((200, 13, 3)),
            "tricks": [
                {
                    "clip_id": "c1",
                    "name": "gainer",
                    "start_frame": 100,
                    "end_frame": 116,
                }
            ],
        }
        _, labels, prov = train.build_dataset(np, [sample], 8, 8, margin_frames=20)
        self.assertEqual(len(labels), 21)
        self.assertEqual(labels.count("background"), 18)
        self.assertEqual(labels.count("gainer"), 3)
        self.assertEqual(
            sorted(start for _, start, _ in prov if start in (80, 88, 120, 128)),
            [],
        )

    def test_window_features_ignore_where_the_sequence_is_centered(self) -> None:
        # The farm centers on the segment, the client on the take: a constant
        # shift of every present joint must not change a window's features
        # (MODEL_CONTRACT.md §3.6.2), missing joints and gap frames included.
        rng = np.random.default_rng(0)
        window = np.concatenate(
            [rng.normal(size=(8, 13, 2)), np.full((8, 13, 1), 0.9)], axis=2
        )
        window[2] = 0.0  # a gap frame
        window[5, 3] = 0.0  # a missing elbow
        window[6, 7] = 0.0  # a missing hip: no root in this frame
        shifted = window.copy()
        present = shifted[:, :, 2] > 0
        shifted[:, :, 0][present] += 3.25
        shifted[:, :, 1][present] -= 1.5
        a = train.featurize_window(np, window)
        b = train.featurize_window(np, shifted)
        self.assertEqual(a.shape, (52,))
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_window_features_by_hand(self) -> None:
        # Two frames, the hips at (+-0.2, 0) then (+-0.2, 1): the root moves
        # down by 1. The head sits 1 above the root in both frames.
        window = np.zeros((2, 13, 3))
        for f, y in enumerate((0.0, 1.0)):
            window[f, 7] = [0.2, y, 1.0]
            window[f, 8] = [-0.2, y, 1.0]
            window[f, 0] = [0.0, y - 1.0, 1.0]
        window[1, 0] = 0.0  # ...but the head is missing in frame 1
        feat = train.featurize_window(np, window)
        pos, speed = feat[:26].reshape(13, 2), feat[26:].reshape(13, 2)
        # Root = mean hip midpoint = (0, 0.5). The head is present in one of
        # two frames, at (0, -1): (position - root) averaged over 2 frames.
        np.testing.assert_allclose(pos[0], [0.0, -0.75])
        np.testing.assert_allclose(pos[7], [0.2, 0.0])
        # Speed counts only pairs where the joint is present in both frames:
        # the hips moved by 1 in y; the head has no such pair.
        np.testing.assert_allclose(speed[7], [0.0, 1.0])
        np.testing.assert_allclose(speed[0], [0.0, 0.0])
        # A missing joint is all zeros.
        np.testing.assert_allclose(pos[3], [0.0, 0.0])

    def test_a_window_without_hips_has_no_root(self) -> None:
        window = np.zeros((4, 13, 3))
        window[:, 0] = [0.5, -1.0, 1.0]  # only the head is seen
        pos = train.featurize_window(np, window)[:26].reshape(13, 2)
        np.testing.assert_allclose(pos[0], [0.5, -1.0])

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
            "keypoints": np.zeros((16, 13, 3)),
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
            "keypoints": np.zeros((16, 13, 3)),
            "tricks": tricks,
        }
        with self.assertRaises(ValueError):
            train.build_dataset(np, [sample], 8, 8)

    # -- class-balance sampling (turnip-ml#38, TRAINING_DESIGN.md §2.7) ----

    def test_class_balance_weights_match_formula(self) -> None:
        # 100:1 name skew, weight ∝ 1/√count: the rare window weighs 10x
        # the frequent one per window.
        labels = ["gainer"] * 3 + ["backflip", "background"]
        weights, is_bg = train.window_sampling_weights(
            labels, {"gainer": 100, "backflip": 1}, 0.5
        )
        self.assertEqual(list(weights[:3]), [0.1, 0.1, 0.1])
        self.assertEqual(weights[3], 1.0)
        self.assertEqual(weights[4], 0.0)
        self.assertEqual(list(is_bg), [False, False, False, False, True])

    def test_combo_window_takes_rarest_name_weight(self) -> None:
        weights, _ = train.window_sampling_weights(
            ["corkscrew+gainer"], {"corkscrew": 100, "gainer": 1}, 0.5
        )
        self.assertEqual(weights[0], 1.0)  # gainer's, not corkscrew's

    def test_window_label_with_unknown_name_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            train.window_sampling_weights(["gainer"], {"corkscrew": 5}, 0.5)

    def test_background_sampled_separately_from_name_weights(self) -> None:
        # Background windows never enter the name-weight stream, and their
        # count is the ratio's alone: name weights can't move it.
        name_idx = np.arange(10, dtype=np.int64)
        bg_idx = np.arange(10, 15, dtype=np.int64)
        samplers = [
            train.make_epoch_sampler(name_idx, np.ones(10), bg_idx, 0.25),
            train.make_epoch_sampler(
                name_idx, np.array([1.0] + [0.01] * 9), bg_idx, 0.25
            ),
        ]
        n_bg = train.background_windows_per_epoch(10, 0.25)
        self.assertEqual(n_bg, 3)  # round(0.25 / 0.75 * 10)
        for i, sampler in enumerate(samplers):
            idx = sampler(np.random.default_rng(7))
            self.assertEqual(len(idx), 13, i)
            self.assertEqual(int((idx >= 10).sum()), n_bg, i)

    def test_background_ratio_zero_samples_no_background(self) -> None:
        sampler = train.make_epoch_sampler(
            np.arange(4, dtype=np.int64),
            np.ones(4),
            np.empty(0, dtype=np.int64),
            0.0,
        )
        idx = sampler(np.random.default_rng(1))
        self.assertEqual(len(idx), 4)
        self.assertTrue((idx < 4).all())

    def test_background_ratio_with_no_background_windows_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            train.make_epoch_sampler(
                np.arange(4, dtype=np.int64),
                np.ones(4),
                np.empty(0, dtype=np.int64),
                0.25,
            )

    def test_epoch_sampling_is_seed_deterministic(self) -> None:
        sampler = train.make_epoch_sampler(
            np.arange(6, dtype=np.int64),
            np.ones(6),
            np.arange(6, 9, dtype=np.int64),
            0.25,
        )
        a = sampler(np.random.default_rng(42))
        b = sampler(np.random.default_rng(42))
        self.assertTrue(np.array_equal(a, b))
        c = sampler(np.random.default_rng(43))
        self.assertFalse(np.array_equal(a, c))

    def test_skewed_names_upsample_per_formula(self) -> None:
        # 100:1 name skew, weight ∝ 1/√count: the rare window is drawn ~10x
        # as often per window as a frequent one (seed-deterministic).
        name_idx = np.arange(101, dtype=np.int64)  # idx 0 = the rare window
        labels = ["rare"] + ["frequent"] * 100
        weights, _ = train.window_sampling_weights(
            labels, {"frequent": 100, "rare": 1}, 0.5
        )
        sampler = train.make_epoch_sampler(
            name_idx, weights, np.empty(0, dtype=np.int64), 0.0
        )
        rng = np.random.default_rng(1234)
        rare_draws = 0
        freq_draws = 0
        for _ in range(300):
            idx = sampler(rng)
            rare_draws += int((idx == 0).sum())
            freq_draws += int((idx != 0).sum())
        ratio = (rare_draws / 1) / (freq_draws / 100)
        self.assertGreater(ratio, 8.0)
        self.assertLess(ratio, 12.0)

    def test_metrics_records_sampling_weights(self) -> None:
        self.assertEqual(self._run(), 0)
        m = self._metrics()
        s = m["sampling"]
        self.assertEqual(s["class_balance_exponent"], 0.5)
        self.assertEqual(s["background_ratio"], 0.2)
        counts = m["train_name_counts"]
        self.assertEqual(list(s["window_weight_per_name"]), sorted(counts))
        want = {name: counts[name] ** -0.5 for name in counts}
        self.assertEqual(s["window_weight_per_name"], want)
        self.assertEqual(
            s["n_background_windows_per_epoch"],
            train.background_windows_per_epoch(s["n_name_windows_per_epoch"], 0.2),
        )
        # The smoke train split really does have both streams.
        self.assertGreater(s["n_name_windows_per_epoch"], 0)
        self.assertGreater(s["n_background_windows_per_epoch"], 0)

    def test_negative_class_balance_exponent_fails_closed(self) -> None:
        p = self._rewrite_config(
            "class_balance_exponent = 0.5", "class_balance_exponent = -0.5"
        )
        self.assertEqual(self._run(config=p), 2)

    def test_background_ratio_must_be_below_one(self) -> None:
        p = self._rewrite_config("background_ratio = 0.2", "background_ratio = 1.0")
        self.assertEqual(self._run(config=p), 2)


if __name__ == "__main__":
    unittest.main()
