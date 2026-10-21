"""Unit tests for Training/evaluate.py.

Run: python -m unittest discover -s Training/tests -t .
"""

from __future__ import annotations

import itertools
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from Training import evaluate


def write_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


class EvaluateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="evaluate-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.samples_dir = self.tmp / "samples"
        self.samples_dir.mkdir()
        self.split_path = self.tmp / "split.json"
        self.pred_path = self.tmp / "predictions.json"
        self.out = self.tmp / "report.json"

    def _sample(self, sid: str, tricks: list[dict]) -> None:
        write_json(
            self.samples_dir / f"{sid}.json",
            {"sample_id": sid, "fps": 10, "tricks": tricks},
        )

    def _split(self, mapping: dict[str, str]) -> None:
        write_json(self.split_path, mapping)

    def _predictions(self, mapping: dict) -> None:
        write_json(self.pred_path, mapping)

    def _argv(self, *extra: str) -> list[str]:
        return [
            "--predictions",
            str(self.pred_path),
            "--split",
            str(self.split_path),
            "--samples-dir",
            str(self.samples_dir),
            "--out",
            str(self.out),
            *extra,
        ]

    def _run(self, *extra: str) -> dict:
        self.assertEqual(evaluate.main(self._argv(*extra)), 0)
        return json.loads(self.out.read_text(encoding="utf-8"))

    def _assert_refused(self) -> ValueError:
        with self.assertRaises(ValueError) as cm:
            evaluate.main(self._argv())
        return cm.exception

    def _basic_holdout(self) -> None:
        # One holdout sample with a single trick; one train sample.
        self._sample("h1", [{"name": "gainer", "start_frame": 4, "end_frame": 20}])
        self._sample("t1", [{"name": "gainer", "start_frame": 0, "end_frame": 8}])
        self._split({"h1": "holdout", "t1": "train"})

    def test_perfect_predictions(self) -> None:
        self._basic_holdout()
        self._predictions(
            {"h1": [{"start_frame": 4, "end_frame": 20, "name": "gainer"}]}
        )
        report = self._run()
        self.assertEqual(report["n_holdout_samples"], 1)
        self.assertEqual(report["n_ground_truth_segments"], 1)
        self.assertEqual(report["n_predicted_segments"], 1)
        self.assertEqual(report["detection_rate"], 1.0)
        self.assertEqual(report["precision"], 1.0)
        self.assertEqual(report["recall"], 1.0)
        self.assertEqual(report["f1"], 1.0)
        self.assertEqual(report["name_accuracy_given_detection"], 1.0)
        self.assertEqual(report["n_background_false_positives"], 0)
        self.assertEqual(report["n_unmatched_predictions"], 0)
        self.assertEqual(
            report["per_name"]["gainer"],
            {
                "support": 1,
                "predicted": 1,
                "true_positives": 1,
                "precision": 1.0,
                "recall": 1.0,
            },
        )

    def test_empty_predictions(self) -> None:
        self._basic_holdout()
        self._predictions({})
        report = self._run()
        self.assertEqual(report["detection_rate"], 0.0)
        self.assertEqual(report["recall"], 0.0)
        self.assertEqual(report["precision"], 0.0)
        self.assertEqual(report["f1"], 0.0)
        self.assertEqual(report["per_sample"]["h1"]["n_detections"], 0)

    def test_no_tricks_no_predictions_is_vacuous(self) -> None:
        self._sample("h1", [])
        self._split({"h1": "holdout"})
        self._predictions({})
        report = self._run()
        self.assertEqual(report["precision"], 1.0)
        self.assertEqual(report["recall"], 1.0)
        self.assertEqual(report["f1"], 1.0)
        self.assertEqual(report["detection_rate"], 1.0)

    def test_byte_identical_reruns(self) -> None:
        self._basic_holdout()
        self._predictions(
            {
                "h1": [
                    {"start_frame": 4, "end_frame": 20, "name": "gainer"},
                    {"start_frame": 40, "end_frame": 48, "name": "cork"},
                ]
            }
        )
        self.assertEqual(evaluate.main(self._argv()), 0)
        first = self.out.read_bytes()
        self.assertEqual(evaluate.main(self._argv()), 0)
        self.assertEqual(self.out.read_bytes(), first)

    def test_iou_threshold_boundary(self) -> None:
        # GT 4..20 (len 16); pred 14..24 (len 10) overlaps 6 frames:
        # IoU = 6 / (16 + 10 - 6) = 6/20 = 0.3. Threshold 0.25 detects it,
        # but the no-intersection-subtraction mutant (IoU = 6/26 ~= 0.23)
        # stays below 0.25, so the subtraction in temporal_iou is guarded.
        self._basic_holdout()
        self._predictions(
            {"h1": [{"start_frame": 14, "end_frame": 24, "name": "gainer"}]}
        )
        default = self._run()
        self.assertEqual(default["n_detections"], 0)
        lowered = self._run("--iou-threshold", "0.25")
        self.assertEqual(lowered["iou_threshold"], 0.25)
        self.assertEqual(lowered["n_detections"], 1)
        self.assertEqual(lowered["n_name_matches"], 1)

    def test_greedy_best_iou_matching(self) -> None:
        # GT (0,10)/gainer with competing preds (0,6)/cork (IoU 0.6) and
        # (0,9)/gainer (IoU 0.9): greedy best-IoU matches gainer and keeps
        # the name match; first-fit would match cork and lose it.
        self._sample("h1", [{"name": "gainer", "start_frame": 0, "end_frame": 10}])
        self._split({"h1": "holdout"})
        self._predictions(
            {
                "h1": [
                    {"start_frame": 0, "end_frame": 6, "name": "cork"},
                    {"start_frame": 0, "end_frame": 9, "name": "gainer"},
                ]
            }
        )
        report = self._run()
        self.assertEqual(report["n_detections"], 1)
        self.assertEqual(report["n_name_matches"], 1)
        self.assertEqual(report["n_unmatched_predictions"], 1)

    def test_iou_threshold_boundary_is_inclusive(self) -> None:
        # GT (4,20) vs pred (12,20): overlap 8, union 16 + 8 - 8 = 16,
        # IoU = 8/16 = 0.5 exactly. The spec'd "IoU >= threshold" detects at
        # threshold 0.5; a strict ">" would drop it.
        self._sample("h1", [{"name": "gainer", "start_frame": 4, "end_frame": 20}])
        self._split({"h1": "holdout"})
        self._predictions(
            {"h1": [{"start_frame": 12, "end_frame": 20, "name": "gainer"}]}
        )
        report = self._run()
        self.assertEqual(report["n_detections"], 1)
        self.assertEqual(report["n_name_matches"], 1)

    def test_name_mismatch_is_detection_not_match(self) -> None:
        self._basic_holdout()
        self._predictions({"h1": [{"start_frame": 4, "end_frame": 20, "name": "cork"}]})
        report = self._run()
        self.assertEqual(report["detection_rate"], 1.0)  # localized...
        self.assertEqual(report["precision"], 0.0)  # ...but misnamed
        self.assertEqual(report["recall"], 0.0)
        self.assertEqual(report["name_accuracy_given_detection"], 0.0)
        self.assertEqual(report["per_name"]["cork"]["predicted"], 1)
        self.assertEqual(report["per_name"]["cork"]["true_positives"], 0)

    def test_combo_name_forms_are_equal(self) -> None:
        self._sample("h1", [{"name": "gainer+cork", "start_frame": 4, "end_frame": 20}])
        self._split({"h1": "holdout"})
        self._predictions(
            {
                "h1": [
                    {
                        "start_frame": 4,
                        "end_frame": 20,
                        "names": ["cork", "gainer"],
                        "is_combo": True,
                    }
                ]
            }
        )
        report = self._run()
        self.assertEqual(report["n_name_matches"], 1)
        self.assertEqual(report["precision"], 1.0)

    # fetch_dataset.py's shape: one entry per (clip_id, canonical name), all
    # on the clip's window.
    CLIP_A = "0b6f1c3e-6a51-4f47-9d57-1f4f3c0a2b11"
    CLIP_B = "7d2e9a40-3c18-4b6e-a2f5-8e0c51d94c22"

    def _combo_clip(self, **extra: object) -> list[dict]:
        return [
            {
                "clip_id": self.CLIP_A,
                "name": n,
                "start_frame": 4,
                "end_frame": 20,
                **extra,
            }
            for n in ("cork", "gainer")
        ]

    def test_combo_clip_is_one_segment_and_name_matches(self) -> None:
        # train.py labels this window "cork+gainer"; predict.py emits it as
        # names ["cork", "gainer"]. Both forms must name-match the clip.
        self._sample("h1", self._combo_clip())
        self._split({"h1": "holdout"})
        for pred in (
            {"names": ["cork", "gainer"], "is_combo": True},
            {"name": "cork+gainer"},
        ):
            with self.subTest(pred=pred):
                self._predictions({"h1": [{"start_frame": 4, "end_frame": 20, **pred}]})
                report = self._run()
                self.assertEqual(report["scoring_version"], evaluate.SCORING_VERSION)
                self.assertEqual(report["n_ground_truth_segments"], 1)
                self.assertEqual(report["n_detections"], 1)
                self.assertEqual(report["n_name_matches"], 1)
                self.assertEqual(report["precision"], 1.0)
                self.assertEqual(report["recall"], 1.0)
                self.assertEqual(report["name_accuracy_given_detection"], 1.0)
                self.assertEqual(report["n_unmatched_predictions"], 0)
                for name in ("cork", "gainer"):
                    self.assertEqual(report["per_name"][name]["support"], 1)
                    self.assertEqual(report["per_name"][name]["true_positives"], 1)
                    self.assertEqual(report["per_name"][name]["recall"], 1.0)

    def test_combo_clip_needs_every_name(self) -> None:
        # Naming one trick of a combo clip localizes it but is not a match.
        self._sample("h1", self._combo_clip())
        self._split({"h1": "holdout"})
        self._predictions({"h1": [{"start_frame": 4, "end_frame": 20, "name": "cork"}]})
        report = self._run()
        self.assertEqual(report["n_ground_truth_segments"], 1)
        self.assertEqual(report["n_detections"], 1)
        self.assertEqual(report["n_name_matches"], 0)
        self.assertEqual(report["per_name"]["cork"]["true_positives"], 0)
        self.assertEqual(report["per_name"]["gainer"]["support"], 1)

    def test_entries_on_one_window_are_one_segment(self) -> None:
        # train.py labels a window from every entry on it, whatever its clip,
        # so ground truth must too: same-window entries are one segment and a
        # name repeated on the window counts once.
        legacy = [
            {"name": n, "start_frame": 4, "end_frame": 20} for n in ("cork", "gainer")
        ]
        two_clips = [
            {"clip_id": c, "name": n, "start_frame": 4, "end_frame": 20}
            for c, n in ((self.CLIP_A, "cork"), (self.CLIP_B, "gainer"))
        ]
        mixed = [
            *self._combo_clip(),
            {"name": "cork", "start_frame": 4, "end_frame": 20},
            {"clip_id": self.CLIP_B, "name": "cork", "start_frame": 4, "end_frame": 20},
        ]
        self._split({"h1": "holdout"})
        self._predictions(
            {"h1": [{"start_frame": 4, "end_frame": 20, "names": ["cork", "gainer"]}]}
        )
        for layout, tricks in (
            ("legacy", legacy),
            ("two_clips", two_clips),
            ("mixed", mixed),
        ):
            with self.subTest(layout=layout):
                self._sample("h1", tricks)
                report = self._run()
                self.assertEqual(report["n_ground_truth_segments"], 1)
                self.assertEqual(report["n_name_matches"], 1)
                self.assertEqual(report["per_name"]["cork"]["support"], 1)
                self.assertEqual(report["per_name"]["gainer"]["support"], 1)

    def test_distinct_windows_stay_separate_segments(self) -> None:
        # A combo clip, a single-name clip and an entry without a clip_id on
        # three windows: three segments, only the combo's names merged.
        self._sample(
            "h1",
            [
                *self._combo_clip(),
                {
                    "clip_id": self.CLIP_B,
                    "name": "cork",
                    "start_frame": 40,
                    "end_frame": 50,
                },
                {"name": "kick", "start_frame": 60, "end_frame": 70},
            ],
        )
        self._split({"h1": "holdout"})
        self._predictions(
            {
                "h1": [
                    {"start_frame": 4, "end_frame": 20, "names": ["cork", "gainer"]},
                    {"start_frame": 40, "end_frame": 50, "name": "cork"},
                    {"start_frame": 60, "end_frame": 70, "name": "kick"},
                ]
            }
        )
        report = self._run()
        self.assertEqual(report["n_ground_truth_segments"], 3)
        self.assertEqual(report["n_name_matches"], 3)
        self.assertEqual(report["per_name"]["cork"]["support"], 2)
        self.assertEqual(report["per_name"]["cork"]["true_positives"], 2)

    def test_is_combo_describes_the_merged_segment(self) -> None:
        self._split({"h1": "holdout"})
        self._predictions({})
        # Each entry of a combo clip may say is_combo: true...
        self._sample("h1", self._combo_clip(is_combo=True))
        self.assertEqual(self._run()["n_ground_truth_segments"], 1)
        # ...but not false, and a single-name window is not a combo.
        for tricks in (
            self._combo_clip(is_combo=False),
            [{"name": "cork", "start_frame": 4, "end_frame": 20, "is_combo": True}],
        ):
            with self.subTest(tricks=tricks):
                self._sample("h1", tricks)
                self._assert_refused()

    def test_clip_entries_disagreeing_on_window_rejected(self) -> None:
        tricks = self._combo_clip()
        tricks[1]["end_frame"] = 21
        self._sample("h1", tricks)
        self._split({"h1": "holdout"})
        self._predictions({})
        err = self._assert_refused()
        self.assertIn("trick #1", str(err))
        self.assertNotIn(self.CLIP_A, str(err))  # logs carry no clip ids

    def test_clip_repeating_a_name_rejected(self) -> None:
        self._sample("h1", [*self._combo_clip(), self._combo_clip()[0]])
        self._split({"h1": "holdout"})
        self._predictions({})
        err = self._assert_refused()
        self.assertIn("trick #2", str(err))
        self.assertNotIn(self.CLIP_A, str(err))

    def test_clip_in_two_holdout_samples_rejected(self) -> None:
        self._sample("h1", self._combo_clip())
        self._sample(
            "h2",
            [
                {
                    "clip_id": self.CLIP_A,
                    "name": "kick",
                    "start_frame": 0,
                    "end_frame": 8,
                }
            ],
        )
        self._split({"h1": "holdout", "h2": "holdout"})
        self._predictions({})
        err = self._assert_refused()
        self.assertIn("h2.json", str(err))
        self.assertNotIn(self.CLIP_A, str(err))

    def test_bad_clip_id_rejected(self) -> None:
        # Present means a non-empty string: null is refused, not "no clip".
        self._split({"h1": "holdout"})
        self._predictions({})
        for clip_id in ("", None, 7):
            with self.subTest(clip_id=clip_id):
                self._sample(
                    "h1",
                    [
                        {
                            "clip_id": clip_id,
                            "name": "cork",
                            "start_frame": 4,
                            "end_frame": 20,
                        }
                    ],
                )
                self._assert_refused()

    def test_combo_clip_report_is_byte_identical(self) -> None:
        # Reruns, and the clip's entries in either file order, give the same
        # report bytes.
        self._split({"h1": "holdout"})
        self._predictions(
            {"h1": [{"start_frame": 4, "end_frame": 20, "names": ["cork", "gainer"]}]}
        )
        reports = []
        for tricks in (self._combo_clip(), self._combo_clip()[::-1]):
            self._sample("h1", tricks)
            for _ in range(2):
                self.assertEqual(evaluate.main(self._argv()), 0)
                reports.append(self.out.read_bytes())
        self.assertEqual(len(set(reports)), 1)

    def test_background_false_positive(self) -> None:
        self._basic_holdout()
        # Far from the GT segment: zero IoU -> hallucination.
        self._predictions(
            {"h1": [{"start_frame": 100, "end_frame": 110, "name": "cork"}]}
        )
        report = self._run()
        self.assertEqual(report["n_background_false_positives"], 1)
        self.assertEqual(report["n_unmatched_predictions"], 1)
        self.assertEqual(report["n_detections"], 0)

    def test_split_discipline_rejects_train_sample(self) -> None:
        self._basic_holdout()
        self._predictions(
            {"t1": [{"start_frame": 0, "end_frame": 8, "name": "gainer"}]}
        )
        self._assert_refused()

    def test_split_discipline_rejects_unknown_sample(self) -> None:
        self._basic_holdout()
        self._predictions(
            {"nope": [{"start_frame": 0, "end_frame": 8, "name": "gainer"}]}
        )
        self._assert_refused()

    def test_invalid_segment_rejected(self) -> None:
        self._basic_holdout()
        self._predictions(
            {"h1": [{"start_frame": 20, "end_frame": 20, "name": "gainer"}]}
        )
        self._assert_refused()

    def test_is_combo_contradiction_rejected(self) -> None:
        self._basic_holdout()
        self._predictions(
            {
                "h1": [
                    {
                        "start_frame": 4,
                        "end_frame": 20,
                        "name": "gainer",
                        "is_combo": True,
                    }
                ]
            }
        )
        self._assert_refused()

    def test_name_and_names_together_rejected(self) -> None:
        self._basic_holdout()
        self._predictions(
            {
                "h1": [
                    {
                        "start_frame": 4,
                        "end_frame": 20,
                        "name": "gainer",
                        "names": ["cork"],
                    }
                ]
            }
        )
        self._assert_refused()

    def test_empty_name_component_rejected(self) -> None:
        self._basic_holdout()
        self._predictions(
            {"h1": [{"start_frame": 4, "end_frame": 20, "name": "gainer+"}]}
        )
        self._assert_refused()

    def test_default_out_sits_next_to_predictions(self) -> None:
        self._basic_holdout()
        self._predictions(
            {"h1": [{"start_frame": 4, "end_frame": 20, "name": "gainer"}]}
        )
        rc = evaluate.main(
            [
                "--predictions",
                str(self.pred_path),
                "--split",
                str(self.split_path),
                "--samples-dir",
                str(self.samples_dir),
            ]
        )
        self.assertEqual(rc, 0)
        self.assertTrue((self.pred_path.parent / "holdout_report.json").is_file())

    def test_bad_iou_threshold_rejected(self) -> None:
        self._basic_holdout()
        self._predictions({})
        self.assertEqual(evaluate.main(self._argv("--iou-threshold", "1.5")), 2)


class GroundTruthSegmentsTest(unittest.TestCase):
    """Direct tests for evaluate.ground_truth_segments (no report re-sort)."""

    def test_output_is_sorted_and_independent_of_entry_order(self) -> None:
        a = "0b6f1c3e-6a51-4f47-9d57-1f4f3c0a2b11"
        b = "7d2e9a40-3c18-4b6e-a2f5-8e0c51d94c22"
        tricks = [
            {"clip_id": b, "name": "cork", "start_frame": 40, "end_frame": 50},
            {"clip_id": a, "name": "gainer", "start_frame": 4, "end_frame": 20},
            {"name": "kick", "start_frame": 60, "end_frame": 70},
            {"clip_id": a, "name": "cork", "start_frame": 4, "end_frame": 20},
            {"name": "gainer", "start_frame": 4, "end_frame": 20},
        ]
        expected = [
            (4, 20, frozenset({"cork", "gainer"})),
            (40, 50, frozenset({"cork"})),
            (60, 70, frozenset({"kick"})),
        ]
        for perm in itertools.permutations(tricks):
            self.assertEqual(evaluate.ground_truth_segments(list(perm), "t"), expected)

    def test_no_tricks_no_segments(self) -> None:
        self.assertEqual(evaluate.ground_truth_segments([], "t"), [])


class MatchSegmentsTest(unittest.TestCase):
    """Direct tests for evaluate.match_segments (greedy best-IoU matcher)."""

    @staticmethod
    def _seg(start: int, end: int) -> tuple[int, int, frozenset[str]]:
        return (start, end, frozenset({"x"}))

    def test_iou_exactly_at_threshold_is_inclusive(self) -> None:
        # GT (0,10) vs pred (0,5): IoU = 5/10 = 0.5 exactly (representable).
        # The code filters with `iou >= threshold`, so the boundary matches.
        gt, pred = [self._seg(0, 10)], [self._seg(0, 5)]
        matches, mg, mp = evaluate.match_segments(gt, pred, 0.5)
        self.assertEqual(matches, [(0, 0, 0.5)])
        self.assertEqual((mg, mp), ({0}, {0}))
        # Just above the boundary the pair is dropped.
        matches, mg, mp = evaluate.match_segments(gt, pred, 0.5000001)
        self.assertEqual((matches, mg, mp), ([], set(), set()))

    def test_tie_broken_by_gt_then_pred_index(self) -> None:
        # One pred overlaps two identical GTs equally: lower gt_idx wins.
        matches, mg, mp = evaluate.match_segments(
            [self._seg(0, 10), self._seg(0, 10)], [self._seg(0, 10)], 0.5
        )
        self.assertEqual(matches, [(0, 0, 1.0)])
        self.assertEqual((mg, mp), ({0}, {0}))
        # One GT, two preds with equal IoU 0.5: lower pred_idx wins.
        matches, mg, mp = evaluate.match_segments(
            [self._seg(0, 10)], [self._seg(5, 10), self._seg(0, 5)], 0.5
        )
        self.assertEqual(matches, [(0, 0, 0.5)])
        self.assertEqual((mg, mp), ({0}, {0}))

    def test_greedy_pick_blocks_later_pair(self) -> None:
        # pred0 (0,10) pairs with gt0 (IoU 1.0) and gt1 (IoU 0.6). Greedy
        # takes the 1.0 pair first, so gt1 stays unmatched even though its
        # pair clears the threshold.
        matches, mg, mp = evaluate.match_segments(
            [self._seg(0, 10), self._seg(0, 6)], [self._seg(0, 10)], 0.5
        )
        self.assertEqual(matches, [(0, 0, 1.0)])
        self.assertEqual((mg, mp), ({0}, {0}))

    def test_multi_pair_dedup_matches_each_side_once(self) -> None:
        # Diagonal pairs are best; cross pairs (IoU 0.6) are skipped because
        # both sides are already taken. Result is sorted by (gt, pred).
        matches, mg, mp = evaluate.match_segments(
            [self._seg(0, 10), self._seg(0, 6)],
            [self._seg(0, 6), self._seg(0, 10)],
            0.5,
        )
        self.assertEqual(matches, [(0, 1, 1.0), (1, 0, 1.0)])
        self.assertEqual((mg, mp), ({0, 1}, {0, 1}))

    def test_no_overlap(self) -> None:
        # Disjoint and merely touching segments both have IoU 0.
        matches, mg, mp = evaluate.match_segments(
            [self._seg(0, 5)], [self._seg(5, 10), self._seg(20, 30)], 0.1
        )
        self.assertEqual((matches, mg, mp), ([], set(), set()))

    def test_empty_inputs(self) -> None:
        seg = [self._seg(0, 10)]
        for gt, pred in (([], []), (seg, []), ([], seg)):
            with self.subTest(gt=len(gt), pred=len(pred)):
                self.assertEqual(
                    evaluate.match_segments(gt, pred, 0.5), ([], set(), set())
                )


if __name__ == "__main__":
    unittest.main()
