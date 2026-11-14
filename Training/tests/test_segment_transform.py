"""Tests for Training/segment_transform.py and its fixture vectors
(docs/MODEL_CONTRACT.md §3.6).

Run: python -m unittest discover -s Training/tests -t .
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import make_segment_vectors as msv
import segment_transform as st

VECTORS = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "segment_transform"
    / "vectors.json"
)


def case(name: str) -> dict:
    data = json.loads(VECTORS.read_text(encoding="utf-8"))
    return next(c for c in data["cases"] if c["name"] == name)


def missing(out: list) -> list[list[str]]:
    return [[st.JOINTS[i] for i, j in enumerate(f) if j is None] for f in out]


class VectorsTest(unittest.TestCase):
    def test_vectors_are_what_the_generator_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "vectors.json"
            msv.main(["--out", str(out)])
            self.assertEqual(out.read_bytes(), VECTORS.read_bytes())

    def test_reference_reproduces_every_vector(self) -> None:
        data = json.loads(VECTORS.read_text(encoding="utf-8"))
        self.assertEqual(data["joints"], list(st.JOINTS))
        for c in data["cases"]:
            with self.subTest(c["name"]):
                out = st.transform(c["input"])
                if c["output"] is None:
                    self.assertIsNone(out)
                    continue
                assert out is not None
                self.assertEqual(missing(out), missing(c["output"]))
                for got_f, want_f in zip(out, c["output"], strict=True):
                    for got, want in zip(got_f, want_f, strict=True):
                        if got is not None:
                            for a, b in zip(got, want, strict=True):
                                self.assertAlmostEqual(a, b, delta=data["tolerance"])
                self.assertEqual(st.model_input(out), c["model_input"])


class RulesTest(unittest.TestCase):
    def test_standing_pose_sits_on_the_standard_body(self) -> None:
        [frame] = case("standing")["output"]
        joint = dict(zip(st.JOINTS, frame, strict=True))
        # The single root is its own center: the hips sit at +-0.2.
        self.assertAlmostEqual(joint["left_hip"][0], 0.2, places=6)
        self.assertAlmostEqual(joint["right_hip"][0], -0.2, places=6)
        self.assertAlmostEqual(joint["left_hip"][1], 0.0, places=6)
        # The neck is 1 torso length up, the shoulders 0.35 either side of it.
        self.assertAlmostEqual(joint["left_shoulder"][1], -1.0, places=6)
        self.assertAlmostEqual(joint["left_shoulder"][0], 0.35, places=6)

    def test_missing_parent_and_coincident_bone(self) -> None:
        self.assertEqual(
            missing(case("missing_elbow")["output"]), [["left_elbow", "left_wrist"]]
        )
        self.assertEqual(
            missing(case("coincident_knee")["output"]),
            [["right_knee", "right_ankle"]],
        )
        self.assertEqual(missing(case("no_face")["output"]), [["head"]])

    def test_a_frame_without_both_hips_is_a_gap(self) -> None:
        c = case("gap_frame")
        self.assertEqual(missing(c["output"])[1], list(st.JOINTS))
        self.assertEqual([row[-1] for row in c["model_input"]], [0.0, 1.0, 0.0])
        self.assertTrue(all(len(row) == 40 for row in c["model_input"]))

    def test_partial_face_head_is_the_mean_of_what_was_seen(self) -> None:
        [frame] = case("partial_face")["output"]
        # Confidence: mean of the nose's 0.9 and the left ear's 0.5.
        self.assertAlmostEqual(frame[0][2], 0.7, places=6)

    def test_output_ignores_where_and_how_big_the_person_is(self) -> None:
        # Moving or scaling the whole take changes nothing: the center and
        # the scale are the sequence's own.
        frames = case("jump")["input"]
        moved = [[[x * 0.5 + 0.2, y * 0.5 + 0.1, c] for x, y, c in f] for f in frames]
        a, b = st.transform(frames), st.transform(moved)
        assert a is not None and b is not None
        for fa, fb in zip(a, b, strict=True):
            for ja, jb in zip(fa, fb, strict=True):
                for va, vb in zip(ja, jb, strict=True):
                    self.assertAlmostEqual(va, vb, places=5)

    def test_center_and_scale_are_the_sequences(self) -> None:
        # The jump's frames grow 1.0, 1.1, 1.2, 1.3: the torso median is the
        # mean of the middle two, 1.15 standing torsos.
        stats = st.sequence_stats(case("jump")["input"])
        standing = st.sequence_stats(case("standing")["input"])
        assert stats is not None and standing is not None
        self.assertAlmostEqual(stats[0] / standing[0], 1.15, places=5)

    def test_malformed_input_is_refused(self) -> None:
        good = case("standing")["input"][0]
        for bad in (
            [good[:16]],
            [[*good[:16], [0.1, 0.2]]],
            [[*good[:16], [0, 0, float("nan")]]],
        ):
            with self.subTest(bad=str(bad)[-40:]), self.assertRaises(st.TransformError):
                st.transform(bad)


if __name__ == "__main__":
    unittest.main()
