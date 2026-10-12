"""Unit tests for Training/tkp1.py against the pose-format spec.

Run with:  python -m unittest discover -s Training/tests -t .
"""

import gzip
import hashlib
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tkp1
from tkp1 import Tkp1Error

# Spec test vector: 3 frames at 10 Hz, frame 1 a gap. Only the nose (0) and
# left shoulder (5) are nonzero; the uncompressed digest is the spec's.
SPEC_SHA256 = "37ba29d3c206f3b9dd41219e29354c480791d000533efdfc8096ee6f0959cac9"


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def frame(points=None):
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    for k, xyz in (points or {}).items():
        kps[k] = list(xyz)
    return kps


SPEC_FRAMES = [
    frame({0: (0.5, 0.2, 0.95), 5: (0.45, 0.35, 0.9)}),
    frame(),
    frame({0: (0.51, 0.21, 0.94), 5: (0.46, 0.36, 0.89)}),
]


def header(**overrides):
    fields = {
        "magic": b"TKP1",
        "version": 1,
        "keypoints": 17,
        "frames": 1,
        "rate": 10.0,
        "source_rate": 30.0,
        "flags": 0,
    }
    fields.update(overrides)
    return tkp1.HEADER.pack(*fields.values())


ONE_FRAME = struct.pack("<51f", *([0.5] * 51))


class SpecVectorTest(unittest.TestCase):
    def test_encoder_reproduces_the_spec_bytes(self):
        raw = tkp1.encode(SPEC_FRAMES)
        self.assertEqual(len(raw), 636)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), SPEC_SHA256)
        self.assertEqual(raw[:24].hex(" "), "54 4b 50 31 01 00 11 00 03 00 00 00 "
                                            "00 00 20 41 00 00 20 41 00 00 00 00")  # fmt: skip
        self.assertEqual(raw[24:36].hex(" "), "00 00 00 3f cd cc 4c 3e 33 33 73 3f")

    def test_decoder_reads_the_spec_vector(self):
        pose = tkp1.decode(tkp1.encode(SPEC_FRAMES))
        self.assertEqual(pose.frame_count, 3)
        self.assertEqual(pose.sample_rate_hz, 10.0)
        self.assertEqual(pose.source_sample_rate_hz, 10.0)
        self.assertEqual([pose.is_gap(f) for f in range(3)], [False, True, False])
        self.assertEqual(pose.keypoints[1], frame())  # gap kept as zeros
        self.assertEqual(pose.keypoints[0][0], [f32(0.5), f32(0.2), f32(0.95)])
        self.assertEqual(pose.keypoints[2][5], [f32(0.46), f32(0.36), f32(0.89)])
        self.assertEqual(len(pose.keypoints[0]), 17)

    def test_gzip_round_trip_is_lossless_and_reproducible(self):
        blob = tkp1.encode_gz(SPEC_FRAMES)
        self.assertEqual(blob, tkp1.encode_gz(SPEC_FRAMES))  # mtime 0
        self.assertEqual(
            tkp1.decode_gz(blob).keypoints,
            tkp1.decode(tkp1.encode(SPEC_FRAMES)).keypoints,
        )

    def test_values_outside_unit_range_are_kept(self):
        pose = tkp1.decode(tkp1.encode([frame({3: (-0.25, 1.5, 0.5)})]))
        self.assertEqual(pose.keypoints[0][3], [-0.25, 1.5, 0.5])


class RejectTest(unittest.TestCase):
    def assert_rejected(self, raw):
        with self.assertRaises(Tkp1Error):
            tkp1.decode(raw)

    def test_valid_baseline(self):
        self.assertEqual(tkp1.decode(header() + ONE_FRAME).frame_count, 1)

    def test_header_fields(self):
        cases = {
            "magic": header(magic=b"TKP2"),
            "version": header(version=2),
            "keypoints": header(keypoints=18),
            "flags": header(flags=1),
            "rate": header(rate=30.0),
            "source-rate-zero": header(source_rate=0.0),
            "source-rate-nan": header(source_rate=float("nan")),
        }
        for name, head in cases.items():
            with self.subTest(field=name):
                self.assert_rejected(head + ONE_FRAME)

    def test_length_must_match_frame_count(self):
        for raw in (
            header() + ONE_FRAME[:-1],
            header() + ONE_FRAME + b"\x00",
            header(frames=2) + ONE_FRAME,
            header()[:23],
        ):
            with self.subTest(length=len(raw)):
                self.assert_rejected(raw)

    def test_non_finite_values(self):
        for bad in (float("nan"), float("inf")):
            with self.subTest(value=bad):
                values = [0.5] * 51
                values[7] = bad
                self.assert_rejected(header() + struct.pack("<51f", *values))

    def test_gzip_framing(self):
        good = gzip.compress(header() + ONE_FRAME, mtime=0)
        self.assertEqual(tkp1.decode_gz(good).frame_count, 1)
        cases = {
            "not-gzip": header() + ONE_FRAME,
            "truncated": good[:-6],
            "two-members": good + good,
            "trailing-garbage": good + b"junk",
        }
        for name, data in cases.items():
            with self.subTest(case=name), self.assertRaises(Tkp1Error):
                tkp1.decode_gz(data)

    def test_decompression_is_bounded(self):
        bomb = gzip.compress(b"\x00" * 10_000, mtime=0)
        with self.assertRaises(Tkp1Error):
            tkp1.gunzip(bomb, max_size=1_000)
        self.assertEqual(len(tkp1.gunzip(bomb, max_size=10_000)), 10_000)

    def test_encoder_refuses_a_gap_frame_with_coordinates(self):
        with self.assertRaises(Tkp1Error):
            tkp1.encode([frame({2: (0.3, 0.4, 0.0)})])
        with self.assertRaises(Tkp1Error):
            tkp1.encode([[[0.0, 0.0, 0.0]] * 16])


if __name__ == "__main__":
    unittest.main()
