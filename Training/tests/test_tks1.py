"""Unit tests for Training/tks1.py (turnip-farm POSE_FORMAT.md §12).

Run: python -m unittest discover -s Training/tests -t .
"""

from __future__ import annotations

import gzip
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tks1

FRAMES = [
    [[0.25 * k, -0.5, 0.75] for k in range(13)],
    [[0.0, 0.0, 0.0] for _ in range(13)],  # a gap frame
    [[0.5, 1.0, 0.5] if k != 4 else [0.0, 0.0, 0.0] for k in range(13)],
]


class Tks1Test(unittest.TestCase):
    def test_round_trip_and_layout(self) -> None:
        raw = tks1.encode(FRAMES)
        self.assertEqual(len(raw), 24 + 3 * 156)
        self.assertEqual(raw[:4], b"TKS1")
        self.assertEqual(struct.unpack_from("<HHIf", raw, 4), (1, 13, 3, 10.0))
        self.assertEqual(raw[16:24], bytes(8))
        # Joint k, channel c of frame f sits at 24 + ((f * 13 + k) * 3 + c) * 4.
        self.assertEqual(
            struct.unpack_from("<f", raw, 24 + ((0 * 13 + 2) * 3) * 4), (0.5,)
        )
        self.assertEqual(tks1.decode(raw), FRAMES)
        self.assertEqual(tks1.decode_gz(tks1.encode_gz(FRAMES)), FRAMES)
        self.assertEqual(tks1.encode_gz(FRAMES), tks1.encode_gz(FRAMES))

    def test_nonconformant_blobs_are_refused(self) -> None:
        good = tks1.encode(FRAMES)

        def patched(offset: int, fmt: str, *values: object) -> bytes:
            raw = bytearray(good)
            struct.pack_into(fmt, raw, offset, *values)
            return bytes(raw)

        cases = {
            "short": good[:10],
            "magic": b"TKP1" + good[4:],
            "version": patched(4, "<H", 2),
            "keypoints": patched(6, "<H", 17),
            "rate": patched(12, "<f", 30.0),
            "reserved": patched(20, "<I", 1),
            "no frames": tks1.HEADER.pack(b"TKS1", 1, 13, 0, 10.0, bytes(8)),
            "length": good + b"\0\0\0\0",
            "frame_count": patched(8, "<I", 4),
            "non-finite": patched(24, "<f", float("nan")),
            # Joint 0 of frame 1 (a gap) given an x but no confidence.
            "missing joint not zero": patched(24 + 13 * 12, "<f", 0.5),
        }
        for name, raw in cases.items():
            with self.subTest(name), self.assertRaises(tks1.Tks1Error):
                tks1.decode(raw)

    def test_gzip_is_one_bounded_member(self) -> None:
        member = tks1.encode_gz(FRAMES)
        for name, data in {
            "not gzip": b"not gzip",
            "two members": member + member,
            "truncated": member[:-4],
        }.items():
            with self.subTest(name), self.assertRaises(tks1.Tks1Error):
                tks1.decode_gz(data)
        oversized = gzip.compress(b"\0" * (tks1.MAX_RAW_BYTES + 1), mtime=0)
        with self.assertRaises(tks1.Tks1Error):
            tks1.decode_gz(oversized)

    def test_encoder_refuses_what_a_decoder_would(self) -> None:
        for name, frames in {
            "no frames": [],
            "wrong joint count": [[[0.0, 0.0, 1.0]] * 12],
            "missing joint not zero": [[[0.5, 0.0, 0.0]] + [[0.0, 0.0, 1.0]] * 12],
        }.items():
            with self.subTest(name), self.assertRaises(tks1.Tks1Error):
                tks1.encode(frames)


if __name__ == "__main__":
    unittest.main()
