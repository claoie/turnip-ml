"""TKS1 segment-blob codec (the farm's training segments).

turnip-farm docs/POSE_FORMAT.md §12. A blob is a single gzip member
wrapping a 24-byte little-endian header and a dense
`frame_count x 13 x (x, y, confidence)` float32 frame array:

    offset  size  field
         0     4  magic "TKS1"
         4     2  format_version   u16  == 1
         6     2  keypoint_count   u16  == 13 (segment_transform.JOINTS)
         8     4  frame_count      u32  >= 1
        12     4  sample_rate_hz   f32  == 10.0
        16     8  reserved              all zero

Coordinates are in torso lengths on the farm's standard body
(Training/segment_transform.py). A joint is missing iff its confidence
is 0.0, and its three floats are then all 0.0; a frame with every joint
missing is a gap frame.

Decoding fails closed on anything a conformant encoder would not write:
bad magic/version/keypoint count, a sample rate other than 10.0, a
nonzero reserved byte, no frames, a length that disagrees with
frame_count, a missing joint with a nonzero coordinate, a non-gzip or
multi-member stream, an oversized payload, or a non-finite value.
Values decode losslessly, as in tkp1.py.
"""

from __future__ import annotations

import gzip
import math
import struct
import sys
from array import array
from collections.abc import Sequence

import segment_transform
import tkp1

MAGIC = b"TKS1"
FORMAT_VERSION = 1
KEYPOINT_COUNT = segment_transform.N_JOINTS  # 13
CHANNELS = 3
CANONICAL_RATE_HZ = 10.0
HEADER = struct.Struct("<4sHHIf8s")
FRAME_BYTES = KEYPOINT_COUNT * CHANNELS * 4  # 156
MAX_FRAMES = tkp1.MAX_FRAMES
MAX_RAW_BYTES = HEADER.size + MAX_FRAMES * FRAME_BYTES

Frame = list[list[float]]  # [joint][x, y, confidence]


class Tks1Error(ValueError):
    """A blob that a conformant encoder would not have produced."""


def decode(raw: bytes) -> list[Frame]:
    """Decode an uncompressed TKS1 blob into [frame][joint][x, y, c]."""
    if len(raw) < HEADER.size:
        raise Tks1Error(f"blob is {len(raw)} bytes, shorter than the header")
    magic, version, keypoints, frames, rate, reserved = HEADER.unpack_from(raw)
    if magic != MAGIC:
        raise Tks1Error(f"bad magic {magic!r}")
    if version != FORMAT_VERSION:
        raise Tks1Error(f"unsupported format_version {version}")
    if keypoints != KEYPOINT_COUNT:
        raise Tks1Error(f"keypoint_count must be {KEYPOINT_COUNT}, got {keypoints}")
    if rate != CANONICAL_RATE_HZ:
        raise Tks1Error(f"sample_rate_hz must be {CANONICAL_RATE_HZ}, got {rate}")
    if reserved != bytes(8):
        raise Tks1Error("reserved bytes must be zero")
    if frames < 1:
        raise Tks1Error("a segment has at least one frame")
    expected = HEADER.size + frames * FRAME_BYTES
    if len(raw) != expected:
        raise Tks1Error(
            f"blob is {len(raw)} bytes; {frames} frames need exactly {expected}"
        )
    values = array("f")
    values.frombytes(raw[HEADER.size :])
    if sys.byteorder == "big":
        values.byteswap()
    if not all(math.isfinite(v) for v in values):
        raise Tks1Error("frame data contains a non-finite value")
    per_frame = KEYPOINT_COUNT * CHANNELS
    out: list[Frame] = []
    for f in range(frames):
        frame = []
        for k in range(KEYPOINT_COUNT):
            at = f * per_frame + k * CHANNELS
            joint = list(values[at : at + CHANNELS])
            if joint[2] == 0.0 and (joint[0] != 0.0 or joint[1] != 0.0):
                raise Tks1Error(f"frame {f} joint {k} is missing but not zero")
            frame.append(joint)
        out.append(frame)
    return out


def decode_gz(data: bytes) -> list[Frame]:
    """Decode a stored `.tks1.gz` blob."""
    try:
        raw = tkp1.gunzip(data, MAX_RAW_BYTES)
    except tkp1.Tkp1Error as e:
        raise Tks1Error(str(e)) from e
    return decode(raw)


def encode(frames: Sequence[Sequence[Sequence[float]]]) -> bytes:
    """Encode 13-joint frames as an uncompressed TKS1 blob (for fixtures)."""
    if not frames:
        raise Tks1Error("a segment has at least one frame")
    values = array("f")
    for f, frame in enumerate(frames):
        if len(frame) != KEYPOINT_COUNT or any(len(j) != CHANNELS for j in frame):
            raise Tks1Error(f"frame {f} must be {KEYPOINT_COUNT} x {CHANNELS}")
        for joint in frame:
            if joint[2] == 0.0 and (joint[0] != 0.0 or joint[1] != 0.0):
                raise Tks1Error(f"frame {f}: a missing joint must be all zeros")
            values.extend(joint)
    if sys.byteorder == "big":
        values.byteswap()
    header = HEADER.pack(
        MAGIC, FORMAT_VERSION, KEYPOINT_COUNT, len(frames), CANONICAL_RATE_HZ, bytes(8)
    )
    return header + values.tobytes()


def encode_gz(frames: Sequence[Sequence[Sequence[float]]]) -> bytes:
    """encode() wrapped in a reproducible single gzip member (mtime 0)."""
    return gzip.compress(encode(frames), mtime=0)
