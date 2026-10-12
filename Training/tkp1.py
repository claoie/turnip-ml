"""TKP1 pose-blob codec (the farm's canonical pose format).

A blob is a single gzip member wrapping a 24-byte little-endian header and
a dense `frame_count x 17 x (x, y, confidence)` float32 frame array:

    offset  size  field
         0     4  magic "TKP1"
         4     2  format_version        u16  == 1
         6     2  keypoint_count        u16  == 17 (MoveNet, COCO order)
         8     4  frame_count           u32
        12     4  sample_rate_hz        f32  canonical 10.0
        16     4  source_sample_rate_hz f32  as-sent rate (provenance)
        20     4  flags                 u32  reserved, == 0

A frame whose 17 confidences are all exactly 0.0 is a gap (missing data):
it is kept as zeros, never dropped, filled or interpolated.

Decoding fails closed on anything a conformant encoder would not write:
bad magic/version/keypoint count, nonzero flags, a length that disagrees
with frame_count, a non-gzip or multi-member stream, or an oversized
payload. Beyond the format's own rules it also rejects a sample rate other
than 10.0 (stored blobs are canonical) and non-finite values.

Values decode losslessly: each float32 becomes the Python float of the
same value, so what training sees is exactly what was stored.
"""

from __future__ import annotations

import gzip
import math
import struct
import sys
import zlib
from array import array
from collections.abc import Sequence
from dataclasses import dataclass

MAGIC = b"TKP1"
FORMAT_VERSION = 1
KEYPOINT_COUNT = 17
CHANNELS = 3
CANONICAL_RATE_HZ = 10.0
HEADER = struct.Struct("<4sHHIffI")
FRAME_BYTES = KEYPOINT_COUNT * CHANNELS * 4  # 204
# Four hours of canonical 10 Hz frames: far beyond any real session, and a
# bound on what a hostile gzip stream can make the decoder allocate.
MAX_FRAMES = 4 * 3600 * 10
MAX_RAW_BYTES = HEADER.size + MAX_FRAMES * FRAME_BYTES

Frame = list[list[float]]  # [keypoint][x, y, confidence]


class Tkp1Error(ValueError):
    """A blob that a conformant encoder would not have produced."""


@dataclass(frozen=True)
class PoseSequence:
    sample_rate_hz: float
    source_sample_rate_hz: float
    keypoints: list[Frame]  # [frame][keypoint][x, y, confidence]

    @property
    def frame_count(self) -> int:
        return len(self.keypoints)

    def is_gap(self, frame: int) -> bool:
        return all(kp[2] == 0.0 for kp in self.keypoints[frame])


def gunzip(data: bytes, max_size: int = MAX_RAW_BYTES) -> bytes:
    """Decompress exactly one gzip member of at most max_size bytes."""
    stream = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
    try:
        raw = stream.decompress(data, max_size + 1)
    except zlib.error as e:
        raise Tkp1Error(f"not a valid gzip stream: {e}") from e
    if len(raw) > max_size or stream.unconsumed_tail:
        raise Tkp1Error(f"decompressed blob exceeds {max_size} bytes")
    if not stream.eof:
        raise Tkp1Error("truncated gzip stream")
    if stream.unused_data:
        raise Tkp1Error("trailing data after the gzip member")
    return raw


def decode(raw: bytes) -> PoseSequence:
    """Decode an uncompressed TKP1 blob."""
    if len(raw) < HEADER.size:
        raise Tkp1Error(f"blob is {len(raw)} bytes, shorter than the header")
    magic, version, keypoints, frames, rate, source_rate, flags = HEADER.unpack_from(
        raw
    )
    if magic != MAGIC:
        raise Tkp1Error(f"bad magic {magic!r}")
    if version != FORMAT_VERSION:
        raise Tkp1Error(f"unsupported format_version {version}")
    if keypoints != KEYPOINT_COUNT:
        raise Tkp1Error(f"keypoint_count must be {KEYPOINT_COUNT}, got {keypoints}")
    if flags != 0:
        raise Tkp1Error(f"reserved flags must be 0, got {flags}")
    expected = HEADER.size + frames * FRAME_BYTES
    if len(raw) != expected:
        raise Tkp1Error(
            f"blob is {len(raw)} bytes; {frames} frames need exactly {expected}"
        )
    if rate != CANONICAL_RATE_HZ:
        raise Tkp1Error(f"sample_rate_hz must be {CANONICAL_RATE_HZ}, got {rate}")
    if not (math.isfinite(source_rate) and source_rate > 0):
        raise Tkp1Error(f"source_sample_rate_hz must be > 0, got {source_rate}")

    values = array("f")
    values.frombytes(raw[HEADER.size :])
    if sys.byteorder == "big":
        values.byteswap()
    if not all(math.isfinite(v) for v in values):
        raise Tkp1Error("frame data contains a non-finite value")
    per_frame = KEYPOINT_COUNT * CHANNELS
    pose = [
        [
            list(
                values[
                    f * per_frame + k * CHANNELS : f * per_frame + (k + 1) * CHANNELS
                ]
            )
            for k in range(KEYPOINT_COUNT)
        ]
        for f in range(frames)
    ]
    return PoseSequence(rate, source_rate, pose)


def decode_gz(data: bytes) -> PoseSequence:
    """Decode a stored `.tkp1.gz` blob."""
    return decode(gunzip(data))


def encode(
    keypoints: Sequence[Sequence[Sequence[float]]],
    source_sample_rate_hz: float = CANONICAL_RATE_HZ,
) -> bytes:
    """Encode canonical 10 Hz frames as an uncompressed TKP1 blob.

    For fixtures and synthetic data. Gap frames must already be all zeros.
    """
    values = array("f")
    for f, frame in enumerate(keypoints):
        if len(frame) != KEYPOINT_COUNT or any(len(kp) != CHANNELS for kp in frame):
            raise Tkp1Error(f"frame {f} must be {KEYPOINT_COUNT} x {CHANNELS}")
        if all(kp[2] == 0.0 for kp in frame) and any(
            v != 0.0 for kp in frame for v in kp
        ):
            raise Tkp1Error(f"gap frame {f} must be all zeros")
        values.extend(v for kp in frame for v in kp)
    if sys.byteorder == "big":
        values.byteswap()
    header = HEADER.pack(
        MAGIC,
        FORMAT_VERSION,
        KEYPOINT_COUNT,
        len(keypoints),
        CANONICAL_RATE_HZ,
        source_sample_rate_hz,
        0,
    )
    return header + values.tobytes()


def encode_gz(
    keypoints: Sequence[Sequence[Sequence[float]]],
    source_sample_rate_hz: float = CANONICAL_RATE_HZ,
) -> bytes:
    """encode() wrapped in a reproducible single gzip member (mtime 0)."""
    return gzip.compress(encode(keypoints, source_sample_rate_hz), mtime=0)
