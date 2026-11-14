"""Reference segment transform: canonical 10 Hz pose -> the TKS1 body frame.

docs/MODEL_CONTRACT.md §3.6 and turnip-farm docs/POSE_FORMAT.md §12. A
model trained on the farm's segments sees poses re-targeted onto a
standard body; the client applies the same transform, without noise, to
the take it runs the model over. This module is the normative reference
for the client, and the fixture vectors in
Training/fixtures/segment_transform/vectors.json (written by
Training/make_segment_vectors.py) are its test cases.

Over a whole sequence this is the farm's `retargetSegment`
(src/ingest/segments.ts) with the noise left out: run over a whole take,
the two agree to float32 rounding.

The transform
-------------
Input: frames of 17 MoveNet keypoints (x, y, confidence), x/y in the
canonical 0-1 source-frame units of TKP1. A keypoint is detected iff its
confidence is > 0.

Per frame:
- root = midpoint of the hips (keypoints 11, 12); a frame missing either
  hip has no root and is a gap frame: all 13 joints missing.
- neck = midpoint of the shoulders (5, 6), missing if either is.
- head = mean of the detected face keypoints 0-4 (x, y and confidence),
  missing if none is.
- The root goes to (root - center) / scale. Every other joint is placed
  from its parent's output position along the observed parent->child
  direction at the standard bone length (BONES). Parents: hips and neck
  from the root; shoulders and head from the neck; elbows from shoulders,
  wrists from elbows; knees from hips, ankles from knees. A joint is
  missing when it is not detected, when its parent is missing, or when it
  coincides with its parent (zero-length bone). The neck itself is not
  output.
- Output confidence: the observed confidence of the joint (the head's is
  the mean over the face keypoints it used).

Over the sequence (the only two multi-frame statistics):
- scale = median over frames with a root and a neck of |neck - root|
  (the torso length, in TKP1 units); the mean of the two middle values
  for an even count. No such frame: the transform has no output.
- center = mean root over frames with a root.

Arithmetic is float64 on the float32 inputs; outputs are rounded to
float32. Fixture vectors compare at an absolute tolerance of 1e-5.
"""

from __future__ import annotations

import math
import statistics
import struct
from collections.abc import Sequence

N_INPUT_KEYPOINTS = 17
JOINTS = (
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
N_JOINTS = len(JOINTS)
# Standard bone lengths in torso lengths (root -> neck = 1), the farm's
# STANDARD_BONES.
BONES = {
    "torso": 1.0,
    "hip_half": 0.2,
    "shoulder_half": 0.35,
    "neck_to_head": 0.45,
    "upper_arm": 0.55,
    "forearm": 0.5,
    "thigh": 0.85,
    "shin": 0.8,
}
# (output joint, its TKP1 keypoint, parent, bone). Parents are named
# points: "root", "neck", or an output joint placed earlier in this list.
# The head's keypoint is None: it is the face mean.
PLACEMENT = (
    ("left_hip", 11, "root", "hip_half"),
    ("right_hip", 12, "root", "hip_half"),
    ("left_shoulder", 5, "neck", "shoulder_half"),
    ("right_shoulder", 6, "neck", "shoulder_half"),
    ("head", None, "neck", "neck_to_head"),
    ("left_elbow", 7, "left_shoulder", "upper_arm"),
    ("right_elbow", 8, "right_shoulder", "upper_arm"),
    ("left_wrist", 9, "left_elbow", "forearm"),
    ("right_wrist", 10, "right_elbow", "forearm"),
    ("left_knee", 13, "left_hip", "thigh"),
    ("right_knee", 14, "right_hip", "thigh"),
    ("left_ankle", 15, "left_knee", "shin"),
    ("right_ankle", 16, "right_knee", "shin"),
)

_F32 = struct.Struct("<f")

Point = tuple[float, float, float]  # x, y, confidence
Frame = Sequence[Sequence[float]]  # 17 x (x, y, confidence)
OutFrame = list[Point | None]  # 13 joints, None where missing


class TransformError(ValueError):
    """Malformed input: the reference refuses rather than guesses."""


def _f32(x: float) -> float:
    return float(_F32.unpack(_F32.pack(x))[0])


def _points(frame: Frame, index: int) -> list[Point | None]:
    if len(frame) != N_INPUT_KEYPOINTS:
        raise TransformError(f"frame {index} has {len(frame)} keypoints, not 17")
    points: list[Point | None] = []
    for k, kp in enumerate(frame):
        if len(kp) != 3 or not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in kp
        ):
            raise TransformError(f"frame {index} keypoint {k} is not (x, y, c)")
        x, y, c = (float(v) for v in kp)
        points.append((x, y, c) if c > 0 else None)
    return points


def _mid(a: Point | None, b: Point | None) -> Point | None:
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, min(a[2], b[2]))


def _head(points: list[Point | None]) -> Point | None:
    face = [p for p in points[:5] if p is not None]
    if not face:
        return None
    n = len(face)
    return (
        sum(p[0] for p in face) / n,
        sum(p[1] for p in face) / n,
        sum(p[2] for p in face) / n,
    )


def sequence_stats(frames: Sequence[Frame]) -> tuple[float, float, float] | None:
    """(scale, center x, center y) over `frames`; None without a torso."""
    torsos: list[float] = []
    roots: list[Point] = []
    for i, frame in enumerate(frames):
        points = _points(frame, i)
        root = _mid(points[11], points[12])
        neck = _mid(points[5], points[6])
        if root is not None:
            roots.append(root)
            if neck is not None:
                torsos.append(math.hypot(neck[0] - root[0], neck[1] - root[1]))
    if not torsos:
        return None
    scale = statistics.median(torsos)
    if not scale > 0:
        return None
    cx = sum(r[0] for r in roots) / len(roots)
    cy = sum(r[1] for r in roots) / len(roots)
    return scale, cx, cy


def retarget_frame(
    frame: Frame, scale: float, cx: float, cy: float, index: int = 0
) -> OutFrame:
    """One frame onto the standard body, given the sequence's statistics."""
    points = _points(frame, index)
    out: OutFrame = [None] * N_JOINTS
    root = _mid(points[11], points[12])
    if root is None:
        return out
    observed: dict[str, Point | None] = {
        "root": root,
        "neck": _mid(points[5], points[6]),
    }
    placed: dict[str, tuple[float, float]] = {
        "root": ((root[0] - cx) / scale, (root[1] - cy) / scale)
    }

    def place(
        parent: str, child: Point | None, bone: str
    ) -> tuple[float, float] | None:
        start, at = observed.get(parent), placed.get(parent)
        if start is None or at is None or child is None:
            return None
        dx, dy = child[0] - start[0], child[1] - start[1]
        d = math.hypot(dx, dy)
        if d == 0:
            return None
        length = BONES[bone]
        return (at[0] + dx / d * length, at[1] + dy / d * length)

    neck_at = place("root", observed["neck"], "torso")
    if neck_at is not None:
        placed["neck"] = neck_at
    head = _head(points)
    for joint, keypoint, parent, bone in PLACEMENT:
        child = head if keypoint is None else points[keypoint]
        at = place(parent, child, bone)
        if at is None or child is None:
            continue
        observed[joint] = child
        placed[joint] = at
        out[JOINTS.index(joint)] = (_f32(at[0]), _f32(at[1]), _f32(child[2]))
    return out


def transform(frames: Sequence[Frame]) -> list[OutFrame] | None:
    """The whole sequence onto the standard body; None without a torso."""
    stats = sequence_stats(frames)
    if stats is None:
        return None
    scale, cx, cy = stats
    return [retarget_frame(f, scale, cx, cy, i) for i, f in enumerate(frames)]


def frames_as_array(frames: Sequence[OutFrame]) -> list[list[list[float]]]:
    """[T][13][x, y, c] with missing joints as zeros: the sample keypoints
    train.py reads, the same layout as a decoded TKS1 blob."""
    return [
        [list(joint) if joint is not None else [0.0, 0.0, 0.0] for joint in frame]
        for frame in frames
    ]


def model_input(frames: Sequence[OutFrame]) -> list[list[float]]:
    """[T, 40] rows: 13 joints x (x, y, c) in JOINTS order, missing joints
    as zeros, then the gap indicator (1.0 when every joint is missing)."""
    rows = []
    for frame in frames:
        row: list[float] = []
        for joint in frame:
            row.extend(joint if joint is not None else (0.0, 0.0, 0.0))
        row.append(1.0 if all(j is None for j in frame) else 0.0)
        rows.append(row)
    return rows
