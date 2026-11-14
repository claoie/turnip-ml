#!/usr/bin/env python3
"""Writes the segment-transform fixture vectors (docs/MODEL_CONTRACT.md §3.6).

Each case is a short canonical 10 Hz sequence of 17 MoveNet keypoints and
the output of the reference transform (Training/segment_transform.py):
13 joints per frame in the TKS1 order, null where missing, and the [T, 40]
model input. The client tests its own transform against these; the cases
cover each rule of the transform once.

    python Training/make_segment_vectors.py \\
        --out Training/fixtures/segment_transform/vectors.json

Deterministic: the cases are fixed literals, input values are float32,
and the output is written with sorted keys.
"""

from __future__ import annotations

import argparse
import copy
import json
import struct
from pathlib import Path
from typing import Any

import segment_transform as st

MISSING = [0.0, 0.0, 0.0]


def f32(x: float) -> float:
    return float(struct.unpack("<f", struct.pack("<f", x))[0])


def kp(x: float, y: float, c: float = 0.9) -> list[float]:
    return [f32(x), f32(y), f32(c)]


# A person standing upright in the middle of the picture, arms slightly out.
STANDING = [
    kp(0.50, 0.20),  # 0 nose
    kp(0.51, 0.19),  # 1 left eye
    kp(0.49, 0.19),  # 2 right eye
    kp(0.53, 0.20),  # 3 left ear
    kp(0.47, 0.20),  # 4 right ear
    kp(0.56, 0.30),  # 5 left shoulder
    kp(0.44, 0.30),  # 6 right shoulder
    kp(0.60, 0.40),  # 7 left elbow
    kp(0.40, 0.40),  # 8 right elbow
    kp(0.62, 0.50),  # 9 left wrist
    kp(0.38, 0.50),  # 10 right wrist
    kp(0.54, 0.55),  # 11 left hip
    kp(0.46, 0.55),  # 12 right hip
    kp(0.55, 0.70),  # 13 left knee
    kp(0.45, 0.70),  # 14 right knee
    kp(0.55, 0.85),  # 15 left ankle
    kp(0.45, 0.85),  # 16 right ankle
]


def frame(**changes: list[float]) -> list[list[float]]:
    """STANDING with keypoints replaced by index: frame(k5=MISSING)."""
    out = copy.deepcopy(STANDING)
    for key, value in changes.items():
        out[int(key[1:])] = value
    return out


def shifted(dx: float, dy: float, scale: float = 1.0) -> list[list[float]]:
    """STANDING moved by (dx, dy) and scaled about the hip midpoint."""
    return [
        kp(0.5 + (x - 0.5) * scale + dx, 0.55 + (y - 0.55) * scale + dy, c)
        for x, y, c in STANDING
    ]


CASES: list[tuple[str, str, list[list[list[float]]]]] = [
    (
        "standing",
        "One upright frame: every joint present, the root at the origin.",
        [frame()],
    ),
    (
        "partial_face",
        (
            "Only the nose and the left ear are detected: the head is their mean, "
            "its confidence the mean of theirs."
        ),
        [frame(k1=MISSING, k2=MISSING, k4=MISSING, k3=kp(0.53, 0.20, 0.5))],
    ),
    (
        "no_face",
        "No face keypoint is detected: the head is missing, nothing else.",
        [frame(**{f"k{i}": MISSING for i in range(5)})],
    ),
    (
        "missing_elbow",
        "The left elbow is not detected, so the left wrist, its child, is missing too.",
        [frame(k7=MISSING)],
    ),
    (
        "missing_shoulder",
        (
            "In the second frame the right shoulder is not detected: there is no "
            "neck, so both shoulders, both arms and the head are missing; the "
            "legs stay. The first frame gives the torso length."
        ),
        [frame(), frame(k6=MISSING)],
    ),
    (
        "coincident_knee",
        (
            "The right knee sits exactly on the right hip: a zero-length bone, so "
            "the knee is missing and the ankle with it."
        ),
        [frame(k14=STANDING[12])],
    ),
    (
        "gap_frame",
        (
            "The middle frame is missing its left hip: no root, so it is a gap "
            "frame. Center and scale come from the other two."
        ),
        [shifted(0.0, 0.0), frame(k11=MISSING), shifted(0.04, -0.02)],
    ),
    (
        "jump",
        (
            "Four frames rising and falling, the body growing in the picture: the "
            "root moves in torso lengths about the sequence's mean root, scaled "
            "by the median torso over four frames (the mean of the middle two)."
        ),
        [
            shifted(0.0, 0.0, 1.0),
            shifted(0.01, -0.10, 1.1),
            shifted(0.02, -0.15, 1.2),
            shifted(0.03, 0.0, 1.3),
        ],
    ),
    (
        "no_torso",
        (
            "Both shoulders are missing in every frame: there is no torso length "
            "to scale by, so the transform has no output (null)."
        ),
        [frame(k5=MISSING, k6=MISSING), frame(k5=MISSING, k6=MISSING)],
    ),
]


def build() -> dict[str, Any]:
    cases = []
    for name, description, frames in CASES:
        out = st.transform(frames)
        cases.append(
            {
                "name": name,
                "description": description,
                "input": frames,
                "output": None
                if out is None
                else [[list(j) if j is not None else None for j in f] for f in out],
                "model_input": None if out is None else st.model_input(out),
            }
        )
    return {
        "joints": list(st.JOINTS),
        "bones": st.BONES,
        "tolerance": 1e-5,
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(build(), indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
