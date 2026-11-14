"""Generate the tiny deterministic smoke-test fixture for train.py.

Writes Training/fixtures/smoke/manifest.jsonl and samples/*.json: eight
synthetic 32-frame @10Hz samples with trick segments. Each is generated as
MoveNet-17 keypoints (make_sample, which make_synthetic_fixtures.py also
uses for its TKP1 fixture poses) and stored after the segment transform
(Training/segment_transform.py), in the TKS1 body frame the farm's
segments arrive in.
Motion signatures are deliberately separable so the smoke run must reach
100% val accuracy — anything less is a training-pipeline bug, not data noise.

  background: near-static pose + tiny jitter
  gainer:     strong vertical oscillation   (high |vy|), arms swinging
              further than the body
  corkscrew:  strong horizontal oscillation (high |vx|), the body
              twisting narrower

Deterministic: the seed fixes every random draw; floats are rounded to 6dp so
the files are byte-stable. Regenerate with:
  python Training/make_smoke_fixture.py --out Training/fixtures/smoke --seed 0
then split with the real splitter (turnip-ml#12):
  python Training/split_dataset.py --in Training/fixtures/smoke/manifest.jsonl \\
      --out Training/fixtures/smoke/split.json --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import segment_transform

FRAMES = 32
FPS = 10

# Plausible standing figure, MoveNet-17 order, 0-1 coords (y down):
# nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles.
BASE_POSE = [
    (0.50, 0.18),
    (0.47, 0.16),
    (0.53, 0.16),
    (0.44, 0.17),
    (0.56, 0.17),
    (0.42, 0.28),
    (0.58, 0.28),
    (0.38, 0.42),
    (0.62, 0.42),
    (0.36, 0.55),
    (0.64, 0.55),
    (0.44, 0.58),
    (0.56, 0.58),
    (0.44, 0.74),
    (0.56, 0.74),
    (0.44, 0.90),
    (0.56, 0.90),
]

# (sample_id, source_id, [(trick_name, start_frame, end_frame)]). source_id
# groups samples as if cut from one source video, so the splitter's
# source grouping has something to keep together.
# Trick boundaries sit midway between window starts (2 mod 4, with the
# smoke config's 8-frame windows at stride 4), so every window is at least
# 6/8 one thing: no window is half trick, half background.
SAMPLES: list[tuple[str, str, list[tuple[str, int, int]]]] = [
    ("smoke-001", "source-a", [("gainer", 2, 22)]),
    ("smoke-002", "source-a", [("corkscrew", 6, 26)]),
    ("smoke-003", "source-a", []),
    ("smoke-004", "source-a", [("gainer", 2, 14), ("corkscrew", 18, 30)]),
    ("smoke-005", "source-b", [("corkscrew", 2, 22)]),
    ("smoke-006", "source-b", [("gainer", 10, 30)]),
    ("smoke-007", "source-c", []),
    ("smoke-008", "source-c", [("gainer", 2, 22)]),
]

AMP = 0.18  # motion amplitude (fraction of frame)
# A trick changes the body's shape, not only its place, since the model
# reads poses about the body's root (train.featurize_window): on a gainer
# the elbows and wrists (MoveNet 7-10) swing further than the body, and a
# corkscrew twists the body, narrowing it by up to TWIST.
ARM_SWING = 1.5
ARMS = range(7, 11)
TWIST = 0.6
WAVES = 2  # oscillation cycles per trick segment
JITTER = 0.0035  # background noise stddev


def trick_offset(name: str, f: int, start: int, end: int) -> tuple[float, float]:
    """(dx, dy) for frame f inside [start, end): sine envelope, 2 waves."""
    t = (f - start) / (end - start)
    # Sine envelope flattened (^0.75): motion ramps up fast and stays strong,
    # so edge windows still carry signal; still exactly 0 at the boundaries.
    env = math.sin(math.pi * t) ** 0.75
    osc = math.sin(2 * math.pi * WAVES * t)
    if name == "gainer":
        return 0.0, AMP * env * osc
    if name == "corkscrew":
        return AMP * env * osc, 0.0
    raise ValueError(f"unknown trick {name!r}")


def make_sample(
    rng: random.Random, tricks: list[tuple[str, int, int]], n_frames: int = FRAMES
) -> list[list[list[float]]]:
    frames = []
    for f in range(n_frames):
        frame = []
        for j, (bx, by) in enumerate(BASE_POSE):
            dx = dy = 0.0
            width = 1.0
            for name, s, e in tricks:
                if s <= f < e:
                    ox, oy = trick_offset(name, f, s, e)
                    if name == "gainer" and j in ARMS:
                        oy *= ARM_SWING
                    if name == "corkscrew":
                        width -= TWIST * abs(ox) / AMP
                    dx += ox
                    dy += oy
            x = 0.5 + (bx - 0.5) * width + dx + rng.gauss(0, JITTER)
            y = by + dy + rng.gauss(0, JITTER)
            score = min(1.0, max(0.5, 0.92 + rng.gauss(0, 0.02)))
            frame.append([round(x, 6), round(y, 6), round(score, 6)])
        frames.append(frame)
    return frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="fixture dir to write")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)

    out = Path(args.out)
    samples_dir = out / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = []
    for sample_id, source_id, tricks in SAMPLES:
        transformed = segment_transform.transform(make_sample(rng, tricks))
        if transformed is None:
            raise ValueError(f"{sample_id}: no torso to scale by")
        keypoints = [
            [[round(v, 6) for v in joint] for joint in frame]
            for frame in segment_transform.frames_as_array(transformed)
        ]
        rec = {
            "sample_id": sample_id,
            "fps": FPS,
            "keypoints": keypoints,
            "tricks": [
                {"name": n, "start_frame": s, "end_frame": e} for n, s, e in tricks
            ],
        }
        with open(samples_dir / f"{sample_id}.json", "w", encoding="utf-8") as fh:
            json.dump(rec, fh)
            fh.write("\n")
        manifest_lines.append(
            json.dumps({"sample_id": sample_id, "source_id": source_id})
        )
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as fh:
        fh.write("\n".join(manifest_lines) + "\n")
    print(f"wrote {len(SAMPLES)} samples to {samples_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
