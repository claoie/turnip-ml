"""Generate a synthetic trick-labeled fixture set for the fixture runner.

SYNTHETIC. Every pose here is generated from its labels with the smoke
fixture's motion model (make_smoke_fixture.py): gainer is vertical
oscillation, corkscrew horizontal, a combo both at once. None of it is
footage, and none of it stands in for the maintainer's fixture set
(Training/fixtures/tricks/). It exists so the unit tests and the training
pipeline's synthetic rehearsal can run the fixture runner and the gate.

Writes, into --out:
  fixtures.json                   the set, in fixture_runner.py's format
  poses/<pose_sha256>.tkp1.gz     one TKP1 blob per fixture

Deterministic: the seed fixes every random draw and tkp1.encode_gz writes
reproducible gzip, so the same seed rewrites the same files. Regenerate
with:
  python Training/make_synthetic_fixtures.py --out data/fixtures --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import tkp1
from make_smoke_fixture import make_sample

# (fixture id, frames, [(names, start_frame, end_frame)]): the core
# vocabulary alone, two tricks in one sequence, a combo, and background.
FIXTURES: list[tuple[str, int, list[tuple[list[str], int, int]]]] = [
    ("synthetic-gainer", 40, [(["gainer"], 8, 24)]),
    ("synthetic-corkscrew", 40, [(["corkscrew"], 12, 28)]),
    ("synthetic-sequence", 48, [(["gainer"], 4, 20), (["corkscrew"], 28, 44)]),
    ("synthetic-combo", 40, [(["corkscrew", "gainer"], 8, 28)]),
    ("synthetic-background", 32, []),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="fixture set dir to write")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)

    out = Path(args.out)
    poses = out / "poses"
    poses.mkdir(parents=True, exist_ok=True)
    entries = []
    for fixture_id, n_frames, segments in FIXTURES:
        motion = [(name, s, e) for names, s, e in segments for name in names]
        blob = tkp1.encode_gz(make_sample(rng, motion, n_frames))
        sha = hashlib.sha256(blob).hexdigest()
        (poses / f"{sha}.tkp1.gz").write_bytes(blob)
        entries.append(
            {
                "id": fixture_id,
                "pose_sha256": sha,
                "segments": [
                    {"start_frame": s, "end_frame": e, "names": sorted(names)}
                    for names, s, e in segments
                ],
            }
        )
    with open(out / "fixtures.json", "w", encoding="utf-8") as fh:
        json.dump({"fixtures": entries}, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {len(entries)} synthetic fixtures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
