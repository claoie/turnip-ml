#!/usr/bin/env python3
"""Fixture runner: score a model on the trick-labeled fixture set.

Writes the fixture regression gate's input (Training/fixture_gate.py,
docs/TRAINING_DESIGN.md §6.3): the model from Training/train.py, run over
every fixture exactly as predict.py runs it over holdout samples (the
model math is train.py's, imported, not duplicated), scored against the
fixtures' labeled segments.

Inputs
------
--model     model.npz from Training/train.py.
--fixtures  the fixture set's fixtures.json. The real one is
            Training/fixtures/tricks/fixtures.json, labeled by the
            maintainer (see its README); tests and the pipeline's synthetic
            rehearsal use make_synthetic_fixtures.py.
--poses     directory of the fixtures' pose blobs, content-addressed as
            <pose_sha256>.tkp1.gz. Never in git: the farm job restores them
            from R2 (training/fixtures/poses/).
--out       score file path (JSON).

Fixture set (fixtures.json)
---------------------------
    {"fixtures": [
      {"id": "gainer-01",
       "pose_sha256": "<sha256 of the .tkp1.gz bytes>",
       "segments": [
         {"start_frame": 12, "end_frame": 40, "names": ["gainer"]}]}]}

- id: unique, lowercase [a-z0-9._-].
- pose_sha256: the pose blob (TKP1, Training/tkp1.py), 10 Hz canonical.
- segments: the maintainer's labels in the pose's frame indices,
  end-exclusive, inside the pose. Names are canonical trick names
  (normalized, no "+", not "background"); a combo is one segment with
  several names. Segments of one fixture do not overlap. A fixture may
  have no segments: a background-only fixture, where anything the model
  fires is a false positive.
Unknown or missing keys fail closed. The set needs at least one labeled
segment.

Output
------
    {"fixture_set": "<64 hex>", "segment_iou": 0.82, "name_accuracy": 0.91}
exactly the gate's input. `fixture_set` is the SHA-256 of a canonical
serialization of the whole set (every fixture's id, pose_sha256 and
segments, sorted) together with the scoring rule (SCORING): adding,
removing or changing a fixture, its labels or its pose, or changing the
rule, changes it, and the gate refuses to compare scores across it.
Reformatting fixtures.json does not change it.

Scores
------
Pooled over the whole set. Per fixture, predicted segments are matched
one-to-one to labeled segments by evaluate.py's greedy best-IoU matching
at temporal IoU >= 0.5, the evaluator's detection threshold.

- segment_iou = sum of matched IoUs / (labeled + predicted - matched):
  every labeled segment missed and every predicted segment left unmatched
  (on a background-only fixture too) counts as IoU 0.
- name_accuracy = matched segments whose name set equals the label's /
  labeled segments: the share of labeled segments found *and* named.
  Unlike the evaluator's name_accuracy_given_detection it does not
  condition on detection, so finding less lowers both scores.

Determinism: same model + fixtures + poses -> byte-identical score file
(sorted keys, scores rounded to 6 decimals, no timestamps).

Fail closed: a malformed fixture set, an empty one, a pose blob that is
missing, does not match its pose_sha256 or does not decode, a segment
past its pose, or an unreadable model are errors (exit 2), never a score.

Privacy: stdout carries the digest, counts and the two scores only.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import evaluate
import numpy as np
import numpy.typing as npt
import predict
import tkp1
import train
from canonicalize import normalize

EXIT_OK = 0
EXIT_ERROR = 2

DETECTION_IOU = 0.5
# Hashed into fixture_set: scores taken under another rule are not
# comparable either. Bump the version whenever the scoring changes.
SCORING = {"version": 1, "detection_iou": DETECTION_IOU}

FIXTURE_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")
HEX64_RE = re.compile(r"[0-9a-f]{64}")
FIXTURE_KEYS = frozenset({"id", "pose_sha256", "segments"})
SEGMENT_KEYS = frozenset({"start_frame", "end_frame", "names"})

Segment = tuple[int, int, frozenset[str]]


class FixtureError(Exception):
    """A fixture set, pose or model the runner refuses to score."""


@dataclass(frozen=True)
class Fixture:
    id: str
    pose_sha256: str
    segments: tuple[Segment, ...]  # sorted by segment_key


def _fail(msg: str) -> NoReturn:
    raise FixtureError(msg)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def segment_key(seg: Segment) -> tuple[int, int, list[str]]:
    # frozensets only order by inclusion; sort names for a total order.
    return seg[0], seg[1], sorted(seg[2])


# ---------------------------------------------------------------------------
# Fixture set
# ---------------------------------------------------------------------------


def _parse_segment(seg: Any, where: str) -> Segment:
    if not isinstance(seg, dict) or set(seg) != SEGMENT_KEYS:
        _fail(f"{where}: needs exactly the keys {sorted(SEGMENT_KEYS)}")
    start, end = seg["start_frame"], seg["end_frame"]
    if not _is_int(start) or not _is_int(end) or not 0 <= start < end:
        _fail(f"{where}: need int 0 <= start_frame < end_frame, got {start}/{end}")
    names = seg["names"]
    if not isinstance(names, list) or not names:
        _fail(f'{where}: "names" must be a non-empty list')
    for name in names:
        if (
            not isinstance(name, str)
            or not name
            or name != normalize(name)
            or "+" in name
            or name == train.BACKGROUND
        ):
            _fail(
                f"{where}: {name!r} is not a canonical trick name "
                '(normalized, no "+", not "background")'
            )
    if len(set(names)) != len(names):
        _fail(f"{where}: duplicate names {sorted(names)}")
    return start, end, frozenset(names)


def _parse_fixture(entry: Any, where: str) -> Fixture:
    if not isinstance(entry, dict) or set(entry) != FIXTURE_KEYS:
        _fail(f"{where}: needs exactly the keys {sorted(FIXTURE_KEYS)}")
    fixture_id = entry["id"]
    if not isinstance(fixture_id, str) or not FIXTURE_ID_RE.fullmatch(fixture_id):
        _fail(f"{where}: id must match {FIXTURE_ID_RE.pattern}, got {fixture_id!r}")
    where = f"fixture {fixture_id!r}"
    pose_sha256 = entry["pose_sha256"]
    if not isinstance(pose_sha256, str) or not HEX64_RE.fullmatch(pose_sha256):
        _fail(f"{where}: pose_sha256 must be 64 lowercase hex chars")
    raw = entry["segments"]
    if not isinstance(raw, list):
        _fail(f'{where}: "segments" must be a list')
    segments = sorted(
        (_parse_segment(s, f"{where} segment #{i}") for i, s in enumerate(raw)),
        key=segment_key,
    )
    for a, b in itertools.pairwise(segments):
        if b[0] < a[1]:
            _fail(
                f"{where}: segments {a[0]}-{a[1]} and {b[0]}-{b[1]} overlap "
                "(a combo is one segment with several names)"
            )
    return Fixture(fixture_id, pose_sha256, tuple(segments))


def load_fixture_set(path: Path) -> list[Fixture]:
    """The validated fixture set, sorted by id."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError, RecursionError) as e:
        _fail(f"cannot read fixture set {path}: {e}")
    if not isinstance(data, dict) or set(data) != {"fixtures"}:
        _fail(f'{path}: must be an object with exactly one key, "fixtures"')
    entries = data["fixtures"]
    if not isinstance(entries, list):
        _fail(f'{path}: "fixtures" must be a list')
    fixtures = [
        _parse_fixture(e, f"{path} fixture #{i}") for i, e in enumerate(entries)
    ]
    ids = [f.id for f in fixtures]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        _fail(f"{path}: duplicate fixture ids {duplicates}")
    if not fixtures:
        _fail(
            f"{path}: the fixture set is empty; the maintainer adds fixtures "
            "(Training/fixtures/tricks/README.md)"
        )
    if not any(f.segments for f in fixtures):
        _fail(f"{path}: no fixture has a labeled segment; nothing to score")
    return sorted(fixtures, key=lambda f: f.id)


def fixture_set_digest(fixtures: list[Fixture]) -> str:
    """SHA-256 of the canonical fixture set plus the scoring rule."""
    canonical = {
        "fixtures": [
            {
                "id": f.id,
                "pose_sha256": f.pose_sha256,
                "segments": [
                    {"start_frame": s, "end_frame": e, "names": sorted(names)}
                    for s, e, names in f.segments
                ],
            }
            for f in sorted(fixtures, key=lambda f: f.id)
        ],
        "scoring": SCORING,
    }
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_pose(poses_dir: Path, fixture: Fixture) -> npt.NDArray[np.float64]:
    """The fixture's keypoints, from a blob verified against pose_sha256."""
    path = poses_dir / f"{fixture.pose_sha256}.tkp1.gz"
    where = f"fixture {fixture.id!r}"
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        _fail(
            f"{where}: pose blob {path.name} not found in {poses_dir} "
            "(R2 training/fixtures/poses/; see Training/fixtures/tricks/README.md)"
        )
    except OSError as e:
        _fail(f"{where}: cannot read {path}: {e}")
    if hashlib.sha256(data).hexdigest() != fixture.pose_sha256:
        _fail(f"{where}: {path} does not match its pose_sha256")
    try:
        pose = tkp1.decode_gz(data)
        keypoints = train._check_keypoints(pose.keypoints, fixture.id)
    except ValueError as e:  # Tkp1Error is a ValueError
        _fail(f"{where}: undecodable pose blob {path.name}: {e}")
    for start, end, _ in fixture.segments:
        if end > pose.frame_count:
            _fail(
                f"{where}: segment {start}-{end} ends past the "
                f"{pose.frame_count}-frame pose"
            )
    return keypoints


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def predict_fixtures(
    model_path: Path, fixtures: list[Fixture], poses_dir: Path
) -> dict[str, list[Segment]]:
    samples = [
        {"sample_id": f.id, "keypoints": load_pose(poses_dir, f), "tricks": []}
        for f in fixtures
    ]
    try:
        ckpt = train.load_checkpoint(model_path)
        raw = predict.predict_samples(ckpt, samples)
        return {
            sid: [
                evaluate.parse_segment(s, f"prediction for fixture {sid!r}")
                for s in segs
            ]
            for sid, segs in raw.items()
        }
    except (ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile) as e:
        # A corrupt npz surfaces as any of these from np.load or the math.
        _fail(f"cannot run model {model_path}: {e}")


def score(
    fixtures: list[Fixture], predictions: dict[str, list[Segment]]
) -> dict[str, Any]:
    """Pooled scores and counts; the scores are unrounded."""
    n_labeled = n_predicted = n_matched = n_named = 0
    iou_sum = 0.0
    for f in fixtures:
        labeled = list(f.segments)
        predicted = sorted(predictions.get(f.id, []), key=segment_key)
        matches, _, _ = evaluate.match_segments(labeled, predicted, DETECTION_IOU)
        n_labeled += len(labeled)
        n_predicted += len(predicted)
        n_matched += len(matches)
        for li, pi, iou in matches:
            iou_sum += iou
            if labeled[li][2] == predicted[pi][2]:
                n_named += 1
    # load_fixture_set guarantees n_labeled >= 1, so neither is 0/0.
    return {
        "segment_iou": iou_sum / (n_labeled + n_predicted - n_matched),
        "name_accuracy": n_named / n_labeled,
        "n_fixtures": len(fixtures),
        "n_labeled": n_labeled,
        "n_predicted": n_predicted,
        "n_matched": n_matched,
    }


def score_file_text(digest: str, scores: dict[str, Any]) -> str:
    out = {
        "fixture_set": digest,
        "segment_iou": round(scores["segment_iou"], 6),
        "name_accuracy": round(scores["name_accuracy"], 6),
    }
    return json.dumps(out, indent=2, sort_keys=True) + "\n"


def run(
    model_path: Path, fixtures_path: Path, poses_dir: Path
) -> tuple[str, dict[str, Any]]:
    """(score file text, scores and counts)."""
    fixtures = load_fixture_set(fixtures_path)
    digest = fixture_set_digest(fixtures)
    scores = score(fixtures, predict_fixtures(model_path, fixtures, poses_dir))
    return score_file_text(digest, scores), {"fixture_set": digest, **scores}


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        _fail(f"cannot write {path}: {e}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, type=Path, help="model.npz")
    p.add_argument("--fixtures", required=True, type=Path, help="fixtures.json")
    p.add_argument(
        "--poses", required=True, type=Path, help="dir of <pose_sha256>.tkp1.gz"
    )
    p.add_argument("--out", required=True, type=Path, help="score file path")
    args = p.parse_args(argv)
    try:
        text, result = run(args.model, args.fixtures, args.poses)
        write_text_atomic(args.out, text)
    except FixtureError as e:
        print(f"fixture_runner: error: {e}", file=sys.stderr)
        return EXIT_ERROR
    print(
        f"scored {result['n_fixtures']} fixtures on fixture set "
        f"{result['fixture_set']}: {result['n_labeled']} labeled segments, "
        f"{result['n_predicted']} predicted, {result['n_matched']} matched; "
        f"segment_iou {result['segment_iou']:.6f}, "
        f"name_accuracy {result['name_accuracy']:.6f}"
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
