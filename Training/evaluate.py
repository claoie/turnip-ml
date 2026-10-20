"""Holdout evaluation for the trick-detection model.

Compares predicted trick segments against ground-truth annotations on the
holdout split and writes a deterministic JSON report next to the model
artifact.

Inputs
------
--predictions   JSON object mapping sample_id -> list of predicted segments,
                in the model output contract (master plan): source-frame
                coordinates, one segment with N names, is_combo derived.
                Each segment: {"start_frame": int, "end_frame": int,
                "name": "a+b" | "names": ["a", "b"], "is_combo": bool?}.
                "background" is not a segment: absence of a segment means
                background.
--split         split.json from Training/split_dataset.py:
                sample_id -> "train" | "val" | "holdout".
--samples-dir   directory of per-sample JSON files ("<sample_id>.json")
                carrying the ground-truth "tricks" list:
                [{"name": str, "start_frame": int, "end_frame": int,
                  "clip_id": str?, "is_combo": bool?}].
                clip_id may be absent; when present it must be a non-empty
                string (null is refused). Entries on one window are one
                segment carrying all their names (see "Ground truth").
--out           report path. Defaults to "holdout_report.json" next to the
                predictions file (i.e. next to the model artifact).
--iou-threshold temporal-IoU threshold for a detection. Default 0.5.

Ground truth
------------
ground_truth_segments() turns a sample's "tricks" into segments, and
train.py labels its windows from the same segments, so what the model
learns and what it is scored against cannot drift apart. The rule: one
segment per distinct (start_frame, end_frame) window, carrying every name
labeled on it. Training/fetch_dataset.py writes one entry per (clip_id,
canonical name), all on the clip's window, so a combo clip is one segment
with N names (TRAINING_DESIGN.md §2.3), and so are two clips labeled on
the same window, which train.py labels "a+b" too. A name repeated on a
window counts once.

clip_id only adds checks: one clip's entries must share a window and must
not repeat a name, and a clip belongs to one holdout sample. An is_combo
flag describes the merged segment, not its own entry.

Reports carry scoring_version (SCORING_VERSION). 2 is this window rule;
reports without the field scored one segment per entry, which split
combo clips into single-name segments that could never name-match.

Metrics
-------
A predicted segment *detects* a ground-truth segment when their temporal
IoU >= threshold (greedy best-IoU matching, each segment matched at most
once). A detection is a *name match* when the predicted name set equals
the ground-truth name set ("a+b" and ["a", "b"] are the same).

- detection_rate: fraction of ground-truth trick segments detected
  (localization only, name-agnostic) -- the issue's "detection rate".
- precision / recall / f1: segment-level, requiring a name match --
  the issue's "clip-detection precision/recall".
- name_accuracy_given_detection: of the detected segments, the fraction
  whose names match.
- per_name: support / predicted / true-positive counts plus
  precision/recall for every trick name seen in ground truth or
  predictions.
- n_background_false_positives: predicted segments with zero IoU against
  every ground-truth segment (pure hallucinations).

On PCK: "PCK" (percentage of correct keypoints) is a pose-estimation
metric, and per the maintainer's 2026-10-06 ruling it is out of scope
here. Since the 2026-10-05 program direction, turnip-ml trains trick
*detection* (pose sequence -> trick segments + names); pose estimation
itself is MoveNet's job, and the repo's former pose-accuracy CI baseline
was removed, so there is no pose baseline to gate against here. There is
no keypoint prediction to score PCK against -- localization is covered by
detection_rate -- so this script evaluates segment detection and naming
instead.

Split discipline (fail closed)
------------------------------
- A prediction for a sample id absent from the split file is an error.
- A prediction for a non-holdout (train/val) sample is an error:
  evaluating anywhere but holdout is a methodology bug.
- A holdout sample with no entry in the predictions file is legal and
  means "the model predicted nothing for this sample".
- Malformed segments, contradictory is_combo flags, a clip whose entries
  disagree on its window or repeat a name, a clip in two holdout samples,
  or missing sample files are errors, never silently skipped. Error
  messages name the sample file and entry index, never the clip_id.

Determinism contract
--------------------
Same inputs -> byte-identical report. Iteration order is always sorted,
floats are rounded to 6 decimals, JSON is dumped with sorted keys and no
timestamps. The report records sha256 of the predictions and split
inputs (content hashes only, no paths) so a report is traceable to the
exact inputs that produced it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, NoReturn

SPLITS = ("train", "val", "holdout")
# The ground-truth rule the report was scored with (module docstring,
# "Ground truth"); bump it whenever the same inputs would score differently.
SCORING_VERSION = 2


# ---------------------------------------------------------------------------
# Loading / validation (fail closed)
# ---------------------------------------------------------------------------


def _fail(msg: str) -> NoReturn:
    raise ValueError(msg)


def load_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        _fail(f"cannot read JSON from {path}: {e}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_names(
    seg: dict[str, Any], where: str, check_combo: bool = True
) -> frozenset[str]:
    """Ground-truth and prediction segments may carry "name" ("a+b") or
    "names" (["a", "b"]); both normalize to the same frozenset.
    check_combo=False leaves is_combo to the caller (ground truth checks it
    against the merged segment)."""
    if "name" in seg and "names" in seg:
        _fail(f'{where}: segment has both "name" and "names"; use exactly one')
    if "names" in seg:
        raw = seg["names"]
        if (
            not isinstance(raw, list)
            or not raw
            or not all(isinstance(n, str) and n for n in raw)
        ):
            _fail(f'{where}: "names" must be a non-empty list of strings')
        names = frozenset(raw)
    elif "name" in seg:
        raw = seg["name"]
        if not isinstance(raw, str) or not raw:
            _fail(f'{where}: "name" must be a non-empty string')
        parts = raw.split("+")
        if any(not p for p in parts):
            _fail(f'{where}: "name" has an empty component in {raw!r}')
        names = frozenset(parts)
    else:
        _fail(f'{where}: segment needs "name" or "names"')
    if check_combo and "is_combo" in seg:
        _check_combo(seg["is_combo"], names, where)
    return names


def _check_combo(flag: Any, names: frozenset[str], where: str) -> None:
    if bool(flag) != (len(names) > 1):
        _fail(f"{where}: is_combo={flag} contradicts names={sorted(names)}")


def parse_segment(
    seg: Any, where: str, check_combo: bool = True
) -> tuple[int, int, frozenset[str]]:
    if not isinstance(seg, dict):
        _fail(f"{where}: segment must be an object, got {type(seg).__name__}")
    start, end = seg.get("start_frame"), seg.get("end_frame")
    if (
        not isinstance(start, int)
        or not isinstance(end, int)
        or isinstance(start, bool)
        or isinstance(end, bool)
    ):
        _fail(f"{where}: start_frame/end_frame must be ints")
    if start < 0 or end <= start:
        _fail(f"{where}: need 0 <= start_frame < end_frame, got {start}/{end}")
    return start, end, normalize_names(seg, where, check_combo)


def load_predictions(path: Path) -> dict[str, list[tuple[int, int, frozenset[str]]]]:
    data = load_json(path)
    if not isinstance(data, dict):
        _fail(f"{path}: predictions must be a JSON object of sample_id -> segments")
    out: dict[str, list[tuple[int, int, frozenset[str]]]] = {}
    for sid, segs in data.items():
        if not isinstance(segs, list):
            _fail(f"{path}: segments for sample {sid!r} must be a list")
        out[sid] = [
            parse_segment(s, f"{path} sample {sid!r} segment #{i}")
            for i, s in enumerate(segs)
        ]
    return out


def load_split(path: Path) -> dict[str, str]:
    data = load_json(path)
    if not isinstance(data, dict):
        _fail(f"{path}: split must be a JSON object of sample_id -> split")
    for sid, split in data.items():
        if split not in SPLITS:
            _fail(f"{path}: sample {sid!r} has unknown split {split!r}")
    return data


def load_ground_truth(
    samples_dir: Path, holdout_ids: list[str]
) -> dict[str, list[tuple[int, int, frozenset[str]]]]:
    gt: dict[str, list[tuple[int, int, frozenset[str]]]] = {}
    clip_owner: dict[str, str] = {}
    for sid in holdout_ids:
        sample_path = samples_dir / f"{sid}.json"
        if not sample_path.is_file():
            _fail(f"holdout sample {sid!r} has no file at {sample_path}")
        sample = load_json(sample_path)
        tricks = sample.get("tricks")
        if not isinstance(tricks, list):
            _fail(f"{sample_path}: missing 'tricks' list")
        gt[sid] = ground_truth_segments(tricks, str(sample_path))
        # Validated above: every entry is an object, every clip_id a string.
        for clip_id in {t["clip_id"] for t in tricks if "clip_id" in t}:
            if clip_owner.setdefault(clip_id, sid) != sid:
                _fail(f"{sample_path}: shares a clip with another holdout sample")
    return gt


def ground_truth_segments(
    tricks: list[Any], where: str
) -> list[tuple[int, int, frozenset[str]]]:
    """A sample's "tricks" as ground-truth segments, sorted: one per distinct
    window, carrying every name labeled on it (module docstring, "Ground
    truth"). train.py labels its windows from these same segments."""
    windows: dict[tuple[int, int], set[str]] = {}
    clips: dict[str, tuple[int, tuple[int, int], set[str]]] = {}
    combo_flags: list[tuple[str, tuple[int, int], Any]] = []
    for i, t in enumerate(tricks):
        entry = f"{where} trick #{i}"
        start, end, names = parse_segment(t, entry, check_combo=False)
        window = (start, end)
        if "clip_id" in t:
            clip_id = t["clip_id"]
            if not isinstance(clip_id, str) or not clip_id:
                _fail(f'{entry}: "clip_id" must be a non-empty string')
            first, clip_window, clip_names = clips.setdefault(
                clip_id, (i, window, set())
            )
            if clip_window != window:
                _fail(f"{entry}: same clip as trick #{first} but another window")
            if clip_names & names:
                _fail(f"{entry}: repeats a name of its clip (trick #{first})")
            clip_names |= names
        windows.setdefault(window, set()).update(names)
        if "is_combo" in t:
            combo_flags.append((entry, window, t["is_combo"]))
    for entry, window, flag in combo_flags:
        _check_combo(flag, frozenset(windows[window]), entry)
    return [(s, e, frozenset(windows[s, e])) for s, e in sorted(windows)]


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def temporal_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0, hi - lo)
    if inter == 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union


def match_segments(
    gt: list[tuple[int, int, frozenset[str]]],
    pred: list[tuple[int, int, frozenset[str]]],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """Greedy best-IoU matching. Returns (matches, matched_gt, matched_pred);
    matches are (gt_idx, pred_idx, iou), sorted deterministically."""
    candidates: list[tuple[float, int, int]] = []
    for gi, (gs, ge, _) in enumerate(gt):
        for pi, (ps, pe, _) in enumerate(pred):
            iou = temporal_iou((gs, ge), (ps, pe))
            if iou >= iou_threshold:
                candidates.append((iou, gi, pi))
    # Highest IoU first; index tie-breaks keep the order deterministic.
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    matches: list[tuple[int, int, float]] = []
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gi, pi, iou))
    matches.sort()
    return matches, matched_gt, matched_pred


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def r6(x: float) -> float:
    return round(float(x), 6)


def prf(tp: int, pred_n: int, gt_n: int) -> tuple[float, float, float]:
    """(precision, recall, f1) with vacuous-truth conventions spelled out:
    no predictions and no ground truth -> perfect; one-sided emptiness -> 0
    on the undefined side."""
    if pred_n == 0 and gt_n == 0:
        return 1.0, 1.0, 1.0
    precision = tp / pred_n if pred_n else 0.0
    recall = tp / gt_n if gt_n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def compute_metrics(
    gt: dict[str, list[tuple[int, int, frozenset[str]]]],
    pred: dict[str, list[tuple[int, int, frozenset[str]]]],
    iou_threshold: float,
) -> dict[str, Any]:
    total_gt = total_pred = 0
    n_detections = n_name_matches = 0
    n_matched_pred = 0
    n_bg_fp = 0
    per_name: dict[str, dict[str, int]] = {}
    per_sample: dict[str, dict[str, int]] = {}

    def name_entry(name: str) -> dict[str, int]:
        return per_name.setdefault(
            name, {"support": 0, "predicted": 0, "true_positives": 0}
        )

    for sid in sorted(gt):
        gsegs = sorted(gt[sid], key=lambda s: (s[0], s[1], sorted(s[2])))
        psegs = sorted(pred[sid], key=lambda s: (s[0], s[1], sorted(s[2])))
        total_gt += len(gsegs)
        total_pred += len(psegs)
        matches, matched_gt, matched_pred = match_segments(gsegs, psegs, iou_threshold)

        det = len(matched_gt)
        nm = 0
        for gi, pi, _ in matches:
            if gsegs[gi][2] == psegs[pi][2]:
                nm += 1
        n_detections += det
        n_name_matches += nm
        n_matched_pred += len(matched_pred)

        for gi, (gs, ge, gnames) in enumerate(gsegs):
            for name in gnames:
                name_entry(name)["support"] += 1
        for pi, (ps, pe, pnames) in enumerate(psegs):
            for name in pnames:
                name_entry(name)["predicted"] += 1
            # Zero IoU against every ground-truth segment: hallucinated.
            if pi not in matched_pred and all(
                temporal_iou((ps, pe), (gs, ge)) == 0.0 for gs, ge, _ in gsegs
            ):
                n_bg_fp += 1
        for gi, pi, _ in matches:
            if gsegs[gi][2] == psegs[pi][2]:
                for name in gsegs[gi][2]:
                    name_entry(name)["true_positives"] += 1

        per_sample[sid] = {
            "n_ground_truth_segments": len(gsegs),
            "n_predicted_segments": len(psegs),
            "n_detections": det,
            "n_name_matches": nm,
        }

    precision, recall, f1 = prf(n_name_matches, total_pred, total_gt)
    detection_rate = n_detections / total_gt if total_gt else 1.0
    name_acc = (
        n_name_matches / n_detections
        if n_detections
        else (1.0 if total_gt == 0 else 0.0)
    )

    per_name_out: dict[str, dict[str, float]] = {}
    for name in sorted(per_name):
        e = per_name[name]
        p, r, _ = prf(e["true_positives"], e["predicted"], e["support"])
        per_name_out[name] = {
            "support": e["support"],
            "predicted": e["predicted"],
            "true_positives": e["true_positives"],
            "precision": r6(p),
            "recall": r6(r),
        }

    return {
        "iou_threshold": iou_threshold,
        "n_holdout_samples": len(gt),
        "n_ground_truth_segments": total_gt,
        "n_predicted_segments": total_pred,
        "n_detections": n_detections,
        "n_name_matches": n_name_matches,
        "detection_rate": r6(detection_rate),
        "precision": r6(precision),
        "recall": r6(recall),
        "f1": r6(f1),
        "name_accuracy_given_detection": r6(name_acc),
        "n_background_false_positives": n_bg_fp,
        "n_unmatched_predictions": total_pred - n_matched_pred,
        "per_name": per_name_out,
        "per_sample": per_sample,
        "scoring_version": SCORING_VERSION,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Evaluate predicted trick segments on the holdout split."
    )
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--samples-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.iou_threshold <= 1.0:
        print("--iou-threshold must be in (0, 1]", file=sys.stderr)
        return 2

    predictions = load_predictions(args.predictions)
    split = load_split(args.split)

    # Split discipline: predictions may only cover holdout samples.
    for sid in predictions:
        if sid not in split:
            _fail(f"prediction for sample {sid!r} not present in {args.split}")
        if split[sid] != "holdout":
            _fail(
                f"prediction for sample {sid!r} is in split "
                f"{split[sid]!r}, not holdout — refusing to evaluate"
            )

    holdout_ids = sorted(sid for sid, s in split.items() if s == "holdout")
    gt = load_ground_truth(args.samples_dir, holdout_ids)
    pred_full = {sid: predictions.get(sid, []) for sid in holdout_ids}

    report = compute_metrics(gt, pred_full, args.iou_threshold)
    report["inputs"] = {
        "predictions_sha256": sha256_file(args.predictions),
        "split_sha256": sha256_file(args.split),
    }

    out = args.out or (args.predictions.parent / "holdout_report.json")
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as e:
        print(f"evaluate.py: error: {e}", file=sys.stderr)
        raise SystemExit(1)
