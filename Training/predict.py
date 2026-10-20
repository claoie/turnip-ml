#!/usr/bin/env python3
"""Run a trained model over one split and write its predicted trick segments.

Produces the predictions file Training/evaluate.py scores: the model from
Training/train.py, applied to every sample of one split (holdout by
default), decoded into trick segments exactly as train.py decodes its val
segments (same windowing, features, standardization and decoding; the
model math is train.py's, imported, not duplicated).

Inputs
------
--model        model.npz from Training/train.py.
--split        split.json from Training/split_dataset.py.
--samples-dir  directory of per-sample JSON files ("<sample_id>.json").
--split-name   which split to predict: holdout (default), val or train.
--out          predictions path (JSON).

Output: {"<sample_id>": [{"start_frame", "end_frame", "names", "is_combo"}]}
for every sample of the split (an empty list when nothing was detected),
in source-frame coordinates; `names` is the sorted canonical name list of
one segment (a combo carries several), the shape evaluate.py reads.

Predictions never see ground truth: each sample's "tricks" are dropped
before windowing, so a label can't leak into what is being evaluated.

Exit codes: 0 = written, 2 = input error (fail closed): a missing or
malformed model, split or sample file, a sample file whose sample_id
disagrees with its name, or a split with no samples to predict.

Determinism: same model + inputs -> byte-identical predictions (sorted
keys, sorted samples, no timestamps). Stdout carries counts only: the
predictions are per-sample and stay in the output file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import train

SPLIT_NAMES = ("holdout", "val", "train")


def load_split_samples(
    split: dict[str, str], samples_dir: Path, split_name: str
) -> list[dict[str, Any]]:
    """The named split's samples, keypoints only (no ground truth)."""
    ids = sorted(sid for sid, s in split.items() if s == split_name)
    if not ids:
        raise ValueError(f"no {split_name} samples in the split: nothing to predict")
    samples = []
    for sid in ids:
        path = samples_dir / f"{sid}.json"
        try:
            with open(path, encoding="utf-8") as fh:
                rec = json.load(fh)
        except (OSError, ValueError) as e:
            raise ValueError(f"cannot read sample {path}: {e}") from e
        if not isinstance(rec, dict) or rec.get("sample_id") != sid:
            raise ValueError(f"sample file {path} does not hold sample {sid!r}")
        keypoints = train._check_keypoints(rec.get("keypoints"), sid)
        samples.append({"sample_id": sid, "keypoints": keypoints, "tricks": []})
    return samples


def predict_split(
    model_path: Path, split_path: Path, samples_dir: Path, split_name: str
) -> dict[str, list[dict[str, Any]]]:
    ckpt = train.load_checkpoint(model_path)
    split = train.load_split(split_path)
    samples = load_split_samples(split, samples_dir, split_name)
    return predict_samples(ckpt, samples)


def predict_samples(
    ckpt: dict[str, Any], samples: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Segments for keypoints-only samples ({"sample_id", "keypoints",
    "tricks": []}); every sample gets an entry, empty when nothing fired.
    Shared with Training/fixture_runner.py."""
    window_frames = int(ckpt["window_frames"])
    window_stride = int(ckpt["window_stride"])
    classes = [str(c) for c in ckpt["classes"]]
    xp = np  # inference is cheap; the GPU seam is a training concern

    predictions: dict[str, list[dict[str, Any]]] = {s["sample_id"]: [] for s in samples}
    windowed = [s for s in samples if s["keypoints"].shape[0] >= window_frames]
    if windowed:
        X_raw, _, prov = train.build_dataset(xp, windowed, window_frames, window_stride)
        X = (np.asarray(X_raw) - ckpt["feat_mean"]) / ckpt["feat_std"]
        pred = [classes[i] for i in train.predict(xp, X, ckpt["W"], ckpt["b"])]
        for seg in train.decode_segments(prov, pred, window_frames):
            predictions[seg["sample_id"]].append(
                {
                    "start_frame": seg["start_frame"],
                    "end_frame": seg["end_frame"],
                    "names": sorted(seg["name"].split("+")),
                    "is_combo": seg["is_combo"],
                }
            )
    return predictions


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, type=Path, help="model.npz")
    p.add_argument("--split", required=True, type=Path, help="split.json")
    p.add_argument("--samples-dir", required=True, type=Path, help="sample JSON dir")
    p.add_argument("--split-name", choices=SPLIT_NAMES, default="holdout")
    p.add_argument("--out", required=True, type=Path, help="predictions JSON path")
    args = p.parse_args(argv)
    try:
        predictions = predict_split(
            args.model, args.split, args.samples_dir, args.split_name
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(predictions, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except (ValueError, TypeError, OSError) as e:
        print(f"predict.py: error: {e}", file=sys.stderr)
        return 2
    n_segments = sum(len(v) for v in predictions.values())
    print(
        f"predicted {n_segments} segments over {len(predictions)} "
        f"{args.split_name} samples"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
