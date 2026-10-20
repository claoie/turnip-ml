#!/usr/bin/env python3
"""Export a trained model to Core ML: a deterministic fp16 .mlpackage zip.

turnip-ml#16, the export step of docs/MODEL_CONTRACT.md §2. The model is
Training/train.py's windowed softmax-linear classifier
(docs/TRAINING_DESIGN.md §4.1), rebuilt as a Core ML ML Program with the
coremltools MIL builder and converted at fp16 (compute and weights).

Inputs
------
--model   model.npz from Training/train.py.
--out     artifact path (.mlpackage.zip).

Model I/O (docs/MODEL_CONTRACT.md §3)
-------------------------------------
Input  `pose`: float32 [T, 52], the canonical 10 Hz sequence: per frame
       17 x (x, y, confidence) in MoveNet-17 order, then the gap
       indicator. window_frames <= T <= tkp1.MAX_FRAMES. The baseline
       reads x and y only; confidence and the gap channel are accepted
       so the input matches the contract the production model will read.
Output `window_probs`: float32 [N, C], N = (T - window_frames) //
       window_stride + 1. Row i is the softmax over `classes` for the
       window covering frames [i * window_stride, i * window_stride +
       window_frames), the same windows and features as train.py.
       Decoding windows into segments is the client's step (contract
       §3.2.1); the manifest's output_desc (`descriptors`) carries the
       window parameters and the class list it needs.

Artifact
--------
A zip of TrickDetector.mlpackage, not a compiled .mlmodelc: compiling
needs macOS (coremlcompiler) and the pipeline runs on Ubuntu, so the
client compiles the package on device (contract §4).

Determinism: same model.npz + same coremltools -> byte-identical zip.
coremltools stamps a conversion date into the model metadata and random
UUIDs into the package manifest; both are replaced (the date dropped, the
UUIDs derived from each item's path), the spec is serialized with
deterministic map order, and the zip is stored (no compression), sorted,
with fixed timestamps and permissions.

Exit codes: 0 = written, 2 = input error (fail closed): unreadable or
malformed model, or coremltools missing.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import tkp1
import train

PACKAGE_NAME = "TrickDetector.mlpackage"
INPUT_NAME = "pose"
OUTPUT_NAME = "window_probs"
INPUT_CHANNELS = tkp1.KEYPOINT_COUNT * tkp1.CHANNELS + 1  # + gap indicator: 52
# Default T the runtime pre-allocates for: a 60 s source.
DEFAULT_FRAMES = 600
# coremltools stamps the wall-clock date here; a timestamp breaks the
# byte-identical artifact, so it is dropped.
CONVERSION_DATE_KEY = "com.github.apple.coremltools.conversion_date"
# Fixed per-item identifiers for the package manifest (coremltools writes
# random UUIDs): uuid5 of the item's path under this namespace.
MANIFEST_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL, "https://github.com/hoiekim/turnip-ml"
)
ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)


class ExportError(Exception):
    """The export refuses to produce an artifact rather than guess."""


def load_model(path: Path) -> dict[str, Any]:
    """The checkpoint, with every field the export reads checked."""
    try:
        ckpt = train.load_checkpoint(path)
    except ValueError as e:
        raise ExportError(str(e)) from e
    classes = [str(c) for c in ckpt["classes"]]
    W, S = int(ckpt["window_frames"]), int(ckpt["window_stride"])
    n_features = 2 * 2 * tkp1.KEYPOINT_COUNT  # mean position + mean speed: 68
    shapes = {
        "W": (n_features, len(classes)),
        "b": (len(classes),),
        "feat_mean": (n_features,),
        "feat_std": (n_features,),
    }
    for key, shape in shapes.items():
        if ckpt[key].shape != shape:
            raise ExportError(f"{path}: {key} has shape {ckpt[key].shape}, not {shape}")
        if not np.all(np.isfinite(ckpt[key])):
            raise ExportError(f"{path}: {key} is not finite")
    if len(set(classes)) != len(classes) or train.BACKGROUND not in classes:
        raise ExportError(f"{path}: classes must be distinct and include background")
    if W < 2 or S < 1 or W > tkp1.MAX_FRAMES:
        raise ExportError(f"{path}: bad window params {W}/{S}")
    if np.any(ckpt["feat_std"] <= 0):
        raise ExportError(f"{path}: feat_std must be positive")
    return {**ckpt, "classes": classes, "window_frames": W, "window_stride": S}


def descriptors(ckpt: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The manifest's input_desc and output_desc for this model."""
    input_desc = {
        "format": "TKP1",
        "format_version": tkp1.FORMAT_VERSION,
        "sample_rate_hz": tkp1.CANONICAL_RATE_HZ,
        "keypoint_count": tkp1.KEYPOINT_COUNT,
        "features": ["x", "y", "confidence", "gap_indicator"],
        "input_name": INPUT_NAME,
        "shape": ["T", INPUT_CHANNELS],
        "min_frames": ckpt["window_frames"],
        "max_frames": tkp1.MAX_FRAMES,
    }
    output_desc = {
        "coordinates": "source-frame",
        "kind": "window_classifier",
        "output_name": OUTPUT_NAME,
        "shape": ["N", len(ckpt["classes"])],
        "window_frames": ckpt["window_frames"],
        "window_stride": ckpt["window_stride"],
        "classes": ckpt["classes"],
        "background_class": train.BACKGROUND,
        "combo_separator": "+",
    }
    return input_desc, output_desc


def build_mlmodel(ckpt: dict[str, Any]) -> Any:
    """The classifier as an fp16 Core ML ML Program (needs coremltools)."""
    try:
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import get_new_symbol
    except ImportError as e:
        raise ExportError(
            "export needs coremltools: pip install -r Training/requirements-export.txt"
        ) from e

    W, S = ckpt["window_frames"], ckpt["window_stride"]
    weight = np.asarray(ckpt["W"], dtype=np.float32).T  # [C, 68]
    bias = np.asarray(ckpt["b"], dtype=np.float32)
    mean = np.asarray(ckpt["feat_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feat_std"], dtype=np.float32)
    n_xy = 2 * tkp1.KEYPOINT_COUNT

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(get_new_symbol(), INPUT_CHANNELS))],
        opset_version=ct.target.iOS15,
    )
    def program(pose: Any) -> Any:
        # [T, 52] -> x, y per joint as [T, 34], joint-major like
        # train.featurize_window's pos.mean(axis=0).ravel().
        kp = mb.slice_by_index(
            x=pose, begin=[0, 0], end=[0, INPUT_CHANNELS - 1], end_mask=[True, False]
        )
        kp = mb.reshape(x=kp, shape=[-1, tkp1.KEYPOINT_COUNT, tkp1.CHANNELS])
        xy = mb.slice_by_index(
            x=kp, begin=[0, 0, 0], end=[0, 0, 2], end_mask=[True, True, False]
        )
        seq = mb.transpose(x=mb.reshape(x=xy, shape=[-1, n_xy]), perm=[1, 0])
        seq = mb.expand_dims(x=seq, axes=[0])  # [1, 34, T]
        # Window i = frames [i*S, i*S + W): mean position over its W frames,
        # mean |velocity| over its W - 1 frame differences.
        pos = mb.avg_pool(x=seq, kernel_sizes=[W], strides=[S], pad_type="valid")
        nxt = mb.slice_by_index(
            x=seq, begin=[0, 0, 1], end=[0, 0, 0], end_mask=[True, True, True]
        )
        prv = mb.slice_by_index(
            x=seq, begin=[0, 0, 0], end=[0, 0, -1], end_mask=[True, True, False]
        )
        speed = mb.avg_pool(
            x=mb.abs(x=mb.sub(x=nxt, y=prv)),
            kernel_sizes=[W - 1],
            strides=[S],
            pad_type="valid",
        )
        feat = mb.concat(values=[pos, speed], axis=1)  # [1, 68, N]
        feat = mb.transpose(x=mb.squeeze(x=feat, axes=[0]), perm=[1, 0])  # [N, 68]
        feat = mb.real_div(x=mb.sub(x=feat, y=mean), y=std)
        logits = mb.linear(x=feat, weight=weight, bias=bias)
        return mb.softmax(x=logits, axis=-1, name=OUTPUT_NAME)

    shape = (
        ct.RangeDim(lower_bound=W, upper_bound=tkp1.MAX_FRAMES, default=DEFAULT_FRAMES),
        INPUT_CHANNELS,
    )
    return ct.convert(
        program,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS15,
        inputs=[ct.TensorType(name=INPUT_NAME, shape=shape, dtype=np.float32)],
    )


def package_files(mlmodel: Any, ckpt: dict[str, Any]) -> dict[str, bytes]:
    """The .mlpackage as {relative path: bytes}, made deterministic."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / PACKAGE_NAME
        mlmodel.save(str(root))
        files = {
            p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*"))
            if p.is_file()
        }
    manifest = json.loads(files.pop("Manifest.json"))
    entries = manifest["itemInfoEntries"]
    ids = {
        old: str(uuid.uuid5(MANIFEST_NAMESPACE, entry["path"])).upper()
        for old, entry in entries.items()
    }
    root_entry = entries[manifest["rootModelIdentifier"]]
    files["Manifest.json"] = (
        json.dumps(
            {
                "fileFormatVersion": manifest["fileFormatVersion"],
                "itemInfoEntries": {ids[k]: v for k, v in entries.items()},
                "rootModelIdentifier": ids[manifest["rootModelIdentifier"]],
            },
            indent=4,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )

    spec_path = "Data/" + root_entry["path"]
    spec = mlmodel.get_spec()
    spec.ParseFromString(files[spec_path])
    meta = spec.description.metadata
    if CONVERSION_DATE_KEY in meta.userDefined:
        del meta.userDefined[CONVERSION_DATE_KEY]
    meta.shortDescription = (
        "Turnip trick detection: windowed classifier over the canonical "
        "10 Hz pose sequence (turnip-ml docs/MODEL_CONTRACT.md)"
    )
    meta.userDefined["turnip.classes"] = json.dumps(ckpt["classes"])
    meta.userDefined["turnip.window_frames"] = str(ckpt["window_frames"])
    meta.userDefined["turnip.window_stride"] = str(ckpt["window_stride"])
    files[spec_path] = spec.SerializeToString(deterministic=True)
    return files


def zip_files(files: dict[str, bytes]) -> bytes:
    """Stored zip of the package under PACKAGE_NAME/, byte-reproducible."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for rel in sorted(files):
            info = zipfile.ZipInfo(f"{PACKAGE_NAME}/{rel}", date_time=ZIP_DATE_TIME)
            info.external_attr = 0o644 << 16
            info.create_system = 3  # unix, whatever the host
            zf.writestr(info, files[rel])
    return buf.getvalue()


def export(model_path: Path) -> bytes:
    ckpt = load_model(model_path)
    return zip_files(package_files(build_mlmodel(ckpt), ckpt))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, type=Path, help="model.npz")
    p.add_argument("--out", required=True, type=Path, help=".mlpackage.zip path")
    args = p.parse_args(argv)
    try:
        data = export(args.model)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(args.out)
    except (ExportError, OSError) as e:
        print(f"export_coreml.py: error: {e}", file=sys.stderr)
        return 2
    print(f"exported {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
