#!/usr/bin/env python3
"""Release checks: no training data ships inside a model artifact.

TRAINING_DESIGN.md §6.2. Every device receives the published model, so
the artifact must carry the model and nothing learned from individual
examples alongside it: no pose sequences, and no per-example
embeddings (`BACKEND_DESIGN.md` §8.1). Training/publish_model.py runs
these checks on the exported artifact before anything is uploaded, so a
failure publishes nothing.

The checks are allow-lists, not searches for known shapes: whatever is
not one of the model's own parameters fails, so a pose sequence or an
embedding table fails however it is stored.

Inputs
------
--model     model.npz from Training/train.py.
--artifact  the .mlpackage zip Training/export_coreml.py wrote from it.

What is checked
---------------
1. The checkpoint holds only CHECKPOINT_KEYS, each at the shape the
   export reads (W [68, C], b [C], classes [C], feat_mean and feat_std
   [68], scalar window parameters). There is no slot for examples.
2. The zip holds exactly the package's model spec, its weight file and
   its manifest (PACKAGE_FILES), and the manifest lists only those
   items.
3. Every value in the Core ML program, constants and operation
   arguments alike, is either
   - a floating-point tensor equal, at the precision it is stored in,
     to one of the model's parameters: feat_mean, 1 / feat_std, W
     (transposed) or b;
   - a floating-point scalar (rank 0): the feature math's own constants,
     such as 0.5 for a midpoint; or
   - a non-floating tensor (shapes, axes, flags, dtype names) of at most
     MAX_SMALL_VALUE elements, each string at most MAX_STRING_LENGTH
     characters.
   A pose window is at least window_frames x 26 values and an embedding
   52, so neither fits the last two rules, and neither equals a
   parameter.
4. The weight file holds only blobs the program references, with no
   bytes after the last one beyond alignment padding.
5. The model's metadata carries only METADATA_KEYS (and the program
   only coremltools' buildInfo, a few short strings), the turnip.* values
   match the checkpoint, and its inputs and outputs are the contract's
   (pose -> window_probs, window_embeddings). `window_embeddings` is a
   function of the device's own input, computed at run time; it stores
   nothing.

Exit codes: 0 = passed, 2 = failed or input error (fail closed). Stdout
and stderr name the failing check and counts, never values.

Needs coremltools (its protobuf definitions), like the export.
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

import export_coreml
import numpy as np

CHECKPOINT_KEYS = frozenset(
    {
        "W",
        "b",
        "classes",
        "feat_mean",
        "feat_std",
        "window_frames",
        "window_stride",
        "background_margin_frames",
    }
)
# Checkpoints from before the §2.5 background margin lack it.
OPTIONAL_CHECKPOINT_KEYS = frozenset({"background_margin_frames"})
SPEC_PATH = "Data/com.apple.CoreML/model.mlmodel"
WEIGHTS_PATH = "Data/com.apple.CoreML/weights/weight.bin"
PACKAGE_FILES = frozenset({"Manifest.json", SPEC_PATH, WEIGHTS_PATH})
METADATA_KEYS = frozenset(
    {
        "com.github.apple.coremltools.source",
        "com.github.apple.coremltools.version",
        "turnip.classes",
        "turnip.window_frames",
        "turnip.window_stride",
    }
)
INPUT_NAMES = (export_coreml.INPUT_NAME,)
OUTPUT_NAMES = (export_coreml.OUTPUT_NAME, export_coreml.EMBEDDING_NAME)
MAX_SMALL_VALUE = 8
MAX_STRING_LENGTH = 64
# Core ML weight file (MIL blob storage v2): a 64-byte file header, then
# per blob a 64-byte metadata record and its data, each 64-byte aligned.
BLOB_ALIGNMENT = 64
BLOB_SENTINEL = 0xDEADBEEF
BLOB_DTYPES = {1: np.float16, 2: np.float32, 3: np.uint8, 4: np.int8}
# MIL proto DataType values of floating-point tensors.
FLOAT_DTYPES = {10: np.float16, 11: np.float32, 12: np.float64}
FP16_RTOL = 1e-3


class ReleaseCheckError(Exception):
    """The artifact fails a release check (or cannot be read)."""


def check_checkpoint(raw: dict[str, Any]) -> None:
    """Check 1: the checkpoint has no key or shape beyond the model's."""
    extra = set(raw) - CHECKPOINT_KEYS
    if extra:
        raise ReleaseCheckError(
            f"checkpoint holds {len(extra)} key(s) outside the model's parameters"
        )
    missing = CHECKPOINT_KEYS - OPTIONAL_CHECKPOINT_KEYS - set(raw)
    if missing:
        raise ReleaseCheckError(f"checkpoint misses {len(missing)} key(s)")
    n_classes = len(raw["classes"])
    n_features = export_coreml.N_FEATURES
    shapes = {
        "W": (n_features, n_classes),
        "b": (n_classes,),
        "classes": (n_classes,),
        "feat_mean": (n_features,),
        "feat_std": (n_features,),
        "window_frames": (),
        "window_stride": (),
        "background_margin_frames": (),
    }
    for key, value in raw.items():
        if np.shape(value) != shapes[key]:
            raise ReleaseCheckError(f"checkpoint {key} is not shaped {shapes[key]}")


def _zip_files(artifact: bytes) -> dict[str, bytes]:
    """Check 2: exactly the package's files, by relative path."""
    prefix = export_coreml.PACKAGE_NAME + "/"
    try:
        with zipfile.ZipFile(io.BytesIO(artifact)) as zf:
            names = zf.namelist()
            if len(set(names)) != len(names) or not all(
                n.startswith(prefix) for n in names
            ):
                raise ReleaseCheckError("artifact zip entries are not one package")
            files = {n[len(prefix) :]: zf.read(n) for n in names}
    except (zipfile.BadZipFile, OSError) as e:
        raise ReleaseCheckError(f"artifact is not a readable zip: {e}") from e
    if set(files) != PACKAGE_FILES:
        raise ReleaseCheckError(
            f"artifact holds {len(set(files) - PACKAGE_FILES)} unexpected and "
            f"misses {len(PACKAGE_FILES - set(files))} expected file(s)"
        )
    try:
        manifest = json.loads(files["Manifest.json"])
        paths = {e["path"] for e in manifest["itemInfoEntries"].values()}
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise ReleaseCheckError("artifact Manifest.json is malformed") from e
    expected = {"com.apple.CoreML/model.mlmodel", "com.apple.CoreML/weights"}
    if paths != expected:
        raise ReleaseCheckError("artifact Manifest.json lists unexpected items")
    return files


def _blobs(weights: bytes) -> dict[int, np.ndarray]:
    """The weight file's blobs by metadata offset (check 4's walk)."""
    blobs: dict[int, np.ndarray] = {}
    if len(weights) < BLOB_ALIGNMENT:
        raise ReleaseCheckError("weight file is truncated")
    (count,) = struct.unpack_from("<I", weights, 0)
    offset = BLOB_ALIGNMENT
    for _ in range(count):
        if offset + BLOB_ALIGNMENT > len(weights):
            raise ReleaseCheckError("weight file is truncated")
        sentinel, dtype, size, data_offset = struct.unpack_from(
            "<IIQQ", weights, offset
        )
        if sentinel != BLOB_SENTINEL or dtype not in BLOB_DTYPES:
            raise ReleaseCheckError("weight file blob header is malformed")
        if data_offset < offset + BLOB_ALIGNMENT or data_offset + size > len(weights):
            raise ReleaseCheckError("weight file blob lies outside the file")
        blobs[offset] = np.frombuffer(
            weights[data_offset : data_offset + size], dtype=BLOB_DTYPES[dtype]
        )
        end = data_offset + size
        offset = -(-end // BLOB_ALIGNMENT) * BLOB_ALIGNMENT
    if len(weights) > offset:
        raise ReleaseCheckError(
            f"weight file has {len(weights) - offset} bytes after its last blob"
        )
    return blobs


def _parameters(raw: dict[str, Any]) -> list[np.ndarray]:
    """The float tensors the export may store, as it stores them."""
    feat_std = np.asarray(raw["feat_std"], dtype=np.float64)
    return [
        np.asarray(raw["feat_mean"], dtype=np.float64),
        1.0 / feat_std,
        np.asarray(raw["W"], dtype=np.float64).T,
        np.asarray(raw["b"], dtype=np.float64),
    ]


def _is_parameter(values: np.ndarray, dtype: Any, params: list[np.ndarray]) -> bool:
    for p in params:
        if values.shape != p.shape:
            continue
        expected = p.astype(dtype).astype(np.float64)
        got = values.astype(np.float64)
        if dtype == np.float16:
            if np.allclose(got, expected, rtol=FP16_RTOL, atol=0.0):
                return True
        elif np.array_equal(got, expected):
            return True
    return False


def _immediate(value: Any, dtype: int, dims: list[int]) -> np.ndarray:
    """An immediate tensor value as an array (shape dims)."""
    tensor = value.immediateValue.tensor
    kind = tensor.WhichOneof("value")
    if kind is None:
        return np.zeros(dims)
    field = getattr(tensor, kind)
    if kind == "bytes":
        if dtype not in FLOAT_DTYPES:
            return np.frombuffer(field.values, dtype=np.uint8)
        return np.frombuffer(field.values, dtype=FLOAT_DTYPES[dtype]).reshape(dims)
    if kind == "strings":
        if any(len(s) > MAX_STRING_LENGTH for s in field.values):
            raise ReleaseCheckError("program holds a string over the length limit")
        return np.array(list(field.values), dtype=object)
    return np.asarray(list(field.values)).reshape(dims)


def _check_value(
    value: Any, blobs: dict[int, np.ndarray], used: set[int], params: list[np.ndarray]
) -> None:
    """Check 3 for one MIL value."""
    vtype = value.type
    if vtype.WhichOneof("type") != "tensorType":
        raise ReleaseCheckError("program holds a non-tensor value")
    tensor_type = vtype.tensorType
    dtype = tensor_type.dataType
    dims = [d.constant.size for d in tensor_type.dimensions]
    kind = value.WhichOneof("value")
    if kind == "blobFileValue":
        blob = value.blobFileValue
        if (
            blob.fileName != "@model_path/weights/weight.bin"
            or blob.offset not in blobs
        ):
            raise ReleaseCheckError("program references a blob the weight file lacks")
        used.add(blob.offset)
        values = blobs[blob.offset]
        if values.size != int(np.prod(dims)):
            raise ReleaseCheckError("program blob size disagrees with its shape")
        values = values.reshape(dims)
    elif kind == "immediateValue":
        try:
            values = _immediate(value, dtype, dims)
        except ValueError as e:
            raise ReleaseCheckError("program value disagrees with its shape") from e
    else:
        raise ReleaseCheckError("program holds a value of unknown storage")
    if dtype in FLOAT_DTYPES:
        if dims and not _is_parameter(values, FLOAT_DTYPES[dtype], params):
            raise ReleaseCheckError(
                f"program holds a float tensor of {values.size} values that is "
                "none of the model's parameters"
            )
    elif values.size > MAX_SMALL_VALUE:
        raise ReleaseCheckError(
            f"program holds a non-float tensor of {values.size} values "
            f"(at most {MAX_SMALL_VALUE})"
        )


def _check_program_attributes(attributes: Any) -> None:
    """The program's only attribute is coremltools' buildInfo: a few short
    strings naming the converter."""
    if set(attributes) - {"buildInfo"}:
        raise ReleaseCheckError("program carries unexpected attributes")
    if "buildInfo" not in attributes:
        return
    info = attributes["buildInfo"]
    if info.WhichOneof("value") != "immediateValue":
        raise ReleaseCheckError("program buildInfo is not an immediate value")
    entries = info.immediateValue.dictionary.values
    if len(entries) > MAX_SMALL_VALUE:
        raise ReleaseCheckError("program buildInfo has too many entries")
    for entry in entries:
        for part in (entry.key, entry.value):
            strings = part.immediateValue.tensor.strings.values
            if len(strings) != 1 or len(strings[0]) > MAX_STRING_LENGTH:
                raise ReleaseCheckError("program buildInfo holds an unexpected value")


def _walk_block(
    block: Any, blobs: dict[int, np.ndarray], used: set[int], params: list[np.ndarray]
) -> None:
    for op in block.operations:
        for attr in op.attributes.values():
            _check_value(attr, blobs, used, params)
        for arg in op.inputs.values():
            for binding in arg.arguments:
                if binding.WhichOneof("binding") == "value":
                    _check_value(binding.value, blobs, used, params)
        for inner in op.blocks:
            _walk_block(inner, blobs, used, params)


def _check_description(spec: Any, raw: dict[str, Any]) -> None:
    """Check 5: metadata and the contract's inputs and outputs only."""
    meta = spec.description.metadata
    keys = set(meta.userDefined)
    if keys != METADATA_KEYS:
        raise ReleaseCheckError("model metadata keys are not the expected set")
    if any(
        len(v) > 4 * MAX_STRING_LENGTH
        for k, v in meta.userDefined.items()
        if not k.startswith("turnip.")
    ):
        raise ReleaseCheckError("model metadata value over the length limit")
    classes = [str(c) for c in raw["classes"]]
    if (
        meta.userDefined["turnip.classes"] != json.dumps(classes)
        or meta.userDefined["turnip.window_frames"] != str(int(raw["window_frames"]))
        or meta.userDefined["turnip.window_stride"] != str(int(raw["window_stride"]))
    ):
        raise ReleaseCheckError("model metadata disagrees with the checkpoint")
    if meta.author or meta.license or meta.versionString:
        raise ReleaseCheckError("model metadata carries unexpected text")
    inputs = tuple(f.name for f in spec.description.input)
    outputs = tuple(f.name for f in spec.description.output)
    if inputs != INPUT_NAMES or outputs != OUTPUT_NAMES:
        raise ReleaseCheckError("model inputs or outputs are not the contract's")
    if any(
        f.shortDescription for f in (*spec.description.input, *spec.description.output)
    ):
        raise ReleaseCheckError("model input or output carries a description")


def check(model_path: Path, artifact: bytes) -> None:
    """Run every check; raises ReleaseCheckError on the first failure."""
    try:
        with np.load(model_path, allow_pickle=False) as z:
            raw = {k: z[k] for k in z.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as e:
        raise ReleaseCheckError(f"cannot read checkpoint {model_path}: {e}") from e
    check_checkpoint(raw)
    files = _zip_files(artifact)
    try:
        from coremltools.proto import Model_pb2
    except ImportError as e:
        raise ReleaseCheckError(
            "release checks need coremltools: "
            "pip install -r Training/requirements-export.txt"
        ) from e
    spec = Model_pb2.Model()
    try:
        spec.ParseFromString(files[SPEC_PATH])
    except Exception as e:  # protobuf's DecodeError has no stable import path
        raise ReleaseCheckError("artifact model spec is not readable") from e
    if spec.WhichOneof("Type") != "mlProgram":
        raise ReleaseCheckError("artifact is not an ML Program")
    _check_description(spec, raw)
    blobs = _blobs(files[WEIGHTS_PATH])
    used: set[int] = set()
    params = _parameters(raw)
    _check_program_attributes(spec.mlProgram.attributes)
    for function in spec.mlProgram.functions.values():
        for attr in function.attributes.values():
            _check_value(attr, blobs, used, params)
        for block in function.block_specializations.values():
            for attr in block.attributes.values():
                _check_value(attr, blobs, used, params)
            _walk_block(block, blobs, used, params)
    if used != set(blobs):
        raise ReleaseCheckError(
            f"weight file holds {len(set(blobs) - used)} blob(s) the program "
            "never reads"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, type=Path, help="model.npz")
    p.add_argument("--artifact", required=True, type=Path, help=".mlpackage.zip")
    args = p.parse_args(argv)
    try:
        check(args.model, args.artifact.read_bytes())
    except (ReleaseCheckError, OSError) as e:
        print(f"release_checks.py: error: {e}", file=sys.stderr)
        return 2
    print("release checks: passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
