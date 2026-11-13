"""Unit tests for Training/release_checks.py (TRAINING_DESIGN.md §6.2).

Each failure case tampers with a real export the way a pose sequence or
an embedding table could end up in it. The artifact cases need
coremltools (Training/requirements-export.txt) and are skipped without
it; the checkpoint cases run everywhere.

Run: python -m unittest discover -s Training/tests -t .
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import shutil
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import export_coreml
import release_checks as rc

HAVE_COREMLTOOLS = importlib.util.find_spec("coremltools") is not None
CLASSES = ["background", "cork", "gainer"]
N_FEATURES = export_coreml.N_FEATURES


def write_model(path: Path, **extra: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    np.savez(
        path,
        W=rng.standard_normal((N_FEATURES, len(CLASSES))),
        b=rng.standard_normal(len(CLASSES)),
        classes=np.array(CLASSES),
        feat_mean=rng.standard_normal(N_FEATURES),
        feat_std=rng.uniform(0.5, 2.0, N_FEATURES),
        window_frames=np.int64(8),
        window_stride=np.int64(4),
        background_margin_frames=np.int64(20),
        **extra,
    )


def unzip(artifact: bytes) -> dict[str, bytes]:
    prefix = export_coreml.PACKAGE_NAME + "/"
    with zipfile.ZipFile(io.BytesIO(artifact)) as zf:
        return {n[len(prefix) :]: zf.read(n) for n in zf.namelist()}


class CheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="release-checks-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def raw(self, **extra: np.ndarray) -> dict:
        path = self.tmp / "model.npz"
        write_model(path, **extra)
        with np.load(path) as z:
            return {k: z[k] for k in z.files}

    def test_the_model_s_own_keys_pass(self) -> None:
        rc.check_checkpoint(self.raw())

    def test_an_extra_key_fails(self) -> None:
        for name, value in (
            ("train_embeddings", np.zeros((10, N_FEATURES))),
            ("pose_window", np.zeros((8, 17, 3))),
        ):
            with self.subTest(name), self.assertRaises(rc.ReleaseCheckError) as cm:
                rc.check_checkpoint(self.raw(**{name: value}))
            self.assertIn("outside the model's parameters", str(cm.exception))

    def test_a_missing_key_fails(self) -> None:
        raw = self.raw()
        del raw["feat_std"]
        with self.assertRaises(rc.ReleaseCheckError):
            rc.check_checkpoint(raw)
        # The margin is optional: checkpoints from before it lack it.
        raw = self.raw()
        del raw["background_margin_frames"]
        rc.check_checkpoint(raw)

    def test_a_parameter_of_another_shape_fails(self) -> None:
        raw = self.raw()
        raw["feat_mean"] = np.zeros((2, N_FEATURES))
        with self.assertRaises(rc.ReleaseCheckError):
            rc.check_checkpoint(raw)


@unittest.skipUnless(HAVE_COREMLTOOLS, "needs coremltools")
class ArtifactTest(unittest.TestCase):
    tmp: Path
    model: Path
    artifact: bytes

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="release-checks-"))
        cls.model = cls.tmp / "model.npz"
        write_model(cls.model)
        cls.artifact = export_coreml.export(cls.model)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, True)

    def assertFails(self, files: dict[str, bytes], message: str) -> None:
        with self.assertRaises(rc.ReleaseCheckError) as cm:
            rc.check(self.model, export_coreml.zip_files(files))
        self.assertIn(message, str(cm.exception))

    def spec(self, files: dict[str, bytes]):
        from coremltools.proto import Model_pb2

        spec = Model_pb2.Model()
        spec.ParseFromString(files[rc.SPEC_PATH])
        return spec

    def const(self, spec, name: str):
        for block in spec.mlProgram.functions["main"].block_specializations.values():
            for op in block.operations:
                if op.type == "const" and op.outputs[0].name == name:
                    return op.attributes["val"]
        raise AssertionError(f"no const {name}")

    def test_the_export_passes(self) -> None:
        rc.check(self.model, self.artifact)
        # zip_files of the unpacked artifact is the artifact: the tampering
        # below changes only what each test changes.
        self.assertEqual(export_coreml.zip_files(unzip(self.artifact)), self.artifact)

    def test_an_extra_file_fails(self) -> None:
        files = unzip(self.artifact)
        files["Data/com.apple.CoreML/train_windows.json"] = b"[]"
        self.assertFails(files, "1 unexpected")

    def test_a_stored_tensor_that_is_no_parameter_fails(self) -> None:
        # Overwrite feat_mean's blob with an embedding: same shape, other
        # values.
        files = unzip(self.artifact)
        spec = self.spec(files)
        offset = self.const(spec, "standardize_center_y_0").blobFileValue.offset
        weights = bytearray(files[rc.WEIGHTS_PATH])
        _, _, size, data_offset = struct.unpack_from("<IIQQ", weights, offset)
        embedding = np.arange(N_FEATURES, dtype=np.float32) / 7
        weights[data_offset : data_offset + size] = embedding.tobytes()
        files[rc.WEIGHTS_PATH] = bytes(weights)
        self.assertFails(files, "none of the model's parameters")

    def test_an_immediate_float_that_is_no_parameter_fails(self) -> None:
        files = unzip(self.artifact)
        spec = self.spec(files)
        bias = self.const(spec, "logits_bias_0").immediateValue.tensor.floats
        shifted = [v + 1.0 for v in bias.values]
        del bias.values[:]
        bias.values.extend(shifted)
        files[rc.SPEC_PATH] = spec.SerializeToString(deterministic=True)
        self.assertFails(files, "none of the model's parameters")

    def test_a_large_non_float_tensor_fails(self) -> None:
        # A pose window quantized to ints in an op argument.
        files = unzip(self.artifact)
        spec = self.spec(files)
        shape = self.const(spec, "reshape_0_shape_0")
        shape.immediateValue.tensor.ints.values.extend(range(100))
        shape.type.tensorType.dimensions[0].constant.size = 103
        files[rc.SPEC_PATH] = spec.SerializeToString(deterministic=True)
        self.assertFails(files, "non-float tensor of 103 values")

    def test_an_unreferenced_blob_fails(self) -> None:
        files = unzip(self.artifact)
        weights = bytearray(files[rc.WEIGHTS_PATH])
        (count,) = struct.unpack_from("<I", weights, 0)
        struct.pack_into("<I", weights, 0, count + 1)
        weights += b"\0" * (-len(weights) % rc.BLOB_ALIGNMENT)
        offset = len(weights)
        data = np.zeros(N_FEATURES, dtype=np.float32).tobytes()
        header = struct.pack(
            "<IIQQ", rc.BLOB_SENTINEL, 2, len(data), offset + rc.BLOB_ALIGNMENT
        )
        weights += header.ljust(rc.BLOB_ALIGNMENT, b"\0") + data
        files[rc.WEIGHTS_PATH] = bytes(weights)
        self.assertFails(files, "1 blob(s) the program never reads")

    def test_bytes_after_the_last_blob_fail(self) -> None:
        files = unzip(self.artifact)
        files[rc.WEIGHTS_PATH] += b"\0" * 128
        self.assertFails(files, "bytes after its last blob")

    def test_an_extra_metadata_key_fails(self) -> None:
        files = unzip(self.artifact)
        spec = self.spec(files)
        spec.description.metadata.userDefined["turnip.examples"] = "[]"
        files[rc.SPEC_PATH] = spec.SerializeToString(deterministic=True)
        self.assertFails(files, "metadata keys")

    def test_metadata_must_match_the_checkpoint(self) -> None:
        files = unzip(self.artifact)
        spec = self.spec(files)
        spec.description.metadata.userDefined["turnip.classes"] = json.dumps(
            ["background"]
        )
        files[rc.SPEC_PATH] = spec.SerializeToString(deterministic=True)
        self.assertFails(files, "disagrees with the checkpoint")

    def test_main_exit_codes(self) -> None:
        good = self.tmp / "good.zip"
        good.write_bytes(self.artifact)
        bad = self.tmp / "bad.zip"
        files = unzip(self.artifact)
        files["extra.bin"] = b"x"
        bad.write_bytes(export_coreml.zip_files(files))
        base = ["--model", str(self.model), "--artifact"]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(rc.main([*base, str(good)]), 0)
        self.assertEqual(out.getvalue(), "release checks: passed\n")
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(rc.main([*base, str(bad)]), 2)
        self.assertIn("1 unexpected", err.getvalue())


if __name__ == "__main__":
    unittest.main()
