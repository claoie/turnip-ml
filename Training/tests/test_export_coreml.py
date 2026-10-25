"""Unit tests for Training/export_coreml.py (turnip-ml#16).

Checkpoint validation and the manifest descriptors run everywhere. The
export itself needs coremltools (Training/requirements-export.txt, which
the training-smoke CI job installs) and is skipped without it; prediction
parity needs the Core ML runtime, so macOS only.

Run: python -m unittest discover -s Training/tests -t .
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import export_coreml
import tkp1
import train

REPO = Path(__file__).resolve().parent.parent.parent
SMOKE = REPO / "Training" / "fixtures" / "smoke"
SMOKE_CONFIG = REPO / "Training" / "config" / "smoke.toml"
HAVE_COREMLTOOLS = importlib.util.find_spec("coremltools") is not None


class SmokeModel(unittest.TestCase):
    tmp: Path
    model: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="export-coreml-test-"))
        with contextlib.redirect_stdout(io.StringIO()):
            train.run(SMOKE_CONFIG, str(cls.tmp / "run"))
        cls.model = cls.tmp / "run" / "model.npz"

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, True)

    def write_model(self, **overrides) -> Path:
        z = dict(np.load(self.model))
        z.update(overrides)
        path = self.tmp / "bad.npz"
        np.savez(path, **z)
        return path


class LoadModelTest(SmokeModel):
    def test_descriptors_describe_the_window_classifier(self) -> None:
        ckpt = export_coreml.load_model(self.model)
        input_desc, output_desc = export_coreml.descriptors(ckpt)
        self.assertEqual(input_desc["format"], "TKP1")
        self.assertEqual(input_desc["shape"], ["T", 52])
        self.assertEqual(input_desc["min_frames"], 8)
        self.assertEqual(input_desc["max_frames"], tkp1.MAX_FRAMES)
        self.assertEqual(
            input_desc["features"], ["x", "y", "confidence", "gap_indicator"]
        )
        self.assertEqual(output_desc["coordinates"], "source-frame")
        self.assertEqual(output_desc["kind"], "window_classifier")
        self.assertEqual(output_desc["classes"], ["background", "corkscrew", "gainer"])
        self.assertEqual(output_desc["shape"], ["N", 3])
        self.assertEqual(
            (output_desc["window_frames"], output_desc["window_stride"]), (8, 4)
        )
        # Contract §3.2.2: the baseline's embedding is on window_probs' grid.
        self.assertEqual(
            output_desc["embedding"],
            {
                "output_name": "window_embeddings",
                "shape": ["N", 68],
                "window_frames": 8,
                "window_stride": 4,
            },
        )
        json.dumps([input_desc, output_desc], allow_nan=False)  # JSONB-safe

    def test_malformed_checkpoints_are_refused(self) -> None:
        z = np.load(self.model)
        cases = {
            "wrong W shape": {"W": np.zeros((10, 3))},
            "non-finite weights": {"b": np.array([0.0, np.nan, 0.0])},
            "no background class": {"classes": np.array(["a", "b", "c"])},
            "zero feature std": {"feat_std": np.zeros_like(z["feat_std"])},
            "one-frame windows": {"window_frames": np.int64(1)},
        }
        for name, overrides in cases.items():
            with self.subTest(name), self.assertRaises(export_coreml.ExportError):
                export_coreml.load_model(self.write_model(**overrides))

    def test_conversion_errors_are_export_errors(self) -> None:
        # Whatever coremltools or a changed package layout raises is a
        # failed export (exit 2), never a traceback.
        for error in (ValueError("bad range"), KeyError("Manifest.json")):
            with (
                self.subTest(type(error).__name__),
                mock.patch.object(export_coreml, "build_mlmodel", return_value=None),
                mock.patch.object(export_coreml, "package_files", side_effect=error),
                self.assertRaisesRegex(export_coreml.ExportError, "conversion failed"),
            ):
                export_coreml.export(self.model)

    def test_missing_coremltools_fails_closed(self) -> None:
        if HAVE_COREMLTOOLS:
            self.skipTest("coremltools is installed")
        out = self.tmp / "model.mlpackage.zip"
        argv = ["--model", str(self.model), "--out", str(out)]
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(export_coreml.main(argv), 2)
        self.assertIn("requirements-export.txt", err.getvalue())
        self.assertFalse(out.exists())


@unittest.skipUnless(HAVE_COREMLTOOLS, "needs coremltools (requirements-export.txt)")
class ExportTest(SmokeModel):
    artifact: bytes

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        with contextlib.redirect_stderr(io.StringIO()):
            cls.artifact = export_coreml.export(cls.model)

    def test_byte_identical_reruns(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(export_coreml.export(self.model), self.artifact)

    def test_zip_layout(self) -> None:
        with zipfile.ZipFile(io.BytesIO(self.artifact)) as zf:
            infos = zf.infolist()
            names = [i.filename for i in infos]
            self.assertEqual(names, sorted(names))
            self.assertEqual(
                names,
                [
                    "TrickDetector.mlpackage/Data/com.apple.CoreML/model.mlmodel",
                    "TrickDetector.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
                    "TrickDetector.mlpackage/Manifest.json",
                ],
            )
            for info in infos:
                self.assertEqual(info.date_time, export_coreml.ZIP_DATE_TIME)
                self.assertEqual(info.compress_type, zipfile.ZIP_STORED)
            manifest = json.loads(zf.read("TrickDetector.mlpackage/Manifest.json"))
        root = manifest["itemInfoEntries"][manifest["rootModelIdentifier"]]
        self.assertEqual(root["path"], "com.apple.CoreML/model.mlmodel")

    def test_spec_is_fp16_with_contract_io_and_no_timestamp(self) -> None:
        from coremltools.proto import MIL_pb2, Model_pb2

        with zipfile.ZipFile(io.BytesIO(self.artifact)) as zf:
            raw = zf.read("TrickDetector.mlpackage/Data/com.apple.CoreML/model.mlmodel")
        spec = Model_pb2.Model()
        spec.ParseFromString(raw)
        meta = spec.description.metadata.userDefined
        self.assertNotIn(export_coreml.CONVERSION_DATE_KEY, meta)
        self.assertEqual(
            json.loads(meta["turnip.classes"]), ["background", "corkscrew", "gainer"]
        )
        self.assertEqual(meta["turnip.window_frames"], "8")
        [inp] = spec.description.input
        self.assertEqual(inp.name, "pose")
        self.assertEqual(
            inp.type.multiArrayType.dataType, Model_pb2.ArrayFeatureType.FLOAT32
        )
        lo, ch = inp.type.multiArrayType.shapeRange.sizeRanges
        self.assertEqual((lo.lowerBound, lo.upperBound), (8, tkp1.MAX_FRAMES))
        self.assertEqual((ch.lowerBound, ch.upperBound), (52, 52))
        self.assertEqual(
            [o.name for o in spec.description.output],
            ["window_probs", "window_embeddings"],
        )
        for out in spec.description.output:
            self.assertEqual(
                out.type.multiArrayType.dataType,
                Model_pb2.ArrayFeatureType.FLOAT32,
                out.name,
            )
        # fp16 up to the window features; the classifier head and the
        # embedding in fp32.
        dtypes = {
            out.name: out.type.tensorType.dataType
            for fn in spec.mlProgram.functions.values()
            for block in fn.block_specializations.values()
            for op in block.operations
            if op.type != "const"
            for out in op.outputs
        }
        fp16, fp32 = MIL_pb2.DataType.FLOAT16, MIL_pb2.DataType.FLOAT32
        self.assertEqual(dtypes["avg_pool_0_cast_fp16"], fp16)
        for name in export_coreml.FP32_OPS:
            self.assertEqual(dtypes[name], fp32, name)

    def test_windows_longer_than_the_default_length(self) -> None:
        from coremltools.proto import Model_pb2

        model = self.write_model(window_frames=np.int64(700))
        with contextlib.redirect_stderr(io.StringIO()):
            artifact = export_coreml.export(model)
        with zipfile.ZipFile(io.BytesIO(artifact)) as zf:
            raw = zf.read("TrickDetector.mlpackage/Data/com.apple.CoreML/model.mlmodel")
        spec = Model_pb2.Model()
        spec.ParseFromString(raw)
        [inp] = spec.description.input
        lo = inp.type.multiArrayType.shapeRange.sizeRanges[0]
        self.assertEqual(lo.lowerBound, 700)
        self.assertEqual(inp.type.multiArrayType.shape[0], 700)

    @unittest.skipUnless(sys.platform == "darwin", "Core ML runs on macOS only")
    def test_tiny_feature_std_stays_finite(self) -> None:
        # 0.1 / 1e-6 overflows fp16 (65504): the fp32 head must not.
        import coremltools as ct

        std = np.load(self.model)["feat_std"].copy()
        std[0] = 1e-6
        model = self.write_model(feat_std=std)
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stderr(io.StringIO()):
                artifact = export_coreml.export(model)
            with zipfile.ZipFile(io.BytesIO(artifact)) as zf:
                zf.extractall(tmp)
            mlmodel = ct.models.MLModel(str(Path(tmp) / export_coreml.PACKAGE_NAME))
            pose = np.full((32, 52), 0.9, dtype=np.float32)
            got = mlmodel.predict({"pose": pose})
        self.assertTrue(np.all(np.isfinite(got["window_probs"])))
        self.assertTrue(np.all(np.isfinite(got["window_embeddings"])))
        np.testing.assert_allclose(got["window_probs"].sum(axis=1), 1.0, atol=1e-3)

    @unittest.skipUnless(sys.platform == "darwin", "Core ML runs on macOS only")
    def test_predictions_match_the_numpy_model(self) -> None:
        import coremltools as ct

        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(io.BytesIO(self.artifact)) as zf:
                zf.extractall(tmp)
            mlmodel = ct.models.MLModel(str(Path(tmp) / export_coreml.PACKAGE_NAME))
            ckpt = export_coreml.load_model(self.model)
            W, S = ckpt["window_frames"], ckpt["window_stride"]
            for path in sorted((SMOKE / "samples").glob("*.json")):
                kps = np.asarray(json.loads(path.read_text())["keypoints"])
                n = kps.shape[0]
                feats = np.stack(
                    [
                        train.featurize_window(np, kps[s : s + W])
                        for s in range(0, n - W + 1, S)
                    ]
                )
                logits = (feats - ckpt["feat_mean"]) / ckpt["feat_std"] @ ckpt["W"]
                want = train._softmax(np, logits + ckpt["b"])
                gap = (kps[:, :, 2].sum(axis=1) == 0)[:, None]
                pose = np.concatenate([kps.reshape(n, 51), gap], axis=1)
                out = mlmodel.predict({"pose": pose.astype(np.float32)})
                got, emb = out["window_probs"], out["window_embeddings"]
                with self.subTest(path.name):
                    self.assertEqual(got.shape, want.shape)
                    np.testing.assert_allclose(got, want, atol=5e-3)
                    np.testing.assert_array_equal(got.argmax(1), want.argmax(1))
                    # The features are fp16 and 1 / feat_std scales their
                    # error up: compare them unstandardized.
                    self.assertEqual(emb.shape, feats.shape)
                    np.testing.assert_allclose(
                        emb * ckpt["feat_std"] + ckpt["feat_mean"], feats, atol=2e-3
                    )
                    # The embedding is exactly what the classifier head reads.
                    np.testing.assert_allclose(
                        train._softmax(np, emb @ ckpt["W"] + ckpt["b"]), got, atol=1e-5
                    )


if __name__ == "__main__":
    unittest.main()
