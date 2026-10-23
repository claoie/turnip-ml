"""Unit tests for Training/publish_model.py (turnip-ml#16).

Export -> upload -> publish against a fake farm and a fake R2, on a run
the real chain produced (train -> predict -> evaluate -> name metrics) from
the smoke fixture under farm-shaped (64-hex) source ids. The export is the
real Core ML one when coremltools is installed (the training-smoke CI job
installs it), else a stand-in.

Run: python -m unittest discover -s Training/tests -t .
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import evaluate
import export_coreml
import name_metrics
import predict
import publish_model
import train

REPO = Path(__file__).resolve().parent.parent.parent
SMOKE = REPO / "Training" / "fixtures" / "smoke"
SMOKE_CONFIG = REPO / "Training" / "config" / "smoke.toml"
HAVE_COREMLTOOLS = importlib.util.find_spec("coremltools") is not None

FARM = "https://farm.example"
KEY = "pipeline-secret-for-tests"
R2_ENV = {
    "R2_ACCOUNT_ID": "account",
    "R2_ACCESS_KEY_ID": "key-id",
    "R2_SECRET_ACCESS_KEY": "secret",
    "R2_BUCKET": "bucket",
}
HEX64 = re.compile(r"[0-9a-f]{64}")


def source_id(sample: str) -> str:
    return hashlib.sha256(sample.encode()).hexdigest()


def stand_in_export(model_path: Path) -> bytes:
    """Deterministic artifact bytes when coremltools is not installed."""
    export_coreml.load_model(model_path)
    return b"PK-stand-in-" + hashlib.sha256(model_path.read_bytes()).digest()


EXPORT = export_coreml.export if HAVE_COREMLTOOLS else stand_in_export


# -- fakes -----------------------------------------------------------------------


def farm_validation_error(body: Any) -> str | None:
    """turnip-farm src/models.ts validatePublishInput, rule for rule.

    JSON null is neither `undefined` nor a string there, so a null optional
    field is rejected like any other wrong type.
    """
    if not isinstance(body, dict):
        return "body must be a JSON object"

    def present(key: str) -> bool:
        return key in body

    def is_int(v: Any) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    def optional_str(key: str, max_len: int) -> bool:
        return not present(key) or (
            isinstance(body[key], str) and len(body[key]) <= max_len
        )

    version = body.get("version")
    if not isinstance(version, str) or not version.strip() or len(version) > 200:
        return "version must be a non-empty string"
    tv: Any = body.get("taxonomy_version")
    if not is_int(tv) or tv < 1:
        return "taxonomy_version must be a positive integer"
    r2_key = body.get("r2_key")
    if not isinstance(r2_key, str) or not r2_key or len(r2_key) > 1024:
        return "r2_key must be a non-empty string"
    if not isinstance(body.get("sha256"), str) or not HEX64.fullmatch(body["sha256"]):
        return "sha256 must be a 64-char lowercase hex digest"
    if present("byte_size") and (
        not is_int(body["byte_size"]) or body["byte_size"] < 0
    ):
        return "byte_size must be a non-negative integer when present"
    if not optional_str("min_client_version", 50):
        return "min_client_version must be a string when present"
    for key in ("input_desc", "output_desc", "val_metrics", "name_metrics"):
        if present(key) and not isinstance(body[key], dict):
            return f"{key} must be a JSON object when present"
    if not optional_str("holdout_report_r2_key", 1024):
        return "holdout_report_r2_key must be a string when present"
    if not optional_str("training_run_id", 200):
        return "training_run_id must be a string when present"
    if present("training_source_ids"):
        ids = body["training_source_ids"]
        if not isinstance(ids, list) or len(ids) > 100000:
            return "training_source_ids must be an array"
        if not all(isinstance(i, str) and HEX64.fullmatch(i) for i in ids):
            return "training_source_ids must be 64-char lowercase hex source ids"
    return None


# The keys validatePublishInput reads; the farm ignores anything else, so
# sending another key would be a silent drift.
FARM_KEYS = {
    "version",
    "taxonomy_version",
    "r2_key",
    "sha256",
    "byte_size",
    "min_client_version",
    "input_desc",
    "output_desc",
    "holdout_report_r2_key",
    "training_run_id",
    "val_metrics",
    "name_metrics",
    "training_source_ids",
}


def error(status: int, code: str) -> tuple[int, Any]:
    return status, {"error": {"code": code, "message": f"farm says {code}"}}


class FakeFarm:
    """GET /api/models/current + POST /api/models, as turnip-farm serves them."""

    def __init__(self, current: str | None = None, published: tuple[str, ...] = ()):
        self.current: dict[str, Any] | None = None
        if current:
            self.current = {
                "version": current,
                "sha256": "0" * 64,
                "r2_key": f"models/{current}.mlpackage.zip",
                "holdout_report_r2_key": f"reports/{current}-holdout.json",
                "promoted_at": "2026-10-01T03:40:00.000Z",
            }
        self.published = set(published) | ({current} if current else set())
        self.posts: list[dict[str, Any]] = []
        self.fail_next_post: int | None = None
        # Insert the model, then lose the answer (a timeout after commit).
        self.drop_next_answer = False
        self.calls = 0

    def transport(self, method, url, headers, body):
        self.calls += 1
        if method == "GET" and url == FARM + "/api/models/current":
            assert "Authorization" not in headers  # public endpoint
            if self.current is None:
                return error(404, "NO_MODEL")
            return 200, {**self.current, "model_type": "trick-detection"}
        if method == "POST" and url == FARM + "/api/models":
            if headers.get("Authorization") != f"Bearer {KEY}":
                return error(401, "UNAUTHENTICATED")
            try:
                payload = json.loads(body)
            except ValueError:
                return error(422, "VALIDATION_ERROR")
            if self.fail_next_post is not None:
                status, self.fail_next_post = self.fail_next_post, None
                return error(status, "INTERNAL")
            if set(payload) - FARM_KEYS:
                raise AssertionError(f"unknown keys sent: {set(payload) - FARM_KEYS}")
            if farm_validation_error(payload):
                return error(422, "VALIDATION_ERROR")
            if payload["version"] in self.published:
                return error(409, "VERSION_EXISTS")
            self.published.add(payload["version"])
            self.current = {
                key: payload[key]
                for key in ("version", "sha256", "r2_key", "holdout_report_r2_key")
            } | {"promoted_at": "2026-10-19T03:40:00.000Z"}
            self.posts.append(payload)
            if self.drop_next_answer:
                self.drop_next_answer = False
                raise publish_model.PublishError("farm request failed: TimeoutError")
            return 201, {
                "id": "00000000-0000-4000-8000-000000000001",
                "version": payload["version"],
                "promoted_at": "2026-10-19T03:40:00.000Z",
                "training_sources_recorded": len(set(payload["training_source_ids"])),
            }
        return error(404, "NOT_FOUND")


class FakeR2:
    def __init__(self, corrupt: bool = False):
        self.objects: dict[str, bytes] = {}
        self.types: dict[str, str] = {}
        self.puts = 0
        self.corrupt = corrupt

    def get(self, key: str) -> bytes | None:
        return self.objects.get(key)

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self.puts += 1
        self.objects[key] = data[:-1] if self.corrupt else data
        self.types[key] = content_type


# -- the run ---------------------------------------------------------------------


class RunFixture:
    """data/dataset + data/run as pipeline.sh leaves them before publishing."""

    def __init__(self, root: Path):
        self.root = root
        self.dataset = root / "dataset"
        self.run_dir = root / "run"
        self.challenger = self.run_dir / "challenger"
        samples = self.dataset / "samples"
        samples.mkdir(parents=True)
        smoke_split = json.loads((SMOKE / "split.json").read_text())
        # source-c (smoke-007/008) to holdout, as the synthetic run pins it.
        smoke_split.update({"smoke-007": "holdout", "smoke-008": "holdout"})
        split, manifest = {}, []
        for name in sorted(smoke_split):
            sid = source_id(name)
            rec = json.loads((SMOKE / "samples" / f"{name}.json").read_text())
            rec["sample_id"] = sid
            (samples / f"{sid}.json").write_text(json.dumps(rec))
            split[sid] = smoke_split[name]
            manifest.append(json.dumps({"sample_id": sid, "source_id": sid}))
        self.train_ids = sorted(s for s, part in split.items() if part == "train")
        (self.dataset / "manifest.jsonl").write_text("\n".join(manifest) + "\n")
        (self.dataset / "dataset_manifest.json").write_text(
            json.dumps(
                {
                    "taxonomy_version": 3,
                    "canonical_names": ["cork", "corkscrew", "gainer"],
                    "samples": len(split),
                }
            )
        )
        self.run_dir.mkdir()
        self.split = self.run_dir / "split.json"
        self.split.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")

        config = root / "config.toml"
        lines = [
            ln
            for ln in SMOKE_CONFIG.read_text().splitlines()
            if not ln.startswith(("split_path", "samples_dir"))
        ]
        lines += [f'split_path = "{self.split}"', f'samples_dir = "{samples}"']
        config.write_text("\n".join(lines) + "\n")
        train.run(config, str(self.challenger))
        preds = self.challenger / "holdout_predictions.json"
        report = self.challenger / "holdout_report.json"
        common = ["--split", str(self.split), "--samples-dir", str(samples)]
        model = ["--model", str(self.challenger / "model.npz")]
        assert predict.main(model + common + ["--out", str(preds)]) == 0
        assert (
            evaluate.main(["--predictions", str(preds), *common, "--out", str(report)])
            == 0
        )
        out = ["--out", str(self.challenger / "name_metrics.json")]
        metrics = ["--metrics", str(self.challenger / "metrics.json")]
        split_arg = ["--split", str(self.split)]
        assert (
            name_metrics.main(["--report", str(report), *metrics, *split_arg, *out])
            == 0
        )


class PublishTestCase(unittest.TestCase):
    tmp_class: str
    fixture: RunFixture

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_class = tempfile.mkdtemp(prefix="publish-fixture-")
        with contextlib.redirect_stdout(io.StringIO()):
            cls.fixture = RunFixture(Path(cls.tmp_class))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp_class, True)

    def setUp(self) -> None:
        # A fresh copy per test: publishing writes into the run directory.
        self.root = Path(tempfile.mkdtemp(prefix="publish-test-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        shutil.copytree(self.fixture.root, self.root, dirs_exist_ok=True)
        self.run_dir = self.root / "run"
        self.dataset = self.root / "dataset"
        self.challenger = self.run_dir / "challenger"
        self.stdout = self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.stderr = self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        env = {"TURNIP_FARM_URL": FARM, "TURNIP_PIPELINE_KEY": KEY, **R2_ENV}
        self.enterContext(mock.patch.dict(os.environ, env, clear=False))
        for name in ("TRAINING_RUN_ID", "MIN_CLIENT_VERSION"):
            self.enterContext(mock.patch.dict(os.environ, {name: ""}))
        self.farm = FakeFarm()
        self.r2 = FakeR2()

    def main(self, *extra: str, farm: FakeFarm | None = None, r2=None) -> int:
        return publish_model.main(
            ["--run", str(self.run_dir), "--dataset", str(self.dataset), *extra],
            transport=(farm or self.farm).transport,
            store=r2 or self.r2,
            exporter=EXPORT,
        )

    def logs(self) -> str:
        return self.stdout.getvalue() + self.stderr.getvalue()


class PublishRoundTripTest(PublishTestCase):
    def test_first_publish_round_trip(self) -> None:
        self.assertEqual(self.main(), 0, self.stderr.getvalue())

        artifact = self.r2.objects["models/trick-v1.0.mlpackage.zip"]
        report = self.r2.objects["reports/trick-v1.0-holdout.json"]
        self.assertEqual(artifact, EXPORT(self.challenger / "model.npz"))
        self.assertEqual(report, (self.challenger / "holdout_report.json").read_bytes())
        self.assertEqual(
            self.r2.types["models/trick-v1.0.mlpackage.zip"], "application/zip"
        )

        [body] = self.farm.posts
        self.assertEqual(body["version"], "trick-v1.0")
        self.assertEqual(body["taxonomy_version"], 3)
        self.assertEqual(body["r2_key"], "models/trick-v1.0.mlpackage.zip")
        self.assertEqual(body["sha256"], hashlib.sha256(artifact).hexdigest())
        self.assertEqual(body["byte_size"], len(artifact))
        self.assertEqual(
            body["holdout_report_r2_key"], "reports/trick-v1.0-holdout.json"
        )
        self.assertEqual(body["training_source_ids"], self.fixture.train_ids)
        self.assertEqual(
            body["name_metrics"],
            json.loads((self.challenger / "name_metrics.json").read_text()),
        )
        ckpt = export_coreml.load_model(self.challenger / "model.npz")
        self.assertEqual(
            (body["input_desc"], body["output_desc"]), export_coreml.descriptors(ckpt)
        )
        self.assertEqual(set(body["val_metrics"]), {"val_accuracy", "n_val_windows"})
        # Unset optional fields are omitted, never sent as null.
        self.assertNotIn("training_run_id", body)
        self.assertNotIn("min_client_version", body)
        self.assertNotIn(None, body.values())

        # The registry record: the exact body, and the farm's answer.
        self.assertEqual(
            json.loads((self.challenger / "publish_request.json").read_text()), body
        )
        record = json.loads((self.challenger / "publish.json").read_text())
        self.assertEqual(record["version"], "trick-v1.0")
        self.assertEqual(
            record["farm"]["training_sources_recorded"], len(self.fixture.train_ids)
        )
        self.assertEqual(
            (self.challenger / publish_model.ARTIFACT_NAME).read_bytes(), artifact
        )

    def test_optional_fields_come_from_the_environment(self) -> None:
        env = {"TRAINING_RUN_ID": "38749450792-1", "MIN_CLIENT_VERSION": "0.2.0"}
        with mock.patch.dict(os.environ, env):
            self.assertEqual(self.main(), 0)
        self.assertEqual(self.farm.posts[0]["training_run_id"], "38749450792-1")
        self.assertEqual(self.farm.posts[0]["min_client_version"], "0.2.0")

    def test_logs_carry_no_ids_urls_or_keys(self) -> None:
        self.assertEqual(self.main(), 0)
        logs = self.logs()
        self.assertIn("trick-v1.0: published", logs)
        for secret in [FARM, KEY, *R2_ENV.values(), *self.fixture.train_ids]:
            self.assertNotIn(secret, logs)

    def test_artifact_and_request_are_deterministic(self) -> None:
        self.assertEqual(self.main(), 0)
        first = self.farm.posts[0]["sha256"]
        request = (self.challenger / "publish_request.json").read_bytes()
        farm, r2 = FakeFarm(), FakeR2()
        self.assertEqual(self.main(farm=farm, r2=r2), 0)
        self.assertEqual(farm.posts[0]["sha256"], first)
        self.assertEqual(
            (self.challenger / "publish_request.json").read_bytes(), request
        )


class RetryTest(PublishTestCase):
    def test_a_lost_answer_is_reconciled_from_the_current_model(self) -> None:
        self.farm.drop_next_answer = True
        self.assertEqual(self.main(), 0, self.stderr.getvalue())
        self.assertEqual(len(self.farm.posts), 1)
        record = json.loads((self.challenger / "publish.json").read_text())
        self.assertTrue(record["farm"]["reconciled"])
        self.assertEqual(record["version"], "trick-v1.0")

    def test_a_rerun_after_publishing_publishes_nothing_more(self) -> None:
        self.assertEqual(self.main(), 0)
        puts = self.r2.puts
        self.assertEqual(self.main(), 0)
        self.assertEqual(len(self.farm.posts), 1)  # no trick-v1.1 duplicate
        self.assertEqual(self.r2.puts, puts)
        self.assertIn("trick-v1.0: already published", self.stdout.getvalue())
        record = json.loads((self.challenger / "publish.json").read_text())
        self.assertTrue(record["farm"]["already_published"])

    def test_a_lost_answer_without_an_insert_stays_an_error(self) -> None:
        self.farm.fail_next_post = 503
        self.assertEqual(self.main(), 2)
        self.assertIn("HTTP 503", self.stderr.getvalue())
        self.assertIsNone(self.farm.current)


class VersionTest(PublishTestCase):
    def test_minor_bump_from_the_current_model(self) -> None:
        farm = FakeFarm(current="trick-v1.3")
        self.assertEqual(self.main(farm=farm), 0)
        self.assertEqual(farm.posts[0]["version"], "trick-v1.4")
        self.assertIn("models/trick-v1.4.mlpackage.zip", self.r2.objects)

    def test_minor_bump_after_a_rollback(self) -> None:
        farm = FakeFarm(current="trick-v1.3-rb1")
        self.assertEqual(self.main(farm=farm), 0)
        self.assertEqual(farm.posts[0]["version"], "trick-v1.4")

    def test_major_bump(self) -> None:
        farm = FakeFarm(current="trick-v1.3")
        self.assertEqual(self.main("--major", farm=farm), 0)
        self.assertEqual(farm.posts[0]["version"], "trick-v2.0")

    def test_explicit_version(self) -> None:
        farm = FakeFarm(current="trick-v1.3")
        self.assertEqual(self.main("--version", "trick-v1.9", farm=farm), 0)
        self.assertEqual(farm.posts[0]["version"], "trick-v1.9")

    def test_malformed_explicit_version_is_refused(self) -> None:
        self.assertEqual(self.main("--version", "v2"), 2)
        self.assertEqual(self.r2.objects, {})

    def test_unknown_current_version_is_refused(self) -> None:
        farm = FakeFarm(current="2026.09.10-1")
        self.assertEqual(self.main(farm=farm), 2)
        self.assertEqual(self.r2.objects, {})

    def test_version_exists_is_an_error(self) -> None:
        # trick-v1.4 exists but is not current (rolled back past).
        farm = FakeFarm(current="trick-v1.3", published=("trick-v1.4",))
        self.assertEqual(self.main(farm=farm), 2)
        self.assertIn("VERSION_EXISTS", self.stderr.getvalue())
        self.assertEqual(farm.posts, [])


class UploadTest(PublishTestCase):
    def test_upload_is_read_back_and_verified(self) -> None:
        r2 = FakeR2(corrupt=True)
        self.assertEqual(self.main(r2=r2), 2)
        self.assertIn("does not match its sha256", self.stderr.getvalue())
        self.assertEqual(self.farm.posts, [])

    def test_other_bytes_at_the_key_are_never_overwritten(self) -> None:
        self.r2.objects["models/trick-v1.0.mlpackage.zip"] = b"someone else's model"
        self.assertEqual(self.main(), 2)
        self.assertIn("immutable", self.stderr.getvalue())
        self.assertEqual(
            self.r2.objects["models/trick-v1.0.mlpackage.zip"], b"someone else's model"
        )
        self.assertEqual(self.r2.puts, 0)
        self.assertEqual(self.farm.posts, [])

    def test_retry_after_a_failed_publish_reuses_the_uploads(self) -> None:
        self.farm.fail_next_post = 500
        self.assertEqual(self.main(), 2)
        self.assertIn("HTTP 500", self.stderr.getvalue())
        self.assertEqual(self.r2.puts, 2)
        self.assertEqual(self.main(), 0)
        self.assertEqual(self.r2.puts, 2)  # same bytes: nothing re-uploaded
        self.assertEqual(self.farm.posts[0]["version"], "trick-v1.0")


class RefusalTest(PublishTestCase):
    def assertNothingHappened(self) -> None:
        self.assertEqual(self.r2.objects, {})
        self.assertEqual(self.farm.posts, [])

    def set_canonical_names(self, names: list[str] | None) -> None:
        path = self.dataset / "dataset_manifest.json"
        manifest = json.loads(path.read_text())
        if names is None:
            del manifest["canonical_names"]
        else:
            manifest["canonical_names"] = names
        path.write_text(json.dumps(manifest))

    def test_uncurated_trick_names_are_never_published(self) -> None:
        # "corkscrew" trained as an unmapped token: a contributor's free
        # text, which must not reach the public manifest.
        self.set_canonical_names(["cork", "gainer"])
        self.assertEqual(self.main(), 2)
        self.assertIn("1 of the model's trick names", self.stderr.getvalue())
        self.assertNotIn("corkscrew", self.logs())
        self.assertNothingHappened()
        self.assertEqual(self.farm.calls, 0)

    def test_a_dataset_without_canonical_names_is_refused(self) -> None:
        self.set_canonical_names(None)
        self.assertEqual(self.main(), 2)
        self.assertIn("canonical_names", self.stderr.getvalue())
        self.assertNothingHappened()

    def test_missing_train_samples_are_counted_not_named(self) -> None:
        manifest = self.dataset / "manifest.jsonl"
        lines = manifest.read_text().splitlines()
        kept = [ln for ln in lines if json.loads(ln)["sample_id"] != self.train_id()]
        manifest.write_text("\n".join(kept) + "\n")
        self.assertEqual(self.main(), 2)
        self.assertIn("1 train samples are not in", self.stderr.getvalue())
        self.assertNotIn(self.train_id(), self.logs())
        self.assertNothingHappened()

    def train_id(self) -> str:
        split = json.loads((self.run_dir / "split.json").read_text())
        return min(sid for sid, part in split.items() if part == "train")

    def test_synthetic_source_ids_are_refused_before_any_side_effect(self) -> None:
        manifest = self.dataset / "manifest.jsonl"
        lines = manifest.read_text().splitlines()
        rec = json.loads(lines[0])
        rec["source_id"] = "source-a"
        manifest.write_text("\n".join([json.dumps(rec), *lines[1:]]) + "\n")
        split = json.loads((self.run_dir / "split.json").read_text())
        self.assertEqual(split[rec["sample_id"]], "train")
        self.assertEqual(self.main(), 2)
        self.assertIn("farm source ids", self.stderr.getvalue())
        self.assertNothingHappened()
        self.assertEqual(self.farm.calls, 0)

    def test_report_from_another_split_is_refused(self) -> None:
        report = json.loads((self.challenger / "holdout_report.json").read_text())
        report["inputs"]["split_sha256"] = "0" * 64
        (self.challenger / "holdout_report.json").write_text(json.dumps(report))
        self.assertEqual(self.main(), 2)
        self.assertNothingHappened()

    def test_malformed_name_metrics_are_refused(self) -> None:
        path = self.challenger / "name_metrics.json"
        metrics = json.loads(path.read_text())
        next(iter(metrics.values()))["support"] = 3
        path.write_text(json.dumps(metrics))
        self.assertEqual(self.main(), 2)
        self.assertNothingHappened()

    def test_missing_taxonomy_version_is_refused(self) -> None:
        (self.dataset / "dataset_manifest.json").write_text("{}")
        self.assertEqual(self.main(), 2)
        self.assertNothingHappened()

    def test_wrong_pipeline_key(self) -> None:
        with mock.patch.dict(os.environ, {"TURNIP_PIPELINE_KEY": "wrong-key"}):
            self.assertEqual(self.main(), 2)
        self.assertIn("HTTP 401 UNAUTHENTICATED", self.stderr.getvalue())
        self.assertNotIn("wrong-key", self.logs())
        self.assertEqual(self.farm.posts, [])

    def test_missing_environment(self) -> None:
        with mock.patch.dict(os.environ, {"TURNIP_FARM_URL": ""}):
            self.assertEqual(self.main(), 2)
        self.assertIn("TURNIP_FARM_URL", self.stderr.getvalue())

    def test_plain_http_farm_is_refused(self) -> None:
        with mock.patch.dict(os.environ, {"TURNIP_FARM_URL": "http://farm.example"}):
            self.assertEqual(self.main(), 2)
        self.assertNothingHappened()


class HttpTransportTest(PublishTestCase):
    """The real urllib transport against a local HTTP stand-in of the farm."""

    def serve(self, farm: FakeFarm) -> str:
        class Handler(BaseHTTPRequestHandler):
            def handle_one(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else None
                headers = {k: v for k, v in self.headers.items()}
                # FakeFarm routes on the public URL; the server is local.
                status, answer = farm.transport(method, FARM + self.path, headers, body)
                data = json.dumps(answer).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                self.handle_one("GET")

            def do_POST(self) -> None:
                self.handle_one("POST")

            def log_message(self, *args: Any) -> None:
                pass

        return self.start(Handler)

    def start(self, handler: type[BaseHTTPRequestHandler]) -> str:
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        # Cleanups run last-in first-out: stop serving, join, then close.
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def run_over_http(self, farm: FakeFarm) -> int:
        with mock.patch.dict(os.environ, {"TURNIP_FARM_URL": self.serve(farm)}):
            return publish_model.main(
                ["--run", str(self.run_dir), "--dataset", str(self.dataset)],
                store=self.r2,
                exporter=EXPORT,
            )

    def test_publish_over_http(self) -> None:
        farm = FakeFarm(current="trick-v1.0")
        self.assertEqual(self.run_over_http(farm), 0, self.stderr.getvalue())
        self.assertEqual(farm.posts[0]["version"], "trick-v1.1")

    def test_redirects_are_never_followed(self) -> None:
        # Following one would send the pipeline key to the Location host.
        seen: list[str | None] = []

        class Elsewhere(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                seen.append(self.headers.get("Authorization"))
                self.send_response(500)
                self.end_headers()

            do_POST = do_GET

            def log_message(self, *args: Any) -> None:
                pass

        elsewhere = self.start(Elsewhere)

        class Redirect(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/api/models/current":
                    body = json.dumps({"error": {"code": "NO_MODEL"}}).encode()
                    self.send_response(404)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.redirect()

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self.redirect()

            def redirect(self) -> None:
                self.send_response(302)
                self.send_header("Location", elsewhere + self.path)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args: Any) -> None:
                pass

        with mock.patch.dict(os.environ, {"TURNIP_FARM_URL": self.start(Redirect)}):
            code = publish_model.main(
                ["--run", str(self.run_dir), "--dataset", str(self.dataset)],
                store=self.r2,
                exporter=EXPORT,
            )
        self.assertEqual(code, 2)
        self.assertIn("HTTP 302", self.stderr.getvalue())
        self.assertEqual(seen, [])

    def test_http_error_status_and_code_are_read(self) -> None:
        farm = FakeFarm(current="trick-v1.0", published=("trick-v1.1",))
        self.assertEqual(self.run_over_http(farm), 2)
        self.assertIn("VERSION_EXISTS", self.stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
