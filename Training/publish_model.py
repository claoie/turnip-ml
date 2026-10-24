#!/usr/bin/env python3
"""Publish a promoted champion: Core ML export -> R2 upload -> POST /api/models.

turnip-ml#16, docs/MODEL_CONTRACT.md §2 steps 2-4. Training/pipeline.sh
runs it only after the fixture gate and the promotion gate have both
passed: an ungated model is never published.

Inputs
------
--run DIR      the pipeline run directory (data/run): split.json and
               challenger/{model.npz, metrics.json, holdout_report.json,
               name_metrics.json}.
--dataset DIR  the dataset directory (data/dataset): manifest.jsonl
               (sample -> source) and dataset_manifest.json (the pinned
               taxonomy_version and its curated canonical names).
--version V    publish as V instead of the next version (see below).
--major        bump the major version: an architecture change, or a
               taxonomy change that alters the output space
               (TRAINING_DESIGN.md §9). Default: a minor bump.

Environment (never argv: values would land in process lists and CI logs)
-----------
TURNIP_FARM_URL, TURNIP_PIPELINE_KEY   the farm and its service key
R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET
                                       the bucket the farm serves models from
TRAINING_RUN_ID     optional: models.training_run_id (the workflow passes
                    the run's R2 record name, <run id>-<attempt>)
MIN_CLIENT_VERSION  optional: models.min_client_version; unset means the
                    model applies to every client

Steps
-----
Every input is validated and the request body is built before any side
effect, so a bad run uploads nothing.
0. Names: every name the model outputs (its classes) and every key of
   name_metrics must be a curated canonical name at the pinned
   taxonomy_version. Unmapped label tokens train (TRAINING_DESIGN.md
   §2.2), but they are a contributor's free text and the manifest is
   public, so a model that names one is not published until the taxonomy
   maps it (dataset/unmapped_labels.log is the queue).
1. Export model.npz with export_coreml.py into
   challenger/TrickDetector.mlpackage.zip; sha256 and byte_size are over
   those exact zip bytes, the bytes the client downloads.
2. Version: GET /api/models/current (public). If the current model is
   already this artifact (same sha256), it was published before, by a run
   whose answer was lost, and nothing more happens. Otherwise 404
   NO_MODEL -> trick-v1.0; trick-v<M>.<m> (a rollback's -rb<N> suffix
   included) -> trick-v<M>.<m+1>, or trick-v<M+1>.0 with --major.
3. Upload the artifact to models/<version>.mlpackage.zip and the holdout
   report to reports/<version>-holdout.json (the farm's reports/ prefix
   for pipeline reports: turnip-farm docs/BACKEND_DESIGN.md manifest
   examples; its docs/DEPLOYMENT.md §9 lists the prefixes). A key that already
   holds other bytes is an error (a published version is immutable); one
   holding the same bytes is kept (a retry after a failed publish). Each
   object is read back and its sha256 checked; a mismatch is an error.
4. POST /api/models with the manifest fields, name_metrics and
   training_source_ids (the train split's sources). The farm sets
   model_type and promoted_at. 409 VERSION_EXISTS is an error: bump the
   version instead. When the answer is lost (a transport failure, a 5xx,
   a malformed body), GET /api/models/current decides: this version with
   this sha256 there means the insert landed.
5. Write challenger/publish_request.json (the exact body) and
   challenger/publish.json (the farm's answer): the registry keeps both
   per version, so a rollback can re-send the champion's own lists
   (MODEL_CONTRACT.md §5).

Requests never follow a redirect: one would carry the pipeline key to
another host. A 3xx is an error.

Exit codes: 0 = published (or already published), 2 = error (fail
closed; nothing is published unless the farm has the model).

Privacy: stdout and stderr carry the version, sizes and counts only,
never URLs, keys, source ids, trick names or farm error text. The
training source ids go to the farm and to the private run record,
nowhere else.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import export_coreml
import name_metrics
import train
from fetch_dataset import (
    FARM_URL_ENV,
    HEX64_RE,
    PIPELINE_KEY_ENV,
    FetchError,
    _write_atomic,
    farm_url_allowed,
    farm_urlopen,
    r2_client_from_env,
)
from split_dataset import _read_jsonl

EXIT_OK = 0
EXIT_ERROR = 2

MODELS_PATH = "/api/models"
CURRENT_PATH = "/api/models/current"
HTTP_TIMEOUT_S = 60.0
MAX_RESPONSE_BYTES = 1024 * 1024
RUN_ID_ENV = "TRAINING_RUN_ID"
MIN_CLIENT_VERSION_ENV = "MIN_CLIENT_VERSION"
FIRST_VERSION = "trick-v1.0"
VERSION_RE = re.compile(r"trick-v([0-9]+)\.([0-9]+)(?:-rb[0-9]+)?")
ARTIFACT_NAME = export_coreml.PACKAGE_NAME + ".zip"
ARTIFACT_TYPE = "application/zip"
REPORT_TYPE = "application/json"

# (method, url, headers, body) -> (HTTP status, parsed JSON body or None)
Transport = Callable[[str, str, dict[str, str], bytes | None], tuple[int, Any]]
Exporter = Callable[[Path], bytes]


class PublishError(Exception):
    """The publish refuses to go on rather than guess."""


class OutcomeUnknown(PublishError):
    """The POST's answer was lost: the farm may or may not have the model."""


class ObjectStore(Protocol):
    def get(self, key: str) -> bytes | None:
        """The object's bytes, or None when the key does not exist."""

    def put(self, key: str, data: bytes, content_type: str) -> None: ...


# -- inputs --------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise PublishError(f"cannot read {path}: {e}") from e


def training_source_ids(split_path: Path, manifest_path: Path) -> list[str]:
    """The sources of the train split: what the model trained on."""
    split = _read_json(split_path)
    if not isinstance(split, dict):
        raise PublishError(f"{split_path} must be an object of sample -> split")
    try:
        records = _read_jsonl(str(manifest_path))
    except (OSError, ValueError) as e:
        raise PublishError(f"cannot read {manifest_path}: {e}") from e
    sources = {
        str(rec.get("sample_id")): rec.get("source_id")
        for rec in records
        if isinstance(rec, dict)
    }
    train_ids = sorted(sid for sid, part in split.items() if part == "train")
    if not train_ids:
        raise PublishError(f"{split_path} has no train samples")
    missing = sum(1 for sid in train_ids if sid not in sources)
    if missing:
        raise PublishError(f"{missing} train samples are not in {manifest_path}")
    ids: set[str] = set()
    for sid in train_ids:
        source = sources[sid]
        if not isinstance(source, str) or HEX64_RE.fullmatch(source) is None:
            raise PublishError(
                "training sources must be farm source ids (64 lowercase hex): "
                "only a dataset fetched from the farm can be published"
            )
        ids.add(source)
    return sorted(ids)


def build_request(run: Path, dataset: Path) -> dict[str, Any]:
    """The POST /api/models body, minus what the version decides."""
    challenger = run / "challenger"
    split_path = run / "split.json"
    try:
        ckpt = export_coreml.load_model(challenger / "model.npz")
    except export_coreml.ExportError as e:
        raise PublishError(str(e)) from e
    input_desc, output_desc = export_coreml.descriptors(ckpt)

    metrics = _read_json(challenger / "metrics.json")
    report = _read_json(challenger / "holdout_report.json")
    manifest = _read_json(dataset / "dataset_manifest.json")
    try:
        if not isinstance(metrics, dict) or not isinstance(manifest, dict):
            raise TypeError("metrics.json and dataset_manifest.json must be objects")
        val_accuracy = name_metrics._unit(
            metrics.get("val_accuracy"), "metrics.json val_accuracy"
        )
        n_val = name_metrics._count(
            metrics.get("n_val_windows"), "metrics.json n_val_windows"
        )
        taxonomy_version = name_metrics._count(
            manifest.get("taxonomy_version"), "dataset_manifest.json taxonomy_version"
        )
        if taxonomy_version < 1:
            raise ValueError("dataset_manifest.json taxonomy_version must be >= 1")
        published_names = name_metrics.validate(
            _read_json(challenger / "name_metrics.json")
        )
    except (ValueError, TypeError) as e:
        raise PublishError(str(e)) from e

    inputs = report.get("inputs") if isinstance(report, dict) else None
    split_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if not isinstance(inputs, dict) or inputs.get("split_sha256") != split_sha:
        raise PublishError("holdout_report.json was not computed on this run's split")

    canonical = manifest.get("canonical_names")
    if not isinstance(canonical, list) or not all(
        isinstance(n, str) for n in canonical
    ):
        raise PublishError(
            "dataset_manifest.json has no canonical_names: rebuild the dataset "
            "with fetch_dataset.py build"
        )
    class_names = {
        name
        for cls in ckpt["classes"]
        if cls != train.BACKGROUND
        for name in cls.split("+")
    }
    uncurated = (class_names | set(published_names)) - set(canonical)
    if uncurated:
        raise PublishError(
            f"{len(uncurated)} of the model's trick names are not curated "
            "canonical names (unmapped labels, see the dataset's "
            "unmapped_labels.log): map them in the taxonomy before publishing, "
            "since the manifest is public"
        )

    body: dict[str, Any] = {
        "taxonomy_version": taxonomy_version,
        "input_desc": input_desc,
        "output_desc": output_desc,
        # The promotion metric of the §4.1 baseline (TRAINING_DESIGN.md §8);
        # holdout metrics live in the report behind holdout_report_r2_key.
        "val_metrics": {"val_accuracy": round(val_accuracy, 6), "n_val_windows": n_val},
        "name_metrics": published_names,
        "training_source_ids": training_source_ids(
            split_path, dataset / "manifest.jsonl"
        ),
    }
    # Optional fields are omitted, never null: the farm accepts a string or
    # nothing (models.ts validatePublishInput).
    for env, key in (
        (RUN_ID_ENV, "training_run_id"),
        (MIN_CLIENT_VERSION_ENV, "min_client_version"),
    ):
        if os.environ.get(env):
            body[key] = os.environ[env]
    return body


# -- farm ----------------------------------------------------------------------


def http_json(
    method: str, url: str, headers: dict[str, str], body: bytes | None
) -> tuple[int, Any]:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with farm_urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
            status, raw = response.status, response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read(MAX_RESPONSE_BYTES + 1)
    except (OSError, http.client.HTTPException, ValueError) as e:
        # Report the type only: the farm URL is configuration.
        raise PublishError(f"farm request failed: {type(e).__name__}") from e
    if len(raw) > MAX_RESPONSE_BYTES:
        raise PublishError(f"farm response exceeds {MAX_RESPONSE_BYTES} bytes")
    try:
        return status, json.loads(raw) if raw else None
    except ValueError:
        return status, None


def _error_code(body: Any) -> str | None:
    error = body.get("error") if isinstance(body, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) else None


class FarmClient:
    def __init__(self, base_url: str, key: str, transport: Transport | None = None):
        if not farm_url_allowed(base_url):
            raise PublishError(f"{FARM_URL_ENV} must be an https:// URL")
        if not key:
            raise PublishError(f"{PIPELINE_KEY_ENV} is empty")
        self.base_url = base_url.rstrip("/")
        self.key = key
        self.transport = transport or http_json

    def current(self) -> dict[str, Any] | None:
        """The current model's manifest; None before the first promotion."""
        headers = {"Accept": "application/json", "User-Agent": "turnip-ml-publish"}
        status, body = self.transport(
            "GET", self.base_url + CURRENT_PATH, headers, None
        )
        if status == 404 and _error_code(body) == "NO_MODEL":
            return None
        if status != 200 or not isinstance(body, dict):
            raise PublishError(f"GET {CURRENT_PATH} failed: HTTP {status}")
        if not isinstance(body.get("version"), str):
            raise PublishError(f"GET {CURRENT_PATH} answered without a version")
        return body

    def publish(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "turnip-ml-publish",
        }
        data = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        try:
            status, answer = self.transport(
                "POST", self.base_url + MODELS_PATH, headers, data
            )
        except PublishError as e:
            raise OutcomeUnknown(str(e)) from e
        if status == 409:
            raise PublishError(
                f"{body['version']} is already published (VERSION_EXISTS); "
                "versions are immutable, publish with --version or --major"
            )
        code = _error_code(answer)
        failed = f"POST {MODELS_PATH} failed: HTTP {status}" + (
            f" {code}" if code else ""
        )
        if status >= 500:
            raise OutcomeUnknown(failed)
        if status != 201:
            raise PublishError(failed)
        if not isinstance(answer, dict) or answer.get("version") != body["version"]:
            raise OutcomeUnknown(
                f"POST {MODELS_PATH} answered 201 without this version"
            )
        return answer


def next_version(current: dict[str, Any] | None, major: bool) -> str:
    if current is None:
        return FIRST_VERSION
    match = VERSION_RE.fullmatch(current["version"])
    if match is None:
        raise PublishError(
            f"current version {current['version']!r} is not trick-v<major>.<minor>; "
            "publish with --version"
        )
    m, n = int(match.group(1)), int(match.group(2))
    return f"trick-v{m + 1}.0" if major else f"trick-v{m}.{n + 1}"


# -- R2 ------------------------------------------------------------------------


class R2Store:
    """The farm's bucket via boto3 (S3-compatible)."""

    def __init__(self) -> None:
        try:
            self.client, self.bucket, self.errors = r2_client_from_env("publish")
        except FetchError as e:
            raise PublishError(str(e)) from e

    def get(self, key: str) -> bytes | None:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            body: bytes = response["Body"].read()
        except self.errors as e:
            error = (getattr(e, "response", None) or {}).get("Error", {})
            if error.get("Code") in ("NoSuchKey", "404"):
                return None
            raise PublishError(f"{key}: R2 read failed: {type(e).__name__}") from e
        return body

    def put(self, key: str, data: bytes, content_type: str) -> None:
        try:
            self.client.put_object(
                Bucket=self.bucket, Key=key, Body=data, ContentType=content_type
            )
        except self.errors as e:
            raise PublishError(f"{key}: R2 upload failed: {type(e).__name__}") from e


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def upload_verified(
    store: ObjectStore, key: str, data: bytes, content_type: str
) -> None:
    """Put `data` at `key` unless the same bytes are there; verify by reading back.

    Other bytes at the key are an error: a published version is immutable.
    """
    want = _sha256(data)
    existing = store.get(key)
    if existing is not None:
        if _sha256(existing) != want:
            raise PublishError(
                f"{key} already holds other bytes; a published version is "
                "immutable, publish with --version or --major"
            )
        return
    store.put(key, data, content_type)
    stored = store.get(key)
    if stored is None or _sha256(stored) != want:
        raise PublishError(f"{key}: the uploaded object does not match its sha256")


# -- publish -------------------------------------------------------------------


def _write_json(path: Path, data: Any) -> None:
    _write_atomic(path, (json.dumps(data, indent=2, sort_keys=True) + "\n").encode())


def _record(body: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": body["version"],
        "r2_key": body["r2_key"],
        "sha256": body["sha256"],
        "byte_size": body["byte_size"],
        "holdout_report_r2_key": body["holdout_report_r2_key"],
        "training_sources": len(body["training_source_ids"]),
        "farm": answer,
    }


def _farm_fields(manifest: dict[str, Any]) -> dict[str, Any]:
    """What a manifest says about its row, for the publish record."""
    return {k: manifest.get(k) for k in ("version", "promoted_at", "sha256")}


def publish(
    run: Path,
    dataset: Path,
    farm: FarmClient,
    store: ObjectStore,
    version: str | None = None,
    major: bool = False,
    exporter: Exporter = export_coreml.export,
) -> dict[str, Any]:
    """Export, upload and publish the run's challenger. Returns the record."""
    challenger = run / "challenger"
    body = build_request(run, dataset)
    report = (challenger / "holdout_report.json").read_bytes()
    try:
        artifact = exporter(challenger / "model.npz")
    except export_coreml.ExportError as e:
        raise PublishError(f"export failed: {e}") from e
    (challenger / ARTIFACT_NAME).write_bytes(artifact)
    sha = _sha256(artifact)

    current = farm.current()
    if current is not None and current.get("sha256") == sha:
        # Published already, by a run whose answer was lost: done.
        done = {
            **body,
            "version": current["version"],
            "r2_key": current.get("r2_key"),
            "sha256": sha,
            "byte_size": len(artifact),
            "holdout_report_r2_key": current.get("holdout_report_r2_key"),
        }
        record = _record(done, {"already_published": True, **_farm_fields(current)})
        _write_json(challenger / "publish.json", record)
        return record

    version = version or next_version(current, major)
    if VERSION_RE.fullmatch(version) is None:
        raise PublishError(f"version {version!r} is not trick-v<major>.<minor>")
    body = {
        **body,
        "version": version,
        "r2_key": f"models/{version}.mlpackage.zip",
        "sha256": sha,
        "byte_size": len(artifact),
        "holdout_report_r2_key": f"reports/{version}-holdout.json",
    }

    upload_verified(store, body["r2_key"], artifact, ARTIFACT_TYPE)
    upload_verified(store, body["holdout_report_r2_key"], report, REPORT_TYPE)
    _write_json(challenger / "publish_request.json", body)
    try:
        answer = farm.publish(body)
    except OutcomeUnknown:
        landed = farm.current()
        if not (
            landed is not None
            and landed.get("version") == version
            and landed.get("sha256") == sha
        ):
            raise
        answer = {"reconciled": True, **_farm_fields(landed)}
    record = _record(body, answer)
    _write_json(challenger / "publish.json", record)
    return record


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, type=Path, help="pipeline run directory")
    p.add_argument("--dataset", required=True, type=Path, help="dataset directory")
    bump = p.add_mutually_exclusive_group()
    bump.add_argument("--version", help="publish as this version")
    bump.add_argument("--major", action="store_true", help="bump the major version")
    return p.parse_args(argv)


def main(
    argv: list[str] | None = None,
    transport: Transport | None = None,
    store: ObjectStore | None = None,
    exporter: Exporter = export_coreml.export,
) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        missing = [n for n in (FARM_URL_ENV, PIPELINE_KEY_ENV) if not os.environ.get(n)]
        if missing:
            raise PublishError(f"missing environment: {', '.join(missing)}")
        farm = FarmClient(
            os.environ[FARM_URL_ENV], os.environ[PIPELINE_KEY_ENV], transport
        )
        record = publish(
            args.run,
            args.dataset,
            farm,
            store or R2Store(),
            args.version,
            args.major,
            exporter,
        )
    except (PublishError, OSError) as e:
        print(f"publish_model: error: {e}", file=sys.stderr)
        return EXIT_ERROR
    farm_answer = record["farm"]
    if farm_answer.get("already_published"):
        how = "already published (the farm's current model is this artifact)"
    elif farm_answer.get("reconciled"):
        how = "published (the answer was lost; the farm's current model confirms it)"
    else:
        how = (
            f"published, {farm_answer.get('training_sources_recorded')} of "
            f"{record['training_sources']} training sources recorded"
        )
    print(f"{record['version']}: {how}; {record['byte_size']}-byte artifact")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
