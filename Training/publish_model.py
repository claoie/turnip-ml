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
               taxonomy_version).
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
1. Export model.npz with export_coreml.py into
   challenger/TrickDetector.mlpackage.zip; sha256 and byte_size are over
   those exact zip bytes, the bytes the client downloads.
2. Version: GET /api/models/current (public). 404 NO_MODEL -> trick-v1.0;
   otherwise trick-v<M>.<m> (a rollback's -rb<N> suffix included) ->
   trick-v<M>.<m+1>, or trick-v<M+1>.0 with --major.
3. Upload the artifact to models/<version>.mlpackage.zip and the holdout
   report to models/<version>.holdout_report.json. A key that already
   holds other bytes is an error (a published version is immutable); one
   holding the same bytes is kept (a retry after a failed publish). Each
   object is read back and its sha256 checked; a mismatch is an error.
4. POST /api/models with the manifest fields, name_metrics and
   training_source_ids (the train split's sources). The farm sets
   model_type and promoted_at. 409 VERSION_EXISTS is an error: bump the
   version instead.
5. Write challenger/publish_request.json (the exact body) and
   challenger/publish.json (the farm's answer): the registry keeps both
   per version, so a rollback can re-send the champion's own lists
   (MODEL_CONTRACT.md §5).

Exit codes: 0 = published, 2 = error (fail closed; nothing is published
unless step 4 returned 201).

Privacy: stdout and stderr carry the version, sizes and counts only,
never URLs, keys, source ids or farm error text. The training source ids
go to the farm and to the private run record, nowhere else.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import export_coreml
from fetch_dataset import (
    FARM_URL_ENV,
    HEX64_RE,
    LOCAL_HOSTS,
    PIPELINE_KEY_ENV,
    R2_ENV,
)

EXIT_OK = 0
EXIT_ERROR = 2

MODELS_PATH = "/api/models"
CURRENT_PATH = "/api/models/current"
HTTP_TIMEOUT_S = 60.0
MAX_RESPONSE_BYTES = 1024 * 1024
RUN_ID_ENV = "TRAINING_RUN_ID"
MIN_CLIENT_VERSION_ENV = "MIN_CLIENT_VERSION"
FIRST_VERSION = "trick-v1.0"
VERSION_RE = re.compile(r"trick-v(\d+)\.(\d+)(?:-rb\d+)?")
ARTIFACT_NAME = export_coreml.PACKAGE_NAME + ".zip"
ARTIFACT_TYPE = "application/zip"
REPORT_TYPE = "application/json"
NAME_METRIC_KEYS = {"precision", "recall", "holdout_count", "train_count"}

# (method, url, headers, body) -> (HTTP status, parsed JSON body or None)
Transport = Callable[[str, str, dict[str, str], bytes | None], tuple[int, Any]]
Exporter = Callable[[Path], bytes]


class PublishError(Exception):
    """The publish refuses to go on rather than guess."""


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


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_unit(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def load_name_metrics(path: Path) -> dict[str, Any]:
    """name_metrics.json (Training/name_metrics.py), shape-checked."""
    data = _read_json(path)
    if not isinstance(data, dict) or not data:
        raise PublishError(f"{path} must be a non-empty object of name -> metrics")
    for name, entry in data.items():
        if not name or not isinstance(entry, dict) or set(entry) != NAME_METRIC_KEYS:
            raise PublishError(
                f"{path}: entries need exactly {sorted(NAME_METRIC_KEYS)}"
            )
        if not (_is_count(entry["holdout_count"]) and _is_count(entry["train_count"])):
            raise PublishError(f"{path}: counts must be non-negative integers")
        for key in ("precision", "recall"):
            if entry[key] is not None and not _is_unit(entry[key]):
                raise PublishError(f"{path}: {key} must be null or in [0, 1]")
    return data


def training_source_ids(split_path: Path, manifest_path: Path) -> list[str]:
    """The sources of the train split: what the model trained on."""
    split = _read_json(split_path)
    if not isinstance(split, dict):
        raise PublishError(f"{split_path} must be an object of sample -> split")
    sources: dict[str, Any] = {}
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        raise PublishError(f"cannot read {manifest_path}: {e}") from e
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError as e:
            raise PublishError(f"{manifest_path} line {i + 1}: {e}") from e
        if not isinstance(rec, dict):
            raise PublishError(f"{manifest_path} line {i + 1} is not an object")
        sources[str(rec.get("sample_id"))] = rec.get("source_id")
    train_ids = sorted(sid for sid, part in split.items() if part == "train")
    if not train_ids:
        raise PublishError(f"{split_path} has no train samples")
    ids = set()
    for sid in train_ids:
        if sid not in sources:
            raise PublishError(f"train sample {sid!r} is not in {manifest_path}")
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
    if not (
        isinstance(metrics, dict)
        and _is_unit(metrics.get("val_accuracy"))
        and _is_count(metrics.get("n_val_windows"))
    ):
        raise PublishError("metrics.json needs val_accuracy and n_val_windows")

    report = _read_json(challenger / "holdout_report.json")
    inputs = report.get("inputs") if isinstance(report, dict) else None
    split_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if not isinstance(inputs, dict) or inputs.get("split_sha256") != split_sha:
        raise PublishError("holdout_report.json was not computed on this run's split")

    manifest = _read_json(dataset / "dataset_manifest.json")
    if not (
        isinstance(manifest, dict)
        and _is_count(manifest.get("taxonomy_version"))
        and manifest["taxonomy_version"] >= 1
    ):
        raise PublishError("dataset_manifest.json needs a taxonomy_version >= 1")

    body: dict[str, Any] = {
        "taxonomy_version": manifest["taxonomy_version"],
        "input_desc": input_desc,
        "output_desc": output_desc,
        # The promotion metric of the §4.1 baseline (TRAINING_DESIGN.md §8);
        # holdout metrics live in the report behind holdout_report_r2_key.
        "val_metrics": {
            "val_accuracy": round(float(metrics["val_accuracy"]), 6),
            "n_val_windows": metrics["n_val_windows"],
        },
        "name_metrics": load_name_metrics(challenger / "name_metrics.json"),
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
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
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
        parsed = urllib.parse.urlsplit(base_url)
        local = parsed.scheme == "http" and parsed.hostname in LOCAL_HOSTS
        if parsed.scheme != "https" and not local:
            raise PublishError(f"{FARM_URL_ENV} must be an https:// URL")
        if not key:
            raise PublishError(f"{PIPELINE_KEY_ENV} is empty")
        self.base_url = base_url.rstrip("/")
        self.key = key
        self.transport = transport or http_json

    def current_version(self) -> str | None:
        """The current model's version; None before the first promotion."""
        headers = {"Accept": "application/json", "User-Agent": "turnip-ml-publish"}
        status, body = self.transport(
            "GET", self.base_url + CURRENT_PATH, headers, None
        )
        if status == 404 and _error_code(body) == "NO_MODEL":
            return None
        version = body.get("version") if isinstance(body, dict) else None
        if status != 200 or not isinstance(version, str):
            raise PublishError(f"GET {CURRENT_PATH} failed: HTTP {status}")
        return version

    def publish(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "turnip-ml-publish",
        }
        data = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        status, answer = self.transport(
            "POST", self.base_url + MODELS_PATH, headers, data
        )
        if status == 409:
            raise PublishError(
                f"{body['version']} is already published (VERSION_EXISTS); "
                "versions are immutable, publish with --version or --major"
            )
        if status != 201 or not isinstance(answer, dict):
            code = _error_code(answer)
            raise PublishError(
                f"POST {MODELS_PATH} failed: HTTP {status}"
                + (f" {code}" if code else "")
            )
        if answer.get("version") != body["version"]:
            raise PublishError(f"POST {MODELS_PATH} answered for another version")
        return answer


def next_version(current: str | None, major: bool) -> str:
    if current is None:
        return FIRST_VERSION
    match = VERSION_RE.fullmatch(current)
    if match is None:
        raise PublishError(
            f"current version {current!r} is not trick-v<major>.<minor>; "
            "publish with --version"
        )
    m, n = int(match.group(1)), int(match.group(2))
    return f"trick-v{m + 1}.0" if major else f"trick-v{m}.{n + 1}"


# -- R2 ------------------------------------------------------------------------


class R2Store:
    """The farm's bucket via boto3 (S3-compatible)."""

    def __init__(self) -> None:
        missing = [name for name in R2_ENV if not os.environ.get(name)]
        if missing:
            raise PublishError(f"missing environment: {', '.join(missing)}")
        try:
            import boto3
            from botocore.config import Config
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError as e:
            raise PublishError("publish needs boto3 to upload to R2") from e
        account, key_id, secret, self.bucket = (os.environ[n] for n in R2_ENV)
        self.client = boto3.client(
            "s3",
            endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
            region_name="auto",
            # R2 rejects the newer default request checksums on some
            # versions; the read-back below is the integrity check.
            config=Config(
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
            ),
        )
        self.errors = (BotoCoreError, ClientError)
        self.client_error = ClientError

    def get(self, key: str) -> bytes | None:
        try:
            body: bytes = self.client.get_object(Bucket=self.bucket, Key=key)[
                "Body"
            ].read()
        except self.client_error as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise PublishError(f"{key}: R2 read failed: {type(e).__name__}") from e
        except self.errors as e:
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
) -> bool:
    """Put `data` at `key` unless it is already there; verify by reading back.

    Returns True when this call uploaded, False when the same bytes were
    already in place. Other bytes at the key are an error.
    """
    want = _sha256(data)
    existing = store.get(key)
    if existing is not None:
        if _sha256(existing) != want:
            raise PublishError(
                f"{key} already holds other bytes; a published version is "
                "immutable, publish with --version or --major"
            )
        return False
    store.put(key, data, content_type)
    stored = store.get(key)
    if stored is None or _sha256(stored) != want:
        raise PublishError(f"{key}: the uploaded object does not match its sha256")
    return True


# -- publish -------------------------------------------------------------------


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


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
    artifact_path = challenger / ARTIFACT_NAME
    artifact_path.write_bytes(artifact)

    version = version or next_version(farm.current_version(), major)
    if VERSION_RE.fullmatch(version) is None:
        raise PublishError(f"version {version!r} is not trick-v<major>.<minor>")
    artifact_key = f"models/{version}.mlpackage.zip"
    report_key = f"models/{version}.holdout_report.json"
    body = {
        **body,
        "version": version,
        "r2_key": artifact_key,
        "sha256": _sha256(artifact),
        "byte_size": len(artifact),
        "holdout_report_r2_key": report_key,
    }

    upload_verified(store, artifact_key, artifact, ARTIFACT_TYPE)
    upload_verified(store, report_key, report, REPORT_TYPE)
    _write_json(challenger / "publish_request.json", body)
    answer = farm.publish(body)
    record = {
        "version": version,
        "r2_key": artifact_key,
        "sha256": body["sha256"],
        "byte_size": body["byte_size"],
        "holdout_report_r2_key": report_key,
        "training_sources": len(body["training_source_ids"]),
        "farm": answer,
    }
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
    print(
        f"published {record['version']}: {record['byte_size']}-byte artifact, "
        f"{record['training_sources']} training sources "
        f"({record['farm'].get('training_sources_recorded')} recorded)"
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
