#!/usr/bin/env python3
"""Training dataset fetch: farm label export + R2 pose blobs -> train.py samples.

Subcommands
-----------
count --out-state FILE [--state FILE] [--out-sources FILE]
    Pages `GET /api/labels/export?since=<watermark>` and prints
    {"new_labels": N, "changed_sources": M}: the label rows changed since
    the watermark in --state (omit --state on the very first run to count
    everything), and the sources they belong to. The newest cursor seen is
    written to --out-state as {"watermark": ...}; the pipeline stores it
    only after the whole run succeeds, so the watermark always means "last
    successful export". --out-sources writes the changed source ids as a
    sorted JSON array: the fixture gate's --ingested list. Taken after the
    dataset pull, it covers every new row the challenger trained on, and
    it includes old sources whose labels were edited (a relabel can poison
    as well as a new upload).

build --out DIR --cache DIR
    Full pull of the export (never incremental: deletions, purges,
    quarantines, releases and replaced label sets change what is
    training-eligible, and the export does not report deletions). Right
    after the pull, every cached blob whose source is absent from it is
    deleted, so data deleted farm-side never outlives one full pull here.
    Then for each source it fetches the pose blob from R2, caches it under
    its r2_key, verifies it against `blob_sha256` (the SHA-256 of the
    stored `.tkp1.gz` bytes) on download and on every cache read (a
    mismatch on read means the source was re-contributed, or the file is
    corrupt: it is fetched again), and decodes it. Writes into DIR, which
    must not exist yet (built in a temporary sibling directory, moved into
    place on success):
      samples/<source_id>.json  train.py's sample format: sample_id (the
                                source id), fps 10, keypoints
                                [frame][17][x, y, confidence] (gap frames
                                kept as zeros), and one trick entry per
                                (clip, canonical name) on the clip window
      manifest.jsonl            {"sample_id", "source_id"} per sample, the
                                split_dataset.py input (splits group by
                                source)
      dataset_manifest.json     pinned taxonomy_version and its curated
                                canonical names, export cursor,
                                counts and sorted sample ids
      unmapped_labels.log       "<count>\t<token>" per unmapped label token,
                                the taxonomy curation queue
    Raw labels are canonicalized at one pinned taxonomy version: the
    farm's current one, i.e. the greatest taxonomy_version in the export's
    taxonomy or rows, so a curation-only bump applies to every label at the
    next run (see canonicalize.py for the per-raw resolution).
    A label that canonicalizes to no name at all (only parentheticals or
    delimiters) contributes nothing; a clip left without names is dropped.

Environment (never argv: values would land in process lists and CI logs)
-----------
TURNIP_FARM_URL       farm base URL (https; http only for localhost)
TURNIP_PIPELINE_KEY   service key sent as a Bearer token
R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET
                      pose-blob bucket (build only; needs boto3)

Privacy: training data carries no account identifier. The farm exports
none, and an export row that carries a user_id is refused rather than
passed along. Stdout and stderr carry counts only, never raw labels,
cursors or pose values; CI logs of this public repository are public. The
dataset and the blob cache belong under data/ (gitignored) and must never
be uploaded as a CI artifact or cache.

Determinism: rows are re-sorted after the pull (keyset pages split sources
arbitrarily), JSON keys are sorted, float32 pose values are written
losslessly, and nothing carries a timestamp, so the same export and blobs
always produce byte-identical files.

Fail closed: a malformed page or row (including one carrying a user_id),
rows that disagree about a source or
clip, a duplicate label row, a stuck or cycling cursor, a clip window
outside its pose sequence, a checksum mismatch or an undecodable blob are
errors (exit 2), never skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import tkp1
from canonicalize import Taxonomy, TaxonomyError

EXIT_OK = 0
EXIT_ERROR = 2

EXPORT_PATH = "/api/labels/export"
HTTP_TIMEOUT_S = 60.0
MAX_PAGE_BYTES = 64 * 1024 * 1024
FPS = 10

FARM_URL_ENV = "TURNIP_FARM_URL"
PIPELINE_KEY_ENV = "TURNIP_PIPELINE_KEY"
R2_ENV = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")

HEX64_RE = re.compile(r"[0-9a-f]{64}")
# The only files the blob cache writes, under <cache>/poses/.
CACHE_ENTRY_RE = re.compile(r"([0-9a-f]{64})\.tkp1\.gz(\.tmp)?")
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")

Transport = Callable[[str, dict[str, str]], Any]
BlobGetter = Callable[[str], bytes]


class FetchError(Exception):
    """The fetch refuses to produce a dataset rather than guess."""


@dataclass(frozen=True)
class LabelRow:
    source_id: str
    r2_key: str
    blob_sha256: str
    clip_id: str
    start_frame: int
    end_frame: int
    trick_name: str
    taxonomy_version: int


@dataclass(frozen=True)
class ExportPull:
    rows: list[LabelRow]
    taxonomy: list[tuple[str, str, int]]
    cursor: str | None  # the newest non-null next_cursor seen


# -- export ------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _hex64(value: Any) -> bool:
    return isinstance(value, str) and HEX64_RE.fullmatch(value) is not None


def parse_row(i: int, row: Any) -> LabelRow:
    fields = {
        "source_id",
        "r2_key",
        "blob_sha256",
        "clip_id",
        "start_frame",
        "end_frame",
        "trick_name",
        "taxonomy_version",
    }
    if not isinstance(row, dict) or not fields <= set(row):
        raise FetchError(f"export row {i} is missing fields")
    if "user_id" in row:
        raise FetchError(
            f"export row {i} carries a user_id; training data must carry no "
            "account identifier"
        )
    source_id = row["source_id"]
    if not _hex64(source_id):
        raise FetchError(f"export row {i}: source_id is not 64 lowercase hex chars")
    if row["r2_key"] != f"poses/{source_id}.tkp1.gz":
        raise FetchError(f"export row {i}: r2_key does not match its source_id")
    if not _hex64(row["blob_sha256"]):
        raise FetchError(f"export row {i}: blob_sha256 is not 64 lowercase hex chars")
    clip_id = row["clip_id"]
    try:
        canonical_clip = str(uuid.UUID(clip_id)) if isinstance(clip_id, str) else None
    except ValueError:
        canonical_clip = None
    if canonical_clip != clip_id:
        raise FetchError(f"export row {i}: clip_id is not a lowercase UUID")
    start, end = row["start_frame"], row["end_frame"]
    if not (_is_int(start) and _is_int(end) and 0 <= start < end):
        raise FetchError(f"export row {i}: needs integer 0 <= start_frame < end_frame")
    name = row["trick_name"]
    if not isinstance(name, str) or not name.strip():
        raise FetchError(f"export row {i}: trick_name must be a non-empty string")
    version = row["taxonomy_version"]
    if not (_is_int(version) and version >= 1):
        raise FetchError(f"export row {i}: taxonomy_version must be an integer >= 1")
    return LabelRow(
        source_id=source_id,
        r2_key=row["r2_key"],
        blob_sha256=row["blob_sha256"],
        clip_id=clip_id,
        start_frame=start,
        end_frame=end,
        trick_name=name,
        taxonomy_version=version,
    )


def parse_taxonomy(i: int, entry: Any) -> tuple[str, str, int]:
    if not isinstance(entry, dict):
        raise FetchError(f"taxonomy entry {i} is not an object")
    raw, canonical, version = (
        entry.get("raw"),
        entry.get("canonical"),
        entry.get("taxonomy_version"),
    )
    if not (
        isinstance(raw, str)
        and raw
        and isinstance(canonical, str)
        and canonical
        and isinstance(version, int)
        and not isinstance(version, bool)
        and version >= 1
    ):
        raise FetchError(f"taxonomy entry {i} needs raw, canonical, taxonomy_version")
    return raw, canonical, version


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: it would carry the pipeline key to whatever
    host the Location names. A 3xx then surfaces as an HTTPError."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: http.client.HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


_FARM_OPENER = urllib.request.build_opener(_NoRedirect)


def farm_urlopen(request: urllib.request.Request, timeout: float) -> Any:
    """urllib.request.urlopen for farm requests, never following redirects."""
    return _FARM_OPENER.open(request, timeout=timeout)


def farm_url_allowed(base_url: str) -> bool:
    """https, or plain http to this machine only."""
    parsed = urllib.parse.urlsplit(base_url)
    local = parsed.scheme == "http" and parsed.hostname in LOCAL_HOSTS
    return parsed.scheme == "https" or local


def http_get_json(url: str, headers: dict[str, str]) -> Any:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with farm_urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
            body = response.read(MAX_PAGE_BYTES + 1)
    except urllib.error.HTTPError as e:
        raise FetchError(f"export request failed: HTTP {e.code}") from e
    except (OSError, http.client.HTTPException, ValueError) as e:
        # URLError and timeouts are OSError. Report the type only: the
        # farm URL is configuration, not something logs need to repeat.
        raise FetchError(f"export request failed: {type(e).__name__}") from e
    if len(body) > MAX_PAGE_BYTES:
        raise FetchError(f"export page exceeds {MAX_PAGE_BYTES} bytes")
    try:
        return json.loads(body)
    except ValueError as e:
        raise FetchError("export response is not valid JSON") from e


class ExportClient:
    def __init__(self, base_url: str, key: str, transport: Transport | None = None):
        if not farm_url_allowed(base_url):
            raise FetchError(f"{FARM_URL_ENV} must be an https:// URL")
        if not key:
            raise FetchError(f"{PIPELINE_KEY_ENV} is empty")
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "User-Agent": "turnip-ml-fetch-dataset",
        }
        self.transport = transport or http_get_json

    def pull(self, since: str | None = None) -> ExportPull:
        """Every row after `since` (None: the whole export)."""
        rows: list[LabelRow] = []
        taxonomy: set[tuple[str, str, int]] = set()
        cursor = since
        seen: set[str | None] = {since}
        while True:
            query = (
                ""
                if cursor is None
                else "?" + urllib.parse.urlencode({"since": cursor})
            )
            page = self.transport(self.base_url + EXPORT_PATH + query, self.headers)
            if not isinstance(page, dict) or not isinstance(page.get("rows"), list):
                raise FetchError("export page must be an object with a rows array")
            page_taxonomy = page.get("taxonomy", [])
            if not isinstance(page_taxonomy, list):
                raise FetchError("export page taxonomy must be an array")
            next_cursor = page.get("next_cursor")
            if next_cursor is not None and not (
                isinstance(next_cursor, str) and next_cursor
            ):
                raise FetchError(
                    "export next_cursor must be a non-empty string or null"
                )
            offset = len(rows)
            rows.extend(parse_row(offset + i, r) for i, r in enumerate(page["rows"]))
            taxonomy.update(parse_taxonomy(i, t) for i, t in enumerate(page_taxonomy))
            if not page["rows"] or next_cursor is None:
                break
            if next_cursor in seen:
                raise FetchError("export cursor did not advance")
            seen.add(next_cursor)
            cursor = next_cursor
        return ExportPull(rows, sorted(taxonomy), cursor)


# -- pose blobs ----------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


class BlobCache:
    """Pose blobs keyed by r2_key; verified on download and on every read."""

    def __init__(self, cache_dir: Path, get: BlobGetter):
        self.cache_dir = cache_dir
        self.get = get

    def fetch(self, r2_key: str, sha256: str) -> bytes:
        # r2_key is validated as poses/<64 hex>.tkp1.gz, so this stays
        # inside the cache directory.
        path = self.cache_dir / r2_key
        try:
            if path.exists():
                data = path.read_bytes()
                if _sha256(data) == sha256:
                    return data
                # Re-contributed source or corrupt entry: the bucket is the
                # truth.
                path.unlink()
        except OSError as e:
            raise FetchError(f"cannot use blob cache {self.cache_dir}: {e}") from e
        try:
            data = self.get(r2_key)
        except FetchError:
            raise
        except Exception as e:
            # Any getter failure is a failed download (exit 2), never a
            # traceback; report the type only, as messages can echo URLs.
            raise FetchError(f"{r2_key}: download failed: {type(e).__name__}") from e
        if not isinstance(data, bytes) or _sha256(data) != sha256:
            raise FetchError(f"{r2_key}: downloaded bytes do not match blob_sha256")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(path, data)
        except OSError as e:
            raise FetchError(f"cannot write blob cache {self.cache_dir}: {e}") from e
        return data

    def prune(self, keep: set[str]) -> int:
        """Delete cached blobs whose source is not in `keep`; returns the count.

        Only files this cache writes are touched (poses/<64 hex>.tkp1.gz and
        leftover .tmp siblings), never anything else in the directory.
        """
        poses = self.cache_dir / "poses"
        pruned = 0
        try:
            entries = sorted(poses.iterdir()) if poses.is_dir() else []
            for path in entries:
                match = CACHE_ENTRY_RE.fullmatch(path.name)
                if match is None or not path.is_file():
                    continue
                if match.group(2) or match.group(1) not in keep:
                    path.unlink()
                    pruned += 1
        except OSError as e:
            raise FetchError(f"cannot prune blob cache {self.cache_dir}: {e}") from e
        return pruned


def r2_client_from_env(purpose: str) -> tuple[Any, str, tuple[type[Exception], ...]]:
    """(boto3 S3 client for R2, bucket, the boto errors a call can raise)."""
    missing = [name for name in R2_ENV if not os.environ.get(name)]
    if missing:
        raise FetchError(f"missing environment: {', '.join(missing)}")
    try:
        import boto3
        from botocore.config import Config
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError as e:
        raise FetchError(f"{purpose} needs boto3 to reach R2") from e
    account, key_id, secret, bucket = (os.environ[name] for name in R2_ENV)
    client = boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        region_name="auto",
        # R2 rejects the newer default request checksums on some versions;
        # callers verify content against their own sha256.
        config=Config(
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )
    return client, bucket, (BotoCoreError, ClientError)


def r2_getter_from_env() -> BlobGetter:
    client, bucket, errors = r2_client_from_env("build (pose blobs)")

    def get(r2_key: str) -> bytes:
        try:
            body: bytes = client.get_object(Bucket=bucket, Key=r2_key)["Body"].read()
        except errors as e:
            raise FetchError(f"{r2_key}: R2 download failed: {type(e).__name__}") from e
        return body

    return get


# -- dataset -------------------------------------------------------------------


def _group(rows: list[LabelRow]) -> dict[str, dict[str, list[LabelRow]]]:
    """source_id -> clip_id -> rows, checking per-source/per-clip agreement."""
    sources: dict[str, dict[str, list[LabelRow]]] = {}
    seen_labels: set[tuple[str, str]] = set()
    for row in sorted(rows, key=lambda r: (r.source_id, r.clip_id, r.trick_name)):
        label = (row.clip_id, row.trick_name)
        if label in seen_labels:
            raise FetchError("export returned a duplicate label row")
        seen_labels.add(label)
        clips = sources.setdefault(row.source_id, {})
        first = next(iter(clips.values()))[0] if clips else row
        if (first.r2_key, first.blob_sha256) != (row.r2_key, row.blob_sha256):
            raise FetchError(f"rows for source {row.source_id} disagree on its blob")
        clip_rows = clips.setdefault(row.clip_id, [])
        if clip_rows and (clip_rows[0].start_frame, clip_rows[0].end_frame) != (
            row.start_frame,
            row.end_frame,
        ):
            raise FetchError(f"rows for clip {row.clip_id} disagree on its window")
        clip_rows.append(row)
    # A clip id belongs to exactly one source.
    owners = Counter(clip for clips in sources.values() for clip in clips)
    if any(n > 1 for n in owners.values()):
        raise FetchError("a clip_id appears under two sources")
    return sources


def _dump(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n"


def build_dataset(pull: ExportPull, blobs: BlobCache, out_dir: Path) -> dict[str, Any]:
    """Write the dataset into out_dir (must not exist). Returns the stats."""
    # Prune first: an empty pull means every cached source is gone, which
    # is exactly when deleting matters.
    pruned = blobs.prune({row.source_id for row in pull.rows})
    if not pull.rows:
        raise FetchError("the export is empty: nothing to train on")
    if out_dir.exists():
        raise FetchError(f"{out_dir} already exists; use a fresh directory per run")
    # The farm's current version: the greatest in the taxonomy (or rows),
    # so curation applies without waiting for labels at the new version.
    pinned = max(
        [row.taxonomy_version for row in pull.rows]
        + [version for _, _, version in pull.taxonomy]
    )
    try:
        taxonomy = Taxonomy(pull.taxonomy, pinned)
    except TaxonomyError as e:
        raise FetchError(f"taxonomy: {e}") from e
    sources = _group(pull.rows)

    try:
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{out_dir.name}.partial-", dir=out_dir.parent)
        )
    except OSError as e:
        raise FetchError(f"cannot create {out_dir}: {e}") from e
    try:
        stats = _write_dataset(pull, sources, taxonomy, blobs, staging)
        staging.rename(out_dir)
    except OSError as e:
        raise FetchError(f"cannot write dataset to {out_dir}: {e}") from e
    finally:
        # Gone after a successful rename; otherwise never leave a partial
        # dataset behind.
        shutil.rmtree(staging, ignore_errors=True)
    # Not in dataset_manifest.json: it describes this cache, not the data.
    return {**stats, "cache_pruned": pruned}


def _write_dataset(
    pull: ExportPull,
    sources: dict[str, dict[str, list[LabelRow]]],
    taxonomy: Taxonomy,
    blobs: BlobCache,
    staging: Path,
) -> dict[str, Any]:
    """Write samples one at a time (a dataset need not fit in memory)."""
    (staging / "samples").mkdir()
    unmapped: Counter[str] = Counter()
    clips_dropped = 0
    sample_ids: list[str] = []
    for source_id, clips in sources.items():
        first = next(iter(clips.values()))[0]
        try:
            pose = tkp1.decode_gz(blobs.fetch(first.r2_key, first.blob_sha256))
        except tkp1.Tkp1Error as e:
            raise FetchError(f"{first.r2_key}: {e}") from e
        tricks: list[dict[str, Any]] = []
        for clip_id, rows in clips.items():
            start, end = rows[0].start_frame, rows[0].end_frame
            if end > pose.frame_count:
                raise FetchError(
                    f"clip {clip_id} ends at frame {end}, past the "
                    f"{pose.frame_count}-frame pose sequence"
                )
            names: set[str] = set()
            for row in rows:
                canonical, tokens = taxonomy.canonicalize(row.trick_name)
                names.update(canonical)
                unmapped.update(tokens)
            if not names:
                clips_dropped += 1
                continue
            tricks.extend(
                {"clip_id": clip_id, "name": n, "start_frame": start, "end_frame": end}
                for n in sorted(names)
            )
        tricks.sort(
            key=lambda t: (t["start_frame"], t["end_frame"], t["clip_id"], t["name"])
        )
        sample = {
            "sample_id": source_id,
            "fps": FPS,
            "keypoints": pose.keypoints,
            "tricks": tricks,
        }
        path = staging / "samples" / f"{source_id}.json"
        path.write_text(_dump(sample), encoding="utf-8")
        sample_ids.append(source_id)

    sample_ids.sort()
    stats = {
        "samples": len(sample_ids),
        "clips": sum(len(clips) for clips in sources.values()) - clips_dropped,
        "clips_dropped_without_names": clips_dropped,
        "label_rows": len(pull.rows),
        "taxonomy_version": taxonomy.version,
        "unmapped_tokens": len(unmapped),
    }
    manifest = {
        **stats,
        # The curated names at the pin: what the farm's vocabulary for this
        # taxonomy_version holds. Unmapped tokens train, but a model that
        # names one is not published (publish_model.py): the manifest is
        # public and the token is a contributor's free text.
        "canonical_names": taxonomy.canonical_names,
        "export_cursor": pull.cursor,
        "sample_ids": sample_ids,
    }
    (staging / "manifest.jsonl").write_text(
        "".join(_dump({"sample_id": s, "source_id": s}) for s in sample_ids),
        encoding="utf-8",
    )
    (staging / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (staging / "unmapped_labels.log").write_text(
        "".join(f"{unmapped[t]}\t{t}\n" for t in sorted(unmapped)), encoding="utf-8"
    )
    return stats


# -- CLI -----------------------------------------------------------------------


def read_state(path: Path) -> str | None:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise FetchError(f"cannot read state file {path}: {e}") from e
    if not isinstance(state, dict) or set(state) != {"watermark"}:
        raise FetchError(f'state file {path} must be {{"watermark": ...}}')
    watermark = state["watermark"]
    if watermark is not None and not (isinstance(watermark, str) and watermark):
        raise FetchError(f"state file {path}: watermark must be a string or null")
    return watermark


def write_json_file(path: Path, data: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(path, _dump(data).encode("utf-8"))
    except OSError as e:
        raise FetchError(f"cannot write {path}: {e}") from e


def write_state(path: Path, watermark: str | None) -> None:
    write_json_file(path, {"watermark": watermark})


def client_from_env(transport: Transport | None) -> ExportClient:
    missing = [n for n in (FARM_URL_ENV, PIPELINE_KEY_ENV) if not os.environ.get(n)]
    if missing:
        raise FetchError(f"missing environment: {', '.join(missing)}")
    return ExportClient(
        os.environ[FARM_URL_ENV], os.environ[PIPELINE_KEY_ENV], transport
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch the training dataset.")
    sub = p.add_subparsers(dest="command", required=True)
    count = sub.add_parser("count", help="count label rows since the watermark")
    count.add_argument("--state", type=Path, help="stored state with the watermark")
    count.add_argument(
        "--out-state", type=Path, required=True, help="where to write the new watermark"
    )
    count.add_argument(
        "--out-sources",
        type=Path,
        help="where to write the changed source ids (the fixture gate's --ingested)",
    )
    build = sub.add_parser("build", help="full pull into a fresh dataset directory")
    build.add_argument("--out", type=Path, required=True, help="dataset directory")
    build.add_argument("--cache", type=Path, required=True, help="pose-blob cache")
    return p.parse_args(argv)


def main(
    argv: list[str] | None = None,
    transport: Transport | None = None,
    blob_getter: BlobGetter | None = None,
) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        client = client_from_env(transport)
        if args.command == "count":
            since = read_state(args.state) if args.state else None
            pull = client.pull(since)
            changed = sorted({row.source_id for row in pull.rows})
            write_state(args.out_state, pull.cursor)
            if args.out_sources:
                write_json_file(args.out_sources, changed)
            result: dict[str, Any] = {
                "new_labels": len(pull.rows),
                "changed_sources": len(changed),
            }
        else:
            blobs = BlobCache(args.cache, blob_getter or r2_getter_from_env())
            result = build_dataset(client.pull(), blobs, args.out)
    except FetchError as e:
        print(f"fetch_dataset: error: {e}", file=sys.stderr)
        return EXIT_ERROR
    print(json.dumps(result, indent=2, sort_keys=True))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
