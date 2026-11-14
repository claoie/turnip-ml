#!/usr/bin/env python3
"""Training dataset fetch: farm segment export + R2 TKS1 blobs -> train.py samples.

The farm exports training pose only as segments: short windows cut
around each labeled clip and re-targeted onto a standard body
(turnip-farm BACKEND_DESIGN.md §1.2.24, POSE_FORMAT.md §12;
TRAINING_DESIGN.md §1, §2). Each export entry is one segment with its
clip's labels and provenance; none carries a source, clip, account or
timestamp.

Subcommands
-----------
count --out-state FILE [--state FILE] [--out-segments FILE]
    Full pull of `GET /api/labels/export` (segments only, no blobs).
    Segment ids are random and the export's cursor is a page position,
    not a change marker, so what changed is read from the segment ids
    themselves: --state holds the ids of the last successful run's pull,
    and every id not in it is new. A re-cut (after any clip or label
    edit on its source) gives a segment a fresh id, so a relabel counts
    as new as well as an upload. Prints {"new_segments": N,
    "new_labels": M}: the new segments and their label count (the
    nightly trigger). --out-state gets this pull's ids as
    {"segment_ids": [...]}; the pipeline stores it only after the whole
    run succeeds. A state from before segments ({"watermark": ...}), or
    no --state, counts every segment as new. --out-segments writes the
    new segment ids as a sorted JSON array: the fixture gate's
    --ingested list. Taken after the dataset pull, it covers every new
    segment the challenger trained on.

build --out DIR --cache DIR
    Full pull of the export (never incremental: deletions, purges,
    quarantines and releases change what is training-eligible, and the
    export does not report deletions). Right after the pull, every
    cached blob whose segment is absent from it is deleted, so data
    deleted farm-side never outlives one full pull here. Then for each
    segment it fetches the TKS1 blob from R2, caches it under its
    r2_key, verifies it against `blob_sha256` on download and on every
    cache read, and decodes it (tks1.py). Writes into DIR, which must not
    exist yet (built in a temporary sibling directory, moved into place
    on success):
      samples/<segment_id>.json  train.py's sample format: sample_id (the
                                 segment id), fps 10, keypoints
                                 [frame][13][x, y, confidence] in TKS1
                                 joint order and torso lengths (missing
                                 joints and gap frames as zeros),
                                 split_bucket, one trick entry per
                                 canonical name on the clip window
                                 (window_start, window_end), and the
                                 label provenance (below)
      manifest.jsonl             {"sample_id", "source_id"} per sample,
                                 the split_dataset.py input. source_id is
                                 "bucket-<split_bucket>": the farm gives
                                 every segment of one source the same
                                 bucket, so splitting by it keeps a
                                 session on one side
      dataset_manifest.json      pinned taxonomy_version and its curated
                                 canonical names, the held-back names,
                                 counts and sorted sample ids
      unmapped_labels.log        "<count>\\t<token>" per unmapped label
                                 token, the taxonomy curation queue
    Raw labels are canonicalized at one pinned taxonomy version: the
    farm's current one, i.e. the greatest taxonomy_version in the export's
    taxonomy or labels, so a curation-only bump applies to every label at
    the next run (see canonicalize.py for the per-raw resolution).
    A label that canonicalizes to no name at all (only parentheticals or
    delimiters) contributes nothing; a segment left without names is
    dropped.

Rare-trick hold-back
--------------------
A trick ships, with its metrics, only once enough distinct people have
contributed it (TRAINING_DESIGN.md §6.2). The farm decides both numbers
(BACKEND_DESIGN.md §1.2.27, §7.3), and every export page carries them:
  trick_contributors   [{"canonical": name, "contributors": count}],
                       distinct contributors per canonical trick under
                       the current taxonomy; a trick missing from the
                       list has 0
  contributor_minimum  the count a trick needs: it trains iff
                       contributors >= contributor_minimum
Both are whole-export values, the same on every page. Read
conservatively all the same: a trick's count is its minimum over the
pages (absent from a page counts 0), the minimum is the greatest any
page gives, and a page without either field holds every trick back
(fail closed). Unmapped tokens have no count, so they are held back
too. A malformed field is an error.

A segment that carries any held-back name is held back whole: no sample
is written for it, so the trick has no name, train count or holdout
metric, and its context frames train as nothing either.
dataset_manifest.json records the held-back names (publish_model.py
refuses a model that names one); stdout carries only how many names and
segments were held back.

Provenance
----------
Every segment carries its clip's provenance (BACKEND_DESIGN.md §1.2.18,
TRAINING_DESIGN.md §1): `proposal_model_version` (null for the heuristic
detector or a manual clip), `suggested_names` (the names that model
pre-filled, or on a blind clip would have; [] when none) and
`suggestions_withheld` (the clip is in the blind sample), and each label
its `origin` ("typed", "vocabulary" or "model"). A sample carries them as
the farm sent them:
  tricks[].origins   the sorted distinct origins of the segment's raw
                     labels that canonicalized to this entry's name: one,
                     unless two raw labels with different origins merged
  clips              {segment_id: {"proposal_model_version",
                     "suggested_names", "suggestions_withheld"}};
                     suggested_names verbatim, in the farm's order, not
                     canonicalized
Nothing here reads them yet; TRAINING_DESIGN.md §6.4 is what will.

Environment (never argv: values would land in process lists and CI logs)
-----------
TURNIP_FARM_URL       farm base URL (https; http only for localhost)
TURNIP_PIPELINE_KEY   service key sent as a Bearer token
R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET
                      segment-blob bucket (build only; needs boto3)

Privacy: training data carries no account or source identifier. The farm
exports none, and a segment that carries a user_id or a source_id is
refused rather than passed along. Stdout and stderr carry counts only,
never raw labels, provenance values, segment ids or pose values; CI logs
of this public repository are public. The dataset and the blob cache
belong under data/ (gitignored) and must never be uploaded as a CI
artifact or cache.

Determinism: segments are re-sorted by id after the pull, JSON keys are
sorted, float32 pose values are written losslessly, and nothing carries a
timestamp, so the same export and blobs always produce byte-identical
files.

Fail closed: a malformed page or segment (including one carrying a
user_id or source_id), a duplicate segment id, a stuck or cycling
cursor, a frame_count that disagrees with the blob, a checksum mismatch
or an undecodable blob are errors (exit 2), never skipped. A segment is
malformed when: a field is missing; segment_id is not a lowercase UUID;
r2_key is not segments/<segment_id>.tks1.gz; blob_sha256 is not 64
lowercase hex; frame_count, window_start, window_end are not integers
with 0 <= window_start < window_end <= frame_count; split_bucket is not
0-9; labels is empty, unsorted or names a trick twice, or a label has a
blank trick_name, a taxonomy_version below 1 or an origin outside the
three; proposal_model_version is neither null nor a non-empty string;
suggested_names is not a list of non-empty strings; suggestions_withheld
is not a boolean; suggested names or a blind clip come without a
proposal_model_version (the farm's clips_suggestions_need_model CHECK);
or a label has origin "model" on a segment with no
proposal_model_version or with suggestions_withheld (both refused by
POST /api/clips, BACKEND_DESIGN.md §7.3).
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

import tks1
from canonicalize import Taxonomy, TaxonomyError

EXIT_OK = 0
EXIT_ERROR = 2

EXPORT_PATH = "/api/labels/export"
HTTP_TIMEOUT_S = 60.0
MAX_PAGE_BYTES = 64 * 1024 * 1024
FPS = 10
SPLIT_BUCKETS = 10

FARM_URL_ENV = "TURNIP_FARM_URL"
PIPELINE_KEY_ENV = "TURNIP_PIPELINE_KEY"
R2_ENV = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")

HEX64_RE = re.compile(r"[0-9a-f]{64}")
# Rare-trick hold-back (TRAINING_DESIGN.md §6.2): the export fields with
# the farm's per-trick count of distinct contributors and the count a
# trick needs before it trains and ships (BACKEND_DESIGN.md §1.2.27).
TRICK_CONTRIBUTORS_FIELD = "trick_contributors"
CONTRIBUTOR_MINIMUM_FIELD = "contributor_minimum"
# A label's origin (BACKEND_DESIGN.md §1.2.18): typed by hand, picked from
# the vocabulary's suggestions, or pre-filled by the trick model and kept.
ORIGINS = ("typed", "vocabulary", "model")
SEGMENT_FIELDS = frozenset(
    {
        "segment_id",
        "r2_key",
        "blob_sha256",
        "frame_count",
        "window_start",
        "window_end",
        "split_bucket",
        "proposal_model_version",
        "suggested_names",
        "suggestions_withheld",
        "labels",
    }
)
LABEL_FIELDS = frozenset({"trick_name", "origin", "taxonomy_version"})
# Fields that would tie a segment to a person or a session.
IDENTIFYING_FIELDS = ("user_id", "source_id")
# The only files the blob cache writes, under <cache>/segments/.
CACHE_ENTRY_RE = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
    r"\.tks1\.gz(\.tmp)?"
)
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")

Transport = Callable[[str, dict[str, str]], Any]
BlobGetter = Callable[[str], bytes]


class FetchError(Exception):
    """The fetch refuses to produce a dataset rather than guess."""


@dataclass(frozen=True)
class Label:
    trick_name: str
    origin: str
    taxonomy_version: int


@dataclass(frozen=True)
class Segment:
    segment_id: str
    r2_key: str
    blob_sha256: str
    frame_count: int
    window_start: int
    window_end: int
    split_bucket: int
    proposal_model_version: str | None
    suggested_names: tuple[str, ...]
    suggestions_withheld: bool
    labels: tuple[Label, ...]


@dataclass(frozen=True)
class ExportPull:
    segments: list[Segment]  # sorted by segment_id
    taxonomy: list[tuple[str, str, int]]
    # Distinct contributors per canonical name, the minimum over the pages
    # (module docstring, "Rare-trick hold-back").
    contributor_counts: dict[str, int]
    # The farm's hold-back threshold; None when a page lacked it, which
    # holds every trick back.
    contributor_minimum: int | None


# -- export ------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _hex64(value: Any) -> bool:
    return isinstance(value, str) and HEX64_RE.fullmatch(value) is not None


def is_segment_id(value: Any) -> bool:
    """A lowercase UUID, as the farm mints segment ids."""
    if not isinstance(value, str):
        return False
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


def _parse_label(i: int, j: int, label: Any) -> Label:
    where = f"export segment {i} label {j}"
    if not isinstance(label, dict) or set(label) != LABEL_FIELDS:
        raise FetchError(f"{where} must be {{trick_name, origin, taxonomy_version}}")
    name = label["trick_name"]
    if not isinstance(name, str) or not name.strip():
        raise FetchError(f"{where}: trick_name must be a non-empty string")
    version = label["taxonomy_version"]
    if not (_is_int(version) and version >= 1):
        raise FetchError(f"{where}: taxonomy_version must be an integer >= 1")
    # Errors name the field, never the value: a label is text.
    origin = label["origin"]
    if not isinstance(origin, str) or origin not in ORIGINS:
        raise FetchError(f"{where}: origin must be one of {', '.join(ORIGINS)}")
    return Label(name, origin, version)


def parse_segment(i: int, entry: Any) -> Segment:
    if not isinstance(entry, dict):
        raise FetchError(f"export segment {i} is not an object")
    for field in IDENTIFYING_FIELDS:
        if field in entry:
            raise FetchError(
                f"export segment {i} carries a {field}; training data must carry "
                "no account or source identifier"
            )
    if not SEGMENT_FIELDS <= set(entry):
        raise FetchError(f"export segment {i} is missing fields")
    segment_id = entry["segment_id"]
    if not is_segment_id(segment_id):
        raise FetchError(f"export segment {i}: segment_id is not a lowercase UUID")
    if entry["r2_key"] != f"segments/{segment_id}.tks1.gz":
        raise FetchError(f"export segment {i}: r2_key does not match its segment_id")
    if not _hex64(entry["blob_sha256"]):
        raise FetchError(
            f"export segment {i}: blob_sha256 is not 64 lowercase hex chars"
        )
    frames, start, end = (
        entry["frame_count"],
        entry["window_start"],
        entry["window_end"],
    )
    if not (
        _is_int(frames)
        and _is_int(start)
        and _is_int(end)
        and 0 <= start < end <= frames
    ):
        raise FetchError(
            f"export segment {i}: needs integers "
            "0 <= window_start < window_end <= frame_count"
        )
    bucket = entry["split_bucket"]
    if not (_is_int(bucket) and 0 <= bucket < SPLIT_BUCKETS):
        raise FetchError(f"export segment {i}: split_bucket must be 0-9")
    raw_labels = entry["labels"]
    if not isinstance(raw_labels, list) or not raw_labels:
        raise FetchError(f"export segment {i}: labels must be a non-empty list")
    labels = tuple(_parse_label(i, j, label) for j, label in enumerate(raw_labels))
    names = [label.trick_name for label in labels]
    if names != sorted(set(names)):
        raise FetchError(
            f"export segment {i}: labels must be sorted with no trick twice"
        )
    model_version = entry["proposal_model_version"]
    if model_version is not None and not (
        isinstance(model_version, str) and model_version
    ):
        raise FetchError(
            f"export segment {i}: proposal_model_version must be a non-empty "
            "string or null"
        )
    suggested = entry["suggested_names"]
    if not isinstance(suggested, list) or not all(
        isinstance(n, str) and n for n in suggested
    ):
        raise FetchError(
            f"export segment {i}: suggested_names must be a list of non-empty strings"
        )
    withheld = entry["suggestions_withheld"]
    if not isinstance(withheld, bool):
        raise FetchError(f"export segment {i}: suggestions_withheld must be a boolean")
    # The farm's clips_suggestions_need_model CHECK: suggested names and the
    # blind sample both need the model that proposed them.
    if model_version is None and (suggested or withheld):
        raise FetchError(
            f"export segment {i}: suggested_names or suggestions_withheld without "
            "a proposal_model_version"
        )
    # POST /api/clips refuses these (BACKEND_DESIGN.md §7.3): a kept
    # suggestion needs a model that proposed it and showed it.
    if any(label.origin == "model" for label in labels) and (
        model_version is None or withheld
    ):
        raise FetchError(
            f"export segment {i}: origin model on a clip whose model showed no names"
        )
    return Segment(
        segment_id=segment_id,
        r2_key=entry["r2_key"],
        blob_sha256=entry["blob_sha256"],
        frame_count=frames,
        window_start=start,
        window_end=end,
        split_bucket=bucket,
        proposal_model_version=model_version,
        suggested_names=tuple(suggested),
        suggestions_withheld=withheld,
        labels=labels,
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


def _is_count(n: Any) -> bool:
    return isinstance(n, int) and not isinstance(n, bool) and n >= 0


def parse_contributor_counts(page: dict[str, Any]) -> dict[str, int]:
    """A page's per-trick contributor counts; {} when the page has none."""
    entries = page.get(TRICK_CONTRIBUTORS_FIELD)
    if entries is None:
        return {}
    where = f"export {TRICK_CONTRIBUTORS_FIELD}"
    if not isinstance(entries, list):
        raise FetchError(f"{where} must be an array")
    counts: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"canonical", "contributors"}:
            raise FetchError(f"{where} entries must be {{canonical, contributors}}")
        name, n = entry["canonical"], entry["contributors"]
        if not (isinstance(name, str) and name and _is_count(n)):
            raise FetchError(f"{where} entries need a name and a count >= 0")
        if name in counts:
            raise FetchError(f"{where} lists a trick twice")
        counts[name] = n
    return counts


def parse_contributor_minimum(page: dict[str, Any]) -> int | None:
    """A page's hold-back threshold; None when the page has none."""
    minimum = page.get(CONTRIBUTOR_MINIMUM_FIELD)
    if minimum is None:
        return None
    if not _is_count(minimum) or minimum < 1:
        raise FetchError(f"export {CONTRIBUTOR_MINIMUM_FIELD} must be a count >= 1")
    return int(minimum)


def _min_counts(pages: list[dict[str, int]]) -> dict[str, int]:
    """Per name, the minimum over the pages; absent from a page counts 0."""
    names = set().union(*pages) if pages else set()
    return {n: min(page.get(n, 0) for page in pages) for n in sorted(names)}


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

    def pull(self) -> ExportPull:
        """The whole export, page by page."""
        segments: list[Segment] = []
        taxonomy: set[tuple[str, str, int]] = set()
        counts: list[dict[str, int]] = []
        minimums: list[int | None] = []
        cursor: str | None = None
        seen: set[str | None] = {None}
        while True:
            query = (
                ""
                if cursor is None
                else "?" + urllib.parse.urlencode({"since": cursor})
            )
            page = self.transport(self.base_url + EXPORT_PATH + query, self.headers)
            if not isinstance(page, dict) or not isinstance(page.get("segments"), list):
                raise FetchError("export page must be an object with a segments array")
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
            offset = len(segments)
            segments.extend(
                parse_segment(offset + i, s) for i, s in enumerate(page["segments"])
            )
            taxonomy.update(parse_taxonomy(i, t) for i, t in enumerate(page_taxonomy))
            counts.append(parse_contributor_counts(page))
            minimums.append(parse_contributor_minimum(page))
            if not page["segments"] or next_cursor is None:
                break
            if next_cursor in seen:
                raise FetchError("export cursor did not advance")
            seen.add(next_cursor)
            cursor = next_cursor
        ids = [s.segment_id for s in segments]
        if len(set(ids)) != len(ids):
            raise FetchError("export returned a segment twice")
        known = [m for m in minimums if m is not None]
        minimum = max(known) if known and len(known) == len(minimums) else None
        return ExportPull(
            sorted(segments, key=lambda s: s.segment_id),
            sorted(taxonomy),
            _min_counts(counts),
            minimum,
        )


# -- segment blobs -------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


class BlobCache:
    """Segment blobs keyed by r2_key; verified on download and on every read."""

    def __init__(self, cache_dir: Path, get: BlobGetter):
        self.cache_dir = cache_dir
        self.get = get

    def fetch(self, r2_key: str, sha256: str) -> bytes:
        # r2_key is validated as segments/<uuid>.tks1.gz, so this stays
        # inside the cache directory.
        path = self.cache_dir / r2_key
        try:
            if path.exists():
                data = path.read_bytes()
                if _sha256(data) == sha256:
                    return data
                # Corrupt entry: the bucket is the truth.
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
        """Delete cached blobs whose segment is not in `keep`; returns the count.

        Only files this cache writes are touched (segments/<uuid>.tks1.gz and
        leftover .tmp siblings), never anything else in the directory.
        """
        directory = self.cache_dir / "segments"
        pruned = 0
        try:
            entries = sorted(directory.iterdir()) if directory.is_dir() else []
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
    client, bucket, errors = r2_client_from_env("build (segment blobs)")

    def get(r2_key: str) -> bytes:
        try:
            body: bytes = client.get_object(Bucket=bucket, Key=r2_key)["Body"].read()
        except errors as e:
            raise FetchError(f"{r2_key}: R2 download failed: {type(e).__name__}") from e
        return body

    return get


# -- dataset -------------------------------------------------------------------


def _provenance(segment: Segment) -> dict[str, Any]:
    """The clip provenance a segment carries, as the sample keeps it."""
    return {
        "proposal_model_version": segment.proposal_model_version,
        "suggested_names": list(segment.suggested_names),
        "suggestions_withheld": segment.suggestions_withheld,
    }


def _dump(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n"


def build_dataset(pull: ExportPull, blobs: BlobCache, out_dir: Path) -> dict[str, Any]:
    """Write the dataset into out_dir (must not exist). Returns the stats."""
    # Prune first: an empty pull means every cached segment is gone, which
    # is exactly when deleting matters.
    pruned = blobs.prune({s.segment_id for s in pull.segments})
    if not pull.segments:
        raise FetchError("the export is empty: nothing to train on")
    if out_dir.exists():
        raise FetchError(f"{out_dir} already exists; use a fresh directory per run")
    # The farm's current version: the greatest in the taxonomy (or labels),
    # so curation applies without waiting for labels at the new version.
    pinned = max(
        [label.taxonomy_version for s in pull.segments for label in s.labels]
        + [version for _, _, version in pull.taxonomy]
    )
    try:
        taxonomy = Taxonomy(pull.taxonomy, pinned)
    except TaxonomyError as e:
        raise FetchError(f"taxonomy: {e}") from e

    try:
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{out_dir.name}.partial-", dir=out_dir.parent)
        )
    except OSError as e:
        raise FetchError(f"cannot create {out_dir}: {e}") from e
    try:
        stats = _write_dataset(pull, taxonomy, blobs, staging)
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
    pull: ExportPull, taxonomy: Taxonomy, blobs: BlobCache, staging: Path
) -> dict[str, Any]:
    """Write samples one at a time (a dataset need not fit in memory)."""
    (staging / "samples").mkdir()
    unmapped: Counter[str] = Counter()
    dropped = 0
    held_back = 0
    held_back_names: set[str] = set()
    sample_ids: list[str] = []
    buckets: dict[str, int] = {}
    for segment in pull.segments:
        # canonical name -> the origins of the raw labels that gave it.
        names: dict[str, set[str]] = {}
        for label in segment.labels:
            canonical, tokens = taxonomy.canonicalize(label.trick_name)
            for n in canonical:
                names.setdefault(n, set()).add(label.origin)
            unmapped.update(tokens)
        if not names:
            dropped += 1
            continue
        rare = {
            n
            for n in names
            if pull.contributor_minimum is None
            or pull.contributor_counts.get(n, 0) < pull.contributor_minimum
        }
        if rare:
            held_back_names |= rare
            held_back += 1
            continue
        data = blobs.fetch(segment.r2_key, segment.blob_sha256)
        try:
            frames = tks1.decode_gz(data)
        except tks1.Tks1Error as e:
            raise FetchError(f"segment {segment.segment_id}: {e}") from e
        if len(frames) != segment.frame_count:
            raise FetchError(
                f"segment {segment.segment_id}: blob has {len(frames)} frames, "
                f"the export says {segment.frame_count}"
            )
        # One entry per canonical name on the clip window: a combo is several
        # entries on one window, read through evaluate.ground_truth_segments.
        tricks = [
            {
                "clip_id": segment.segment_id,
                "name": n,
                "origins": sorted(names[n]),
                "start_frame": segment.window_start,
                "end_frame": segment.window_end,
            }
            for n in sorted(names)
        ]
        sample = {
            "sample_id": segment.segment_id,
            "fps": FPS,
            "keypoints": frames,
            "split_bucket": segment.split_bucket,
            "tricks": tricks,
            "clips": {segment.segment_id: _provenance(segment)},
        }
        path = staging / "samples" / f"{segment.segment_id}.json"
        path.write_text(_dump(sample), encoding="utf-8")
        sample_ids.append(segment.segment_id)
        buckets[segment.segment_id] = segment.split_bucket

    stats = {
        "samples": len(sample_ids),
        "segments": len(pull.segments),
        "segments_dropped_without_names": dropped,
        "segments_held_back": held_back,
        "names_held_back": len(held_back_names),
        "label_rows": sum(len(s.labels) for s in pull.segments),
        "taxonomy_version": taxonomy.version,
        "unmapped_tokens": len(unmapped),
    }
    manifest = {
        **stats,
        # The curated names at the pin: what the farm's vocabulary for this
        # taxonomy_version holds. Unmapped tokens train only if the farm
        # counts them, and a model that names one is not published
        # (publish_model.py): the manifest is public and the token is a
        # contributor's free text.
        "canonical_names": taxonomy.canonical_names,
        # Names with fewer than the farm's contributor_minimum contributors
        # (null: no minimum, so every name): never trained or published
        # (TRAINING_DESIGN.md §6.2).
        "held_back_names": sorted(held_back_names),
        "contributor_minimum": pull.contributor_minimum,
        "sample_ids": sample_ids,
    }
    (staging / "manifest.jsonl").write_text(
        "".join(
            _dump({"sample_id": s, "source_id": f"bucket-{buckets[s]}"})
            for s in sample_ids
        ),
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


def read_state(path: Path) -> set[str]:
    """The segment ids of the last successful run's pull.

    A state from before segments ({"watermark": ...}) names no segment, so
    every segment counts as new once, on the first run after the switch.
    """
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise FetchError(f"cannot read state file {path}: {e}") from e
    if isinstance(state, dict) and set(state) == {"watermark"}:
        return set()
    if not isinstance(state, dict) or set(state) != {"segment_ids"}:
        raise FetchError(f'state file {path} must be {{"segment_ids": [...]}}')
    ids = state["segment_ids"]
    if not isinstance(ids, list) or not all(is_segment_id(s) for s in ids):
        raise FetchError(f"state file {path}: segment_ids must be segment ids")
    return set(ids)


def write_json_file(path: Path, data: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(path, _dump(data).encode("utf-8"))
    except OSError as e:
        raise FetchError(f"cannot write {path}: {e}") from e


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
    count = sub.add_parser("count", help="count segments new since the last run")
    count.add_argument("--state", type=Path, help="the last run's segment ids")
    count.add_argument(
        "--out-state", type=Path, required=True, help="where to write this pull's ids"
    )
    count.add_argument(
        "--out-segments",
        type=Path,
        help="where to write the new segment ids (the fixture gate's --ingested)",
    )
    build = sub.add_parser("build", help="full pull into a fresh dataset directory")
    build.add_argument("--out", type=Path, required=True, help="dataset directory")
    build.add_argument("--cache", type=Path, required=True, help="segment-blob cache")
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
            previous = read_state(args.state) if args.state else set()
            pull = client.pull()
            new = [s for s in pull.segments if s.segment_id not in previous]
            write_json_file(
                args.out_state, {"segment_ids": [s.segment_id for s in pull.segments]}
            )
            if args.out_segments:
                write_json_file(args.out_segments, [s.segment_id for s in new])
            result: dict[str, Any] = {
                "new_segments": len(new),
                "new_labels": sum(len(s.labels) for s in new),
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
