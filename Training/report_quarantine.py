#!/usr/bin/env python3
"""Send the fixture gate's quarantine rows to the farm: POST /api/quarantine.

turnip-ml#51, docs/TRAINING_DESIGN.md §7 step 2. On a fixture regression
fixture_gate.py writes quarantine.json, one row per source changed since
the watermark. This puts those sources under an open quarantine farm-side,
so the export leaves them out until the maintainer releases or purges them
(farm master plan §6). The workflow's farm job runs it after a regression
and advances the watermark only once it succeeds; nothing else sends.

Inputs
------
--quarantine FILE   the gate's quarantine.json: a JSON array of
                    {"source_id", "reason", "metrics"} rows.

Environment (never argv: values would land in process lists and CI logs)
-----------
TURNIP_FARM_URL, TURNIP_PIPELINE_KEY   the farm and its service key

Requests
--------
Every row is validated and every body serialized before the first
request, so a malformed file sends nothing. A row has exactly source_id
(a farm source id: 64 lowercase hex), reason (non-empty, at most 1000
characters) and metrics (an object or null), the rules of the farm's
readQuarantineInput (turnip-farm src/quarantine.ts). Any other key is
refused, so nothing account-shaped can ride along, and so is a source
listed twice. Rows sharing a reason and metrics go in one body
{source_ids, reason, metrics}, ids sorted, at most 10000 per body (the
farm's limit); bodies go out in a fixed order. An empty array sends
nothing.

A body counts as sent only when the farm answers 2xx with
{quarantined, already_open, unknown} accounting for every id it was
sent, each exactly once. already_open ids were quarantined before: the
farm keeps one open row per source, so a resubmission is a no-op and a
re-run after a failure is safe. unknown ids were deleted by their owner
after the export pull; there is nothing left to quarantine.

Failure
-------
A transport failure or a 5xx is retried, up to ATTEMPTS tries in all
(the farm may have committed before the answer was lost; the retry then
answers already_open). Any other answer that is not 2xx (401: fix the
pipeline key; 422: the body broke the farm's rules; a 3xx) fails at once,
and so does a 2xx that does not account for the ids sent. The first body
that fails stops the run; the ones before it stay sent, and a re-run
sends them again harmlessly. Requests never follow a redirect: one would
carry the pipeline key to another host.

Exit codes: 0 = every body sent (or nothing to send), 2 = error (fail
closed: the workflow then keeps the watermark held).

Privacy: stdout and stderr carry counts, HTTP statuses and farm error
codes only, never URLs, keys, source ids or the farm's error messages
(a 422's details name the offending id).
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fetch_dataset import (
    FARM_URL_ENV,
    HEX64_RE,
    PIPELINE_KEY_ENV,
    farm_url_allowed,
    farm_urlopen,
)

EXIT_OK = 0
EXIT_ERROR = 2

QUARANTINE_PATH = "/api/quarantine"
# readQuarantineInput's limits.
MAX_SOURCE_IDS = 10000
MAX_REASON_LEN = 1000
ROW_KEYS = frozenset({"source_id", "reason", "metrics"})
ANSWER_KEYS = ("quarantined", "already_open", "unknown")
# The farm checks and inserts every id of a body in one transaction.
HTTP_TIMEOUT_S = 120.0
# The answer lists every id sent back: about 70 bytes each at 10000 ids.
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
ATTEMPTS = 3
RETRY_DELAYS_S = (10.0, 30.0)  # before the second and the third try
# Farm error codes are constants such as VALIDATION_ERROR; anything else
# is not printed.
ERROR_CODE_RE = re.compile(r"[A-Z][A-Z_]{0,63}")

# (method, url, headers, body) -> (HTTP status, parsed JSON body or None)
Transport = Callable[[str, str, dict[str, str], bytes | None], tuple[int, Any]]
Sleep = Callable[[float], None]


class QuarantineError(Exception):
    """The report refuses to go on rather than guess."""


class Transient(QuarantineError):
    """A failure a resubmission may cure: a transport failure or a 5xx."""


# -- inputs --------------------------------------------------------------------


def _js_length(text: str) -> int:
    """String.length in JavaScript: UTF-16 code units, as the farm counts."""
    return len(text.encode("utf-16-le")) // 2


def load_rows(path: Path) -> list[dict[str, Any]]:
    """The gate's rows, validated by the farm's rules. Errors name no id."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as e:
        raise QuarantineError(f"cannot read {path}: {type(e).__name__}") from e
    if not isinstance(data, list):
        raise QuarantineError(f"{path} must contain a JSON array of rows")
    seen: set[str] = set()
    for i, row in enumerate(data):
        if not isinstance(row, dict) or set(row) != ROW_KEYS:
            raise QuarantineError(
                f"row {i} must have exactly the keys source_id, reason and metrics"
            )
        source_id, reason, metrics = row["source_id"], row["reason"], row["metrics"]
        if not isinstance(source_id, str) or HEX64_RE.fullmatch(source_id) is None:
            raise QuarantineError(
                f"row {i}: source_id must be a farm source id (64 lowercase hex)"
            )
        if source_id in seen:
            raise QuarantineError(f"row {i}: the source is listed twice")
        seen.add(source_id)
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or _js_length(reason) > MAX_REASON_LEN
        ):
            raise QuarantineError(
                f"row {i}: reason must be a non-empty string of at most "
                f"{MAX_REASON_LEN} characters"
            )
        if metrics is not None and not isinstance(metrics, dict):
            raise QuarantineError(f"row {i}: metrics must be a JSON object or null")
    return data


def _encode(data: Any) -> bytes:
    try:
        text = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, RecursionError) as e:
        raise QuarantineError(f"cannot serialize a request: {type(e).__name__}") from e
    return text.encode("utf-8")


def build_requests(rows: list[dict[str, Any]]) -> list[tuple[list[str], bytes]]:
    """(source ids, serialized body) per request, in a fixed order."""
    groups: dict[bytes, tuple[str, Any, list[str]]] = {}
    for row in rows:
        key = _encode([row["reason"], row["metrics"]])
        group = groups.setdefault(key, (row["reason"], row["metrics"], []))
        group[2].append(row["source_id"])
    requests = []
    for key in sorted(groups):
        reason, metrics, ids = groups[key]
        ids = sorted(ids)
        for start in range(0, len(ids), MAX_SOURCE_IDS):
            chunk = ids[start : start + MAX_SOURCE_IDS]
            body = {"source_ids": chunk, "reason": reason, "metrics": metrics}
            requests.append((chunk, _encode(body)))
    return requests


# -- farm ----------------------------------------------------------------------


def http_json(
    method: str, url: str, headers: dict[str, str], body: bytes | None
) -> tuple[int, Any]:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        try:
            with farm_urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
                status, raw = response.status, response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as e:
            status, raw = e.code, e.read(MAX_RESPONSE_BYTES + 1)
    except (OSError, http.client.HTTPException, ValueError) as e:
        # Report the type only: the farm URL is configuration.
        raise Transient(f"farm request failed: {type(e).__name__}") from e
    if len(raw) > MAX_RESPONSE_BYTES:
        raise QuarantineError(f"farm response exceeds {MAX_RESPONSE_BYTES} bytes")
    try:
        return status, json.loads(raw) if raw else None
    except (ValueError, RecursionError):
        return status, None


def _error_code(body: Any) -> str | None:
    error = body.get("error") if isinstance(body, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    if isinstance(code, str) and ERROR_CODE_RE.fullmatch(code):
        return code
    return None


def check_answer(sent: list[str], answer: Any) -> dict[str, int]:
    """Counts per answer list, once the answer accounts for every id sent."""
    if not isinstance(answer, dict):
        raise QuarantineError(f"POST {QUARANTINE_PATH} answered without a JSON object")
    counts: dict[str, int] = {}
    accounted: list[str] = []
    for key in ANSWER_KEYS:
        ids = answer.get(key)
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise QuarantineError(f"POST {QUARANTINE_PATH} answered without {key}")
        counts[key] = len(ids)
        accounted.extend(ids)
    # `sent` holds no duplicates, so equal sizes and sets mean each id once.
    if len(accounted) != len(sent) or set(accounted) != set(sent):
        raise QuarantineError(
            f"POST {QUARANTINE_PATH} answered for {len(accounted)} sources, "
            f"not the {len(sent)} sent"
        )
    return counts


class FarmClient:
    def __init__(
        self,
        base_url: str,
        key: str,
        transport: Transport | None = None,
        sleep: Sleep = time.sleep,
    ):
        if not farm_url_allowed(base_url):
            raise QuarantineError(f"{FARM_URL_ENV} must be an https:// URL")
        if not key:
            raise QuarantineError(f"{PIPELINE_KEY_ENV} is empty")
        self.base_url = base_url.rstrip("/")
        self.key = key
        self.transport = transport or http_json
        self.sleep = sleep

    def _post(self, ids: list[str], data: bytes) -> dict[str, int]:
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "turnip-ml-quarantine",
        }
        status, answer = self.transport(
            "POST", self.base_url + QUARANTINE_PATH, headers, data
        )
        if 200 <= status < 300:
            return check_answer(ids, answer)
        code = _error_code(answer)
        failed = f"POST {QUARANTINE_PATH} failed: HTTP {status}" + (
            f" {code}" if code else ""
        )
        if status >= 500:
            raise Transient(failed)
        raise QuarantineError(failed)

    def quarantine(self, ids: list[str], data: bytes) -> dict[str, int]:
        """POST one body, retrying what a resubmission may cure."""
        attempt = 1
        while True:
            try:
                return self._post(ids, data)
            except Transient as e:
                if attempt == ATTEMPTS:
                    raise
                print(
                    f"report_quarantine: {e} (try {attempt} of {ATTEMPTS}); retrying",
                    file=sys.stderr,
                )
                self.sleep(RETRY_DELAYS_S[attempt - 1])
                attempt += 1


def report(farm: FarmClient, requests: list[tuple[list[str], bytes]]) -> dict[str, int]:
    """Send every request in order; totals per answer list."""
    totals = dict.fromkeys(ANSWER_KEYS, 0)
    for n, (ids, data) in enumerate(requests, start=1):
        try:
            counts = farm.quarantine(ids, data)
        except QuarantineError as e:
            raise QuarantineError(
                f"{e} (request {n} of {len(requests)}; {n - 1} sent before it)"
            ) from e
        for key in ANSWER_KEYS:
            totals[key] += counts[key]
    return totals


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--quarantine",
        required=True,
        type=Path,
        help="the fixture gate's quarantine.json",
    )
    return p.parse_args(argv)


def main(
    argv: list[str] | None = None,
    transport: Transport | None = None,
    sleep: Sleep = time.sleep,
) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        requests = build_requests(load_rows(args.quarantine))
        if not requests:
            print("report_quarantine: nothing to quarantine (0 sources)")
            return EXIT_OK
        missing = [n for n in (FARM_URL_ENV, PIPELINE_KEY_ENV) if not os.environ.get(n)]
        if missing:
            raise QuarantineError(f"missing environment: {', '.join(missing)}")
        farm = FarmClient(
            os.environ[FARM_URL_ENV], os.environ[PIPELINE_KEY_ENV], transport, sleep
        )
        totals = report(farm, requests)
    except QuarantineError as e:
        print(f"report_quarantine: error: {e}", file=sys.stderr)
        return EXIT_ERROR
    sources = sum(totals.values())
    print(
        f"report_quarantine: {sources} sources in {len(requests)} request(s): "
        f"{totals['quarantined']} quarantined, {totals['already_open']} already "
        f"open, {totals['unknown']} unknown (deleted since the export pull)"
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
