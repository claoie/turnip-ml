"""Unit tests for Training/report_quarantine.py (turnip-ml#51).

The fixture gate's quarantine.json -> POST /api/quarantine against a fake
farm that validates like turnip-farm's readQuarantineInput and keeps one
open quarantine per source, like its data_quarantine_open_uniq index.
Hermetic: no network beyond a local HTTP stand-in, no secrets.

Run: python -m unittest discover -s Training/tests -t .
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fixture_gate
import report_quarantine

FARM = "https://farm.example"
KEY = "pipeline-secret-for-tests"
HEX64 = re.compile(r"[0-9a-f]{64}")
BEARER = re.compile(r"^Bearer (\S+)$", re.IGNORECASE)
REASON = "fixture regression: challenger composite 0.400000 < champion 0.900000"
METRICS = {"delta": -0.5, "fixture_set": "f" * 64, "tolerance": 0.01}


def source_id(n: int) -> str:
    return hashlib.sha256(f"source-{n}".encode()).hexdigest()


def ids(count: int) -> list[str]:
    return [source_id(n) for n in range(count)]


def rows(
    sources: list[str], reason: str = REASON, metrics: Any = METRICS
) -> list[dict[str, Any]]:
    return fixture_gate.quarantine_rows(sources, reason, metrics)


CODES = {401: "UNAUTHENTICATED", 404: "NOT_FOUND", 422: "VALIDATION_ERROR"}


def error(status: int, **details: Any) -> tuple[int, Any]:
    code = CODES.get(status, "INTERNAL")
    body: dict[str, Any] = {"code": code, "message": f"farm says {code}"}
    if details:
        body["details"] = details
    return status, {"error": body}


# -- the fake farm ---------------------------------------------------------------


def js_length(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def farm_validation_error(body: Any) -> str | None:
    """turnip-farm src/quarantine.ts readQuarantineInput, rule for rule."""
    if not isinstance(body, dict):
        return "body must be a JSON object"
    source_ids, reason = body.get("source_ids"), body.get("reason")
    if not isinstance(source_ids, list) or not 0 < len(source_ids) <= 10000:
        return "source_ids must be a non-empty array"
    if not all(isinstance(i, str) and HEX64.fullmatch(i) for i in source_ids):
        return "source_ids must be 64-char lowercase hex video ids"
    if not isinstance(reason, str) or not reason.strip() or js_length(reason) > 1000:
        return "reason must be a non-empty string"
    metrics = body.get("metrics")
    if metrics is not None and not isinstance(metrics, dict):
        return "metrics must be a JSON object or null"
    return None


class FakeFarm:
    """POST /api/quarantine as turnip-farm serves it."""

    def __init__(self, known: list[str] | None = None):
        # Sources the farm still has; None means every well-formed id.
        self.known = None if known is None else set(known)
        self.open: dict[str, tuple[str, Any]] = {}
        self.posts: list[dict[str, Any]] = []
        self.calls = 0
        # Statuses answered before the farm does anything, one per POST.
        self.fail: list[int] = []
        # Every POST fails with a 503 once this many bodies have landed.
        self.down_after: int | None = None
        # Insert, then lose the answer (a timeout after commit).
        self.drop_next_answer = False
        # Rewrites a well-formed answer, to model a farm that drifted.
        self.tamper: Any = None

    def transport(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> tuple[int, Any]:
        self.calls += 1
        if method != "POST" or url != FARM + "/api/quarantine":
            return error(404)
        match = BEARER.match(headers.get("Authorization", "").strip())
        if match is None or match.group(1) != KEY:
            return error(401)
        if self.fail:
            # A 422's details name the offending id, as the farm's do.
            return error(self.fail.pop(0), id=source_id(0))
        if self.down_after is not None and len(self.posts) >= self.down_after:
            return error(503)
        try:
            payload = json.loads(body or b"")
        except ValueError:
            return error(422)
        if farm_validation_error(payload):
            return error(422)
        if set(payload) - {"source_ids", "reason", "metrics"}:
            raise AssertionError(f"unknown keys sent: {set(payload)}")
        self.posts.append(payload)
        answer: dict[str, list[str]] = {
            "quarantined": [],
            "already_open": [],
            "unknown": [],
        }
        for sid in dict.fromkeys(payload["source_ids"]):
            if self.known is not None and sid not in self.known:
                answer["unknown"].append(sid)
            elif sid in self.open:
                answer["already_open"].append(sid)
            else:
                self.open[sid] = (payload["reason"].strip(), payload.get("metrics"))
                answer["quarantined"].append(sid)
        if self.drop_next_answer:
            self.drop_next_answer = False
            raise report_quarantine.Transient("farm request failed: TimeoutError")
        if self.tamper is not None:
            return 200, self.tamper(answer)
        return 200, answer


# -- tests -----------------------------------------------------------------------


class ReportTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = self.enterContext(tempfile.TemporaryDirectory(prefix="quarantine-"))
        self.path = Path(tmp) / "quarantine.json"
        self.stdout = self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.stderr = self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        env = {"TURNIP_FARM_URL": FARM, "TURNIP_PIPELINE_KEY": KEY}
        self.enterContext(mock.patch.dict(os.environ, env))
        self.farm = FakeFarm()
        self.sleeps: list[float] = []

    def write(self, data: Any) -> None:
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    def main(self, farm: FakeFarm | None = None) -> int:
        return report_quarantine.main(
            ["--quarantine", str(self.path)],
            transport=(farm or self.farm).transport,
            sleep=self.sleeps.append,
        )

    def logs(self) -> str:
        return self.stdout.getvalue() + self.stderr.getvalue()

    def clear_logs(self) -> None:
        for stream in (self.stdout, self.stderr):
            stream.seek(0)
            stream.truncate()

    def assertLogsArePrivate(self) -> None:
        logs = self.logs()
        self.assertIsNone(HEX64.search(logs), "a source id reached the log")
        for secret in (KEY, FARM, "farm says"):
            self.assertNotIn(secret, logs)


class RoundTripTest(ReportTestCase):
    def test_rows_become_one_request(self) -> None:
        sources = ids(3)
        self.write(rows(sources))
        self.assertEqual(self.main(), 0, self.stderr.getvalue())
        self.assertEqual(len(self.farm.posts), 1)
        self.assertEqual(
            self.farm.posts[0],
            {"source_ids": sorted(sources), "reason": REASON, "metrics": METRICS},
        )
        self.assertEqual(set(self.farm.open), set(sources))
        self.assertIn(
            "3 sources in 1 request(s): 3 quarantined", self.stdout.getvalue()
        )
        self.assertLogsArePrivate()

    def test_the_gates_own_output_is_sent(self) -> None:
        out = self.path.parent / "gate"
        champion, challenger, ingested = (
            self.path.parent / name
            for name in ("champion.json", "challenger.json", "ingested.json")
        )
        for path, score in ((champion, 0.9), (challenger, 0.4)):
            path.write_text(
                json.dumps(
                    {
                        "fixture_set": "f" * 64,
                        "segment_iou": score,
                        "name_accuracy": score,
                    }
                )
            )
        ingested.write_text(json.dumps(ids(4)))
        _, code = fixture_gate.run(
            champion,
            challenger,
            ingested,
            out,
            0.01,
            self.path.parent / "registry.json",
        )
        self.assertEqual(code, fixture_gate.EXIT_REGRESSION)
        self.path = out / fixture_gate.QUARANTINE_FILE
        self.assertEqual(self.main(), 0, self.stderr.getvalue())
        self.assertEqual(set(self.farm.open), set(ids(4)))
        reason, metrics = self.farm.open[source_id(0)]
        self.assertTrue(reason.startswith("fixture regression"))
        self.assertEqual(metrics["delta"], -0.5)

    def test_chunks_at_the_farms_limit(self) -> None:
        sources = ids(2 * report_quarantine.MAX_SOURCE_IDS + 1)
        self.write(rows(sources))
        self.assertEqual(self.main(), 0, self.stderr.getvalue())
        sent = [post["source_ids"] for post in self.farm.posts]
        self.assertEqual([len(chunk) for chunk in sent], [10000, 10000, 1])
        self.assertEqual([i for chunk in sent for i in chunk], sorted(sources))
        self.assertEqual(len(self.farm.open), len(sources))
        self.assertIn("20001 sources in 3 request(s)", self.stdout.getvalue())

    def test_a_full_chunk_is_one_request(self) -> None:
        self.write(rows(ids(report_quarantine.MAX_SOURCE_IDS)))
        self.assertEqual(self.main(), 0, self.stderr.getvalue())
        self.assertEqual(len(self.farm.posts), 1)

    def test_each_reason_and_metrics_gets_its_own_body(self) -> None:
        sources = ids(6)
        self.write(
            rows(sources[:2])
            + rows(sources[2:4], reason="another reason")
            + rows(sources[4:], metrics=None)
        )
        self.assertEqual(self.main(), 0, self.stderr.getvalue())
        bodies = sorted(
            (p["reason"], json.dumps(p["metrics"]), p["source_ids"])
            for p in self.farm.posts
        )
        self.assertEqual(
            bodies,
            sorted(
                [
                    (REASON, json.dumps(METRICS), sorted(sources[:2])),
                    ("another reason", json.dumps(METRICS), sorted(sources[2:4])),
                    (REASON, "null", sorted(sources[4:])),
                ]
            ),
        )

    def test_requests_are_deterministic(self) -> None:
        sources = ids(5)
        self.write(list(reversed(rows(sources))))
        first = report_quarantine.build_requests(report_quarantine.load_rows(self.path))
        self.write(rows(sources))
        again = report_quarantine.build_requests(report_quarantine.load_rows(self.path))
        self.assertEqual(first, again)

    def test_empty_quarantine_sends_nothing(self) -> None:
        self.write([])
        with mock.patch.dict(os.environ, {"TURNIP_PIPELINE_KEY": ""}):
            self.assertEqual(self.main(), 0, self.stderr.getvalue())
        self.assertEqual(self.farm.calls, 0)
        self.assertIn("nothing to quarantine", self.stdout.getvalue())

    def test_unknown_sources_are_reported_not_fatal(self) -> None:
        sources = ids(3)
        self.write(rows(sources))
        farm = FakeFarm(known=sources[:2])
        self.assertEqual(self.main(farm), 0, self.stderr.getvalue())
        self.assertIn(
            "2 quarantined, 0 already open, 1 unknown", self.stdout.getvalue()
        )


class IdempotenceTest(ReportTestCase):
    def test_resubmission_changes_nothing(self) -> None:
        sources = ids(4)
        self.write(rows(sources))
        self.assertEqual(self.main(), 0)
        before = dict(self.farm.open)
        self.assertEqual(self.main(), 0, self.stderr.getvalue())
        self.assertEqual(self.farm.open, before)
        self.assertIn("0 quarantined, 4 already open", self.stdout.getvalue())

    def test_a_rerun_after_a_failed_chunk_finishes_the_job(self) -> None:
        sources = ids(report_quarantine.MAX_SOURCE_IDS + 1)
        self.write(rows(sources))
        # The first chunk lands; the second fails on every try.
        self.farm.down_after = 1
        self.assertEqual(self.main(), 2)
        self.assertEqual(len(self.farm.open), report_quarantine.MAX_SOURCE_IDS)
        self.assertIn(
            "HTTP 503 INTERNAL (request 2 of 2; 1 sent", self.stderr.getvalue()
        )

        self.farm.down_after = None
        self.clear_logs()
        self.assertEqual(self.main(), 0, self.stderr.getvalue())
        self.assertEqual(set(self.farm.open), set(sources))
        self.assertIn("1 quarantined, 10000 already open", self.stdout.getvalue())

    def test_a_lost_answer_is_retried_into_already_open(self) -> None:
        sources = ids(3)
        self.write(rows(sources))
        self.farm.drop_next_answer = True
        self.assertEqual(self.main(), 0, self.stderr.getvalue())
        self.assertEqual(len(self.farm.posts), 2)
        self.assertEqual(self.sleeps, [report_quarantine.RETRY_DELAYS_S[0]])
        self.assertIn("0 quarantined, 3 already open", self.stdout.getvalue())
        self.assertIn("TimeoutError (try 1 of 3); retrying", self.stderr.getvalue())


class FailureTest(ReportTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.sources = ids(3)
        self.write(rows(self.sources))

    def test_5xx_is_retried(self) -> None:
        self.farm.fail = [503]
        self.assertEqual(self.main(), 0, self.stderr.getvalue())
        self.assertEqual(set(self.farm.open), set(self.sources))
        self.assertEqual(self.sleeps, [10.0])

    def test_5xx_on_every_try_is_an_error(self) -> None:
        self.farm.fail = [500, 502, 503]
        self.assertEqual(self.main(), 2)
        self.assertEqual(self.farm.calls, report_quarantine.ATTEMPTS)
        self.assertEqual(self.sleeps, list(report_quarantine.RETRY_DELAYS_S))
        self.assertIn(
            "error: POST /api/quarantine failed: HTTP 503", self.stderr.getvalue()
        )
        self.assertEqual(self.farm.open, {})
        self.assertLogsArePrivate()

    def test_422_fails_at_once_without_echoing_the_farm(self) -> None:
        self.farm.fail = [422]
        self.assertEqual(self.main(), 2)
        self.assertEqual(self.farm.calls, 1)
        self.assertEqual(self.sleeps, [])
        self.assertIn("HTTP 422 VALIDATION_ERROR", self.stderr.getvalue())
        # The farm's details carried an id and its message free text.
        self.assertLogsArePrivate()

    def test_wrong_pipeline_key(self) -> None:
        with mock.patch.dict(os.environ, {"TURNIP_PIPELINE_KEY": "wrong-key"}):
            self.assertEqual(self.main(), 2)
        self.assertEqual(self.farm.calls, 1)
        self.assertIn("HTTP 401 UNAUTHENTICATED", self.stderr.getvalue())
        self.assertNotIn("wrong-key", self.logs())
        self.assertEqual(self.farm.open, {})

    def test_other_4xx_fails_at_once(self) -> None:
        self.farm.fail = [404]
        self.assertEqual(self.main(), 2)
        self.assertEqual(self.farm.calls, 1)
        self.assertIn("HTTP 404 NOT_FOUND", self.stderr.getvalue())

    def test_unprintable_error_codes_are_dropped(self) -> None:
        def odd(
            method: str, url: str, headers: dict[str, str], body: bytes | None
        ) -> tuple[int, Any]:
            return 400, {"error": {"code": f"BAD {source_id(0)}"}}

        code = report_quarantine.main(
            ["--quarantine", str(self.path)], transport=odd, sleep=self.sleeps.append
        )
        self.assertEqual(code, 2)
        self.assertIn("HTTP 400 (request 1 of 1", self.stderr.getvalue())
        self.assertLogsArePrivate()

    def test_an_answer_that_misses_an_id_is_an_error(self) -> None:
        cases = {
            "drops one": lambda a: {**a, "quarantined": a["quarantined"][1:]},
            "repeats one": lambda a: {
                **a,
                "already_open": a["quarantined"][:1],
            },
            "names a stranger": lambda a: {
                **a,
                "quarantined": [*a["quarantined"][1:], "e" * 64],
            },
            "lacks a list": lambda a: {"quarantined": a["quarantined"]},
            "not an object": lambda a: [a],
            "empty": lambda a: None,
        }
        for name, tamper in cases.items():
            with self.subTest(name):
                farm = FakeFarm()
                farm.tamper = tamper
                self.clear_logs()
                self.assertEqual(self.main(farm), 2)
                # Not retried: the farm answered, just not as agreed.
                self.assertEqual(farm.calls, 1)
                self.assertIn("POST /api/quarantine answered", self.stderr.getvalue())
                self.assertLogsArePrivate()

    def test_missing_environment(self) -> None:
        with mock.patch.dict(os.environ, {"TURNIP_FARM_URL": ""}):
            self.assertEqual(self.main(), 2)
        self.assertIn("missing environment: TURNIP_FARM_URL", self.stderr.getvalue())
        self.assertEqual(self.farm.calls, 0)

    def test_plain_http_farm_is_refused(self) -> None:
        with mock.patch.dict(os.environ, {"TURNIP_FARM_URL": "http://farm.example"}):
            self.assertEqual(self.main(), 2)
        self.assertEqual(self.farm.calls, 0)


class RefusalTest(ReportTestCase):
    """A malformed file sends nothing at all, and the error names no id."""

    def assertRefused(self, data: Any, message: str) -> None:
        self.write(data)
        self.clear_logs()
        self.assertEqual(self.main(), 2)
        self.assertEqual(self.farm.calls, 0)
        self.assertIn(message, self.stderr.getvalue())
        self.assertIsNone(HEX64.search(self.logs()))

    def test_malformed_rows_are_refused_before_any_request(self) -> None:
        good = rows(ids(2))
        cases: dict[str, tuple[Any, str]] = {
            "not an array": ({"rows": good}, "must contain a JSON array"),
            "an account id rides along": (
                [good[0], {**good[1], "user_id": "u-1"}],
                "row 1 must have exactly the keys",
            ),
            "a key missing": (
                [{"source_id": source_id(0), "reason": REASON}],
                "row 0 must have exactly the keys",
            ),
            "a listed twice source": ([good[0], good[1], good[0]], "listed twice"),
            "an empty reason": ([{**good[0], "reason": "  "}], "row 0: reason"),
            "a long reason": ([{**good[0], "reason": "x" * 1001}], "row 0: reason"),
            "a list of metrics": ([{**good[0], "metrics": [1]}], "row 0: metrics"),
            "an uppercase id": (
                [{**good[0], "source_id": source_id(0).upper()}],
                "row 0: source_id",
            ),
        }
        for name, (data, message) in cases.items():
            with self.subTest(name):
                self.assertRefused(data, message)

    def test_synthetic_source_ids_never_reach_the_farm(self) -> None:
        # The synthetic rehearsal's sources (make_smoke_fixture.py).
        self.assertRefused(rows(["source-a", "source-b"]), "row 0: source_id")

    def test_reason_length_is_counted_as_the_farm_counts_it(self) -> None:
        # U+1F6F9 (skateboard) is two UTF-16 code units in JavaScript.
        self.write(rows(ids(1), reason="x" * 998 + "\U0001f6f9"))
        self.assertEqual(self.main(), 0, self.stderr.getvalue())
        self.farm = FakeFarm()
        self.assertRefused(rows(ids(1), reason="x" * 999 + "\U0001f6f9"), "reason")

    def test_unreadable_files_are_refused(self) -> None:
        self.path.write_text("[{")
        self.assertEqual(self.main(), 2)
        self.assertIn("JSONDecodeError", self.stderr.getvalue())
        self.path.unlink()
        self.assertEqual(self.main(), 2)
        self.assertIn("FileNotFoundError", self.stderr.getvalue())
        self.assertEqual(self.farm.calls, 0)

    def test_non_finite_metrics_are_refused(self) -> None:
        self.path.write_text(json.dumps(rows(ids(1), metrics={"delta": float("nan")})))
        self.assertEqual(self.main(), 2)
        self.assertEqual(self.farm.calls, 0)


class HttpTransportTest(ReportTestCase):
    """The real urllib transport against a local HTTP stand-in of the farm."""

    def start(self, handler: type[BaseHTTPRequestHandler]) -> str:
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        # Cleanups run last-in first-out: stop serving, join, then close.
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def serve(self, farm: FakeFarm, raw: bytes | None = None) -> str:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                headers = {k: v for k, v in self.headers.items()}
                # FakeFarm routes on the public URL; the server is local.
                status, answer = farm.transport("POST", FARM + self.path, headers, body)
                data = raw if raw is not None else json.dumps(answer).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                pass

        return self.start(Handler)

    def main_at(self, url: str) -> int:
        with mock.patch.dict(os.environ, {"TURNIP_FARM_URL": url}):
            return report_quarantine.main(
                ["--quarantine", str(self.path)], sleep=self.sleeps.append
            )

    def test_quarantine_over_http(self) -> None:
        sources = ids(report_quarantine.MAX_SOURCE_IDS + 1)
        self.write(rows(sources))
        self.assertEqual(self.main_at(self.serve(self.farm)), 0, self.stderr.getvalue())
        self.assertEqual(set(self.farm.open), set(sources))
        self.assertEqual(len(self.farm.posts), 2)

    def test_a_non_json_5xx_is_retried(self) -> None:
        self.write(rows(ids(2)))
        self.farm.fail = [502, 502, 502]
        url = self.serve(self.farm, raw=b"<html>Bad Gateway</html>")
        self.assertEqual(self.main_at(url), 2)
        self.assertEqual(self.farm.calls, 3)
        self.assertIn("HTTP 502 (request 1 of 1", self.stderr.getvalue())

    def test_redirects_are_never_followed(self) -> None:
        # Following one would send the pipeline key to the Location host.
        seen: list[str | None] = []

        class Elsewhere(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                seen.append(self.headers.get("Authorization"))
                self.send_response(500)
                self.end_headers()

            do_GET = do_POST

            def log_message(self, *args: Any) -> None:
                pass

        elsewhere = self.start(Elsewhere)

        class Redirect(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self.send_response(307)
                self.send_header("Location", elsewhere + self.path)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args: Any) -> None:
                pass

        self.write(rows(ids(2)))
        self.assertEqual(self.main_at(self.start(Redirect)), 2)
        self.assertIn("HTTP 307", self.stderr.getvalue())
        self.assertEqual(seen, [])
        self.assertEqual(self.sleeps, [])

    def test_a_dead_farm_is_retried_then_an_error(self) -> None:
        self.write(rows(ids(2)))
        # Nothing listens on a port once its server is closed.
        closed = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        dead = f"http://127.0.0.1:{closed.server_address[1]}"
        closed.server_close()
        self.assertEqual(self.main_at(dead), 2)
        self.assertEqual(len(self.sleeps), report_quarantine.ATTEMPTS - 1)
        self.assertIn("farm request failed: URLError", self.stderr.getvalue())
        self.assertNotIn(dead, self.logs())


if __name__ == "__main__":
    unittest.main()
