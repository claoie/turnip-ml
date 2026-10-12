"""Unit tests for Training/fetch_dataset.py.

Hermetic: a fake paginated export and a fake blob store, no network.
Run with:  python -m unittest discover -s Training/tests -t .
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
import urllib.parse
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fetch_dataset
import tkp1
from fetch_dataset import EXIT_ERROR, EXIT_OK, BlobCache, ExportClient, FetchError

FARM = "https://farm.example"
KEY = "pipeline-secret"
SRC_A, SRC_B, SRC_C = "a" * 64, "b" * 64, "c" * 64
CLIP = {n: f"c11b0000-0000-4000-8000-{n:012d}" for n in range(1, 10)}
TAXONOMY = [
    {"raw": "cork", "canonical": "cork", "taxonomy_version": 1},
    {"raw": "corkscrew", "canonical": "cork", "taxonomy_version": 1},
    {"raw": "gainer", "canonical": "gainer", "taxonomy_version": 1},
    {"raw": "hook", "canonical": "hook", "taxonomy_version": 1},
    {"raw": "scoot", "canonical": "scoot", "taxonomy_version": 2},
]


def pose(frames, seed):
    """Deterministic keypoints; frame 2 is a gap."""
    out = []
    for f in range(frames):
        if f == 2:
            out.append([[0.0, 0.0, 0.0] for _ in range(17)])
            continue
        out.append(
            [
                [((seed + f + k) % 10) / 10, ((seed * k + f) % 7) / 7, 0.5 + k / 100]
                for k in range(17)
            ]
        )
    return out


BLOBS = {
    sid: tkp1.encode_gz(pose(40, seed))
    for seed, sid in enumerate((SRC_A, SRC_B, SRC_C))
}


def row(source, clip, name, start=5, end=15, user=None, version=1, **overrides):
    data = {
        "source_id": source,
        "user_id": user or {SRC_A: "user-1", SRC_B: "user-1", SRC_C: "user-2"}[source],
        "r2_key": f"poses/{source}.tkp1.gz",
        "blob_sha256": hashlib.sha256(BLOBS[source]).hexdigest(),
        "clip_id": CLIP[clip],
        "start_frame": start,
        "end_frame": end,
        "trick_name": name,
        "taxonomy_version": version,
    }
    data.update(overrides)
    return data


# Export order is the farm's cursor order, which interleaves sources.
ROWS = [
    row(SRC_B, 3, "Cork"),
    row(SRC_A, 1, "hook - scoot (combo)", 20, 30, version=2),
    row(SRC_C, 5, "Mystery Flip", 0, 40),
    row(SRC_A, 1, "Gainer", 20, 30),
    row(SRC_A, 2, "corkscrew"),
    row(SRC_B, 4, "(combo)", 30, 35),
]


class FakeExport:
    """Paginated export; cursors are opaque strings; records requests."""

    def __init__(self, rows=ROWS, page_size=2, null_on_last_page=False):
        self.rows = list(rows)
        self.page_size = page_size
        self.null_on_last_page = null_on_last_page
        self.requests = []

    def __call__(self, url, headers):
        self.requests.append((url, headers))
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        start = int(query["since"][0].removeprefix("cur-")) + 1 if query else 0
        page = self.rows[start : start + self.page_size]
        last = start + len(page) - 1
        exhausted = last >= len(self.rows) - 1
        next_cursor = f"cur-{last}" if page else None
        if exhausted and self.null_on_last_page:
            next_cursor = None
        versions = {r["taxonomy_version"] for r in page}
        return {
            "rows": page,
            "taxonomy": [t for t in TAXONOMY if t["taxonomy_version"] in versions],
            "next_cursor": next_cursor,
        }


class FakeBlobs:
    def __init__(self, blobs=None):
        self.blobs = dict(blobs or {f"poses/{s}.tkp1.gz": b for s, b in BLOBS.items()})
        self.calls = []

    def __call__(self, r2_key):
        self.calls.append(r2_key)
        return self.blobs[r2_key]


class FetchTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache = self.root / "cache"
        self.blobs = FakeBlobs()
        self.stdout = self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.stderr = self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        self.enterContext(
            mock.patch.dict(
                os.environ,
                {fetch_dataset.FARM_URL_ENV: FARM, fetch_dataset.PIPELINE_KEY_ENV: KEY},
            )
        )

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, export=None, out="dataset"):
        client = ExportClient(FARM, KEY, export or FakeExport())
        return fetch_dataset.build_dataset(
            client.pull(), BlobCache(self.cache, self.blobs), self.root / out
        )

    def sample(self, source, out="dataset"):
        path = self.root / out / "samples" / f"{source}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def tree(self, out):
        base = self.root / out
        return {
            str(p.relative_to(base)): p.read_bytes()
            for p in sorted(base.rglob("*"))
            if p.is_file()
        }

    def assert_build_refused(self, rows):
        out = f"refused-{len(list(self.root.glob('refused-*')))}"
        with self.assertRaises(FetchError):
            self.build(FakeExport(rows), out=out)
        self.assertFalse((self.root / out).exists())
        self.assertEqual(list(self.root.glob(".*partial*")), [])


class BuildTest(FetchTestCase):
    def test_dataset_layout_and_samples(self):
        stats = self.build()
        self.assertEqual(
            stats,
            {
                "samples": 3,
                "clips": 4,
                "clips_dropped_without_names": 1,
                "label_rows": 6,
                "taxonomy_version": 2,
                "unmapped_tokens": 1,
            },
        )
        a = self.sample(SRC_A)
        self.assertEqual(a["sample_id"], SRC_A)
        self.assertEqual(a["user_id"], "user-1")
        self.assertEqual(a["fps"], 10)
        self.assertEqual(len(a["keypoints"]), 40)
        self.assertEqual(len(a["keypoints"][0]), 17)
        self.assertEqual(a["keypoints"][2], [[0.0, 0.0, 0.0]] * 17)  # gap kept
        self.assertEqual(a["keypoints"], tkp1.decode_gz(BLOBS[SRC_A]).keypoints)
        # Clip 1 carries a combo across two label rows: one trick entry per
        # canonical name on the same window. Clip 2's alias is canonicalized.
        self.assertEqual(
            [
                (t["clip_id"], t["name"], t["start_frame"], t["end_frame"])
                for t in a["tricks"]
            ],
            [
                (CLIP[2], "cork", 5, 15),
                (CLIP[1], "gainer", 20, 30),
                (CLIP[1], "hook", 20, 30),
                (CLIP[1], "scoot", 20, 30),
            ],
        )
        # Clip 4 ("(combo)") canonicalizes to nothing and is dropped.
        self.assertEqual(
            [t["clip_id"] for t in self.sample(SRC_B)["tricks"]], [CLIP[3]]
        )
        self.assertEqual(self.sample(SRC_C)["tricks"][0]["name"], "mystery flip")

    def test_manifests(self):
        self.build()
        manifest = (self.root / "dataset" / "manifest.jsonl").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            [json.loads(line) for line in manifest.splitlines()],
            [
                {"sample_id": SRC_A, "user_id": "user-1"},
                {"sample_id": SRC_B, "user_id": "user-1"},
                {"sample_id": SRC_C, "user_id": "user-2"},
            ],
        )
        meta = json.loads((self.root / "dataset" / "dataset_manifest.json").read_text())
        self.assertEqual(meta["sample_ids"], [SRC_A, SRC_B, SRC_C])
        self.assertEqual(meta["taxonomy_version"], 2)
        self.assertEqual(meta["export_cursor"], "cur-5")
        log = (self.root / "dataset" / "unmapped_labels.log").read_text(
            encoding="utf-8"
        )
        self.assertEqual(log, "1\tmystery flip\n")

    def test_byte_identical_across_page_sizes(self):
        self.build(FakeExport(page_size=1), out="p1")
        self.build(FakeExport(page_size=4), out="p4")
        self.build(FakeExport(page_size=100), out="p100")
        reference = self.tree("p1")
        self.assertEqual(len(reference), 6)
        for out in ("p4", "p100"):
            self.assertEqual(self.tree(out), reference, out)

    def test_null_cursor_on_the_last_page(self):
        # A farm that answers null on the final partial page still yields
        # the full dataset; only the recorded cursor lags one page behind.
        self.build(FakeExport(page_size=4), out="ref")
        self.build(FakeExport(page_size=4, null_on_last_page=True), out="null")
        ref, null = self.tree("ref"), self.tree("null")
        meta_ref = json.loads(ref.pop("dataset_manifest.json"))
        meta_null = json.loads(null.pop("dataset_manifest.json"))
        self.assertEqual(null, ref)
        self.assertEqual(meta_ref.pop("export_cursor"), "cur-5")
        self.assertEqual(meta_null.pop("export_cursor"), "cur-3")
        self.assertEqual(meta_null, meta_ref)

    def test_blobs_are_cached_by_content_hash(self):
        self.build(out="first")
        self.assertEqual(len(self.blobs.calls), 3)
        self.build(out="second")
        self.assertEqual(len(self.blobs.calls), 3)  # all served from cache
        # A corrupt cache entry is discarded and refetched.
        entry = self.cache / f"{hashlib.sha256(BLOBS[SRC_A]).hexdigest()}.tkp1.gz"
        entry.write_bytes(b"corrupt")
        self.build(out="third")
        self.assertEqual(self.blobs.calls[3:], [f"poses/{SRC_A}.tkp1.gz"])
        self.assertEqual(entry.read_bytes(), BLOBS[SRC_A])

    def test_checksum_mismatch_fails_closed(self):
        self.blobs.blobs[f"poses/{SRC_B}.tkp1.gz"] = BLOBS[SRC_A]
        with self.assertRaises(FetchError):
            self.build()
        self.assertFalse((self.root / "dataset").exists())
        self.assertFalse(
            (
                self.cache / f"{hashlib.sha256(BLOBS[SRC_B]).hexdigest()}.tkp1.gz"
            ).exists()
        )

    def test_undecodable_blob_fails_closed(self):
        bad = b"not gzip"
        rows = [row(SRC_A, 1, "cork", blob_sha256=hashlib.sha256(bad).hexdigest())]
        self.blobs.blobs[f"poses/{SRC_A}.tkp1.gz"] = bad
        self.assert_build_refused(rows)

    def test_clip_past_the_pose_sequence(self):
        self.assert_build_refused([row(SRC_A, 1, "cork", 30, 41)])
        # Ending exactly at the last frame boundary is fine.
        self.build(FakeExport([row(SRC_A, 1, "cork", 30, 40)]))

    def test_out_dir_must_be_fresh(self):
        (self.root / "dataset").mkdir()
        with self.assertRaises(FetchError):
            self.build()
        self.assertEqual(self.blobs.calls, [])

    def test_empty_export(self):
        self.assert_build_refused([])


class RowValidationTest(FetchTestCase):
    def test_malformed_rows(self):
        cases = {
            "source-id": row(SRC_A, 1, "cork", source_id="A" * 64),
            "r2-key": row(SRC_A, 1, "cork", r2_key=f"poses/{SRC_B}.tkp1.gz"),
            "r2-key-path": row(SRC_A, 1, "cork", r2_key=f"../poses/{SRC_A}.tkp1.gz"),
            "sha": row(SRC_A, 1, "cork", blob_sha256=None),
            "clip-not-uuid": row(SRC_A, 1, "cork", clip_id="clip-1"),
            "clip-uppercase": row(SRC_A, 1, "cork", clip_id=CLIP[1].upper()),
            "bool-frame": row(SRC_A, 1, "cork", start_frame=True),
            "empty-window": row(SRC_A, 1, "cork", 10, 10),
            "negative-start": row(SRC_A, 1, "cork", -1, 10),
            "empty-name": row(SRC_A, 1, "  "),
            "version-zero": row(SRC_A, 1, "cork", version=0),
            "user-whitespace": row(SRC_A, 1, "cork", user="user 1"),
        }
        for name, data in cases.items():
            with self.subTest(case=name):
                self.assert_build_refused([data])
        missing = row(SRC_A, 1, "cork")
        del missing["blob_sha256"]
        self.assert_build_refused([missing])

    def test_rows_that_disagree(self):
        cases = {
            "source-owner": [
                row(SRC_A, 1, "cork"),
                row(SRC_A, 2, "hook", user="user-9"),
            ],
            "clip-window": [row(SRC_A, 1, "cork", 5, 15), row(SRC_A, 1, "hook", 5, 16)],
            "duplicate-label": [row(SRC_A, 1, "cork"), row(SRC_A, 1, "cork")],
            "clip-two-sources": [row(SRC_A, 1, "cork"), row(SRC_B, 1, "hook")],
        }
        for name, rows in cases.items():
            with self.subTest(case=name):
                self.assert_build_refused(rows)

    def test_malformed_taxonomy(self):
        def export(url, headers):
            return {
                "rows": [row(SRC_A, 1, "cork")],
                "taxonomy": [
                    {"raw": "cork", "canonical": "Cork", "taxonomy_version": 1}
                ],
                "next_cursor": None,
            }

        with self.assertRaises(FetchError):
            self.build(export)


class PaginationTest(FetchTestCase):
    def test_requests_carry_the_key_and_the_cursor(self):
        export = FakeExport(page_size=4)
        pull = ExportClient(FARM + "/", KEY, export).pull()
        self.assertEqual(len(pull.rows), 6)
        urls = [url for url, _ in export.requests]
        self.assertEqual(
            urls,
            [
                f"{FARM}/api/labels/export",
                f"{FARM}/api/labels/export?since=cur-3",
                f"{FARM}/api/labels/export?since=cur-5",
            ],
        )
        self.assertEqual(export.requests[0][1]["Authorization"], f"Bearer {KEY}")

    def test_cursor_is_url_encoded(self):
        export = FakeExport()
        with contextlib.suppress(Exception):
            ExportClient(FARM, KEY, export).pull(since="a b&c=d")
        self.assertEqual(
            export.requests[0][0], f"{FARM}/api/labels/export?since=a+b%26c%3Dd"
        )

    def test_stuck_or_cycling_cursor(self):
        for cursors in (["x", "x"], ["x", "y", "x"]):
            with self.subTest(cursors=cursors):
                pages = iter(cursors)

                def export(url, headers, pages=pages):
                    return {"rows": [row(SRC_A, 1, "cork")], "next_cursor": next(pages)}

                with self.assertRaises(FetchError):
                    ExportClient(FARM, KEY, export).pull()

    def test_malformed_pages(self):
        for page in (
            [],
            {"rows": {}},
            {"rows": [], "taxonomy": {}},
            {"rows": [row(SRC_A, 1, "cork")], "next_cursor": 7},
            {"rows": [row(SRC_A, 1, "cork")], "next_cursor": ""},
        ):
            with self.subTest(page=page), self.assertRaises(FetchError):
                ExportClient(FARM, KEY, lambda url, headers, p=page: p).pull()

    def test_url_must_be_https_unless_local(self):
        for url in ("http://farm.example", "ftp://farm.example", "farm.example"):
            with self.subTest(url=url), self.assertRaises(FetchError):
                ExportClient(url, KEY)
        for url in ("http://localhost:3000", "http://127.0.0.1:3000"):
            ExportClient(url, KEY)
        with self.assertRaises(FetchError):
            ExportClient(FARM, "")


class HttpTest(unittest.TestCase):
    def test_http_errors_report_the_status_only(self):
        error = urllib.error.HTTPError(
            FARM, 401, "Unauthorized", None, io.BytesIO(b"{}")
        )
        with (
            mock.patch("urllib.request.urlopen", side_effect=error),
            self.assertRaisesRegex(FetchError, "HTTP 401"),
        ):
            fetch_dataset.http_get_json(FARM, {})

    def test_transport_failures_and_bad_json(self):
        with (
            mock.patch("urllib.request.urlopen", side_effect=TimeoutError()),
            self.assertRaisesRegex(FetchError, "TimeoutError"),
        ):
            fetch_dataset.http_get_json(FARM, {})
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"not json"
        with (
            mock.patch("urllib.request.urlopen", return_value=response),
            self.assertRaises(FetchError),
        ):
            fetch_dataset.http_get_json(FARM, {})


class MainTest(FetchTestCase):
    def test_count_then_count_again_from_the_stored_watermark(self):
        export = FakeExport(page_size=4)
        state = self.root / "state.json"
        code = fetch_dataset.main(
            ["count", "--out-state", str(state)], transport=export
        )
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(json.loads(self.stdout.getvalue()), {"new_labels": 6})
        self.assertEqual(json.loads(state.read_text()), {"watermark": "cur-5"})

        export.rows.append(row(SRC_C, 6, "hook", 20, 30))
        self.stdout.truncate(0)
        self.stdout.seek(0)
        newer = self.root / "newer.json"
        code = fetch_dataset.main(
            ["count", "--state", str(state), "--out-state", str(newer)],
            transport=export,
        )
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(json.loads(self.stdout.getvalue()), {"new_labels": 1})
        self.assertEqual(json.loads(newer.read_text()), {"watermark": "cur-6"})

    def test_count_with_nothing_new_keeps_the_watermark(self):
        state = self.root / "state.json"
        state.write_text('{"watermark": "cur-5"}')
        out = self.root / "out.json"
        code = fetch_dataset.main(
            ["count", "--state", str(state), "--out-state", str(out)],
            transport=FakeExport(),
        )
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(json.loads(self.stdout.getvalue()), {"new_labels": 0})
        self.assertEqual(json.loads(out.read_text()), {"watermark": "cur-5"})

    def test_malformed_state_file(self):
        for content in ("not json", "{}", '{"watermark": 5}', '{"watermark": ""}'):
            with self.subTest(state=content):
                state = self.root / "state.json"
                state.write_text(content)
                code = fetch_dataset.main(
                    [
                        "count",
                        "--state",
                        str(state),
                        "--out-state",
                        str(self.root / "o"),
                    ],
                    transport=FakeExport(),
                )
                self.assertEqual(code, EXIT_ERROR)

    def test_build_via_main_prints_counts_only(self):
        argv = ["build", "--out", str(self.root / "ds"), "--cache", str(self.cache)]
        code = fetch_dataset.main(argv, transport=FakeExport(), blob_getter=self.blobs)
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(json.loads(self.stdout.getvalue())["samples"], 3)
        output = self.stdout.getvalue() + self.stderr.getvalue()
        for secret in ("user-1", "user-2", "Mystery", "corkscrew", "cur-", KEY):
            self.assertNotIn(secret, output)

    def test_build_failure_exits_2_without_leaking(self):
        self.blobs.blobs[f"poses/{SRC_B}.tkp1.gz"] = BLOBS[SRC_A]
        argv = ["build", "--out", str(self.root / "ds"), "--cache", str(self.cache)]
        code = fetch_dataset.main(argv, transport=FakeExport(), blob_getter=self.blobs)
        self.assertEqual(code, EXIT_ERROR)
        self.assertIn("blob_sha256", self.stderr.getvalue())
        for secret in ("user-1", "Mystery", KEY):
            self.assertNotIn(secret, self.stderr.getvalue())

    def test_missing_environment(self):
        with mock.patch.dict(os.environ, {fetch_dataset.PIPELINE_KEY_ENV: ""}):
            code = fetch_dataset.main(
                ["count", "--out-state", str(self.root / "o")], transport=FakeExport()
            )
        self.assertEqual(code, EXIT_ERROR)
        self.assertIn(fetch_dataset.PIPELINE_KEY_ENV, self.stderr.getvalue())

    def test_r2_getter_requires_its_environment(self):
        with (
            mock.patch.dict(os.environ, {name: "" for name in fetch_dataset.R2_ENV}),
            self.assertRaisesRegex(FetchError, "R2_ACCOUNT_ID"),
        ):
            fetch_dataset.r2_getter_from_env()

    @unittest.skipIf(importlib.util.find_spec("boto3"), "boto3 is installed")
    def test_r2_getter_requires_boto3(self):
        with (
            mock.patch.dict(os.environ, {name: "x" for name in fetch_dataset.R2_ENV}),
            self.assertRaisesRegex(FetchError, "boto3"),
        ):
            fetch_dataset.r2_getter_from_env()


if __name__ == "__main__":
    unittest.main()
