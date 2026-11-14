"""Unit tests for Training/fetch_dataset.py (the farm's segment export).

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
import evaluate
import fetch_dataset
import predict
import tks1
import train
from fetch_dataset import EXIT_ERROR, EXIT_OK, BlobCache, ExportClient, FetchError

FARM = "https://farm.example"
KEY = "pipeline-secret"
SEG = {n: f"5e600000-0000-4000-8000-{n:012d}" for n in range(1, 10)}
TAXONOMY = [
    {"raw": "cork", "canonical": "cork", "taxonomy_version": 1},
    {"raw": "corkscrew", "canonical": "cork", "taxonomy_version": 1},
    {"raw": "gainer", "canonical": "gainer", "taxonomy_version": 1},
    {"raw": "hook", "canonical": "hook", "taxonomy_version": 1},
    {"raw": "scoot", "canonical": "scoot", "taxonomy_version": 2},
]


def frames(n, seed):
    """Deterministic 13-joint frames; frame 2 is a gap, joint 5 of frame 3
    is missing."""
    out = []
    for f in range(n):
        if f == 2:
            out.append([[0.0, 0.0, 0.0] for _ in range(13)])
            continue
        frame = [
            [((seed + f + k) % 10) / 10 - 0.5, ((seed * k + f) % 7) / 7, 0.5 + k / 100]
            for k in range(13)
        ]
        if f == 3:
            frame[5] = [0.0, 0.0, 0.0]
        out.append(frame)
    return out


FRAME_COUNTS = {1: 30, 2: 20, 3: 20, 4: 15, 5: 40}
BLOBS = {n: tks1.encode_gz(frames(count, n)) for n, count in FRAME_COUNTS.items()}


def key(n):
    return f"segments/{SEG[n]}.tks1.gz"


def segment(n, labels, start=5, end=15, bucket=0, **overrides):
    """An export entry: labels are (trick_name, origin, taxonomy_version)."""
    data = {
        "segment_id": SEG[n],
        "r2_key": key(n),
        "blob_sha256": hashlib.sha256(BLOBS[n]).hexdigest() if n in BLOBS else "0" * 64,
        "frame_count": FRAME_COUNTS.get(n, 20),
        "window_start": start,
        "window_end": end,
        "split_bucket": bucket,
        # The farm's defaults: typed names on a heuristic or manual clip.
        "proposal_model_version": None,
        "suggested_names": [],
        "suggestions_withheld": False,
        "labels": [
            {"trick_name": name, "origin": origin, "taxonomy_version": version}
            for name, origin, version in labels
        ],
    }
    data.update(overrides)
    return data


MODEL = "trick-v1.2"
# Segment 1: the model pre-filled two names; the user kept one, removed the
# other and typed a combo. Segment 2: a blind clip, named from the
# vocabulary.
PROPOSED = {
    "proposal_model_version": MODEL,
    "suggested_names": ["gainer", "cork"],
    "suggestions_withheld": False,
}
BLIND = {
    "proposal_model_version": MODEL,
    "suggested_names": ["cork"],
    "suggestions_withheld": True,
}

# Export order is the farm's (by id); the fake serves them unsorted so the
# re-sort shows.
SEGMENTS = [
    segment(3, [("Cork", "typed", 1)], bucket=1),
    segment(
        1,
        [("Gainer", "model", 1), ("hook - scoot (combo)", "typed", 2)],
        start=5,
        end=25,
        **PROPOSED,
    ),
    segment(5, [("Mystery Flip", "typed", 1)], start=0, end=40, bucket=2),
    segment(2, [("corkscrew", "vocabulary", 1)], **BLIND),
    segment(4, [("(combo)", "typed", 1)], start=5, end=10, bucket=1),
]

# The farm's hold-back threshold, and enough contributors for every name
# in SEGMENTS to train.
MINIMUM = 5
COUNTS = {
    name: MINIMUM + 4 for name in ("cork", "gainer", "hook", "scoot", "mystery flip")
}


def contributors(counts):
    """A {name: count} dict in the farm's trick_contributors form."""
    return [{"canonical": n, "contributors": c} for n, c in sorted(counts.items())]


class FakeExport:
    """Paginated export; cursors are opaque strings; records requests."""

    def __init__(
        self,
        segments=SEGMENTS,
        page_size=2,
        null_on_last_page=False,
        taxonomy=TAXONOMY,
        counts=COUNTS,
        minimum=MINIMUM,
    ):
        self.segments = list(segments)
        self.page_size = page_size
        self.null_on_last_page = null_on_last_page
        self.taxonomy = taxonomy
        # trick_contributors and contributor_minimum on every page. None
        # leaves the field out; a callable maps the page's start index to
        # the value; a counts dict is sent in the farm's list form, any
        # other counts value as it is.
        self.counts = counts
        self.minimum = minimum
        self.requests = []

    def __call__(self, url, headers):
        self.requests.append((url, headers))
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        start = int(query["since"][0].removeprefix("cur-")) + 1 if query else 0
        page = self.segments[start : start + self.page_size]
        last = start + len(page) - 1
        exhausted = last >= len(self.segments) - 1
        next_cursor = f"cur-{last}" if page else None
        if exhausted and self.null_on_last_page:
            next_cursor = None
        body = {"segments": page, "taxonomy": self.taxonomy, "next_cursor": next_cursor}
        counts = self.counts(start) if callable(self.counts) else self.counts
        if counts is not None:
            if isinstance(counts, dict):
                counts = contributors(counts)
            body[fetch_dataset.TRICK_CONTRIBUTORS_FIELD] = counts
        minimum = self.minimum(start) if callable(self.minimum) else self.minimum
        if minimum is not None:
            body[fetch_dataset.CONTRIBUTOR_MINIMUM_FIELD] = minimum
        return body


class FakeBlobs:
    def __init__(self, blobs=None):
        self.blobs = dict(blobs or {key(n): b for n, b in BLOBS.items()})
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

    def sample(self, n, out="dataset"):
        path = self.root / out / "samples" / f"{SEG[n]}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def has_sample(self, n, out="dataset"):
        return (self.root / out / "samples" / f"{SEG[n]}.json").exists()

    def manifest(self, out="dataset"):
        path = self.root / out / "dataset_manifest.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def tree(self, out):
        base = self.root / out
        return {
            str(p.relative_to(base)): p.read_bytes()
            for p in sorted(base.rglob("*"))
            if p.is_file()
        }

    def assert_build_refused(self, segments):
        out = f"refused-{len(list(self.root.glob('refused-*')))}"
        with self.assertRaises(FetchError):
            self.build(FakeExport(segments), out=out)
        self.assertFalse((self.root / out).exists())
        self.assertEqual(list(self.root.glob(".*partial*")), [])

    def main_build(self, export, out="ds"):
        argv = ["build", "--out", str(self.root / out), "--cache", str(self.cache)]
        return fetch_dataset.main(argv, transport=export, blob_getter=self.blobs)


class BuildTest(FetchTestCase):
    def test_dataset_layout_and_samples(self):
        stats = self.build()
        self.assertEqual(
            stats,
            {
                "samples": 4,
                "segments": 5,
                "segments_dropped_without_names": 1,
                "segments_held_back": 0,
                "names_held_back": 0,
                "label_rows": 6,
                "taxonomy_version": 2,
                "unmapped_tokens": 1,
                "cache_pruned": 0,
            },
        )
        one = self.sample(1)
        self.assertEqual(
            set(one),
            {"sample_id", "fps", "keypoints", "split_bucket", "tricks", "clips"},
        )
        self.assertEqual(one["sample_id"], SEG[1])
        self.assertEqual(one["fps"], 10)
        self.assertEqual(one["split_bucket"], 0)
        self.assertEqual(one["keypoints"], tks1.decode_gz(BLOBS[1]))
        self.assertEqual(one["keypoints"][2], [[0.0, 0.0, 0.0]] * 13)  # gap kept
        # A combo across two labels: one entry per canonical name, all on the
        # segment's clip window.
        self.assertEqual(
            [
                (t["clip_id"], t["name"], t["start_frame"], t["end_frame"])
                for t in one["tricks"]
            ],
            [(SEG[1], n, 5, 25) for n in ("gainer", "hook", "scoot")],
        )
        self.assertEqual(self.sample(2)["tricks"][0]["name"], "cork")  # alias
        self.assertEqual(self.sample(5)["tricks"][0]["name"], "mystery flip")
        # Segment 4 ("(combo)") canonicalizes to nothing and is dropped.
        self.assertFalse(self.has_sample(4))

    def test_samples_carry_provenance(self):
        self.build()
        one, two = self.sample(1), self.sample(2)
        self.assertEqual(
            {t["name"]: t["origins"] for t in one["tricks"]},
            {"gainer": ["model"], "hook": ["typed"], "scoot": ["typed"]},
        )
        self.assertEqual(
            one["clips"],
            {
                SEG[1]: {
                    "proposal_model_version": MODEL,
                    "suggested_names": ["gainer", "cork"],
                    "suggestions_withheld": False,
                }
            },
        )
        self.assertEqual(two["tricks"][0]["origins"], ["vocabulary"])
        self.assertIs(two["clips"][SEG[2]]["suggestions_withheld"], True)

    def test_merged_names_keep_every_origin(self):
        # Two raw labels that canonicalize to one name merge their origins.
        both = segment(3, [("Cork", "typed", 1), ("corkscrew", "vocabulary", 1)])
        self.build(FakeExport([both]))
        [entry] = self.sample(3)["tricks"]
        self.assertEqual(entry["origins"], ["typed", "vocabulary"])

    def test_legitimate_provenance_is_accepted(self):
        for name, overrides in {
            "heuristic": {},
            "proposed": PROPOSED,
            "blind": BLIND,
            "model with no names": {"proposal_model_version": MODEL},
        }.items():
            with self.subTest(name):
                origin = "model" if name == "proposed" else "typed"
                entry = segment(3, [("Cork", origin, 1)], **overrides)
                self.build(FakeExport([entry]), out=f"ok-{name}")

    def test_consumers_load_the_samples(self):
        self.build()
        samples = self.root / "dataset" / "samples"
        split = {SEG[1]: "train", SEG[2]: "train", SEG[3]: "val", SEG[5]: "holdout"}
        tr, va = train.load_samples(samples, split)
        self.assertEqual([s["sample_id"] for s in tr], [SEG[1], SEG[2]])
        self.assertEqual(tr[0]["keypoints"].shape, (30, 13, 3))
        self.assertEqual(len(va), 1)
        gt = evaluate.load_ground_truth(samples, [SEG[5]])
        self.assertEqual(gt[SEG[5]], [(0, 40, frozenset({"mystery flip"}))])
        [held] = predict.load_split_samples(split, samples, "holdout")
        self.assertEqual(held["tricks"], [])

    def test_manifests(self):
        self.build()
        lines = (self.root / "dataset" / "manifest.jsonl").read_text().splitlines()
        # Splits group by the farm's split_bucket, never by a source.
        self.assertEqual(
            [json.loads(line) for line in lines],
            [
                {"sample_id": SEG[1], "source_id": "bucket-0"},
                {"sample_id": SEG[2], "source_id": "bucket-0"},
                {"sample_id": SEG[3], "source_id": "bucket-1"},
                {"sample_id": SEG[5], "source_id": "bucket-2"},
            ],
        )
        meta = self.manifest()
        self.assertEqual(meta["sample_ids"], [SEG[1], SEG[2], SEG[3], SEG[5]])
        self.assertNotIn("cache_pruned", meta)  # describes the cache, not the data
        self.assertEqual(meta["taxonomy_version"], 2)
        # Curated names only: "mystery flip" trains but is no canonical name.
        self.assertEqual(meta["canonical_names"], ["cork", "gainer", "hook", "scoot"])
        self.assertEqual(meta["held_back_names"], [])
        self.assertEqual(meta["contributor_minimum"], MINIMUM)
        log = (self.root / "dataset" / "unmapped_labels.log").read_text()
        self.assertEqual(log, "1\tmystery flip\n")

    def test_pin_is_the_farms_current_taxonomy_version(self):
        # A curation-only bump: version 3 maps a raw string no label was
        # confirmed under at v3. It applies to every label at the next run.
        curated = [
            *TAXONOMY,
            {"raw": "mystery flip", "canonical": "mystery", "taxonomy_version": 3},
        ]
        counts = {**COUNTS, "mystery": COUNTS["cork"]}
        stats = self.build(FakeExport(taxonomy=curated, counts=counts))
        self.assertEqual(stats["taxonomy_version"], 3)
        self.assertEqual(stats["unmapped_tokens"], 0)
        self.assertEqual(self.sample(5)["tricks"][0]["name"], "mystery")

    def test_byte_identical_across_page_sizes(self):
        self.build(FakeExport(page_size=1), out="one")
        self.build(FakeExport(page_size=10), out="ten")
        self.assertEqual(self.tree("one"), self.tree("ten"))

    def test_null_cursor_on_the_last_page(self):
        export = FakeExport(page_size=10, null_on_last_page=True)
        self.assertEqual(self.build(export)["samples"], 4)
        self.assertEqual(len(export.requests), 1)

    def test_blobs_are_cached_by_r2_key(self):
        self.build(out="first")
        self.assertEqual(len(self.blobs.calls), 4)
        entry = self.cache / key(1)
        self.assertEqual(entry.read_bytes(), BLOBS[1])
        self.build(out="second")
        self.assertEqual(len(self.blobs.calls), 4)  # all served from cache
        # A corrupt cache entry is discarded and refetched.
        entry.write_bytes(b"corrupt")
        self.build(out="third")
        self.assertEqual(self.blobs.calls[4:], [key(1)])
        self.assertEqual(entry.read_bytes(), BLOBS[1])

    def test_cache_is_pruned_to_the_pull(self):
        self.build(out="first")
        directory = self.cache / "segments"
        foreign = self.cache / "notes.txt"
        foreign.write_text("not ours")
        (directory / "readme.md").write_text("not ours")
        (directory / f"{SEG[1]}.tks1.gz.tmp").write_bytes(b"interrupted write")
        # Segment 5 was deleted farm-side, so it is absent from the next pull.
        kept = [s for s in SEGMENTS if s["segment_id"] != SEG[5]]
        stats = self.build(FakeExport(kept), out="second")
        self.assertEqual(stats["cache_pruned"], 2)  # 5's blob and the leftover
        self.assertEqual(
            sorted(p.name for p in directory.iterdir()),
            sorted([f"{SEG[n]}.tks1.gz" for n in (1, 2, 3)] + ["readme.md"]),
        )
        self.assertTrue(foreign.exists())

    def test_empty_pull_prunes_everything_then_refuses(self):
        self.build(out="first")
        with self.assertRaises(FetchError):
            self.build(FakeExport([]), out="second")
        self.assertEqual(list((self.cache / "segments").glob("*.tks1.gz")), [])

    def test_failed_pull_never_prunes(self):
        self.build(out="first")

        def broken(url, headers):
            raise FetchError("export request failed: HTTP 500")

        with self.assertRaises(FetchError):
            self.build(broken, out="second")
        self.assertEqual(len(list((self.cache / "segments").glob("*.tks1.gz"))), 4)

    def test_checksum_mismatch_fails_closed(self):
        self.blobs.blobs[key(2)] = BLOBS[1]
        with self.assertRaises(FetchError):
            self.build()
        self.assertFalse((self.root / "dataset").exists())
        self.assertFalse((self.cache / key(2)).exists())

    def test_undecodable_blob_fails_closed(self):
        bad = b"not gzip"
        self.blobs.blobs[key(3)] = bad
        self.assert_build_refused(
            [
                segment(
                    3,
                    [("Cork", "typed", 1)],
                    blob_sha256=hashlib.sha256(bad).hexdigest(),
                )
            ]
        )

    def test_frame_count_must_match_the_blob(self):
        self.assert_build_refused(
            [segment(3, [("Cork", "typed", 1)], frame_count=21, end=21)]
        )

    def test_any_download_failure_fails_closed(self):
        url = "https://acct.r2.example/bucket/segments/x?X-Amz-Signature=SECRET"
        for error in (KeyError("missing"), RuntimeError(url), OSError(url)):
            with self.subTest(error=type(error).__name__):

                def failing(r2_key, error=error):
                    raise error

                self.blobs = failing
                with self.assertRaisesRegex(FetchError, type(error).__name__) as ctx:
                    self.build(out=f"dl-{type(error).__name__}")
                self.assertNotIn("SECRET", str(ctx.exception))
                self.assertFalse((self.root / f"dl-{type(error).__name__}").exists())

    def test_out_dir_must_be_fresh(self):
        (self.root / "dataset").mkdir()
        with self.assertRaises(FetchError):
            self.build()
        self.assertEqual(self.blobs.calls, [])

    def test_empty_export(self):
        self.assert_build_refused([])


class HoldBackTest(FetchTestCase):
    def test_a_rare_trick_holds_back_its_whole_segment(self):
        # scoot is one contributor short: segment 1 (gainer + hook + scoot)
        # is held back whole, gainer and hook included.
        counts = {**COUNTS, "scoot": MINIMUM - 1}
        self.assertEqual(self.main_build(FakeExport(counts=counts)), EXIT_OK)
        stats = json.loads(self.stdout.getvalue())
        self.assertEqual(stats["segments_held_back"], 1)
        self.assertEqual(stats["names_held_back"], 1)
        self.assertEqual(stats["samples"], 3)
        self.assertFalse(self.has_sample(1, out="ds"))
        self.assertEqual(self.manifest(out="ds")["held_back_names"], ["scoot"])
        # Its blob is never fetched; stdout carries counts, never the name.
        self.assertNotIn(key(1), self.blobs.calls)
        self.assertNotIn("scoot", self.stdout.getvalue())

    def test_minimum_contributors_ship(self):
        stats = self.build(FakeExport(counts={**COUNTS, "scoot": MINIMUM}))
        self.assertEqual(stats["segments_held_back"], 0)
        self.assertTrue(self.has_sample(1))

    def test_the_farms_minimum_decides(self):
        counts = {**COUNTS, "scoot": MINIMUM}
        stats = self.build(FakeExport(counts=counts, minimum=MINIMUM + 1))
        self.assertEqual(stats["segments_held_back"], 1)
        self.assertEqual(self.manifest()["contributor_minimum"], MINIMUM + 1)

    def test_no_minimum_holds_every_trick_back(self):
        stats = self.build(FakeExport(minimum=None))
        self.assertEqual(stats["samples"], 0)
        self.assertEqual(stats["segments_held_back"], 4)
        self.assertIsNone(self.manifest()["contributor_minimum"])

    def test_the_minimum_is_the_greatest_over_pages(self):
        def minimum(start):
            return 10 if start == 2 else MINIMUM

        stats = self.build(FakeExport(minimum=minimum))
        self.assertEqual(stats["samples"], 0)  # every count is MINIMUM + 4 = 9

    def test_a_page_without_the_minimum_holds_every_trick_back(self):
        def minimum(start):
            return None if start == 4 else MINIMUM

        self.assertEqual(self.build(FakeExport(minimum=minimum))["samples"], 0)

    def test_no_counts_hold_every_trick_back(self):
        stats = self.build(FakeExport(counts=None))
        self.assertEqual(stats["samples"], 0)
        self.assertEqual(stats["names_held_back"], 5)

    def test_a_name_without_a_count_is_held_back(self):
        counts = {n: c for n, c in COUNTS.items() if n != "mystery flip"}
        self.build(FakeExport(counts=counts))
        self.assertFalse(self.has_sample(5))
        self.assertTrue(self.has_sample(3))

    def test_counts_are_the_minimum_over_pages(self):
        # One page says cork has few contributors; the others say many.
        def counts(start):
            return {**COUNTS, "cork": 1} if start == 2 else COUNTS

        self.build(FakeExport(counts=counts))
        self.assertFalse(self.has_sample(2))  # corkscrew -> cork
        self.assertFalse(self.has_sample(3))
        self.assertTrue(self.has_sample(1))

    def test_a_page_without_counts_holds_every_trick_back(self):
        def counts(start):
            return None if start == 4 else COUNTS

        self.assertEqual(self.build(FakeExport(counts=counts))["samples"], 0)

    def test_malformed_counts_fail_closed(self):
        good = {"canonical": "cork", "contributors": 9}
        bad_counts = [
            "cork",
            [["cork", 9]],
            [{**good, "contributors": -1}],
            [{**good, "contributors": "9"}],
            [{**good, "contributors": True}],
            [{**good, "canonical": ""}],
            [{**good, "extra": 1}],
            [good, good],
        ]
        exports = [
            (FakeExport(counts=lambda start, b=b: b), "trick_contributors")
            for b in bad_counts
        ] + [
            (FakeExport(minimum=b), "contributor_minimum")
            for b in (0, -1, "5", True, 2.5)
        ]
        for i, (export, field) in enumerate(exports):
            with self.subTest(field=field, case=i):
                self.stderr.truncate(0)
                self.assertEqual(self.main_build(export, out=f"o{i}"), EXIT_ERROR)
                self.assertIn(field, self.stderr.getvalue())
                self.assertFalse((self.root / f"o{i}").exists())


class SegmentValidationTest(FetchTestCase):
    def test_malformed_segments(self):
        cork = [("Cork", "typed", 1)]
        cases = {
            "not an object": [],
            "missing field": {
                k: v for k, v in segment(3, cork).items() if k != "split_bucket"
            },
            "segment_id not a uuid": segment(3, cork, segment_id="seg-3"),
            "segment_id upper case": segment(3, cork, segment_id=SEG[3].upper()),
            "r2_key elsewhere": segment(3, cork, r2_key=f"poses/{SEG[3]}.tks1.gz"),
            "blob_sha256": segment(3, cork, blob_sha256="ABC"),
            "window reversed": segment(3, cork, start=10, end=5),
            "window past the segment": segment(3, cork, start=5, end=21),
            "window negative": segment(3, cork, start=-1, end=5),
            "window not ints": segment(3, cork, start=5.0, end=15),
            "frame_count bool": segment(3, cork, frame_count=True),
            "split_bucket 10": segment(3, cork, bucket=10),
            "split_bucket str": segment(3, cork, bucket="1"),
            "no labels": segment(3, []),
            "labels unsorted": segment(
                3, [("gainer", "typed", 1), ("Cork", "typed", 1)]
            ),
            "label twice": segment(3, [("Cork", "typed", 1), ("Cork", "typed", 1)]),
            "blank name": segment(3, [(" ", "typed", 1)]),
            "version 0": segment(3, [("Cork", "typed", 0)]),
            "bad origin": segment(3, [("Cork", "suggested", 1)]),
            "label extra key": {
                **segment(3, cork),
                "labels": [{**segment(3, cork)["labels"][0], "x": 1}],
            },
        }
        for name, entry in cases.items():
            with self.subTest(name):
                self.assert_build_refused([entry])

    def test_identifying_fields_are_refused(self):
        for field in ("user_id", "source_id"):
            with self.subTest(field):
                entry = segment(3, [("Cork", "typed", 1)], **{field: "a" * 64})
                self.assert_build_refused([entry])

    def test_malformed_provenance(self):
        cork = [("Cork", "typed", 1)]
        cases = {
            "version not a string": segment(3, cork, proposal_model_version=7),
            "version empty": segment(3, cork, proposal_model_version=""),
            "names not a list": segment(
                3, cork, suggested_names="cork", proposal_model_version=MODEL
            ),
            "blank name": segment(
                3, cork, suggested_names=[""], proposal_model_version=MODEL
            ),
            "withheld not bool": segment(
                3, cork, suggestions_withheld=1, proposal_model_version=MODEL
            ),
            "names without model": segment(3, cork, suggested_names=["cork"]),
            "blind without model": segment(3, cork, suggestions_withheld=True),
            "model origin without model": segment(3, [("Cork", "model", 1)]),
            "model origin on blind clip": segment(3, [("Cork", "model", 1)], **BLIND),
        }
        for name, entry in cases.items():
            with self.subTest(name):
                self.assert_build_refused([entry])

    def test_a_segment_served_twice_fails_closed(self):
        self.assert_build_refused([SEGMENTS[0], SEGMENTS[2], SEGMENTS[0]])

    def test_malformed_taxonomy(self):
        bad = [{"raw": "cork", "canonical": "Cork", "taxonomy_version": 1}]
        with self.assertRaises(FetchError):
            self.build(FakeExport(taxonomy=bad))


class PaginationTest(FetchTestCase):
    def test_requests_carry_the_key_and_the_cursor(self):
        export = FakeExport(page_size=3)
        pull = ExportClient(FARM + "/", KEY, export).pull()
        self.assertEqual(len(pull.segments), 5)
        self.assertEqual(
            [s.segment_id for s in pull.segments], sorted(SEG[n] for n in range(1, 6))
        )
        self.assertEqual(
            [url for url, _ in export.requests],
            [
                f"{FARM}/api/labels/export",
                f"{FARM}/api/labels/export?since=cur-2",
                f"{FARM}/api/labels/export?since=cur-4",
            ],
        )
        self.assertEqual(export.requests[0][1]["Authorization"], f"Bearer {KEY}")

    def test_stuck_or_cycling_cursor(self):
        for cursors in (["x", "x"], ["x", "y", "x"]):
            with self.subTest(cursors=cursors):
                pages = iter(cursors)
                entries = iter(SEGMENTS)

                def export(url, headers, pages=pages, entries=entries):
                    return {"segments": [next(entries)], "next_cursor": next(pages)}

                with self.assertRaises(FetchError):
                    ExportClient(FARM, KEY, export).pull()

    def test_malformed_pages(self):
        entry = SEGMENTS[0]
        for page in (
            [],
            {"segments": {}},
            {"rows": [entry]},
            {"segments": [], "taxonomy": {}},
            {"segments": [entry], "next_cursor": 7},
            {"segments": [entry], "next_cursor": ""},
        ):
            with self.subTest(page=str(page)[:40]), self.assertRaises(FetchError):
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
            mock.patch("fetch_dataset.farm_urlopen", side_effect=error),
            self.assertRaisesRegex(FetchError, "HTTP 401"),
        ):
            fetch_dataset.http_get_json(FARM, {})

    def test_transport_failures_and_bad_json(self):
        with (
            mock.patch("fetch_dataset.farm_urlopen", side_effect=TimeoutError()),
            self.assertRaisesRegex(FetchError, "TimeoutError"),
        ):
            fetch_dataset.http_get_json(FARM, {})
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"not json"
        with (
            mock.patch("fetch_dataset.farm_urlopen", return_value=response),
            self.assertRaises(FetchError),
        ):
            fetch_dataset.http_get_json(FARM, {})


class CountTest(FetchTestCase):
    def count(self, export, state=None, segments_out=None):
        argv = ["count", "--out-state", str(self.root / "new_state.json")]
        if state is not None:
            argv += ["--state", str(state)]
        if segments_out is not None:
            argv += ["--out-segments", str(segments_out)]
        self.stdout.truncate(0)
        self.stdout.seek(0)
        code = fetch_dataset.main(argv, transport=export)
        return code, (json.loads(self.stdout.getvalue()) if code == EXIT_OK else None)

    def test_first_run_counts_everything(self):
        new = self.root / "new.json"
        code, result = self.count(FakeExport(), segments_out=new)
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(result, {"new_segments": 5, "new_labels": 6})
        self.assertEqual(
            json.loads(new.read_text()), sorted(SEG[n] for n in range(1, 6))
        )
        state = json.loads((self.root / "new_state.json").read_text())
        self.assertEqual(state, {"segment_ids": sorted(SEG[n] for n in range(1, 6))})

    def test_only_segments_absent_from_the_state_are_new(self):
        self.count(FakeExport(SEGMENTS[:3]))
        state = self.root / "state.json"
        (self.root / "new_state.json").rename(state)
        # Segment 2 is new; segment 3 was re-cut away (absent, not new).
        later = [SEGMENTS[1], SEGMENTS[2], SEGMENTS[3]]
        new = self.root / "new.json"
        code, result = self.count(FakeExport(later), state=state, segments_out=new)
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(result, {"new_segments": 1, "new_labels": 1})
        self.assertEqual(json.loads(new.read_text()), [SEG[2]])

    def test_nothing_new(self):
        self.count(FakeExport())
        state = self.root / "state.json"
        (self.root / "new_state.json").rename(state)
        new = self.root / "new.json"
        code, result = self.count(FakeExport(), state=state, segments_out=new)
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(result, {"new_segments": 0, "new_labels": 0})
        self.assertEqual(json.loads(new.read_text()), [])

    def test_a_state_from_before_segments_counts_everything(self):
        state = self.root / "state.json"
        state.write_text('{"watermark": "old-row-cursor"}')
        code, result = self.count(FakeExport(), state=state)
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(result["new_segments"], 5)

    def test_malformed_state_file(self):
        for content in (
            "not json",
            "{}",
            '{"segment_ids": "x"}',
            '{"segment_ids": ["not-a-uuid"]}',
            '{"segment_ids": [], "watermark": null}',
        ):
            with self.subTest(state=content):
                state = self.root / "state.json"
                state.write_text(content)
                self.assertEqual(self.count(FakeExport(), state=state)[0], EXIT_ERROR)

    def test_malformed_provenance_fails_count_too(self):
        bad = segment(3, [("Cork", "suggested", 1)])
        self.assertEqual(self.count(FakeExport([bad]))[0], EXIT_ERROR)
        self.assertFalse((self.root / "new_state.json").exists())


class MainTest(FetchTestCase):
    def test_build_via_main_prints_counts_only(self):
        self.assertEqual(self.main_build(FakeExport()), EXIT_OK)
        self.assertEqual(json.loads(self.stdout.getvalue())["samples"], 4)
        output = self.stdout.getvalue() + self.stderr.getvalue()
        for secret in ("Mystery", "corkscrew", "cur-", KEY, MODEL, "vocabulary"):
            self.assertNotIn(secret, output)
        for segment_id in SEG.values():
            self.assertNotIn(segment_id, output)

    def test_malformed_provenance_exits_2_without_leaking(self):
        cases = {
            "origin": segment(3, [("Cork", "leaked-origin", 1)]),
            "version": segment(3, [("Cork", "typed", 1)], proposal_model_version=[1]),
            "names": segment(3, [("Cork", "typed", 1)], suggested_names=["leaked"]),
        }
        for i, (name, entry) in enumerate(cases.items()):
            with self.subTest(case=name):
                self.stderr.truncate(0)
                self.stderr.seek(0)
                code = self.main_build(FakeExport([entry]), out=f"ds{i}")
                self.assertEqual(code, EXIT_ERROR)
                self.assertIn("export segment 0", self.stderr.getvalue())
                self.assertNotIn("leaked", self.stderr.getvalue())
                self.assertEqual(self.stdout.getvalue(), "")

    def test_build_failure_exits_2_without_leaking(self):
        self.blobs.blobs[key(2)] = BLOBS[1]
        self.assertEqual(self.main_build(FakeExport()), EXIT_ERROR)
        self.assertIn("blob_sha256", self.stderr.getvalue())
        for secret in ("Mystery", KEY):
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

    @unittest.skipUnless(importlib.util.find_spec("boto3"), "boto3 not installed")
    def test_r2_getter_reads_the_bucket_and_hides_error_details(self):
        from botocore.exceptions import ClientError

        env = {
            "R2_ACCOUNT_ID": "acct",
            "R2_ACCESS_KEY_ID": "key-id",
            "R2_SECRET_ACCESS_KEY": "SECRET",
            "R2_BUCKET": "segments-bucket",
        }
        client = mock.MagicMock()
        client.get_object.return_value = {"Body": io.BytesIO(BLOBS[1])}
        with (
            mock.patch.dict(os.environ, env),
            mock.patch("boto3.client", return_value=client) as make_client,
        ):
            get = fetch_dataset.r2_getter_from_env()
            self.assertEqual(get(key(1)), BLOBS[1])
            client.get_object.side_effect = ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "SECRET detail"}},
                "GetObject",
            )
            with self.assertRaisesRegex(FetchError, "ClientError") as ctx:
                get(key(2))
        self.assertNotIn("SECRET", str(ctx.exception))
        self.assertEqual(make_client.call_args.args, ("s3",))
        self.assertEqual(
            make_client.call_args.kwargs["endpoint_url"],
            "https://acct.r2.cloudflarestorage.com",
        )
        client.get_object.assert_called_with(Bucket="segments-bucket", Key=key(2))

    @unittest.skipIf(importlib.util.find_spec("boto3"), "boto3 is installed")
    def test_r2_getter_requires_boto3(self):
        with (
            mock.patch.dict(os.environ, {name: "x" for name in fetch_dataset.R2_ENV}),
            self.assertRaisesRegex(FetchError, "boto3"),
        ):
            fetch_dataset.r2_getter_from_env()


if __name__ == "__main__":
    unittest.main()
