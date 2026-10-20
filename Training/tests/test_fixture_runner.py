"""Unit tests for Training/fixture_runner.py and make_synthetic_fixtures.py.

Hermetic: synthetic fixture sets only (make_synthetic_fixtures.py or
built here), and the smoke model trained on the committed smoke fixture.
Run with:  python -m unittest discover -s Training/tests -t .
"""

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fixture_gate
import fixture_runner
import make_synthetic_fixtures
import tkp1
import train
from fixture_runner import (
    EXIT_ERROR,
    EXIT_NOTHING_TO_SCORE,
    EXIT_OK,
    EXIT_OTHER_SET,
    Fixture,
    FixtureError,
    NothingToScore,
    fixture_set_digest,
    load_fixture_set,
    score,
)

REPO = Path(__file__).resolve().parent.parent.parent
SMOKE_CONFIG = REPO / "Training" / "config" / "smoke.toml"
REAL_SET = REPO / "Training" / "fixtures" / "tricks" / "fixtures.json"
SHA_A = "a" * 64
SHA_B = "b" * 64


def seg(start, end, *names):
    return {"start_frame": start, "end_frame": end, "names": list(names)}


def fixture(fixture_id="f1", sha=SHA_A, segments=None):
    return {
        "id": fixture_id,
        "pose_sha256": sha,
        "segments": [seg(4, 12, "gainer")] if segments is None else segments,
    }


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.stdout = self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.stderr = self.enterContext(contextlib.redirect_stderr(io.StringIO()))

    def tearDown(self):
        self.tmp.cleanup()

    def write_set(self, data, name="fixtures.json"):
        path = self.root / name
        path.write_text(data if isinstance(data, str) else json.dumps(data))
        return path

    def synthetic_set(self, name="synthetic"):
        out = self.root / name
        make_synthetic_fixtures.main(["--out", str(out), "--seed", "0"])
        return out / "fixtures.json", out / "poses"


class LoadFixtureSetTest(TempDirTestCase):
    def test_valid_set_is_sorted_and_normalized(self):
        path = self.write_set(
            {
                "fixtures": [
                    fixture("b", segments=[seg(20, 30, "gainer", "corkscrew")]),
                    fixture("a", segments=[seg(10, 15, "gainer"), seg(0, 5, "hook")]),
                ]
            }
        )
        fixtures = load_fixture_set(path)
        self.assertEqual([f.id for f in fixtures], ["a", "b"])
        self.assertEqual(
            fixtures[0].segments,
            ((0, 5, frozenset({"hook"})), (10, 15, frozenset({"gainer"}))),
        )
        self.assertEqual(fixtures[1].segments[0][2], {"corkscrew", "gainer"})

    def test_background_only_fixture_allowed(self):
        path = self.write_set(
            {"fixtures": [fixture("bg", segments=[]), fixture("trick")]}
        )
        self.assertEqual(load_fixture_set(path)[0].segments, ())

    def test_real_set_is_well_formed(self):
        # Whatever the maintainer's set holds (it ships empty), it parses;
        # its pose blobs live in R2, so they are not checked here.
        fixture_runner.read_fixture_set(REAL_SET)

    def test_nothing_to_score(self):
        for case, data in (
            ("empty", {"fixtures": []}),
            ("only background", {"fixtures": [fixture(segments=[])]}),
        ):
            with self.subTest(case=case):
                path = self.write_set(data)
                fixture_runner.read_fixture_set(path)  # well-formed
                with self.assertRaises(NothingToScore):
                    load_fixture_set(path)

    def test_duplicated_keys_fail_closed(self):
        for case, text in (
            (
                "fixture key",
                '{"fixtures": [{"id": "f1", "id": "f2", "pose_sha256": "'
                + SHA_A
                + '", "segments": []}]}',
            ),
            (
                "segment key",
                '{"fixtures": [{"id": "f1", "pose_sha256": "'
                + SHA_A
                + '", "segments": [{"start_frame": 0, "end_frame": 4, '
                '"names": ["hook"], "names": ["gainer"]}]}]}',
            ),
            ("top-level key", '{"fixtures": [], "fixtures": []}'),
        ):
            with (
                self.subTest(case=case),
                self.assertRaisesRegex(FixtureError, "duplicated keys"),
            ):
                load_fixture_set(self.write_set(text))

    def test_malformed_sets_fail_closed(self):
        bad = {
            "not json": "{",
            "not an object": [],
            "extra top-level key": {"fixtures": [fixture()], "notes": "x"},
            "fixtures not a list": {"fixtures": {}},
            "empty": {"fixtures": []},
            "only background": {"fixtures": [fixture(segments=[])]},
            "missing key": {"fixtures": [{"id": "f1", "pose_sha256": SHA_A}]},
            "extra key": {"fixtures": [{**fixture(), "note": "gainer"}]},
            "bad id": {"fixtures": [fixture("Gainer 1")]},
            "duplicate id": {"fixtures": [fixture("f1"), fixture("f1", SHA_B)]},
            "short sha": {"fixtures": [fixture(sha="abc")]},
            "upper sha": {"fixtures": [fixture(sha="A" * 64)]},
            "segments not a list": {"fixtures": [fixture(segments={})]},
            "segment extra key": {
                "fixtures": [fixture(segments=[{**seg(0, 4, "gainer"), "x": 1}])]
            },
            "float frame": {"fixtures": [fixture(segments=[seg(0.0, 4, "gainer")])]},
            "bool frame": {"fixtures": [fixture(segments=[seg(False, 4, "gainer")])]},
            "negative start": {"fixtures": [fixture(segments=[seg(-1, 4, "gainer")])]},
            "empty span": {"fixtures": [fixture(segments=[seg(4, 4, "gainer")])]},
            "no names": {"fixtures": [fixture(segments=[seg(0, 4)])]},
            "combo string": {
                "fixtures": [fixture(segments=[seg(0, 4, "gainer+corkscrew")])]
            },
            "not normalized": {"fixtures": [fixture(segments=[seg(0, 4, "Gainer")])]},
            "background name": {
                "fixtures": [fixture(segments=[seg(0, 4, "background")])]
            },
            "duplicate names": {
                "fixtures": [fixture(segments=[seg(0, 4, "gainer", "gainer")])]
            },
            "overlap": {
                "fixtures": [
                    fixture(segments=[seg(0, 10, "gainer"), seg(8, 14, "corkscrew")])
                ]
            },
        }
        for case, data in bad.items():
            with self.subTest(case=case), self.assertRaises(FixtureError):
                load_fixture_set(self.write_set(data))

    def test_adjacent_segments_do_not_overlap(self):
        path = self.write_set(
            {"fixtures": [fixture(segments=[seg(0, 8, "gainer"), seg(8, 14, "hook")])]}
        )
        self.assertEqual(len(load_fixture_set(path)[0].segments), 2)

    def test_unreadable_set(self):
        with self.assertRaises(FixtureError):
            load_fixture_set(self.root / "missing.json")


BASE = {
    "fixtures": [
        fixture("a", SHA_A, [seg(0, 8, "gainer"), seg(10, 20, "hook", "scoot")]),
        fixture("b", SHA_B, [seg(4, 12, "corkscrew")]),
    ]
}


class DigestTest(TempDirTestCase):
    def digest(self, data):
        return fixture_set_digest(load_fixture_set(self.write_set(data)))

    def test_hex_digest_the_gate_accepts(self):
        digest = self.digest(BASE)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertTrue(fixture_gate.is_identifier(digest))

    def test_stable_across_formatting_and_order(self):
        reordered = {
            "fixtures": [
                {
                    "segments": [seg(4, 12, "corkscrew")],
                    "pose_sha256": SHA_B,
                    "id": "b",
                },
                fixture(
                    "a", SHA_A, [seg(10, 20, "scoot", "hook"), seg(0, 8, "gainer")]
                ),
            ]
        }
        self.assertEqual(self.digest(BASE), self.digest(reordered))
        pretty = json.dumps(BASE, indent=4)
        self.assertEqual(self.digest(BASE), self.digest(pretty))

    def test_any_fixture_change_changes_it(self):
        base = self.digest(BASE)
        a, b = BASE["fixtures"]
        changes = {
            "label renamed": [
                fixture("a", SHA_A, [seg(0, 8, "gainer"), seg(10, 20, "hook")]),
                b,
            ],
            "combo name added": [
                a,
                fixture("b", SHA_B, [seg(4, 12, "corkscrew", "gainer")]),
            ],
            "boundary moved": [a, fixture("b", SHA_B, [seg(4, 13, "corkscrew")])],
            "pose changed": [a, fixture("b", "c" * 64, [seg(4, 12, "corkscrew")])],
            "segment removed": [a, fixture("b", SHA_B, [])],
            "fixture removed": [a],
            "fixture added": [a, b, fixture("c", SHA_A, [])],
            "fixture renamed": [a, fixture("c", SHA_B, [seg(4, 12, "corkscrew")])],
        }
        digests = {base}
        for case, fixtures in changes.items():
            with self.subTest(case=case):
                digest = self.digest({"fixtures": fixtures})
                self.assertNotIn(digest, digests)
                digests.add(digest)

    def test_scoring_rule_is_part_of_it(self):
        fixtures = load_fixture_set(self.write_set(BASE))
        before = fixture_set_digest(fixtures)
        original = dict(fixture_runner.SCORING)
        try:
            fixture_runner.SCORING["version"] = original["version"] + 1
            self.assertNotEqual(fixture_set_digest(fixtures), before)
        finally:
            fixture_runner.SCORING.clear()
            fixture_runner.SCORING.update(original)


class ScoreTest(unittest.TestCase):
    def fixture(self, fixture_id, *segments):
        return Fixture(fixture_id, SHA_A, tuple(segments))

    def test_perfect(self):
        f = self.fixture("a", (0, 10, frozenset({"gainer"})))
        s = score([f], {"a": [(0, 10, frozenset({"gainer"}))]})
        self.assertEqual((s["segment_iou"], s["name_accuracy"]), (1.0, 1.0))

    def test_partial_overlap_and_wrong_name(self):
        f = self.fixture(
            "a", (0, 10, frozenset({"gainer"})), (20, 30, frozenset({"hook"}))
        )
        # IoU 0.8 named right; IoU 1.0 named wrong.
        s = score(
            [f],
            {"a": [(0, 8, frozenset({"gainer"})), (20, 30, frozenset({"scoot"}))]},
        )
        self.assertAlmostEqual(s["segment_iou"], 1.8 / 2)
        self.assertEqual(s["name_accuracy"], 0.5)

    def test_combo_needs_every_name(self):
        f = self.fixture("a", (0, 10, frozenset({"gainer", "hook"})))
        s = score([f], {"a": [(0, 10, frozenset({"gainer"}))]})
        self.assertEqual((s["segment_iou"], s["name_accuracy"]), (1.0, 0.0))

    def test_below_detection_threshold_is_a_miss_and_a_false_positive(self):
        f = self.fixture("a", (0, 10, frozenset({"gainer"})))
        # IoU 4/14 < 0.5: unmatched on both sides.
        s = score([f], {"a": [(6, 14, frozenset({"gainer"}))]})
        self.assertEqual((s["segment_iou"], s["name_accuracy"]), (0.0, 0.0))
        self.assertEqual((s["n_matched"], s["n_predicted"]), (0, 1))

    def test_missed_segment_lowers_both(self):
        f = self.fixture(
            "a", (0, 10, frozenset({"gainer"})), (20, 30, frozenset({"hook"}))
        )
        s = score([f], {"a": [(0, 10, frozenset({"gainer"}))]})
        self.assertEqual((s["segment_iou"], s["name_accuracy"]), (0.5, 0.5))

    def test_false_positive_on_background_fixture_lowers_iou(self):
        fixtures = [
            self.fixture("a", (0, 10, frozenset({"gainer"}))),
            self.fixture("bg"),
        ]
        predictions = {
            "a": [(0, 10, frozenset({"gainer"}))],
            "bg": [(0, 5, frozenset({"hook"}))],
        }
        s = score(fixtures, predictions)
        self.assertEqual((s["segment_iou"], s["name_accuracy"]), (0.5, 1.0))

    def test_nothing_predicted(self):
        f = self.fixture("a", (0, 10, frozenset({"gainer"})))
        s = score([f], {"a": []})
        self.assertEqual((s["segment_iou"], s["name_accuracy"]), (0.0, 0.0))

    def test_pooled_over_fixtures(self):
        fixtures = [
            self.fixture("a", (0, 10, frozenset({"gainer"}))),
            self.fixture(
                "b", (0, 10, frozenset({"hook"})), (20, 30, frozenset({"hook"}))
            ),
        ]
        s = score(fixtures, {"a": [(0, 10, frozenset({"gainer"}))], "b": []})
        self.assertAlmostEqual(s["segment_iou"], 1 / 3)
        self.assertAlmostEqual(s["name_accuracy"], 1 / 3)


class PoseTest(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.poses = self.root / "poses"
        self.poses.mkdir()
        frames = [[[0.5, 0.5, 0.9]] * 17] * 20
        self.blob = tkp1.encode_gz(frames)
        self.sha = hashlib.sha256(self.blob).hexdigest()

    def fixture(self, sha=None, end=12):
        return Fixture("f1", sha or self.sha, ((4, end, frozenset({"gainer"})),))

    def test_verified_pose_loads(self):
        (self.poses / f"{self.sha}.tkp1.gz").write_bytes(self.blob)
        kps = fixture_runner.load_pose(self.poses, self.fixture())
        self.assertEqual(kps.shape, (20, 17, 3))

    def test_missing_blob(self):
        with self.assertRaisesRegex(FixtureError, "not found"):
            fixture_runner.load_pose(self.poses, self.fixture())

    def test_tampered_blob_is_refused(self):
        (self.poses / f"{self.sha}.tkp1.gz").write_bytes(self.blob + b"\0")
        with self.assertRaisesRegex(FixtureError, "does not match its pose_sha256"):
            fixture_runner.load_pose(self.poses, self.fixture())

    def test_undecodable_blob_is_refused(self):
        junk = b"not a pose"
        sha = hashlib.sha256(junk).hexdigest()
        (self.poses / f"{sha}.tkp1.gz").write_bytes(junk)
        with self.assertRaisesRegex(FixtureError, "undecodable"):
            fixture_runner.load_pose(self.poses, self.fixture(sha))

    def test_segment_past_the_pose(self):
        (self.poses / f"{self.sha}.tkp1.gz").write_bytes(self.blob)
        with self.assertRaisesRegex(FixtureError, "past the 20-frame pose"):
            fixture_runner.load_pose(self.poses, self.fixture(end=21))


class SyntheticFixturesTest(TempDirTestCase):
    def test_deterministic_and_valid(self):
        fixtures_a, poses_a = self.synthetic_set("a")
        fixtures_b, poses_b = self.synthetic_set("b")
        self.assertEqual(fixtures_a.read_bytes(), fixtures_b.read_bytes())
        for blob in sorted(poses_a.iterdir()):
            self.assertEqual(blob.read_bytes(), (poses_b / blob.name).read_bytes())
        fixtures = load_fixture_set(fixtures_a)
        self.assertEqual(len(fixtures), len(make_synthetic_fixtures.FIXTURES))
        for f in fixtures:
            fixture_runner.load_pose(poses_a, f)  # every blob verifies


class RunnerEndToEndTest(TempDirTestCase):
    """Runner output -> fixture_gate.py on a synthetic model."""

    @classmethod
    def setUpClass(cls):
        cls.model_dir = tempfile.TemporaryDirectory()
        with contextlib.redirect_stdout(io.StringIO()):
            train.run(SMOKE_CONFIG, cls.model_dir.name)
        cls.model = Path(cls.model_dir.name) / "model.npz"

    @classmethod
    def tearDownClass(cls):
        cls.model_dir.cleanup()

    def setUp(self):
        super().setUp()
        self.fixtures, self.poses = self.synthetic_set()

    def run_runner(self, model, out, fixtures=None):
        return fixture_runner.main(
            [
                "score",
                "--model",
                str(model),
                "--fixtures",
                str(fixtures or self.fixtures),
                "--poses",
                str(self.poses),
                "--out",
                str(out),
            ]
        )

    def blind_model(self):
        """The smoke model with every window forced to background."""
        ckpt = train.load_checkpoint(self.model)
        classes = [str(c) for c in ckpt["classes"]]
        ckpt["W"] = np.zeros_like(ckpt["W"])
        ckpt["b"] = np.where(np.array(classes) == train.BACKGROUND, 1.0, 0.0)
        path = self.root / "blind.npz"
        np.savez(path, **ckpt)
        return path

    def gate(self, champion, challenger):
        ingested = self.root / "ingested.json"
        ingested.write_text(json.dumps(["d" * 64]))
        out = self.root / "gate"
        code = fixture_gate.main(
            [
                "--champion",
                str(champion),
                "--challenger",
                str(challenger),
                "--ingested",
                str(ingested),
                "--out",
                str(out),
                "--registry",
                str(self.root / "registry.json"),
            ],
            send=lambda url, payload: None,
        )
        return code, json.loads((out / "quarantine.json").read_text())

    def test_score_file_is_byte_identical_on_rerun(self):
        a, b = self.root / "a.json", self.root / "b.json"
        self.assertEqual(self.run_runner(self.model, a), EXIT_OK)
        self.assertEqual(self.run_runner(self.model, b), EXIT_OK)
        self.assertEqual(a.read_bytes(), b.read_bytes())
        scores = json.loads(a.read_text())
        self.assertEqual(
            sorted(scores), ["fixture_set", "name_accuracy", "segment_iou"]
        )
        self.assertEqual(
            a.read_text(), json.dumps(scores, indent=2, sort_keys=True) + "\n"
        )
        # The gate accepts it as is.
        self.assertEqual(fixture_gate.load_scores(a), scores)

    def test_changed_fixture_changes_fixture_set(self):
        a, b = self.root / "a.json", self.root / "b.json"
        self.run_runner(self.model, a)
        data = json.loads(self.fixtures.read_text())
        data["fixtures"][0]["segments"][0]["end_frame"] -= 1
        changed = self.write_set(data, "changed.json")
        self.run_runner(self.model, b, changed)
        self.assertNotEqual(
            json.loads(a.read_text())["fixture_set"],
            json.loads(b.read_text())["fixture_set"],
        )

    def test_same_model_passes_the_gate(self):
        champion, challenger = (
            self.root / "champion.json",
            self.root / "challenger.json",
        )
        self.run_runner(self.model, champion)
        self.run_runner(self.model, challenger)
        code, quarantine = self.gate(champion, challenger)
        self.assertEqual(code, fixture_gate.EXIT_PASS)
        self.assertEqual(quarantine, [])

    def test_blind_challenger_regresses(self):
        champion, challenger = (
            self.root / "champion.json",
            self.root / "challenger.json",
        )
        self.run_runner(self.model, champion)
        self.run_runner(self.blind_model(), challenger)
        blind = json.loads(challenger.read_text())
        self.assertEqual((blind["segment_iou"], blind["name_accuracy"]), (0.0, 0.0))
        code, quarantine = self.gate(champion, challenger)
        self.assertEqual(code, fixture_gate.EXIT_REGRESSION)
        self.assertEqual([row["source_id"] for row in quarantine], ["d" * 64])

    def test_scores_from_another_fixture_set_are_refused_by_the_gate(self):
        champion, challenger = (
            self.root / "champion.json",
            self.root / "challenger.json",
        )
        data = json.loads(self.fixtures.read_text())
        data["fixtures"].pop()
        self.run_runner(self.model, champion, self.write_set(data, "older.json"))
        self.run_runner(self.model, challenger)
        ingested = self.root / "ingested.json"
        ingested.write_text("[]")
        with self.assertRaisesRegex(fixture_gate.GateError, "fixture sets differ"):
            fixture_gate.run(
                champion,
                challenger,
                ingested,
                self.root / "gate",
                fixture_gate.DEFAULT_TOLERANCE,
                self.root / "registry.json",
            )

    def test_errors_exit_2_and_write_nothing(self):
        out = self.root / "scores.json"
        not_a_zip = self.root / "corrupt.npz"
        not_a_zip.write_bytes(b"PK\x03\x04 not really a zip")
        bare_npy = self.root / "array.npz"
        with open(bare_npy, "wb") as fh:
            np.save(fh, np.zeros(3))
        truncated = self.root / "truncated.npz"
        truncated.write_bytes(self.model.read_bytes()[:-200])
        empty = self.write_set({"fixtures": []}, "empty.json")
        cases = {
            "missing model": (self.root / "none.npz", self.fixtures),
            "not a zip": (not_a_zip, self.fixtures),
            "bare .npy": (bare_npy, self.fixtures),
            "truncated npz": (truncated, self.fixtures),
            "empty set": (self.model, empty),
        }
        for case, (model, fixtures) in cases.items():
            with self.subTest(case=case):
                self.assertEqual(self.run_runner(model, out, fixtures), EXIT_ERROR)
                self.assertFalse(out.exists())

    def test_pose_shorter_than_the_window_is_refused(self):
        window = int(train.load_checkpoint(self.model)["window_frames"])
        blob = tkp1.encode_gz([[[0.5, 0.5, 0.9]] * 17] * (window - 1))
        sha = hashlib.sha256(blob).hexdigest()
        (self.poses / f"{sha}.tkp1.gz").write_bytes(blob)
        data = json.loads(self.fixtures.read_text())
        data["fixtures"].append({"id": "too-short", "pose_sha256": sha, "segments": []})
        out = self.root / "scores.json"
        code = self.run_runner(self.model, out, self.write_set(data, "short.json"))
        self.assertEqual(code, EXIT_ERROR)
        self.assertIn(f"{window}-frame window", self.stderr.getvalue())
        self.assertFalse(out.exists())

    def test_check(self):
        background = json.loads(self.fixtures.read_text())
        for f in background["fixtures"]:
            f["segments"] = []
        cases = {
            "scoreable": (self.fixtures, EXIT_OK),
            "empty": (
                self.write_set({"fixtures": []}, "e.json"),
                EXIT_NOTHING_TO_SCORE,
            ),
            "background only": (
                self.write_set(background, "bg.json"),
                EXIT_NOTHING_TO_SCORE,
            ),
            "malformed": (self.write_set({"fixtures": {}}, "m.json"), EXIT_ERROR),
        }
        for case, (fixtures, code) in cases.items():
            with self.subTest(case=case):
                argv = ["check", "--fixtures", str(fixtures)]
                self.assertEqual(fixture_runner.main(argv), code)

    def test_compare(self):
        scores = self.root / "scores.json"
        self.run_runner(self.model, scores)
        other = json.loads(scores.read_text())
        other["fixture_set"] = "f" * 64
        unreadable = self.root / "unreadable.json"
        unreadable.write_text("{")
        cases = {
            "same set": (self.fixtures, scores, EXIT_OK),
            "none recorded": (self.fixtures, self.root / "none.json", EXIT_OTHER_SET),
            "another set": (
                self.fixtures,
                self.write_set(other, "other.json"),
                EXIT_OTHER_SET,
            ),
            "unreadable": (self.fixtures, unreadable, EXIT_OTHER_SET),
            "malformed set": (
                self.write_set({"fixtures": {}}, "m.json"),
                scores,
                EXIT_ERROR,
            ),
        }
        for case, (fixtures, recorded, code) in cases.items():
            with self.subTest(case=case):
                argv = ["compare", "--fixtures", str(fixtures)]
                argv += ["--scores", str(recorded)]
                self.assertEqual(fixture_runner.main(argv), code)

    def test_stdout_carries_aggregates_only(self):
        self.run_runner(self.model, self.root / "scores.json")
        line = self.stdout.getvalue()
        self.assertIn("segment_iou", line)
        self.assertNotIn("synthetic-", line)  # no fixture ids


if __name__ == "__main__":
    unittest.main()
