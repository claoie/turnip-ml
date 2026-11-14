"""End-to-end tests for Training/pipeline.sh: fixture runner -> fixture
gate -> promotion gate, and the explicit stops.

Hermetic: each test runs the real script in a temporary copy of
Training/ (train.py resolves its config paths against the directory it
lives in, so a symlink would write into the real repository's data/), on
the synthetic smoke dataset and a synthetic fixture set, with an
environment built from scratch. No network, no secrets.
Run with:  python -m unittest discover -s Training/tests -t .
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SOURCE_A = "a" * 64
SOURCE_B = "b" * 64
# What a test run inherits from the caller; every pipeline knob
# (FROM_SCRATCH, BOOTSTRAP_CHAMPION, ...) is set by the test or not at all.
INHERITED_ENV = ("HOME", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT")
FULL_SLOT = ["champion.json", "fixture_scores.json", "metrics.json", "model.npz"]
# The smoke dataset has 4 train and 2 holdout sources, far below the
# membership-inference check's minimum, so the template's copy of the check
# is made lenient: every test but the one that checks the real constants
# exercises the gates behind it.
MI_STRICT = "MAX_AUC = 0.60\nMIN_SOURCES = 20\n"
MI_LENIENT = "MAX_AUC = 1.0\nMIN_SOURCES = 1\n"

# Built once for the module (see setUpModule).
BIN: Path
TEMPLATE: Path


def run_pipeline(cwd, **env):
    full_env = {k: os.environ[k] for k in INHERITED_ENV if k in os.environ}
    full_env.update(
        PATH=f"{BIN}{os.pathsep}{os.environ.get('PATH', '')}",
        HOLDOUT_IDS="data/holdout_ids.txt",
        SPLIT_RATIOS="0.5 0.25 0.25",
        FIXTURES="data/fixtures/fixtures.json",
        FIXTURE_POSES="data/fixtures/poses",
        INGESTED="data/ingested.json",
        REGISTRY="data/registry.json",
        GITHUB_STEP_SUMMARY=str(Path(cwd) / "summary.md"),
    )
    full_env.update(env)
    return subprocess.run(
        ["bash", "Training/pipeline.sh"],
        cwd=cwd,
        env=full_env,
        capture_output=True,
        text=True,
        check=False,
    )


def setUpModule():
    global BIN, TEMPLATE
    tmp = tempfile.TemporaryDirectory()
    unittest.addModuleCleanup(tmp.cleanup)
    base = Path(tmp.name)
    # `python` on PATH is this interpreter, as in CI after setup-python.
    # A wrapper, not a symlink: a symlinked venv interpreter outside its
    # venv no longer finds the venv's packages.
    BIN = base / "bin"
    BIN.mkdir()
    wrapper = BIN / "python"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    # The template every test copies: Training/, the smoke dataset, a
    # synthetic fixture set, pinned holdout ids, and a seed champion
    # bootstrapped once from them.
    TEMPLATE = base / "template"
    shutil.copytree(
        REPO / "Training",
        TEMPLATE / "Training",
        ignore=shutil.ignore_patterns("__pycache__", "registry.json"),
    )
    set_mi_constants(TEMPLATE, MI_LENIENT)
    for script, out in (
        ("make_smoke_fixture.py", "data/dataset"),
        ("make_synthetic_fixtures.py", "data/fixtures"),
    ):
        subprocess.run(
            [sys.executable, f"Training/{script}", "--out", out],
            cwd=TEMPLATE,
            check=True,
            capture_output=True,
        )
    (TEMPLATE / "data" / "holdout_ids.txt").write_text("smoke-008\n")
    (TEMPLATE / "data" / "ingested.json").write_text("[]\n")
    result = run_pipeline(TEMPLATE, BOOTSTRAP_CHAMPION="true")
    if result.returncode != 0:
        raise RuntimeError(f"bootstrap run failed:\n{result.stderr}")
    shutil.move(TEMPLATE / "data" / "run", TEMPLATE / "data" / "bootstrap")
    (TEMPLATE / "data" / "registry.json").unlink(missing_ok=True)


def set_mi_constants(root, constants):
    """Rewrite the membership-inference thresholds in a copy of Training/."""
    path = Path(root) / "Training" / "membership_inference.py"
    text = path.read_text()
    current = MI_LENIENT if MI_LENIENT in text else MI_STRICT
    if current not in text:
        raise RuntimeError("membership_inference.py constants moved")
    path.write_text(text.replace(current, constants))


class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        case = tempfile.TemporaryDirectory()
        self.addCleanup(case.cleanup)
        self.root = Path(case.name) / "repo"
        shutil.copytree(TEMPLATE, self.root, symlinks=True)
        self.run_dir = self.root / "data" / "run"
        self.champion = self.root / "data" / "champion"

    def pipeline(self, **env):
        result = run_pipeline(self.root, **env)
        # Public-log rule: no sample (= source) id reaches stdout or stderr.
        self.assertNotIn("smoke-0", result.stdout + result.stderr)
        self.assertNotIn(SOURCE_A, result.stdout + result.stderr)
        return result

    def ok(self, **env):
        result = self.pipeline(**env)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def load(self, path):
        return json.loads(Path(path).read_text())

    def outcome(self):
        return self.load(self.run_dir / "outcome.json")

    def write(self, name, data):
        path = self.root / "data" / name
        path.write_text(data if isinstance(data, str) else json.dumps(data))
        return f"data/{name}"

    def seed_champion(self, **overrides):
        """The bootstrapped champion; overrides edit its recorded files."""
        shutil.copytree(
            self.root / "data" / "bootstrap" / "champion_update", self.champion
        )
        for name, edit in overrides.items():
            path = self.champion / name
            path.write_text(json.dumps({**self.load(path), **edit}))

    def update_files(self):
        update = self.run_dir / "champion_update"
        return sorted(p.name for p in update.iterdir()) if update.exists() else []

    def assert_slot_verifies(self):
        """The staged slot passes the workflow's store-time check."""
        update = self.run_dir / "champion_update"
        pointer = self.root / "data" / "pointer.json"
        for args in (
            ["pointer", "--manifest", update / "champion.json"]
            + ["--version", "test-1", "--out", pointer],
            ["verify", "--pointer", pointer, "--dir", update],
        ):
            subprocess.run(
                [sys.executable, "Training/champion_slot.py", *map(str, args)],
                cwd=self.root,
                check=True,
                capture_output=True,
            )


def outcome(
    gate,
    promotion,
    rescored=False,
    scoreable=True,
    leakage_flagged=False,
    leakage_streak=0,
    membership="pass",
    from_scratch=False,
    refresh=False,
):
    return {
        "champion_rescored": rescored,
        "fixture_gate": gate,
        "fixtures_scoreable": scoreable,
        "from_scratch": from_scratch,
        "refresh": refresh,
        "leakage_flagged": leakage_flagged,
        "leakage_streak": leakage_streak,
        "membership_inference": membership,
        "promotion": promotion,
    }


class StopsTest(PipelineTestCase):
    def test_bootstrap_stages_the_first_champion_with_its_scores(self):
        bootstrap = self.root / "data" / "bootstrap"
        self.assertEqual(
            self.load(bootstrap / "outcome.json"),
            outcome("not-run", "bootstrap", from_scratch=True),
        )
        update = bootstrap / "champion_update"
        self.assertEqual(sorted(p.name for p in update.iterdir()), FULL_SLOT)
        self.assertEqual(
            (update / "fixture_scores.json").read_bytes(),
            (bootstrap / "challenger" / "fixture_scores.json").read_bytes(),
        )

    def test_no_champion_scores_the_challenger_and_stops(self):
        result = self.ok()
        self.assertEqual(
            self.outcome(), outcome("not-run", "not-run", from_scratch=True)
        )
        self.assertTrue((self.run_dir / "challenger" / "fixture_scores.json").exists())
        self.assertEqual(self.update_files(), [])
        self.assertIn("no champion yet", result.stdout)
        self.assertIn("No champion changed", result.stdout)
        summary = (self.root / "summary.md").read_text()
        self.assertIn("Challenger fixture scores", summary)

    def test_nothing_to_score_stops_before_scoring(self):
        background = {
            "fixtures": [
                {**f, "segments": []}
                for f in self.load(self.root / "data/fixtures/fixtures.json")[
                    "fixtures"
                ]
            ]
        }
        for case, data in (
            ("empty", '{"fixtures": []}\n'),
            ("background only", background),
        ):
            with self.subTest(case):
                fixtures = self.write("unscoreable.json", data)
                self.seed_champion()
                # Nothing to gate, so no changed-source list is needed.
                result = self.ok(FIXTURES=fixtures, INGESTED="data/missing.json")
                self.assertEqual(
                    self.outcome(),
                    outcome("not-run", "not-run", scoreable=False),
                )
                self.assertFalse(
                    (self.run_dir / "challenger" / "fixture_scores.json").exists()
                )
                self.assertIn("nothing to score yet", result.stdout)
                shutil.rmtree(self.champion)

    def test_stale_challenger_scores_are_not_reported(self):
        self.ok()  # scores the challenger on the synthetic set
        self.ok(FIXTURES=self.write("empty.json", '{"fixtures": []}'))
        self.assertFalse((self.run_dir / "challenger" / "fixture_scores.json").exists())
        summary = (self.root / "summary.md").read_text()
        last_run = summary.rsplit("## Training pipeline run", 1)[1]
        self.assertNotIn("Challenger fixture scores", last_run)

    def test_inconsistent_starts_fail_fast(self):
        self.seed_champion()
        with self.subTest("bootstrap over a champion"):
            result = self.pipeline(BOOTSTRAP_CHAMPION="true")
            self.assertEqual(result.returncode, 2)
            self.assertIn("a champion exists", result.stderr)
        with self.subTest("no changed-source list"):
            result = self.pipeline(INGESTED="data/missing.json")
            self.assertEqual(result.returncode, 2)
            self.assertIn("changed-source list", result.stderr)
        with self.subTest("malformed fixture set"):
            result = self.pipeline(FIXTURES=self.write("bad.json", '{"fixtures": {}}'))
            self.assertEqual(result.returncode, 2)
            self.assertIn("malformed", result.stderr)
        with self.subTest("bootstrap without fixtures"):
            shutil.rmtree(self.champion)
            result = self.pipeline(
                BOOTSTRAP_CHAMPION="true",
                FIXTURES=self.write("empty.json", '{"fixtures": []}'),
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("add fixtures first", result.stderr)
        self.assertFalse(self.run_dir.exists())  # refused before training


class GatesTest(PipelineTestCase):
    def test_pass_then_archive_against_recorded_scores(self):
        self.seed_champion()
        self.ok()
        self.assertEqual(self.outcome(), outcome("pass", "archive"))
        self.assertEqual(self.update_files(), [])
        registry = self.load(self.root / "data" / "registry.json")
        # Gate records only: the log also carries leakage_gap records (#42),
        # which have no gate/decision fields.
        gates = [r for r in registry if "decision" in r]
        self.assertEqual([r.get("gate") for r in gates], ["fixture-regression", None])
        self.assertEqual([r["decision"] for r in gates], ["pass", "archive"])
        summary = (self.root / "summary.md").read_text()
        self.assertIn("Champion fixture scores (recorded)", summary)
        self.assertIn("Fixture gate: pass", summary)

    def test_registry_accumulates_across_runs(self):
        self.seed_champion()
        self.ok()
        self.ok()
        registry = self.load(self.root / "data" / "registry.json")
        gates = [r for r in registry if "decision" in r]
        self.assertEqual(
            [r["decision"] for r in gates], ["pass", "archive", "pass", "archive"]
        )
        # Without TRAINING_RUN_ID the leakage step names a run by a hash of
        # its inputs and results: the identical second run is the same run
        # to the streak (#42), so it appends no second record.
        gaps = [r for r in registry if r.get("kind") == "leakage_gap"]
        self.assertEqual(len(gaps), 1)
        self.assertTrue(gaps[0]["run_id"].startswith("sha256:"))

    def test_leakage_records_are_keyed_by_training_run_id(self):
        self.seed_champion()
        for run_id in ("40296249195-1", "40296249195-1", "40296249196-1"):
            self.ok(TRAINING_RUN_ID=run_id)
        registry = self.load(self.root / "data" / "registry.json")
        gaps = [r for r in registry if r.get("kind") == "leakage_gap"]
        self.assertEqual(
            [r["run_id"] for r in gaps], ["40296249195-1", "40296249196-1"]
        )
        summary = (self.root / "summary.md").read_text()
        self.assertIn("holdout windows", summary)

    def test_regression_skips_promotion_and_quarantines(self):
        self.seed_champion(
            **{"fixture_scores.json": {"segment_iou": 1.0, "name_accuracy": 1.0}}
        )
        self.write("ingested.json", [SOURCE_B, SOURCE_A])
        result = self.ok()
        self.assertEqual(self.outcome(), outcome("regression", "not-run"))
        self.assertFalse((self.run_dir / "promotion.json").exists())
        self.assertEqual(self.update_files(), [])
        quarantine = self.load(self.run_dir / "fixture_gate" / "quarantine.json")
        self.assertEqual([r["source_id"] for r in quarantine], [SOURCE_A, SOURCE_B])
        decision = self.load(self.run_dir / "fixture_gate" / "decision.json")
        self.assertEqual(decision["alert_status"], "not-configured")
        self.assertIn("regression", result.stdout)

    def test_stale_scores_rescore_the_champion_then_promote(self):
        self.seed_champion(
            **{
                "fixture_scores.json": {"fixture_set": "f" * 64},
                "metrics.json": {"val_accuracy": 0.5},
            }
        )
        result = self.ok()
        self.assertEqual(self.outcome(), outcome("pass", "promote", rescored=True))
        self.assertIn("another fixture set", result.stderr)
        # The champion's scores on the current set equal its bootstrap ones.
        bootstrap_scores = (
            self.root / "data" / "bootstrap" / "challenger" / "fixture_scores.json"
        )
        self.assertEqual(
            (self.run_dir / "champion" / "fixture_scores.json").read_bytes(),
            bootstrap_scores.read_bytes(),
        )
        # Promotion stages the challenger, its fixture scores recorded.
        self.assertEqual(self.update_files(), FULL_SLOT)
        update = self.run_dir / "champion_update"
        challenger = self.run_dir / "challenger"
        for name in ("fixture_scores.json", "metrics.json", "model.npz"):
            self.assertEqual(
                (update / name).read_bytes(), (challenger / name).read_bytes(), name
            )
        self.assert_slot_verifies()

    def test_missing_scores_rescore_without_promotion(self):
        self.seed_champion()
        (self.champion / "fixture_scores.json").unlink()
        result = self.ok()
        self.assertEqual(self.outcome(), outcome("pass", "archive", rescored=True))
        # The whole slot is staged: the champion as it is, new scores.
        self.assertEqual(self.update_files(), FULL_SLOT)
        update = self.run_dir / "champion_update"
        for name in ("metrics.json", "model.npz"):
            self.assertEqual(
                (update / name).read_bytes(), (self.champion / name).read_bytes()
            )
        self.assertEqual(
            (update / "fixture_scores.json").read_bytes(),
            (self.run_dir / "champion" / "fixture_scores.json").read_bytes(),
        )
        self.assert_slot_verifies()
        self.assertIn("no fixture scores recorded", result.stderr)
        self.assertIn("re-scored", result.stdout)

    def test_only_a_gated_promotion_is_published(self):
        # publish_model.py fails without the farm's environment, so a run
        # that reaches it exits 2: that is how these runs show whether it
        # was called.
        with self.subTest("bootstrap"):
            result = self.ok(BOOTSTRAP_CHAMPION="true", PUBLISH="true")
            self.assertIn("never published", result.stdout)
        with self.subTest("archive"):
            self.seed_champion()
            result = self.ok(PUBLISH="true")
            self.assertEqual(self.outcome()["promotion"], "archive")
            self.assertIn("nothing was promoted", result.stdout)
        with self.subTest("promote"):
            shutil.rmtree(self.champion)
            self.seed_champion(**{"metrics.json": {"val_accuracy": 0.5}})
            result = self.pipeline(PUBLISH="true")
            self.assertEqual(result.returncode, 2)
            self.assertIn("publish_model: error: missing environment", result.stderr)
        with self.subTest("promote without PUBLISH"):
            result = self.ok()
            self.assertEqual(self.outcome()["promotion"], "promote")
            self.assertIn("PUBLISH is not true", result.stdout)
            self.assertFalse((self.run_dir / "challenger" / "publish.json").exists())

    def test_failed_membership_inference_archives_the_challenger(self):
        # The real constants: the smoke dataset's 2 holdout sources are
        # under the minimum, so the check fails and a challenger that
        # would otherwise promote is archived.
        set_mi_constants(self.root, MI_STRICT)
        self.seed_champion(**{"metrics.json": {"val_accuracy": 0.5}})
        result = self.ok(PUBLISH="true")
        self.assertEqual(self.outcome(), outcome("pass", "archive", membership="fail"))
        self.assertEqual(self.update_files(), [])
        self.assertIn("membership-inference check", result.stdout)
        self.assertIn("nothing was promoted", result.stdout)
        report = self.load(self.run_dir / "challenger" / "holdout_report.json")
        self.assertIs(report["membership_inference"]["passed"], False)
        self.assertEqual(report["membership_inference"]["n_holdout_sources"], 2)
        promotion = self.load(self.run_dir / "promotion.json")
        self.assertEqual(promotion["decision"], "archive")
        summary = (self.root / "summary.md").read_text()
        self.assertIn("Membership inference (§6.2): fail", summary)

    def test_scheduled_retrain_replaces_an_equal_champion(self):
        # The from-scratch challenger matches the champion's val accuracy
        # (the smoke data is separable): a normal run archives it, a
        # refresh promotes it.
        self.seed_champion()
        champion_acc = self.load(self.champion / "metrics.json")["val_accuracy"]
        self.ok(FROM_SCRATCH="true")
        challenger = self.load(self.run_dir / "challenger" / "metrics.json")
        self.assertEqual(challenger["val_accuracy"], champion_acc)
        self.assertEqual(self.outcome(), outcome("pass", "archive", from_scratch=True))
        result = self.ok(FROM_SCRATCH="true", REFRESH="true")
        self.assertEqual(
            self.outcome(),
            outcome("pass", "promote", from_scratch=True, refresh=True),
        )
        self.assertEqual(self.update_files(), FULL_SLOT)
        self.assertIn("replaced the champion", result.stdout)
        promotion = self.load(self.run_dir / "promotion.json")
        self.assertIs(promotion["refresh"], True)

    def test_refresh_needs_from_scratch(self):
        self.seed_champion()
        result = self.pipeline(REFRESH="true")
        self.assertEqual(result.returncode, 2)
        self.assertIn("REFRESH=true needs FROM_SCRATCH=true", result.stderr)
        self.assertFalse(self.run_dir.exists())

    def test_a_refresh_that_fails_membership_inference_is_archived(self):
        # A scheduled retrain needs no improvement, but the
        # membership-inference check still applies to it.
        set_mi_constants(self.root, MI_STRICT)
        self.seed_champion()
        result = self.ok(FROM_SCRATCH="true", REFRESH="true")
        self.assertEqual(
            self.outcome(),
            outcome(
                "pass",
                "archive",
                membership="fail",
                from_scratch=True,
                refresh=True,
            ),
        )
        self.assertEqual(self.update_files(), [])
        self.assertIn("membership-inference check", result.stdout)
        self.assertIn("the next run retrains from scratch again", result.stdout)
        promotion = self.load(self.run_dir / "promotion.json")
        self.assertEqual(promotion["decision"], "archive")
        self.assertIs(promotion["refresh"], True)
        self.assertIs(promotion["membership_inference"]["passed"], False)

    def test_champion_without_metrics_runs_the_gate_only(self):
        self.seed_champion()
        (self.champion / "metrics.json").unlink()
        result = self.ok()
        self.assertEqual(self.outcome(), outcome("pass", "not-run"))
        self.assertFalse((self.run_dir / "promotion.json").exists())
        self.assertIn("no metrics.json to compare against", result.stdout)
        self.assertIn("No champion changed", result.stdout)

    def test_rerun_is_byte_identical(self):
        self.seed_champion()
        self.ok()
        first = {
            name: (self.run_dir / name).read_bytes()
            for name in (
                "outcome.json",
                "challenger/fixture_scores.json",
                "challenger/holdout_report.json",
                "fixture_gate/quarantine.json",
            )
        }
        self.ok()
        for name, data in first.items():
            self.assertEqual((self.run_dir / name).read_bytes(), data, name)


if __name__ == "__main__":
    unittest.main()
