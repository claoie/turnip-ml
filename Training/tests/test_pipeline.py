"""End-to-end tests for Training/pipeline.sh: fixture runner -> fixture
gate -> promotion gate, and the explicit stops.

Hermetic: each test runs the real script in a temporary copy of
Training/ (train.py resolves its config paths against the directory it
lives in, so a symlink would write into the real repository's data/), on
the synthetic smoke dataset and a synthetic fixture set. No network, no
secrets.
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


class PipelineTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        # `python` on PATH is this interpreter, as in CI after setup-python.
        # A wrapper, not a symlink: a symlinked venv interpreter outside its
        # venv no longer finds the venv's packages.
        cls.bin = base / "bin"
        cls.bin.mkdir()
        wrapper = cls.bin / "python"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        wrapper.chmod(0o755)
        # The template every test copies: Training/, the smoke dataset, a
        # synthetic fixture set, pinned holdout ids, and a seed champion
        # bootstrapped once from them.
        cls.template = base / "template"
        shutil.copytree(
            REPO / "Training",
            cls.template / "Training",
            ignore=shutil.ignore_patterns("__pycache__", "registry.json"),
        )
        cls.py(cls.template, "Training/make_smoke_fixture.py", "--out", "data/dataset")
        cls.py(
            cls.template,
            "Training/make_synthetic_fixtures.py",
            "--out",
            "data/fixtures",
        )
        (cls.template / "data" / "holdout_ids.txt").write_text("smoke-008\n")
        (cls.template / "data" / "ingested.json").write_text("[]\n")
        result = cls.pipeline_in(cls.template, BOOTSTRAP_CHAMPION="true")
        assert result.returncode == 0, result.stderr
        shutil.move(cls.template / "data" / "run", cls.template / "data" / "bootstrap")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def py(cls, cwd, *args):
        subprocess.run(
            [sys.executable, *args], cwd=cwd, check=True, capture_output=True
        )

    @classmethod
    def pipeline_in(cls, cwd, **env):
        full_env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("GITHUB_", "TURNIP_"))
        }
        full_env.update(
            PATH=f"{cls.bin}{os.pathsep}{os.environ.get('PATH', '')}",
            HOLDOUT_IDS="data/holdout_ids.txt",
            SPLIT_RATIOS="0.5 0.25 0.25",
            FIXTURES="data/fixtures/fixtures.json",
            FIXTURE_POSES="data/fixtures/poses",
            INGESTED="data/ingested.json",
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

    def setUp(self):
        self.case = tempfile.TemporaryDirectory()
        self.root = Path(self.case.name) / "repo"
        shutil.copytree(self.template, self.root, symlinks=True)
        self.run_dir = self.root / "data" / "run"
        self.champion = self.root / "data" / "champion"

    def tearDown(self):
        self.case.cleanup()

    def pipeline(self, **env):
        result = self.pipeline_in(self.root, **env)
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


class StopsTest(PipelineTestCase):
    def test_bootstrap_stages_the_first_champion_with_its_scores(self):
        bootstrap = self.root / "data" / "bootstrap"
        self.assertEqual(
            self.load(bootstrap / "outcome.json")["promotion"], "bootstrap"
        )
        update = bootstrap / "champion_update"
        self.assertEqual(
            sorted(p.name for p in update.iterdir()),
            ["fixture_scores.json", "metrics.json", "model.npz"],
        )
        self.assertEqual(
            (update / "fixture_scores.json").read_bytes(),
            (bootstrap / "challenger" / "fixture_scores.json").read_bytes(),
        )

    def test_no_champion_scores_the_challenger_and_stops(self):
        result = self.ok()
        self.assertEqual(
            self.outcome(),
            {
                "champion_rescored": False,
                "fixture_gate": "not-run",
                "n_fixtures": 5,
                "promotion": "not-run",
            },
        )
        self.assertTrue((self.run_dir / "challenger" / "fixture_scores.json").exists())
        self.assertEqual(self.update_files(), [])
        self.assertIn("no champion yet", result.stdout)
        self.assertIn("No champion changed", result.stdout)
        summary = (self.root / "summary.md").read_text()
        self.assertIn("Challenger fixture scores", summary)

    def test_empty_fixture_set_stops_before_scoring(self):
        empty = self.root / "data" / "empty.json"
        empty.write_text('{"fixtures": []}\n')
        self.seed_champion()
        result = self.ok(FIXTURES=str(empty))
        self.assertEqual(self.outcome()["fixture_gate"], "not-run")
        self.assertEqual(self.outcome()["n_fixtures"], 0)
        self.assertFalse((self.run_dir / "challenger" / "fixture_scores.json").exists())
        self.assertIn("has no fixtures yet", result.stdout)

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
        with self.subTest("bootstrap without fixtures"):
            (self.root / "data" / "empty.json").write_text('{"fixtures": []}')
            shutil.rmtree(self.champion)
            result = self.pipeline(
                BOOTSTRAP_CHAMPION="true", FIXTURES="data/empty.json"
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("add fixtures first", result.stderr)
            self.seed_champion()
        with self.subTest("champion without metrics"):
            (self.champion / "metrics.json").unlink()
            result = self.pipeline()
            self.assertEqual(result.returncode, 2)
            self.assertIn("no metrics.json", result.stderr)
        self.assertFalse(self.run_dir.exists())  # refused before training


class GatesTest(PipelineTestCase):
    def test_pass_then_archive_against_recorded_scores(self):
        self.seed_champion()
        self.ok()
        self.assertEqual(
            self.outcome(),
            {
                "champion_rescored": False,
                "fixture_gate": "pass",
                "n_fixtures": 5,
                "promotion": "archive",
            },
        )
        self.assertEqual(self.update_files(), [])
        registry = self.load(self.run_dir / "registry.json")
        self.assertEqual(
            [r.get("gate") for r in registry], ["fixture-regression", None]
        )
        self.assertEqual([r["decision"] for r in registry], ["pass", "archive"])
        summary = (self.root / "summary.md").read_text()
        self.assertIn("Champion fixture scores (recorded)", summary)
        self.assertIn("Fixture gate: pass", summary)

    def test_regression_skips_promotion_and_quarantines(self):
        self.seed_champion(
            **{"fixture_scores.json": {"segment_iou": 1.0, "name_accuracy": 1.0}}
        )
        (self.root / "data" / "ingested.json").write_text(
            json.dumps([SOURCE_B, SOURCE_A])
        )
        result = self.ok()
        self.assertEqual(self.outcome()["fixture_gate"], "regression")
        self.assertEqual(self.outcome()["promotion"], "not-run")
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
        self.ok()
        self.assertEqual(
            self.outcome(),
            {
                "champion_rescored": True,
                "fixture_gate": "pass",
                "n_fixtures": 5,
                "promotion": "promote",
            },
        )
        # The champion's scores on the current set equal its bootstrap ones.
        bootstrap_scores = (
            self.root / "data" / "bootstrap" / "challenger" / "fixture_scores.json"
        )
        self.assertEqual(
            (self.run_dir / "champion" / "fixture_scores.json").read_bytes(),
            bootstrap_scores.read_bytes(),
        )
        # Promotion stages the challenger, its fixture scores recorded.
        self.assertEqual(
            self.update_files(), ["fixture_scores.json", "metrics.json", "model.npz"]
        )
        update = self.run_dir / "champion_update"
        challenger = self.run_dir / "challenger"
        for name in ("fixture_scores.json", "metrics.json", "model.npz"):
            self.assertEqual(
                (update / name).read_bytes(), (challenger / name).read_bytes(), name
            )

    def test_missing_scores_rescore_without_promotion(self):
        self.seed_champion()
        (self.champion / "fixture_scores.json").unlink()
        result = self.ok()
        self.assertTrue(self.outcome()["champion_rescored"])
        self.assertEqual(self.outcome()["promotion"], "archive")
        # Only the re-scored scores are staged for the champion slot.
        self.assertEqual(self.update_files(), ["fixture_scores.json"])
        self.assertIn("none recorded", result.stderr)
        self.assertIn("re-scored", result.stdout)

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
