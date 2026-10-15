"""The training pipeline's two configs differ only in init_checkpoint.

Run with:  python -m unittest discover -s Training/tests -t .
"""

import sys
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train

CONFIG = Path(__file__).resolve().parent.parent / "config"


class PipelineConfigsTest(unittest.TestCase):
    def load(self, name):
        with open(CONFIG / name, "rb") as fh:
            return tomllib.load(fh)

    def test_configs_differ_only_in_init_checkpoint(self):
        scratch = self.load("pipeline.toml")
        finetune = self.load("pipeline-finetune.toml")
        self.assertEqual(scratch.pop("init_checkpoint"), "")
        self.assertEqual(finetune.pop("init_checkpoint"), "data/champion/model.npz")
        self.assertEqual(scratch, finetune)

    def test_configs_pass_train_validation(self):
        for name in ("pipeline.toml", "pipeline-finetune.toml"):
            with self.subTest(config=name):
                train.load_config(CONFIG / name)

    def test_hyperparameters_match_the_full_run_template(self):
        full = self.load("full.toml")
        pipeline = self.load("pipeline.toml")
        for key in (
            "seed",
            "window_frames",
            "window_stride",
            "learning_rate",
            "epochs",
            "l2",
        ):
            self.assertEqual(pipeline[key], full[key], key)


if __name__ == "__main__":
    unittest.main()
