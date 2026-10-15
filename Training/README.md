# Training

Trick-detection training pipeline for Turnip: pose keypoint sequences in,
trick segments (start/end frame) + trick names out. The iOS app consumes
what this repo validates; see the root README for repo-wide conventions.

## Layout

- `split_dataset.py` — deterministic train/val/holdout splitter grouped by
  `source_id` (turnip-ml#12). See its docstring for the full contract.
- `train.py` — champion training / fine-tuning script (turnip-ml#13).
  Hermetic TOML config, deterministic, CI smoke-tested.
- `make_smoke_fixture.py` — generates the tiny synthetic smoke fixture.
- `config/smoke.toml` — smoke-test hyperparameters (used in CI).
- `config/full.toml` — full-run template (droplet / GPU worker).
- `fixtures/smoke/` — tiny synthetic fixture (see its README).
- `requirements.txt` — pinned training dependencies.
- `tests/` — unit tests (`python -m unittest discover -s Training/tests -t .`).

## Determinism contract

Every randomized step in this directory is seeded, and the seed is part of
the artifact: the same manifest + seed + Python interpreter version always
yields the byte-identical output. No timestamps, no unseeded RNG, no
dependence on input order. A training run that cannot be reproduced
byte-for-byte on the same interpreter is a bug.

## Splitting a manifest

```sh
python Training/split_dataset.py \
  --in manifest.jsonl --out split.json --seed 0 \
  --holdout-ids admin_holdout.txt
```

`manifest.jsonl` carries one JSON object per line with `sample_id` and
`source_id` (extra fields are ignored). Splits group by source because
training data carries no account identifier (`docs/TRAINING_DESIGN.md`
§2.6). `admin_holdout.txt` lists the admin-curated holdout sample ids, one
per line — those samples, and every sibling sample from the same sources,
are pinned to holdout and never trained on (poisoning defense). `split.json` maps each sample id to
`train` / `val` / `holdout` with sorted keys.

## Training the champion

```sh
pip install -r Training/requirements.txt
python Training/train.py --config Training/config/smoke.toml
```

`train.py` implements turnip-ml#13:

- **Hermetic config.** Every hyperparameter lives in the TOML file —
  unknown or missing keys fail closed, so a typo'd hyperparameter can never
  silently fall back to a default. `--out` optionally overrides `out_dir`.
- **Split discipline.** Trains on the `train` split only; `val` is for
  evaluation; `holdout` samples are never loaded for training (the
  poisoning defense from #12 is enforced at load time, not just by
  convention). A val label never seen in train fails closed.
- **Windowing.** Samples are cut into overlapping windows
  (`window_frames` / `window_stride`); each window is labeled by the trick
  covering at least half of it, else `background`. A segment carrying N
  names labels the window `"a+b"` (sorted) — `is_combo` is derived from the
  `"+"`, per the model output contract (master plan): source-frame
  coordinates, one segment with N names.
- **Model (smoke).** Softmax linear classifier over per-joint mean position
  + mean speed, trained with full-batch gradient descent (seeded). The
  architecture is deliberately tiny: the smoke test proves the pipeline
  end-to-end, not the model. `Training/README.md` "Full run" covers the
  real architecture seam.
- **Fine-tuning.** Set `init_checkpoint` to a previous run's `model.npz`
  to continue training the champion instead of starting from scratch;
  class list and window params must match or it fails closed.
- **Artifacts.** The out dir gets `config.toml` (byte-exact snapshot of the
  effective config), `model.npz` (weights, class list, feature stats,
  window params), and `metrics.json` (sorted keys: seed, losses, val
  accuracy overall + per class, decoded val segments, config sha256).
  Re-running the same config + data yields byte-identical `metrics.json`.

The CI smoke test (`.github/workflows/training-smoke.yml`) runs the script
on the synthetic fixture and the unit tests. The fixture is separable by
construction, so the smoke run must reach 100% val accuracy — anything less
fails the run.

## Full run

Real training runs on GitHub Actions: `.github/workflows/training-pipeline.yml`,
nightly at 03:17 UTC or by dispatch, on CPU runners for now (#24 tracks a
GPU worker). The compute seam stays one variable, `compute` in the config,
read in exactly one place (`train.py::_select_backend`):

```toml
compute = "cpu"   # Actions CPU runner / smoke
compute = "gpu"   # GPU worker (cupy-backed); fails closed without it
```

The workflow has two modes:

- **`farm`** (the schedule, or dispatch with `source=farm`) runs from the
  `training` Environment, which holds `vars.TURNIP_FARM_URL` and the secrets
  `TURNIP_PIPELINE_KEY`, `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`,
  `R2_SECRET_ACCESS_KEY` and `R2_BUCKET`. Until `TURNIP_FARM_URL` is set it
  does nothing. Otherwise:
  1. restores the watermark, champion and pinned holdout ids from
     `training/` in R2;
  2. `fetch_dataset.py count`: skips the run at 50 or fewer new label rows
     (dispatch `force` overrides);
  3. `fetch_dataset.py build`: full pull into `data/dataset`;
  4. `fetch_dataset.py count --out-sources`: the changed-source list;
  5. `pipeline.sh` (below);
  6. records `data/run` under `training/runs/<run id>/` in R2, then
     advances the watermark.
- **`synthetic`** (dispatch default) rehearses the same chain on the smoke
  fixture, with no secrets, and uploads its run directory as an artifact.

`Training/pipeline.sh` is the shared chain: `split_dataset.py` →
`train.py` (`config/pipeline-finetune.toml` when a champion was restored,
else `config/pipeline.toml`; dispatch `from_scratch` forces the latter) →
`predict.py` on holdout → `evaluate.py`. It stops before the fixture gate,
which needs the fixture runner (#27); promotion and publishing (#16) come
after the gate, so a run changes no champion yet.

Nothing per-sample reaches the log or an artifact in `farm` mode: this
repository's logs and artifacts are public. Run it locally the same way:
put a dataset in `data/dataset/` (`fetch_dataset.py build`, or
`make_smoke_fixture.py --out data/dataset`) and run `Training/pipeline.sh`.

Swapping the smoke model for the real architecture (temporal conv net over
the keypoint sequence) is a `build_dataset`/`train_classifier`-level change
behind the same config + artifact contract; the smoke model stays as the
fast CI path.
