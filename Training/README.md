# Training

Trick-detection training pipeline for Turnip: pose keypoint sequences in,
trick segments (start/end frame) + trick names out. The iOS app consumes
what this repo validates; see the root README for repo-wide conventions.

## Layout

- `fetch_dataset.py` — builds the dataset from the farm's segment export
  and the R2 `TKS1` segment blobs (`count` and `build`). See its
  docstring for the full contract.
- `tks1.py` — the `TKS1` segment-blob codec (turnip-farm
  `POSE_FORMAT.md` §12).
- `split_dataset.py` — deterministic train/val/holdout splitter grouped by
  the manifest's `source_id`, the farm's split bucket for segments
  (turnip-ml#12). See its docstring for the full contract.
- `train.py` — champion training / fine-tuning script (turnip-ml#13).
  Hermetic TOML config, deterministic, CI smoke-tested.
- `membership_inference.py` — the membership-inference check: can the
  model's outputs tell training segments from holdout ones? Written
  into the holdout report; the promotion gate archives a model that
  fails it (`docs/TRAINING_DESIGN.md` §6.2).
- `name_metrics.py` — per-name holdout precision/recall plus holdout and
  train counts (from `train.py`'s `metrics.json`), the manifest's
  `name_metrics` (turnip-ml#41).
- `export_coreml.py` — exports a trained model to a byte-reproducible fp16
  Core ML `.mlpackage` zip (turnip-ml#16).
- `publish_model.py` — publishes a promoted champion: export, release
  checks, R2 upload with read-back hash checks, `POST /api/models`
  (turnip-ml#16).
- `release_checks.py` — confirms an exported artifact carries nothing
  but the model's parameters: no pose sequences, no per-example
  embeddings (`docs/TRAINING_DESIGN.md` §6.2).
- `segment_transform.py` — the reference segment transform: canonical
  pose onto the farm's standard body (`TKS1`), which a client applies
  before a segment-trained model (`docs/MODEL_CONTRACT.md` §3.6).
- `make_segment_vectors.py` — writes its fixture vectors,
  `fixtures/segment_transform/vectors.json`, which the client tests
  against.
- `make_smoke_fixture.py` — generates the tiny synthetic smoke fixture.
- `fixture_runner.py` — scores a model on the trick-labeled fixture set
  and writes the fixture gate's input (turnip-ml#27); also the one place
  that decides whether a set can be scored (`check`) and whether recorded
  scores belong to it (`compare`). See its docstring for the fixture
  format and the scores.
- `champion_slot.py` — the champion's file hashes: the manifest
  `pipeline.sh` stages, and the pointer the workflow stores and verifies.
- `fixture_gate.py` — fixture regression gate (anti-poisoning).
- `report_quarantine.py` — sends a fixture regression's `quarantine.json`
  to the farm (`POST /api/quarantine`, turnip-ml#51); `farm` mode only.
- `promote.py` — champion/challenger promotion gate; reads the holdout
  report's membership-inference check, and `--refresh` for a scheduled
  retrain.
- `retrain_schedule.py` — the fixed retrain schedule: whether a
  from-scratch retrain is due, and recording one
  (`docs/TRAINING_DESIGN.md` §2.1).
- `make_synthetic_fixtures.py` — generates a synthetic fixture set for
  the tests and the pipeline's synthetic rehearsal.
- `config/smoke.toml` — smoke-test hyperparameters (used in CI).
- `config/full.toml` — full-run template (droplet / GPU worker).
- `fixtures/smoke/` — tiny synthetic fixture (see its README).
- `fixtures/tricks/` — the maintainer's trick-labeled fixture set for
  the fixture gate; ships empty (see its README to add fixtures).
- `requirements.txt` — pinned training dependencies.
- `requirements-export.txt` — `requirements.txt` plus pinned `coremltools`
  for the export.
- `tests/` — unit tests (`python -m unittest discover -s Training/tests -t .`).

## Determinism contract

Every randomized step in this directory is seeded, and the seed is part of
the artifact: the same manifest + seed + Python interpreter version always
yields the byte-identical output. No timestamps, no unseeded RNG, no
dependence on input order. A training run that cannot be reproduced
byte-for-byte on the same interpreter is a bug.

## The dataset

`fetch_dataset.py build` writes one `samples/<segment_id>.json` per farm
segment: its 13 `TKS1` joints at 10 Hz (on the farm's standard body, in
torso lengths), its `split_bucket`, and one `tricks` entry per canonical
name on the segment's clip window. It also writes `manifest.jsonl` (the
splitter's input, grouping each segment by `bucket-<split_bucket>`) and
`dataset_manifest.json`. Samples also carry each label's provenance
(`docs/TRAINING_DESIGN.md` §1, §6.4):

- `tricks[].origins` — how the name was given: `typed`, `vocabulary` or
  `model` (a model suggestion the user kept). Sorted; more than one only
  when two raw labels with different origins canonicalize to the same
  name.
- `clips` — keyed by the segment id, the clip's `proposal_model_version` (`null` for
  the heuristic detector or a manual clip), `suggested_names` (what that
  model pre-filled, or on a blind clip would have, verbatim; `[]` when
  none) and `suggestions_withheld` (the clip is in the blind sample).

All four export fields are required and validated against the farm's
schema; a missing or malformed one fails the build (exit 2). Nothing reads
the provenance yet: turnip-ml#40 adds the rubber-stamp measures and the
down-weighting. The smoke fixture carries none, as `train.py`'s sample
format does not require it.

A trick with fewer distinct contributors than the farm's
`contributor_minimum` is held back whole. The counts come from the
export's `trick_contributors`; a trick it doesn't list has none, and an
export without either field holds every trick back. A segment carrying a
held-back trick gets no sample at all, and `dataset_manifest.json` lists
the held-back names (`docs/TRAINING_DESIGN.md` §6.2).

`train.py` reads samples in that body frame. The smoke fixture is
generated as MoveNet keypoints and stored after the same transform
(`segment_transform.py`), so it looks like farm data to every script.

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
  coordinates, one segment with N names. Segments come from
  `evaluate.ground_truth_segments` (one per distinct window, with every
  name labeled on it), the same ground truth the holdout evaluation
  scores against.
- **Class balance.** Each epoch's windows are resampled, not reused whole:
  name windows are drawn with replacement ∝
  `train_count**-class_balance_exponent` per canonical name — a combo
  window takes its rarest name's weight — so a frequent name doesn't drown
  the rest while a rare name isn't repeated into memorization; background
  windows are sampled separately at `background_ratio` of the epoch
  sample (`docs/TRAINING_DESIGN.md` §2.7, §2.5). The scheme, the per-name
  counts it read, and the effective per-name weights land in
  `metrics.json` under `sampling` for reproducibility.
- **Model (smoke).** Softmax linear classifier over per-joint mean position
  + mean speed, trained with gradient descent on class-balanced epoch
  samples (seeded). The architecture is deliberately tiny: the smoke test
  proves the pipeline end-to-end, not the model. `Training/README.md`
  "Full run" covers the real architecture seam.
- **Fine-tuning.** Set `init_checkpoint` to a previous run's `model.npz`
  to continue training the champion instead of starting from scratch;
  class list and window params must match or it fails closed.
- **Artifacts.** The out dir gets `config.toml` (byte-exact snapshot of the
  effective config), `model.npz` (weights, class list, feature stats,
  window params), and `metrics.json` (sorted keys: seed, losses, val
  accuracy overall + per class, decoded val segments, config sha256, the
  class-balance sampling record).
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
  `R2_SECRET_ACCESS_KEY` and `R2_BUCKET`, plus `TURNIP_DISCORD_WEBHOOK_URL`
  for the fixture gate's regression alert (optional: unset, the gate
  records the alert as not configured). Until `TURNIP_FARM_URL` is set it
  does nothing. Otherwise:
  1. restores from `training/` in R2: the watermark, the champion, the
     gates' decision log (`registry.json`), the pinned holdout ids and
     the fixture set's pose blobs (`fixtures/poses/`). The champion is the
     version `champion/current.json` points to
     (`champion/versions/<version>/`: `model.npz`, `metrics.json`,
     `fixture_scores.json`), each file verified against the pointer's
     SHA-256 (`champion_slot.py verify`). A hand-placed, unversioned
     `champion/model.npz` is still restored, with a warning that it
     cannot be verified;
  2. `retrain_schedule.py due`: a full retrain is due when the last one
     (`training/state/retrain_state.json`) is 30 days old or none is
     recorded. Then the run trains from scratch, with `REFRESH=true`:
     the promotion gate replaces the champion unless the challenger is
     more than 2% worse;
  3. `fetch_dataset.py count`: skips the run at 50 or fewer labels on
     segments new since the watermark (the last successful run's segment
     ids) unless a retrain is due (dispatch `force` overrides);
  4. `fetch_dataset.py build`: full pull into `data/dataset`;
  5. `fetch_dataset.py count --out-segments`: the new-segment list;
  6. `pipeline.sh` (below);
  7. records `data/run` under `training/runs/<run id>/` in R2; stores
     what `pipeline.sh` staged in `data/run/champion_update/` as a new
     champion version, and only then rewrites `champion/current.json`, so
     an upload that fails partway changes nothing; then
     `retrain_schedule.py record` notes today as the last full retrain
     when a from-scratch challenger became the champion; stores the
     decision log;
  8. after a fixture regression, sends `quarantine.json` to the farm with
     `report_quarantine.py` (below). The list holds segment ids, which
     the farm's endpoint doesn't take yet, so the send fails, and with
     it the run (`docs/TRAINING_DESIGN.md` §7);
  9. advances the watermark. After a regression only once the farm has
     accepted every quarantine request: a failed send fails the run and
     the watermark holds, so the next run's new-segment list still
     covers the suspect segments.
- **`synthetic`** (dispatch default) rehearses the same chain on the smoke
  fixture and a synthetic fixture set, with no secrets, twice: a first run
  with no champion bootstraps one, a second runs the fixture gate and the
  promotion gate against it, with the champion stored and verified in
  between as the farm job does (minus R2). Both run directories and the
  decision log are uploaded as an artifact.

`Training/pipeline.sh` is the shared chain: `split_dataset.py` →
`train.py` (`config/pipeline-finetune.toml` when a champion was restored,
else `config/pipeline.toml`; dispatch `from_scratch` forces the latter) →
`predict.py` on holdout → `evaluate.py` → `membership_inference.py` →
`name_metrics.py` →
`fixture_runner.py` on the challenger → `fixture_gate.py` against the
champion's recorded fixture scores (re-scoring the champion once when they
are missing or from another fixture set) → `promote.py` if the gate passed
→ `publish_model.py` if it promoted, in `farm` mode only (`PUBLISH=true`).
A promoted challenger is staged with its fixture scores in
`data/run/champion_update/`, as a whole slot with its manifest (a re-score
stages the champion's own files with the new scores). Both gates append to the decision log (`REGISTRY`,
default `Training/registry.json`). Its explicit stops, each named in the
job summary:

- the fixture set (`fixtures/tricks/`) has nothing to score yet (no
  fixture with a labeled segment): no gate, no promotion;
- no champion yet: nothing to compare against, so nothing is promoted
  unless the maintainer dispatches `bootstrap_champion` (with `force` when
  there are 50 or fewer new label rows), which makes that run's challenger
  the first champion. Bootstrap records the champion's fixture scores, so
  it is refused while the fixture set has nothing to score;
- a fixture regression: promotion is skipped;
- a challenger that failed the membership-inference check: the
  promotion gate archives it. The smoke fixture has 2 holdout sources,
  under the check's minimum of 20, so the `synthetic` rehearsal's
  second run always archives;
- a champion without `metrics.json`: the fixture gate runs, but the
  promotion gate has nothing to compare against and does not;
- publishing follows a gated promotion only: a bootstrapped champion
  passed no fixture gate, so it becomes the fine-tune base and gate
  baseline but is never published; the first gated promotion is.

## Publishing a champion

```sh
python Training/publish_model.py --run data/run --dataset data/dataset
```

`publish_model.py` reads the farm and R2 settings from the environment
(the `farm` variables above, plus optional `TRAINING_RUN_ID` and
`MIN_CLIENT_VERSION`) and, in this order:

1. Builds the whole `POST /api/models` body first: the descriptors of the
   exported model, `val_metrics`, `name_metrics.json`, the dataset's
   pinned `taxonomy_version`, and `training_segment_ids` (the train
   split's segments). A dataset that isn't the farm's (segment ids are
   UUIDs) is refused before anything is uploaded, so `synthetic` mode can
   never publish. So is a model whose trick names include one that isn't a
   curated canonical name (`dataset_manifest.json`'s `canonical_names`):
   unmapped labels train, but they are contributors' free text and the
   manifest is public. Map them in the taxonomy first
   (`unmapped_labels.log` is the queue).
2. Exports the champion with `export_coreml.py`. The pipeline runs on
   Ubuntu, so the artifact is a `.mlpackage` zip the client compiles on
   device, not a compiled `.mlmodelc` (`docs/MODEL_CONTRACT.md` §2).
   `release_checks.py` then checks the zip holds nothing but the
   model's parameters; a failure stops the publish before any upload.
3. Picks the version: the minor bump of the farm's current model, or
   `trick-v1.0` before the first; `--major` or `--version` override it.
   If the current model already is this artifact (same sha256), an
   earlier run published it and lost the answer: nothing more is done.
   If the current model reads another input format (`input_desc.format`,
   `TKP1` before the switch to segments), the publish is a major bump and
   needs `MIN_CLIENT_VERSION`, the first app version that applies
   `docs/MODEL_CONTRACT.md` §3.6; without it, nothing is published.
4. Uploads the artifact to `models/<version>.mlpackage.zip` and the holdout
   report to `reports/<version>-holdout.json`, reads each back and checks its sha256. It never overwrites a key that
   holds other bytes.
5. Calls `POST /api/models` and records the request body and the farm's
   answer in `data/run/challenger/` (`publish_request.json`,
   `publish.json`). When the answer is lost (timeout, 5xx), the farm's
   current model decides whether the insert landed. Requests never follow
   a redirect, which would carry the pipeline key to another host.

It prints the version, sizes and counts only.

## Quarantining a regression's sources

```sh
python Training/report_quarantine.py --quarantine data/run/fixture_gate/quarantine.json
```

`report_quarantine.py` reads `TURNIP_FARM_URL` and `TURNIP_PIPELINE_KEY`
from the environment and, in this order:

1. Validates every row of the gate's `quarantine.json` by the farm's rules
   before sending anything: `source_id` a farm source id (64 lowercase
   hex), `reason` non-empty and at most 1000 characters, `metrics` an
   object or null, no other key, no source listed twice. `synthetic`
   mode's sources are not farm ids, so it can never quarantine. A
   segment id is refused with its own message: the farm quarantines
   sources only, and segments carry none. An empty file sends nothing.
2. Sends one `POST /api/quarantine` body `{source_ids, reason, metrics}`
   per reason and metrics, at most 10000 ids each (the farm's limit), in
   a fixed order.
3. Counts a body as sent only when the farm answers 2xx and its
   `quarantined`, `already_open` and `unknown` lists account for every id
   sent. `unknown` sources were deleted by their owner after the export
   pull; that is not an error. A transport failure or a 5xx is retried
   (three tries in all); a 401, a 422 or a 3xx fails at once, with exit 2.
   The farm skips sources already under an open quarantine, so a retry
   or a re-run after a failure is safe. Requests never follow a redirect.

It prints counts only: never source ids, nor the farm's error messages (a
422's details name the offending id).

Nothing per-sample reaches the log or an artifact in `farm` mode: this
repository's logs and artifacts are public, so the gates' decision records
go to files (`data/run/fixture_gate/decision.json`,
`data/run/promotion.json`, the decision log), not the log. Run it locally
the same way: put a dataset in `data/dataset/` (`fetch_dataset.py build`,
or `make_smoke_fixture.py --out data/dataset`), the new-segment list
in `data/run/ingested.json` (`[]` will do locally; it is required only
when the fixture gate will run, i.e. with a champion and a scoreable
fixture set), and run `Training/pipeline.sh`. Point `FIXTURES` and `FIXTURE_POSES` at
`make_synthetic_fixtures.py --out data/fixtures` output
(`data/fixtures/fixtures.json`, `data/fixtures/poses`) to rehearse the
gates without the real set.

Swapping the smoke model for the real architecture (temporal conv net over
the keypoint sequence) is a `build_dataset`/`train_classifier`-level change
behind the same config + artifact contract; the smoke model stays as the
fast CI path.
