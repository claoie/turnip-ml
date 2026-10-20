# Trick-labeled fixture set (maintainer-labeled)

The fixture regression gate's yardstick (`docs/TRAINING_DESIGN.md` §6.3):
fixed (pose sequence, labeled segments) pairs that every nightly
challenger is scored on (`Training/fixture_runner.py`) and compared
against the champion with (`Training/fixture_gate.py`).

**The labels here are the maintainer's.** Nobody else adds, edits or
"fixes" a label in `fixtures.json`, and no label is ever inferred from
footage or copied from the farm export (CONTRIBUTING rule 1). The set
ships empty. Until it has a fixture with a labeled segment, the pipeline
runs no fixture gate, promotes nothing, and says so in its job summary.

## Layout

- `fixtures.json`: the set, labels plus each pose's SHA-256. In git.
- Pose blobs: **not in git** (CONTRIBUTING rule 3). TKP1 `.tkp1.gz`
  files (`Training/tkp1.py`), content-addressed in the private R2 bucket
  as `training/fixtures/poses/<sha256>.tkp1.gz`. The nightly farm job
  restores them to `data/fixture-poses/`. The runner hashes every blob
  before decoding it and refuses to score when one is missing or does
  not match its `pose_sha256`.

## Format

```json
{
  "fixtures": [
    {
      "id": "gainer-01",
      "pose_sha256": "<sha256 of the .tkp1.gz bytes>",
      "segments": [
        {"start_frame": 12, "end_frame": 40, "names": ["gainer"]}
      ]
    },
    {
      "id": "combo-hook-scoot-01",
      "pose_sha256": "<sha256>",
      "segments": [
        {"start_frame": 30, "end_frame": 75, "names": ["hook", "scoot"]}
      ]
    },
    {"id": "background-standing-01", "pose_sha256": "<sha256>", "segments": []}
  ]
}
```

(The names above only show the shape. They are not labels.)

- `id`: unique, lowercase `[a-z0-9._-]`.
- `pose_sha256`: SHA-256 of the `.tkp1.gz` bytes, exactly as stored.
- `segments`: frame indices into that pose at its canonical 10 Hz,
  `start_frame` inclusive and `end_frame` exclusive. `names` are
  canonical names at the current taxonomy (lowercase, single spaces, no
  `+`). A combo is one segment that lists every name. Segments of one
  fixture must not overlap. An empty list makes a background fixture:
  anything the model detects there counts against it.

Any other key, or a missing or duplicated one, fails closed. Every pose
must be at least as long as the model's window (16 frames, 1.6 s, for
the pipeline model): a shorter one is invisible to the model, so the
runner refuses to score it. Check the file with
`python Training/fixture_runner.py check --fixtures Training/fixtures/tricks/fixtures.json`.
Aim to cover every
core-vocabulary trick, the known combos, and some background
(standing, setup, walking between tricks).

## Adding a fixture

1. Pick a pose sequence you can keep indefinitely. A fixture blob lives
   outside the farm's data lifecycle: deleting the source farm-side does
   not delete the fixture. Your own recordings are the safe choice.
2. Get its TKP1 blob. For a source you uploaded, that is the stored
   blob its export row's `r2_key` points to:

   ```sh
   r2() { aws --endpoint-url "https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com" s3 "$@"; }
   r2 cp "s3://$R2_BUCKET/<r2_key>" pose.tkp1.gz
   ```

3. Store it under its hash, in R2 and locally:

   ```sh
   sha=$(shasum -a 256 pose.tkp1.gz | cut -d' ' -f1)
   r2 cp pose.tkp1.gz "s3://$R2_BUCKET/training/fixtures/poses/$sha.tkp1.gz"
   mkdir -p data/fixture-poses && cp pose.tkp1.gz "data/fixture-poses/$sha.tkp1.gz"
   ```

4. Add its entry to `fixtures.json` with `pose_sha256` set to `$sha`
   and your segments.
5. Check that it scores, against any model:

   ```sh
   python Training/fixture_runner.py score --model data/champion/model.npz \
     --fixtures Training/fixtures/tricks/fixtures.json \
     --poses data/fixture-poses --out data/fixture_scores.json
   ```

6. Commit `fixtures.json` (never the blob) in a PR.

The first champion needs fixtures: a `bootstrap_champion` dispatch
records its fixture scores, so it is refused while this set has nothing
to score. Add fixtures first, then bootstrap.

Any change to the set changes its `fixture_set` digest. The next nightly
run then re-scores the champion on the new set once and stores those
scores with it, so the gate always compares scores from the same set.

## Not this

`Training/make_synthetic_fixtures.py` generates a synthetic set for the
unit tests and the pipeline's synthetic rehearsal. Its labels are the
generator's own, not trick labels. Never copy them here.
