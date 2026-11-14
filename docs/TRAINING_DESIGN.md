# Trick-Detection Training Design (`turnip-ml`)

**Scope.** This document is the training-side detailed design for `turnip-ml`,
implementing §6 of the master plan
(`https://github.com/hoiekim/turnip-farm/blob/main/docs/MASTER_PLAN.md`).
It covers dataset construction, the training task and architecture,
evaluation, the anti-poisoning quarantine loop, promotion, and versioning.
Companion docs: the farm's `BACKEND_DESIGN.md` (tables, export endpoint)
and `POSE_FORMAT.md` (pose wire format). In-flight implementation this design
aligns with: PR #17 (deterministic stratified splitter), PR #18 (champion
training script + CI smoke test), PR #19 (holdout evaluation), PR #20
(champion/challenger promotion gate). Where this document proposes beyond
what those PRs implement, it says so explicitly.

## 1. Data contract: what training consumes

- Only **confirmed** clips + labels. The farm has no draft/pending clip
  lifecycle (master plan §1–§2), so everything reachable via
  `GET /api/labels/export` is training-eligible by construction.
- **Segments, never whole-session pose** (`BACKEND_DESIGN.md` §1.2.24,
  `POSE_FORMAT.md` §12). The export serves one **segment** per labeled
  clip: the clip window plus 0.5 s (5 frames) of context on each side,
  re-targeted onto a standard body. It is hip-centered and scaled by
  the median torso length; every bone is set to a standard length along
  its observed direction; keypoints 0–4 collapse to one head point; and
  light noise is added. The blob is `TKS1`: 13 joints at 10 Hz, in
  torso lengths (`Training/tks1.py`). The farm keeps at most one segment
  per clip, never two of a source that overlap or lie within 1 s, so a
  session can't be stitched back together.
- Privacy posture: segments + labels only. Never video, and no account
  or source identifier: a segment carries a fresh random id, no
  timestamp, and a `split_bucket` (0–9, the same for every segment of
  one source) so splits can keep a session on one side without knowing
  which session it was. The farm links a segment to its source only
  internally, and deletes that link with the source and the account
  (`BACKEND_DESIGN.md` §1.2.24). That makes the data de-linked and hard
  to re-identify, not anonymous: a segment can still be matched to a
  video of the same clip (`BACKEND_DESIGN.md` §9 Q9).
- The export excludes segments of quarantined sources (master plan §4)
  and presents one entry per segment with its clip window and every
  label on it — see §2.2.
- Each segment also carries its labels' provenance (`BACKEND_DESIGN.md`
  §1.2.18): each label's `origin` (`typed` | `vocabulary` | `model`),
  and the clip's `proposal_model_version`, `suggested_names` and
  `suggestions_withheld`. Training will use them to measure and limit the
  model learning from its own suggestions (§6.4, proposed — not yet
  implemented, turnip-ml#40).
- On-device personalization (`MODEL_CONTRACT.md` §3.5) adds nothing to
  the export. It trains on the device from the user's own library and
  uploads nothing it learns.

## 2. Dataset construction

### 2.1 Export and watermarking

- **Change detection.** Segment ids are random and the export's cursor
  is only a page position, so change is read from the ids themselves.
  The watermark is the last successful run's set of segment ids, stored
  in the training workspace (not in the farm). The nightly job pulls
  the export (entries only, no blobs), counts the labels on segments
  whose id is not in the watermark, and proceeds only when they exceed
  50 (master plan §6). Any clip or label edit re-cuts its source's
  segments under fresh ids, so a relabel counts as new, as an upload
  does. The watermark advances only when a run completes. A watermark
  from before segments (an export cursor) counts every segment as new
  once.
- **Dataset pull.** The dataset itself is always built from a **full
  pull**. The export carries no deletions (`BACKEND_DESIGN.md` §6): a
  segment whose source was deleted by its owner or purged from
  quarantine, or that was re-cut, simply stops appearing. A dataset
  accumulated from incremental pulls would keep training on deleted
  data.
- **Entry contents.** Each segment carries its `TKS1` blob's R2
  reference and digest, its frame count, the clip window inside it, its
  `split_bucket`, the clip's provenance, and its labels (raw name,
  `origin`, `taxonomy_version`). There is no account or source
  identifier; an entry carrying a `user_id` or `source_id` is refused.
- **Blob cache.** Segment blobs may be cached locally, keyed by `r2_key`
  and verified against `blob_sha256`. After each full pull, every
  cached blob whose segment is absent from the pull is deleted.
- **Scheduled retrain.** A full pull keeps deleted data out of new
  training, but the nightly run fine-tunes the champion, whose weights
  carry everything it was ever trained on, and it runs only past 50 new
  label rows. So the farm job also retrains from scratch on a fixed
  schedule (`Training/retrain_schedule.py`). Once the last full retrain
  is 30 days old, or none is recorded, the next nightly run trains from
  scratch whatever the label count. Its promotion gate runs as a
  refresh (§8), so the challenger replaces the champion unless it is
  clearly worse. *(Judgment call on 30 days, §10 Q12.)*
  - A from-scratch challenger that becomes the champion (promoted or
    bootstrapped) records the date in R2
    (`training/state/retrain_state.json`).
  - Until one lands, every nightly run tries again. Data deleted right
    after a full retrain leaves the published model within about 30
    days, plus any nights a gate refuses the refresh.

### 2.2 Canonicalization (raw → canonical)

Applied **at dataset build time**, pinned to a single `taxonomy_version`
for the whole training run. The farm preserves raw free-text strings
forever; training consumes canonical names via `label_taxonomy`.

- **The pin is the farm's current version:** the greatest
  `taxonomy_version` in the taxonomy the export ships, not the greatest
  among the label rows. A curation-only bump (new mappings, no new
  labels yet) therefore applies to every label at the next run.
- **Versions are append-only:** a bump adds only new or changed
  mappings, so each raw string resolves to its mapping with the greatest
  `taxonomy_version` ≤ the pin (the farm's `(raw, taxonomy_version DESC)`
  lookup). The export must therefore ship the whole table, not only the
  versions present in its rows (`BACKEND_DESIGN.md` §7.3).

Deterministic pipeline per label string:

1. Normalize: lowercase, trim, collapse internal whitespace.
2. Strip parentheticals like `(combo)`.
3. Look the whole label up first (as typed, then without
   parentheticals). On a miss, split it on `-`, `/`, `+`, `,` and look up
   each token — e.g. `"hook - scoot - gainer - cartfull (combo)"` becomes
   `[hook, scoot, gainer, cartfull]` on one window. Whole-label lookup
   comes first so a mapped name that contains a delimiter (`b-twist`) is
   not split.
4. **Judgment call:** tokens with no mapping keep their normalized form
   as the canonical name and are appended to a curation queue
   (`unmapped_labels.log`) for later taxonomy work. They train only if
   the farm's contributor counts cover them; otherwise the rare-trick
   hold-back (§6.2) holds their segments back. Rationale: failing
   closed on unmapped names would starve early training when the
   vocabulary is still growing; the log feeds the taxonomy curation loop
   (master plan §9, open question 1). A model trained on an unmapped
   name is not published, though: the manifest is public and the name
   is a contributor's free text (`MODEL_CONTRACT.md` §1), so publishing
   waits until the taxonomy maps it.

Segment → Label shape (decided): the farm stores **one label row per
`(clip_id, trick_name)`** (`BACKEND_DESIGN.md` §1.1), and each segment
carries its clip's whole label set, sorted, never empty. The dataset
builder canonicalizes every raw name of a segment through
`label_taxonomy` at the pinned `taxonomy_version` (above) and writes one
trick entry per canonical name, all on the segment's clip window. A
segment whose names all canonicalize to nothing is dropped.

### 2.3 Training targets

- One **target segment** per training segment: its clip window
  `(window_start, window_end)`, in the segment's own 10 Hz frames. The
  frames outside it are the segment's context.
- Multi-label: a combo clip with N canonical names becomes one segment
  with an N-hot target vector over the canonical vocabulary (vocabulary
  size fixed at the pinned `taxonomy_version`).
- `is_combo` is derived (`len(trick_names) > 1`); the combo MVP does not
  sub-segment individual tricks (master plan §6).

### 2.4 Gap handling

- Gap frames (all-zero confidence; in a segment, any frame missing a
  hip) are **missing data**, as are missing joints (one joint's
  confidence 0, e.g. its parent was not seen). They break interpolation
  and are never synthesized across (master plan §3).
- Input per frame: the 13 `TKS1` joints × `(x, y, confidence)` **plus
  one binary gap-indicator channel** = 40 dims per frame
  (`MODEL_CONTRACT.md` §3.6). The indicator lets the model distinguish
  "athlete at the root" from "no observation".
- The baseline's window features (`train.featurize_window`) read joints
  only where present: positions about the window's root (the mean hip
  midpoint over frames with both hips), and speeds over frame pairs
  where the joint is present in both. They don't depend on where the
  sequence was centered, so the farm's per-segment center and the
  client's per-take center give the same features
  (`MODEL_CONTRACT.md` §3.6.2).
- Loss masking: any frame-level auxiliary loss ignores gap frames.
  Segment-level targets are unaffected — a labeled window may contain
  gaps; the label still applies to the window.
- Segments are consumed whole in the dataset. Windowing is an
  architecture/training detail, not a dataset one.

### 2.5 Background negatives

- Background negatives come only from each segment's **context
  frames**: the farm exports nothing of a session outside its segments
  (§1). A window is background when the clip window covers less than
  half of it (`train.label_for_window`), so a background window lies
  mostly in context.
- Context is 5 frames on each side, so a window must be shorter than 10
  frames for a background window to exist at all. The pipeline configs
  use 8-frame windows at stride 4 *(judgment call)*: about one
  background window per segment end.
- **The ±2 s margin no longer applies** (`background_margin_seconds =
  0` in the pipeline configs): every context frame is within 0.5 s of
  the clip, so any margin would leave no background. Boundaries are
  where labelers disagree, so these background windows carry some of
  the label noise the margin used to keep out. The margin stays a config
  constant for data that has longer unlabeled stretches, such as the
  smoke fixture.

### 2.6 Splits (aligns with PR #17)

- Grouped by the farm's `split_bucket`, target 80/10/10
  train/val/holdout: every segment of one source has the same bucket, so
  a session lands in one split. `fetch_dataset.py` writes the bucket as
  the manifest's group (`source_id = "bucket-<n>"`), which
  `split_dataset.py` groups by. There are 10 buckets, so a split takes
  whole tenths of the data, and the ratios hold only to that grain.
- **Person leakage is a measured risk.** The export carries no account
  identifier (§1), so one person's different videos can land in both
  train and val. With few contributors this is not a rounding error:
  one prolific contributor can be a large share of both splits, the
  model is rewarded for learning that person, and val is what the
  promotion gate reads (§8). Three measures (master plan §6, decided
  2026-10-15):
  - The admin-curated holdout is built from footage of people outside
    the contributor pool — the maintainer and volunteers recorded for
    the purpose, who agree not to contribute through the app. With no
    account on contributions this is an agreement, not a check the
    pipeline can run.
  - Every run reports `val − holdout` for the promotion metric and
    segment F1 in `holdout_report.json` and the nightly summary.
  - A gap above 0.05 absolute on three consecutive promoted runs is
    flagged in the Discord summary and reopens how splits group
    sources — e.g. a per-install contributor tag used only to group
    splits, which links one person's sources to each other but not to
    an account. It does not by itself reopen unlinking (master plan §1).
    *(Judgment call on both numbers.)*
- Deterministic: seeded RNG, sorted ids, byte-identical `split.json`
  (PR #17's `split_dataset.py` is the implementation).
- Admin-curated holdout pinning: a pinned sample id pulls its whole
  group into holdout — the poisoning-defense anchor. With segments a
  sample id is a segment id, which changes whenever its source is
  re-cut, and its group is a whole split bucket, about a tenth of all
  contributions. Pinning therefore no longer selects the admin-curated
  footage (§10 Q13). The holdout set stays admin-curated in intent
  (master plan §6), composed as above.
- The holdout split is **never loaded for training** — enforced at load
  time by the training script (PR #18), not by convention.
- The dataset manifest (sample ids, taxonomy_version, held-back names)
  and the split (split assignment, split seed) are written next to the
  dataset; a training run is reproducible from them alone.

### 2.7 Class balance

The app's Profile rewards naming clips per trick (master plan §5.6), so
contributions will crowd toward the tricks most people do — gainers,
backflips — and a long tail of rarer ones will stay thin.

- **Judgment call:** sample training windows with weight
  ∝ 1/√(count of the canonical name) rather than uniformly, so a
  frequent name does not drown the rest while a rare name with a
  handful of examples is not repeated into memorization. A combo
  window takes the weight of its rarest name. The scheme is a config
  constant; the per-name counts it reads are this run's train-split
  counts — the same numbers the run later publishes as `train_count`
  in `name_metrics` (§6.1).
- Background negatives (§2.5) are sampled separately and keep their
  own ratio.

## 3. Task definition

- **Input:** pose key sequence, `T × 52`, canonical 10 Hz pose format (features
  per §2.4).
- **Output:** trick segments in **source-frame coordinates** — absolute
  frame indices into the source's canonical 10 Hz sequence — each with
  `trick_names: [...]` (canonical names) and derived `is_combo`.
- This is detection + naming, not pose estimation. Pose estimation is
  MoveNet's job; fine-tuning MoveNet is off the table (master plan §6).

## 4. Architecture

### 4.1 In-flight baseline (PR #18)

The current training script implements a **windowed softmax-linear
classifier** over per-joint mean position + mean speed, with an
`init_checkpoint` path that continues from the champion (the fine-tune
path). The CI smoke test trains it on a synthetic fixture. This is the
MVP that proves the pipeline end-to-end; it is not the production
architecture. Its export also returns the standardized window features
the softmax layer reads, as the embedding output the app's on-device
personalization uses (`MODEL_CONTRACT.md` §3.2.2).

### 4.2 Proposed production architecture

- **Temporal encoder** over the keypoint sequence (52-dim with gap
  channel): a **dilated temporal convolutional network (TCN)** is the
  first choice — small footprint, strong local temporal structure, and
  straightforward Core ML conversion. A small Transformer encoder is the
  fallback if long-range context proves decisive; it costs more
  parameters and latency.
- **Segment proposal head:** anchor-free per-frame boundary regression
  (predict start/end offsets per frame) or per-frame BIO tagging with
  boundary refinement. Either emits candidate segments with scores.
- **Per-segment classification head:** multi-label classifier over the
  canonical vocabulary — sigmoid per class, not softmax, because combos
  carry multiple names. Those per-class sigmoids are the output
  contract's `name_scores` (`MODEL_CONTRACT.md` §3.2), which the app
  thresholds before pre-filling a name, so their calibration matters
  as much as their ranking.
- **Embedding output:** the encoder's output pooled per window,
  `D ≤ 128`, exported next to the heads (`MODEL_CONTRACT.md` §3.2.2).
  The app's on-device personalization ranks names by how close a clip
  is to the user's own named clips in this space (§3.5 there), so it
  works best when the encoder places windows of one trick close
  together.
  *(Judgment call on 128: the app stores one embedding per saved clip.)*
- **Rationale:** tricks are temporal patterns in joint trajectories;
  frame-wise classifiers (the §4.1 baseline) cannot model duration or
  boundaries. The encoder → proposal → classification split mirrors the
  output contract directly (segments + names).
- **On-device constraint (proposal — unmeasured):** the published model
  must run on-device via Core ML. Budget: ≤ 2 M parameters and < 100 ms
  inference for a 60 s (600-frame) sequence on a recent iPhone.
  Architecture choices that break the budget don't ship, regardless of
  validation score.

### 4.3 Post-training conversion

`coremltools` → a Core ML ML Program package (`.mlpackage`, fp16),
zipped byte-reproducibly and uploaded to R2; `POST /api/models`
publishes the champion with its manifest (see `MODEL_CONTRACT.md` §2).
Conversion is a pipeline step, not a manual action
(`Training/export_coreml.py`, run by `Training/publish_model.py` after
promotion). The client compiles the package on device: compiling to
`.mlmodelc` needs macOS, and the pipeline runs on Ubuntu. The §4.1
baseline exports as a window classifier whose per-window probabilities
the client decodes into segments (`MODEL_CONTRACT.md` §3.2.1). The
export also writes the embedding output and its `output_desc.embedding`
descriptor (`MODEL_CONTRACT.md` §3.2.2).

## 5. Training loop

Nightly pipeline order:

1. Count the labels on segments new since the watermark; when they
   exceed 50, take a full pull of `GET /api/labels/export` and prune the
   blob cache (§2.1).
2. Build the dataset (§2): canonicalize at the pinned
   `taxonomy_version`, hold back rare tricks (§6.2), write one sample per
   segment with its `TKS1` frames and (segment, name) entries;
   background comes from the segments' context frames.
3. Split deterministically by `split_bucket` (§2.6).
4. Train the challenger on the train split, initializing from the
   current champion (`init_checkpoint` — the standing fine-tune path;
   from-scratch training is the exception: architecture change or
   taxonomy major bump, decided by the maintainer). Val split for model
   selection; holdout never loaded.
5. Fixture regression gate (§6.3) — the fixture runner scores the
   challenger on the trick-labeled fixture set, and the gate compares
   that against the champion's recorded fixture scores. Its quarantine
   candidates (§7) come from a second `since=<watermark>` pull, taken
   after step 1's full pull; on completion the watermark advances to
   that pull's newest cursor. After a regression it advances only once
   the farm has accepted the quarantine (§7 step 2): until then the
   suspect sources are still in the export, so the watermark holds and
   the next run's candidates still cover them.
6. Holdout evaluation (§6.2) — report written next to the artifact.
7. Promotion gate (§8) — champion/challenger on the validation metric.
8. Publish (§9, `MODEL_CONTRACT.md`).

Further loop properties:

- **Cadence:** nightly cron or manual dispatch; skipped when new labels
  since the watermark are ≤ 50 (master plan §6).
- **Determinism:** seeded RNG everywhere, sorted iteration order, no
  timestamps in artifacts — re-runs yield byte-identical `metrics.json`
  (established by PR #18; required of all future training code).
- **Config:** hermetic TOML; unknown/missing keys fail closed (PR #18).
- **Compute:** `compute = "cpu"|"gpu"` remains the single GPU-worker
  seam (PR #18); the nightly job targets the GPU worker when available.
- **Fixture scores** are recorded with the champion when it is
  promoted, next to its model and val metrics (and in the gate's
  registry record), so the nightly job compares against the champion's
  fixture scores without re-running the champion. A change to the
  fixture set re-scores the champion once (§6.3).

## 6. Evaluation

### 6.1 Metrics (aligns with PR #19)

`Training/evaluate.py` scores a predictions file against the holdout
split and writes a deterministic `holdout_report.json` next to the
artifact:

- **Detection rate** — temporal IoU ≥ 0.5, name-agnostic (did we find
  the trick?).
- **Segment precision / recall / F1** — name match required (did we
  find it *and* name it?).
- **Name accuracy given detection** — of the detected segments, how many
  names are right.
- **Per-name precision/recall** — exposes weak vocabulary entries and
  guides taxonomy curation. Published with each promoted model as
  `name_metrics` (`MODEL_CONTRACT.md` §1): per canonical name,
  `precision`, `recall`, `holdout_count`, and `train_count` (that
  name's training examples). It is what the app shows users as "the
  model now knows this trick this well", so a name with fewer than 5
  holdout examples publishes its counts with `null` precision and
  recall rather than a number that small a sample can't support.
  *(Judgment call on 5.)* Precision is also `null` for a name the
  model never predicted on the holdout (0 of 0 is undefined).
  `Training/name_metrics.py` writes it as `name_metrics.json` next to
  the holdout report, taking `train_count` from `train.py`'s
  `metrics.json` (`train_name_counts`), the counts §2.7 trains with.
- **Background false-positive count** — spurious segments fired in
  unlabeled regions.
- **By origin** — name accuracy and segment F1 split by label `origin`
  and by blind vs. not (§6.4), reported, not gated. **Proposed (not yet
  implemented)** — tracked as turnip-ml#40; the evaluator does not split
  by origin yet.

Additionally recommended for the champion/challenger comparison:
**mAP at tIoU ∈ {0.3, 0.5, 0.7}** as the segment-quality summary. The
evaluator (PR #19) computes the core set above except the By-origin split
(§6.4, proposed); mAP is the proposed promotion-facing roll-up.

Deliberately **not** PCK: PCK is a pose-estimation metric. Pose
estimation is MoveNet's job, and fine-tuning it is off the table
(per the PR #19 discussion).

### 6.2 Holdout protocol

- The holdout split is admin-curated and pinned (PR #17). The evaluator
  rejects predictions for non-holdout or unknown samples — fail closed,
  never silently evaluated (PR #19).
- The evaluator is decoupled from training: it scores predictions
  files, so it works with any architecture emitting the output contract,
  including the §4.2 proposal, without duplicating model math.
- The holdout report is written next to the model artifact, uploaded to
  R2, and referenced from the model manifest (`holdout_report_r2_key`).
  It is recorded for every nightly candidate; the promotion decision
  itself is the validation-metric gate (§8).
- The report carries the `val − holdout` gap that measures person
  leakage (§2.6). The holdout's people are outside the contributor
  pool, so a model that learned individuals rather than tricks shows
  up as a gap here.
- **Anonymity evidence: membership inference.** Every device receives
  the model, so treating it as anonymous needs evidence
  (`BACKEND_DESIGN.md` §9 Q8, EDPB Opinion 28/2024). The holdout report
  carries a membership-inference check (`Training/membership_inference.py`):
  can the model's outputs tell training data from holdout data better
  than chance? The unit is a sample, which is one segment of farm data
  (the code calls it a source).
  - **Attack:** a confidence attack (Yeom et al., 2018). Each window
    scores the model's probability for its true label, windowed and
    standardized as training saw it; a segment scores the mean over its
    windows. A window whose label the model has no class for is left
    out and counted.
  - **Measure:** the ROC AUC of training-segment scores against
    holdout-segment scores. 0.5 is chance.
  - **Pass** iff AUC ≤ 0.60 and each side has at least 20 segments.
    *(Judgment call on both, §10 Q6.)* With 20 independent holdout
    segments a model that leaks nothing still scores above 0.60 in about
    7% of runs; fewer give no evidence either way, so the check fails
    rather than pass on noise. Segments of one source are not
    independent, so the real chance spread is somewhat wider.
  - The holdout's people are outside the contributor pool (§2.6), so a
    model that learned individuals fails as well as one that memorized
    sources. That errs on the side of failing.
  - The promotion gate archives a challenger that did not pass (§8).
- **Scheduled retrain.** Models are retrained from scratch on a set
  schedule, so deleted data leaves within one cycle (§2.1, scheduled
  retrain).
- **Release checks: nothing but the model ships.** No pose sequences
  and no per-example embeddings ship inside a model artifact (the
  second is the farm's privacy plan, `BACKEND_DESIGN.md` §8.1).
  `Training/publish_model.py` runs `Training/release_checks.py` on the
  exported zip before anything is uploaded, and a failure publishes
  nothing. The checks are allow-lists, so whatever is not one of the
  model's parameters fails however it is stored:
  - the checkpoint holds only the model's keys, at the shapes the
    export reads;
  - the zip holds only the package's spec, weight file and manifest;
  - every floating-point value in the Core ML program equals one of the
    model's parameters (`feat_mean`, `1 / feat_std`, `W`, `b`), and
    every other value (shapes, axes, flags) has at most 8 elements;
  - the weight file holds only blobs the program reads;
  - the metadata, inputs and outputs are the contract's
    (`MODEL_CONTRACT.md` §3). The `window_embeddings` output is computed
    on the device from its own input; it stores nothing.
- **Rare-trick hold-back.** A trick, with its metrics, ships only once
  enough distinct people have contributed it (`BACKEND_DESIGN.md` §8.1).
  - Both numbers come from the farm, which alone knows the accounts
    (`BACKEND_DESIGN.md` §1.2.27, §7.3). Every export page carries
    `trick_contributors`, distinct contributors per canonical trick
    (a trick missing from it has 0), and `contributor_minimum`, the
    count a trick needs (5, the farm's judgment call, §10 Q12). A trick
    trains iff `contributors >= contributor_minimum`.
  - Both are the same on every page, but read conservatively: a trick's
    count is its minimum over the pages and the minimum is the greatest
    any page gives.
  - Fail closed: a page without either field holds every trick back,
    and an unmapped token (§2.2) has no count.
  - `fetch_dataset.py build` holds back every segment carrying a name
    below the minimum, whole, combos included. No sample is written for
    it, so the trick has no name, `train_count` or holdout metric, and
    its context frames don't train as background either.
  - `dataset_manifest.json` lists the held-back names, and
    `publish_model.py` refuses a model that names one. Logs carry only
    how many names and segments were held back.

### 6.3 Fixture regression gate (anti-poisoning)

- **Fixtures:** a trick-labeled fixture set (`Training/fixtures/tricks/`)
  — fixed (pose sequence, labeled segments) pairs covering the core
  vocabulary and known combos, plus background-only sequences. The
  labels are the maintainer's, never inferred from footage or taken
  from the export, so nothing contributed can move them. The set's
  `fixtures.json` (labels plus each pose's SHA-256) is versioned with
  the repo; adding a fixture is a code change, never data-dependent.
  The pose blobs stay out of git, content-addressed in the private
  bucket (`training/fixtures/poses/<sha256>.tkp1.gz`), and are verified
  against their committed hashes before use. The set guards the trick
  model's *output*. Nothing guards its pose *input* any more: the
  pose-accuracy fixtures went with the pose-accuracy harness (PR #22),
  as pose estimation is MoveNet's job (§3).
- **Fixture runner** (`Training/fixture_runner.py`): runs a model over
  every fixture exactly as the holdout predictor runs it, and pools two
  scores over the set. Labeled and predicted segments are matched
  one-to-one at temporal IoU ≥ 0.5, with the evaluator's greedy
  best-IoU matching:
  - `segment_iou` = Σ matched IoU ÷ (labeled + predicted − matched):
    a missed segment and a spurious one (on a background fixture too)
    each count as IoU 0.
  - `name_accuracy` = labeled segments matched with the exact name set
    ÷ labeled segments. Unlike §6.1's name accuracy given detection, it
    does not condition on detection, so finding less lowers both
    scores.

  Its output, `{"fixture_set", "segment_iou", "name_accuracy"}`, is
  byte-identical on re-run. `fixture_set` is a SHA-256 over the whole
  set (ids, labels, pose hashes) and the scoring rule, so any fixture
  change changes it, and the gate refuses to compare scores across it.
- **Gate:** after training a candidate, run it over the fixtures and
  compare segment IoU + name accuracy against the champion's recorded
  fixture scores (`Training/fixture_gate.py`).
- **Recorded champion scores:** a challenger's fixture scores are
  stored with it when it is promoted, next to its model and metrics, so
  the nightly job never re-runs the champion. Each champion is a version
  in the private bucket (`training/champion/versions/<version>/`), made
  current by a pointer written last (`training/champion/current.json`)
  that carries every file's SHA-256. A failed upload therefore leaves
  the old champion whole, and every restore is hash-verified. When the
  scores are missing or carry another `fixture_set` (the set or the
  scoring rule changed), the pipeline re-scores the champion once and
  stores it again as a new version with the new scores.
- **First champion:** with no champion there is nothing to compare
  against, and the nightly path never promotes ungated: the run records
  the challenger's fixture scores and stops. The maintainer seeds the
  first champion explicitly (pipeline dispatch `bootstrap_champion`);
  the gate guards every promotion after it. The bootstrapped champion is
  the fine-tune base and gate baseline but is never published: only a
  gated promotion is (§8). *(Judgment call: an automatic first champion
  would be the one promotion nothing checks.)*
- **Nothing to score:** until the set has a fixture with a labeled
  segment, the gate and promotion don't run, and a bootstrap is refused
  (the first champion needs fixture scores); the pipeline says so and
  changes no champion. The fixture runner is the one judge of this
  (`fixture_runner.py check`). A fixture shorter than the model's window
  is refused rather than scored: the model cannot see it.
- **Regression policy (judgment call):** regression iff
  `challenger_fixture_score < champion_fixture_score − 0.01` (absolute,
  on the composite fixture score, the mean of the two). Any regression
  beyond that tolerance triggers the quarantine flow (§7) and skips
  promotion. No human override exists in the nightly path; the
  maintainer can re-run with an explicit override flag, which is
  logged.

### 6.4 Provenance and rubber-stamping

**Proposed (not yet implemented)** — tracked as turnip-ml#40. The
measures below describe the intended design, not current behavior:
nothing computes them yet, and the nightly summary reports none of
them.

Once the model pre-fills names (`MODEL_CONTRACT.md` §3.3), keeping its
suggestion is the cheapest way to name a clip, and the app's Profile
rewards naming clips. A model trained on names it suggested itself
learns its own mistakes back. Label provenance (§1) makes that
measurable:

- **Keep rate** — on model-proposed clips that were not blind, the
  share of `suggested_names` (the names the model pre-filled) that the
  user kept: labels with `origin: model` ÷ `suggested_names`.
- **Blind agreement** — on blind clips (`suggestions_withheld`), the
  share of `suggested_names` (the names the model would have
  pre-filled) that the user gave anyway, without seeing them.
- **Rubber-stamp gap** = keep rate − blind agreement, per proposing
  model version. Users who judge each suggestion keep it about as
  often as they would have named it unprompted; a large positive gap
  means suggestions are being kept that would not have been given.

All three are computed over the export, so over contributed clips
only, and reported in the nightly summary next to the §6.1 metrics by
origin. A contributed clip always carries at least one name, so a clip
whose suggestions were all removed still appears, through the name the
user gave instead. What the measures cannot see is clips never
contributed, and users may contribute the clips they are surest of —
read the keep rate as a lower bound on how much rubber-stamping the
app's naming as a whole sees. How they are used:

- **Down-weighting.** Training weights labels with `origin: model` on
  non-blind clips by a config constant, default 1.0. When the gap
  stays above 0.1 the maintainer lowers it; the change is a reviewed
  config change, never automatic. *(Judgment call on 0.1.)*
- **Independent evaluation.** Blind clips in the val split are scored
  separately: the model's name accuracy against names given without
  its influence, comparable across model versions.
- **No penalty for blindness.** The app counts a blind clip exactly
  like any other everywhere, the Profile included
  (`turnip-ios/docs/PROFILE_DESIGN.md` §6), so users have no reason to
  treat the blind sample differently, and the sample stays unbiased.

Provenance is client-reported and unverifiable, like the names
themselves; a client that lies about it is a poisoning case, which the
fixture gate (§6.3) remains the defense against.

On-device personalization (`MODEL_CONTRACT.md` §3.5) leaves these
measures as they are. It never pre-fills, so `suggested_names` and
`origin: model` stay the central model's. Its ranking is withheld on
blind clips along with the model's, so blind names stay independent of
both. What it does change, the export cannot see: a name picked from a
personally ranked list arrives as `origin: vocabulary`, and a window the
personal keep layer passed looks like any other model proposal
(`MODEL_CONTRACT.md` §6 item 9).

## 7. Quarantine mechanics (anti-poisoning)

On fixture regression (master plan §6), in order:

1. **Discord alert** via webhook with the metrics delta (challenger vs
   champion fixture scores) and the affected `source_id`s (step 2). **Never account IDs:** a Discord message outlives the farm's
   30-day attribution window (`BACKEND_DESIGN.md` §8).
2. **Quarantine the sources changed since the watermark:** one
   `data_quarantine` row per source —
   `(source_id, reason, metrics JSONB snapshot, quarantined_at)`, with
   no `user_id`.
   - **Which sources:** every source with a label row changed since the
     previous successful watermark. That is new uploads and relabels of
     old sources alike (a relabel can poison as well as an upload), and
     the same rows the trigger counts (§2.1). It is not "sources with
     `created_at` after the watermark", which would miss relabels.
   - **How the list is built:** a `since=<watermark>` pull taken after
     the dataset pull, so the list covers every new row the challenger
     trained on. A source that changes between the two pulls is
     over-quarantined, which is harmless.
   - **How it reaches the farm:** the pipeline sends the source ids
     explicitly, so the alert and the quarantine list are the same. The
     farm inserts one row per id and skips sources already under an
     open quarantine. The farm does not pick sources by time window: it
     does not know the pipeline's watermark. The gate writes the list
     as `quarantine.json`; the nightly job sends it with
     `Training/report_quarantine.py` (`POST /api/quarantine`, at most
     10000 ids per request) and holds the watermark until the farm has
     accepted every request (§5).
   - **With segments this step is blocked.** The export carries no
     source ids (§1), so the list holds the new **segment** ids
     (`fetch_dataset.py count --out-segments`, §2.1), and
     `POST /api/quarantine` takes source ids only.
     `report_quarantine.py` refuses a segment id with a message that
     says so. A regression therefore still skips promotion and writes
     `quarantine.json`, but the run fails at the send and the watermark
     holds, so every later run lists those segments again. Unblocking
     it needs the farm to quarantine by segment id, resolving each
     through `segment_sources` to its source (§10 Q14).
3. **Training exclusion:** the export excludes sources with an
   unresolved quarantine row (`resolution IS NULL`) or
   `resolution='purged'`; `resolution='released'` rows are re-included.
4. **Skip promotion** for that night's candidate, regardless of its
   validation metrics.
5. **Review workflow:** the maintainer reviews the quarantine queue and
   marks each row **released** (good data — re-included next night) or
   **purged** (bad data — excluded permanently). To find who uploaded a
   purged source, the maintainer reads the farm's `source_attribution`,
   which links a source to the account that last wrote it until 30
   days after that write. Repeated purges traced to one account feed
   the `users.is_blocked` decision, which remains a manual call. Once a
   source is past the window, nobody can tell who uploaded it.
6. The next nightly run retries with the quarantine applied. The loop
   cannot promote poisoned data even when validation metrics look fine,
   because the fixture gate (§6.3) runs **before** the promotion gate.

**Volume spikes.** The Profile makes contribution volume spiky — a new
release, a popular achievement — and a regression on a spike night
quarantines every source changed that night, not only the bad ones.
Accepted for now: the review releases the good sources, and the cost is
a day's delay for them. Revisit if a spike night's queue is more than
one review can clear.

## 8. Promotion (aligns with PR #20)

- **Champion/challenger:** `Training/promote.py` compares the
  challenger's `metrics.json` against the champion's. Promote iff the
  challenger passed the membership-inference check in its holdout
  report (§6.2) and beats the champion by **≥ 1% relative** improvement
  on the configured validation metric —
  `(challenger − champion) / champion ≥ 0.01` (PR #20). A challenger
  that did not pass the check archives whatever its metric, and a
  report without the check is an input error.
- **Refresh:** a scheduled retrain (§2.1) exists to replace the
  champion, whose weights still carry data deleted since it was
  trained, so it need not beat the champion. `promote.py --refresh`
  promotes it unless it is more than 2% worse, relative
  (`(challenger − champion) / champion ≥ −0.02`). *(Judgment call on
  2%.)* The fixture gate runs before it as for every challenger, and
  a refresh that did not pass the membership-inference check archives
  like any other.
- **Judgment call:** for the production architecture (§4.2), the
  promotion metric should be the segment-quality composite (segment F1
  / mAP), not raw window accuracy — `val_accuracy` (PR #20's default)
  is the right default for the §4.1 windowed baseline only. The gate's
  `--metric` flag already supports this; the nightly config chooses
  the metric appropriate to the active architecture.
- Exit codes: 0 = promote, 3 = archive, 2 = input error (fail closed).
  Each decision is appended to the model registry (local
  `Training/registry.json`; the nightly job keeps it in the private
  bucket as `training/registry.json`, next to every champion version;
  the farm's `models` table once `POST /api/models` exists).
- **Gate order:** holdout evaluation and the membership-inference check
  (§6.2, recorded) → fixture regression (§6.3) → promotion gate. A
  candidate that regresses on fixtures never reaches champion/challenger
  comparison. A bootstrapped first champion (§6.3) passes no gate, the
  check included; it is never published.
- **Rollback:** the registry and the farm `models` table retain every
  promoted champion. Rollback publishes a new `models` row that
  re-promotes the previous champion's artifacts — a new unique
  `version` (`trick-v<major>.<minor>-rb<N>`, e.g. `trick-v1.3-rb1`;
  versions are never re-published, so the rollback needs its own) with
  `promoted_at = now()`. The manifest follows because "current" is the
  row with the greatest non-null `promoted_at` (BACKEND_DESIGN.md §4).
  No retraining, no re-point endpoint — see MODEL_CONTRACT.md §5 for
  the full procedure.

## 9. Versioning

- **Model version:** `trick-v<major>.<minor>` (judgment call).
  Minor = retrain on new data, same architecture and taxonomy major.
  Major = architecture change, or a taxonomy change that alters the
  output space (canonical renames/removals).
- **Taxonomy coupling:** every model records the `taxonomy_version` it
  trained on (in `models.taxonomy_version` and the manifest). A
  taxonomy change that renames or removes canonical names forces a
  model major bump — otherwise the model's name outputs would disagree
  with the current vocabulary. Additive vocabulary growth (new names
  only) is a minor bump.
- `taxonomy_version` itself is a monotonic int, bumped when canonical
  names are added or changed (master plan §2).
- **What a promoted model publishes beyond its artifact:** its
  `name_metrics` (§6.1) and `training_segment_ids`, the segment ids of
  its train split, both sent with `POST /api/models`. The farm maps the
  segments still linked to a source through `segment_sources`, and
  turns them into each owner's `trained_in` (`BACKEND_DESIGN.md`
  §1.2.20, §7.3); a de-linked or re-cut segment matches nothing. The registry keeps both per version, so a rollback
  re-sends the re-promoted champion's own list (§8).

## 10. Open questions

1. Unmapped raw labels (§2.2) — keep-training-with-logging vs
   fail-closed exclusion.
2. Background margin width (§2.5) — ±2 s is a starting guess; tune
   against validation.
3. Fixture regression tolerance (§6.3) — 0.01 absolute is a starting
   point.
4. On-device budget (§4.2) — ≤ 2 M params / < 100 ms needs on-device
   measurement before it can gate anything.
5. Promotion metric for the production architecture (§8) — segment
   composite vs window accuracy.
6. Membership-inference check (§6.2) — a confidence attack, passing at
   AUC ≤ 0.60 with at least 20 segments a side, is a starting point. A
   stronger attack (shadow models, per-window rather than per-source
   scores) or val sources as the non-members, which share the
   contributor pool, may replace it.
7. Person-leakage threshold (§2.6) — 0.05 absolute over three promoted
   runs is a starting point.
8. Class-balance scheme (§2.7) — 1/√count is a starting point.
9. Down-weighting kept suggestions (§6.4) — the 0.1 rubber-stamp gap
   that prompts it, and how far to lower the weight.
10. Landed or bailed (master plan §9 Q4) — if clips gain an optional
    outcome, whether a bail trains as its trick, as a separate class,
    or not at all.
11. Offline check of on-device personalization (`MODEL_CONTRACT.md`
    §3.5) — contributions lose their account link after 30 days, and
    the export never carries it, so the export cannot group them by
    user. Replaying personalization per
    source (built on a source's earlier clips, scored on its later ones)
    is the closest offline proxy, and the contract's fixed clip-embedding
    rule makes that replay match the client.
12. Rare-trick hold-back (§6.2) — the farm's `contributor_minimum` of 5
    distinct contributors before a trick ships is a starting point. The
    retrain schedule that bounds how long deleted data stays in a model
    (§2.1) is a starting point too: a from-scratch retrain every 30 days
    that replaces the champion unless it is more than 2% worse. Open:
    what to do when a gate keeps refusing the refresh, beyond retrying
    nightly.
13. Admin-curated holdout with segments (§2.6) — a pinned sample id is
    now a segment id, which changes on every re-cut, and it pulls a
    whole split bucket (about a tenth of all data) into holdout. The
    curated holdout footage needs a way to be named that survives
    re-cuts, such as a farm-side holdout flag carried as its own bucket.
14. Quarantine with segments (§7) — `POST /api/quarantine` takes source
    ids, and segments carry none, so a fixture regression can no longer
    be quarantined: the run fails at the send and the watermark holds.
    It needs the farm to accept segment ids and resolve them to sources
    through `segment_sources`.
15. Windows on segments (§2.5) — 8-frame windows at stride 4 are the
    largest that leave any background in a segment's 5 frames of
    context. Longer context from the farm, or background segments cut
    from unlabeled stretches, would let windows grow back to 16 frames
    and the margin return.
