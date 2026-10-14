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
  `GET /api/labels/export?since=` is training-eligible by construction.
- Privacy posture: pose keypoints (pose format) + labels only. Never
  video, and no account identifier: the farm links a contribution to an
  account only internally, until 30 days after its last write, and
  never exports that link (master plan §1; `BACKEND_DESIGN.md`
  §1.2.17).
- The export excludes quarantined sources (master plan §4) and presents
  one **(source, clip, canonical trick name)** row per label — see §2.2.

## 2. Dataset construction

### 2.1 Export and watermarking

- **Change detection.** The nightly job first pulls
  `GET /api/labels/export?since=<watermark>` and proceeds only when the
  new label rows exceed 50 (master plan §6). The watermark is the
  export cursor (the farm's opaque `next_cursor`) of the last successful
  run, stored in the training workspace (not in the farm). It advances
  only when a run completes.
- **Dataset pull.** The dataset itself is always built from a **full
  pull** (no `since`). The export carries no deletions
  (`BACKEND_DESIGN.md` §6): a source deleted by its owner, by account
  deletion, or by a quarantine purge, and a label dropped when a clip's
  label set is replaced, simply stop appearing. A dataset accumulated
  from incremental pulls would keep training on deleted data.
- **Row contents.** Each row carries the pose-blob R2 reference and its
  digest, the clip window, the raw label string, and the
  `taxonomy_version` in effect at label time. There is no account
  identifier.
- **Blob cache.** Pose blobs may be cached locally, keyed by `r2_key`
  and verified against `blob_sha256`. After each full pull, every
  cached blob whose source is absent from the pull is deleted.

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
   (`unmapped_labels.log`) for later taxonomy work. Rationale: failing
   closed on unmapped names would starve early training when the
   vocabulary is still growing; the log feeds the taxonomy curation loop
   (master plan §9, open question 1).

Clip → Label shape (decided): the farm stores **one label row per
`(clip_id, trick_name)`** (`BACKEND_DESIGN.md` §1.1) — one-to-many at
every level (Source → Clip → Label). The export is already
label-granular (§1), so the dataset builder consumes one
`(clip_id, raw trick name)` row per export row and canonicalizes each
through `label_taxonomy` at the pinned `taxonomy_version` (above).

### 2.3 Training targets

- One **segment** per confirmed clip window: `(start_frame, end_frame)`
  in source-frame coordinates (absolute indices into the source's
  canonical 10 Hz pose sequence).
- Multi-label: a combo clip with N canonical names becomes one segment
  with an N-hot target vector over the canonical vocabulary (vocabulary
  size fixed at the pinned `taxonomy_version`).
- `is_combo` is derived (`len(trick_names) > 1`); the combo MVP does not
  sub-segment individual tricks (master plan §6).

### 2.4 Gap handling

- pose-format gap frames (all-zero confidence) are **missing data**. They break
  interpolation and are never synthesized across (master plan §3).
- Input features per frame: 17×3 `(x, y, confidence)` **plus one binary
  gap-indicator channel** = 52 dims per frame. The indicator lets the
  model distinguish "athlete at origin" from "no observation".
- Loss masking: any frame-level auxiliary loss ignores gap frames.
  Segment-level targets are unaffected — a labeled window may contain
  gaps; the label still applies to the window.
- Sequences are consumed at full source length in the dataset. Windowing,
  if needed for memory, is an architecture/training detail, not a
  dataset one.

### 2.5 Background negatives

- Non-labeled regions of contributed sources serve as background
  negatives (master plan §6).
- **Judgment call:** exclude a ±2 s margin around every labeled window
  from the background pool. Boundaries are where labelers disagree, and
  training on them as hard negatives injects label noise. The margin
  width is a config constant to be tuned empirically.

### 2.6 Splits (aligns with PR #17)

- Grouped by `source_id`, target 80/10/10 train/val/holdout: all clips
  of one source land in one split. The export carries no account
  identifier (§1), so one person's different videos can land in
  different splits, which can flatter validation scores slightly. The
  admin-curated holdout is the check on that. (PR #17 groups by
  `user_id`; it needs to group by `source_id`.)
- Deterministic: seeded RNG, sorted ids, byte-identical `split.json`
  (PR #17's `split_dataset.py` is the implementation).
- Admin-curated holdout pinning: a pinned sample id pulls its whole
  source into holdout — the poisoning-defense anchor. The holdout set
  stays admin-curated (master plan §6).
- The holdout split is **never loaded for training** — enforced at load
  time by the training script (PR #18), not by convention.
- The dataset manifest (sample ids, split assignment, taxonomy_version,
  export watermark, split seed) is written next to the dataset; a
  training run is reproducible from the manifest alone.

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
architecture.

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
  carry multiple names.
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

`coremltools` → Core ML (`.mlmodelc`), uploaded to R2; `POST
/api/models` publishes the champion with its manifest (see
`MODEL_CONTRACT.md`). Conversion is a pipeline step, not a manual
action.

## 5. Training loop

Nightly pipeline order:

1. Count new labels since the watermark; when they exceed 50, take a
   full pull of `GET /api/labels/export` and prune the blob cache
   (§2.1).
2. Build the dataset (§2): canonicalize at the pinned
   `taxonomy_version`, expand to (clip, name) pairs, attach gap-masked
   pose-format sequences, sample background negatives.
3. Split deterministically by `source_id` (§2.6).
4. Train the challenger on the train split, initializing from the
   current champion (`init_checkpoint` — the standing fine-tune path;
   from-scratch training is the exception: architecture change or
   taxonomy major bump, decided by the maintainer). Val split for model
   selection; holdout never loaded.
5. Fixture regression gate (§6.3) — challenger vs champion on fixtures.
   Its quarantine candidates (§7) come from a second
   `since=<watermark>` pull, taken after step 1's full pull; on
   completion the watermark advances to that pull's newest cursor.
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
- **Fixture scores** are recorded in the model registry alongside val
  metrics, so the nightly job compares against the champion's fixture
  scores without re-running the champion.

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
  guides taxonomy curation.
- **Background false-positive count** — spurious segments fired in
  unlabeled regions.

Additionally recommended for the champion/challenger comparison:
**mAP at tIoU ∈ {0.3, 0.5, 0.7}** as the segment-quality summary. The
evaluator (PR #19) computes the core set above; mAP is the proposed
promotion-facing roll-up.

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
- **Proposed (not yet implemented): anonymity evidence.** Every device
  receives the model, so treating it as anonymous needs evidence
  (`BACKEND_DESIGN.md` §9 Q8, EDPB Opinion 28/2024). The holdout report
  gains a membership-inference check: can the model's outputs tell
  training sources from holdout sources better than chance? Release
  checks also confirm that no pose sequences ship inside the model
  artifact.

### 6.3 Fixture regression gate (anti-poisoning)

- **Fixtures:** the existing pose-accuracy fixtures (guarding the
  *input* to the trick model) **plus** a trick-labeled fixture set —
  fixed (pose sequence, labeled segments) pairs covering the core
  vocabulary and known combos. Fixtures are versioned with the repo;
  adding a fixture is a code change, never data-dependent.
- **Gate:** after training a candidate, run it over the fixtures and
  compare segment IoU + name accuracy against the champion's recorded
  fixture scores.
- **Regression policy (judgment call):** regression iff
  `challenger_fixture_score < champion_fixture_score − 0.01` (absolute,
  on the composite fixture score). Any regression beyond that tolerance
  triggers the quarantine flow (§7) and skips promotion. No human
  override exists in the nightly path; the maintainer can re-run with
  an explicit override flag, which is logged.

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
     does not know the pipeline's watermark.
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

## 8. Promotion (aligns with PR #20)

- **Champion/challenger:** `Training/promote.py` compares the
  challenger's `metrics.json` against the champion's. Promote iff the
  challenger beats the champion by **≥ 1% relative** improvement on the
  configured validation metric —
  `(challenger − champion) / champion ≥ 0.01` (PR #20).
- **Judgment call:** for the production architecture (§4.2), the
  promotion metric should be the segment-quality composite (segment F1
  / mAP), not raw window accuracy — `val_accuracy` (PR #20's default)
  is the right default for the §4.1 windowed baseline only. The gate's
  `--metric` flag already supports this; the nightly config chooses
  the metric appropriate to the active architecture.
- Exit codes: 0 = promote, 3 = archive, 2 = input error (fail closed).
  Each decision is appended to the model registry (local
  `Training/registry.json`; the farm's `models` table once
  `POST /api/models` exists).
- **Gate order:** fixture regression (§6.3) → holdout evaluation
  (§6.2, recorded) → promotion gate. A candidate that regresses on
  fixtures never reaches champion/challenger comparison.
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
6. Membership-inference check (§6.2) — which attack, and what
   advantage over chance counts as passing.
