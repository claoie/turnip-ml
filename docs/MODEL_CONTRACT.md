# Model Manifest + On-Device Inference Contract (`turnip-ml` → `turnip-ios`)

**Scope.** This document is the contract between the training pipeline
and the iOS client: the model manifest format, the publishing flow, and
exactly what the client must do to run the trick-detection model
on-device. It implements the "Export + deploy" step of master plan §6
(`https://github.com/hoiekim/turnip-farm/blob/main/docs/MASTER_PLAN.md`)
and the OTA flow of §5.4. Companion docs: the farm's
`BACKEND_DESIGN.md` (`models` table, `POST /api/models`,
`GET /api/models/current`), `POSE_FORMAT.md` (pose wire format), and
`TRAINING_DESIGN.md` (how champions are produced).

## 1. Manifest format

Served by `GET /api/models/current` — public, no sign-in
(`BACKEND_DESIGN.md` §1.2.19), so users who never contribute still get
the model — assembled from the farm's `models` row (plus
`label_taxonomy` for the vocabulary). The JSON keys match the farm
column names exactly, so an implementer reading `BACKEND_DESIGN.md`
finds every field:

```jsonc
{
  "version": "trick-v1.3",            // models.version (trick-v<major>.<minor>)
  "model_type": "trick-detection",    // models.model_type
  "taxonomy_version": 7,              // models.taxonomy_version
  "r2_key": "models/trick-v1.3.mlpackage.zip",
                                      // models.r2_key
  "url": "<presigned R2 GET, 15-minute expiry>",
                                      // derived at serve time from models.r2_key
  "sha256": "<hex of the zip bytes>", // models.sha256 (NOT NULL)
  "byte_size": 1843200,               // models.byte_size
  "min_client_version": "0.2.0",      // models.min_client_version
  "input_desc": {                     // models.input_desc (JSONB)
    "format": "TKP1",
    "format_version": 1,
    "sample_rate_hz": 10.0,           // canonical input rate the model expects
    "keypoint_count": 17,
    "features": ["x", "y", "confidence", "gap_indicator"],
    "input_name": "pose",             // the Core ML input feature
    "shape": ["T", 52],
    "min_frames": 16,                 // shorter sequences have no window
    "max_frames": 144000              // TKP1's 4-hour bound
  },
  "output_desc": {                    // models.output_desc (JSONB), §3.2.1
    "coordinates": "source-frame",    // absolute indices into the canonical 10 Hz sequence
    "kind": "window_classifier",
    "output_name": "window_probs",    // the Core ML output feature, [N, C]
    "shape": ["N", 4],
    "window_frames": 16,
    "window_stride": 8,
    "classes": ["background", "cork", "cork+gainer", "gainer"],
    "background_class": "background",
    "combo_separator": "+"
  },
  "vocabulary": [                     // assembled from label_taxonomy
    {"canonical": "cork", "aliases": ["cork 720", "corkscrew"]},
    {"canonical": "gainer", "aliases": ["moon kick"]},
    "..."
  ],
  "val_metrics": {                    // models.val_metrics (JSONB)
    "val_accuracy": 0.91,             // the promotion metric (TRAINING_DESIGN.md §8)
    "n_val_windows": 5120
  },
  "name_metrics": {                   // models.name_metrics (JSONB)
    "cork":   {"precision": 0.88, "recall": 0.79, "holdout_count": 24, "train_count": 310},
    "gainer": {"precision": 0.90, "recall": 0.84, "holdout_count": 31, "train_count": 412}
    // ...one entry per canonical name
  },
  "holdout_report_r2_key": "reports/trick-v1.3-holdout.json",
                                      // models.holdout_report_r2_key
  "training_run_id": "nightly-2026-10-07",
                                      // models.training_run_id
  "promoted_at": "2026-10-07T07:12:00Z"
                                      // models.promoted_at (RFC 3339 UTC)
}
```

Field provenance — every manifest key has a farm-side source; there
are no phantom fields:

- `version`, `model_type`, `taxonomy_version`, `r2_key`, `sha256`,
  `byte_size`, `min_client_version`, `input_desc`, `output_desc`,
  `val_metrics`, `name_metrics`, `holdout_report_r2_key`,
  `training_run_id`, `promoted_at` — the `models` row, one key per
  column, same name.
  `models.id` (row UUID) is internal and not exposed in the manifest.
- `vocabulary` — the full canonical vocabulary at the row's
  `taxonomy_version`, assembled server-side from `label_taxonomy`
  (`canonical` names + `aliases`), not stored on the `models` row.
- `url` — not a column: a presigned R2 GET for `r2_key`, minted at
  serve time (15-minute expiry). The client downloads the artifact
  from `url`; `r2_key` is retained for provenance and debugging.

Notes:

- `vocabulary` is the full canonical vocabulary at `taxonomy_version`,
  with aliases for display and search. The client MUST use this embedded
  vocabulary for the active model — never mix it with a cached
  vocabulary from another version (§5).
- `score` is per-segment confidence in [0, 1]; the client thresholds it
  (§3.3). `name_scores` are per-name confidences, one per entry of
  `trick_names` (§3.2).
- `name_metrics` holds, per canonical name the model was evaluated on,
  holdout `precision` and `recall` (`TRAINING_DESIGN.md` §6.1's
  per-name metrics) plus `holdout_count` and `train_count`, the
  number of examples behind them. It is the app's impact feedback: how
  well this model knows each trick, how that moved since the previous
  model, and which tricks it is still learning (few training
  examples). A name with too few holdout examples to score carries its
  counts and `null` precision and recall rather than a misleading
  number; `precision` is also `null` for a name the model never
  predicted on the holdout (undefined, not 0%).
- Every trick name the manifest carries (`name_metrics` keys,
  `output_desc.classes`) is a curated canonical name at
  `taxonomy_version`. The pipeline does not publish a model trained on
  an unmapped label (`TRAINING_DESIGN.md` §2.2): that label is a
  contributor's free text, and this manifest is public.
- `input_desc` and `output_desc` describe the exported Core ML model's
  features; the pipeline writes them from the model it exports
  (`Training/export_coreml.py`). The §4.1 baseline's `output_desc` is a
  window classifier's (§3.2.1); the production model's will describe its
  segment outputs (§3.2).
- `val_metrics` carries the active architecture's promotion metric, for
  the baseline `val_accuracy` over `n_val_windows` windows. Holdout
  metrics are in the report behind `holdout_report_r2_key`.
- `promoted_at` is RFC 3339 UTC. All sizes are bytes; the checksum is
  over the exact zip bytes the client downloads.
- `min_client_version` NULL means the update applies to all client
  versions.

## 2. Publishing flow

1. The nightly pipeline promotes a challenger (TRAINING_DESIGN.md §8).
2. Export (`Training/export_coreml.py`): `coremltools` → a Core ML ML
   Program package (`.mlpackage`, fp16 weights and compute), zipped
   byte-reproducibly; `sha256` + `byte_size` are over the zip. The zip
   holds one directory, `TrickDetector.mlpackage/`. It is not
   a compiled `.mlmodelc`: compiling needs macOS (`coremlcompiler`) and
   the pipeline runs on Ubuntu, so the client compiles the package on
   device (§4).
3. Upload the artifact to `models/trick-v<...>.mlpackage.zip` and the
   holdout report to `reports/trick-v<...>-holdout.json` in R2 (`models/`
   and `reports/` are the pipeline's prefixes in the shared bucket).
   Each object is read back and its sha256 checked. A key that already
   holds other bytes is never overwritten.
4. `POST /api/models` (service-key scoped; `Training/publish_model.py`,
   which runs steps 2–4 after promotion): inserts the `models` row —
   `version`, `model_type`, `taxonomy_version`, `r2_key`, `sha256` (NOT
   NULL, 64-char hex of the artifact bytes), `byte_size`,
   `min_client_version`, `input_desc`, `output_desc`, `val_metrics`
   JSONB, `name_metrics` JSONB, `holdout_report_r2_key`,
   `training_run_id`, and `promoted_at = now()` (`model_type` and
   `promoted_at` are the farm's; the request omits them) — plus
   `training_source_ids`, the sources the model trained on, which the
   farm records in `model_training_sources` so each owner can see their
   contribution went in (`BACKEND_DESIGN.md` §1.2.20). The current model
   is defined as the row with the greatest non-null `promoted_at`
   (BACKEND_DESIGN.md §4), so promotion IS the insert — there is no
   `current` flag to flip. Versions are immutable: an existing `version`
   → `409 VERSION_EXISTS` (bump the version instead). Old rows are
   retained for rollback.
5. `GET /api/models/current` serves the manifest above: version, URL,
   checksum, `taxonomy_version`, `name_metrics`, and the vocabulary
   list.

Endpoint naming note: an earlier draft of this contract referenced
`GET /api/models/latest`; the merged master plan (§4) names the
endpoint `GET /api/models/current`. This document follows the master
plan.

## 3. On-device inference contract

### 3.1 Input

- The client feeds the model a canonical **10 Hz pose-format frame sequence**:
  `T × 17 × 3` float32 `(x, y, confidence)`, x/y normalized 0–1 in
  source-frame coordinates, gaps as all-zero-confidence frames.
- If the client's analysis rate differs from 10 Hz, the client resamples
  to 10 Hz first, using the master-plan rules (linear interpolation of
  x/y/confidence; **gaps break interpolation** — never synthesize pose
  across missing data).
- **Judgment call:** the client appends the gap-indicator channel
  (1.0 on gap frames, 0.0 otherwise) to build the `T × 52` model input
  itself, rather than the exported model deriving it. Rationale: the
  "all confidences are zero" test is trivial client-side and keeps the
  Core ML graph purely numeric.

### 3.2 Output

Per inference call, the model returns segments in **source-frame
coordinates**:

```jsonc
{"start_frame": 120, "end_frame": 175, "score": 0.91,
 "trick_names": ["cork", "gainer"], "name_scores": [0.88, 0.31],
 "is_combo": true}
```

- `start_frame` / `end_frame` are absolute indices into the canonical
  10 Hz sequence the client fed in. The client maps them back to its own
  timeline (video timestamps) via its resampling map.
- `score` is the segment's confidence that a trick is there;
  `name_scores[i]` is the confidence in `trick_names[i]` — the
  per-class sigmoid output of the classification head
  (`TRAINING_DESIGN.md` §4.2). Same order and length as `trick_names`,
  which lists every name the model offers for the segment, highest
  first. *(Judgment call: a parallel array rather than
  `[{name, score}]` objects, so `trick_names` keeps the shape the master
  plan's output contract already gives it.)*
- `is_combo` is always `trick_names.length > 1`, kept explicit for
  client convenience (display, routing). The client re-derives it from
  the names it actually pre-fills (§3.3), which can be fewer.
- Combo MVP: one segment with N names — no sub-segmentation of
  individual tricks.

#### 3.2.1 Baseline output (the window classifier)

The model the pipeline ships today is the §4.1 baseline of
`TRAINING_DESIGN.md`, a windowed softmax classifier. A Core ML graph
cannot emit §3.2's variable-length segment list from it, so it returns
per-window class probabilities, and the client decodes them into §3.2
segments. The manifest's `output_desc` carries what decoding needs
(§1).

- Output `window_probs`: float32 `[N, C]`, `N = (T − window_frames) ÷
  window_stride + 1` (integer division). Row `i` is the softmax over
  `classes` for the window covering frames
  `[i · window_stride, i · window_stride + window_frames)`.
- Each window takes its highest-probability class. Consecutive windows
  with the same class, other than `background_class`, merge into one
  segment from the first window's start frame to the last window's end
  frame. This is the decoding training scores (`Training/train.py`,
  `decode_segments`).
- `trick_names` is the class split on `combo_separator`: a combo is one
  class (`"cork+gainer"`). `is_combo` is `trick_names.length > 1`.
- **Judgment call:** `score` is the mean, over the merged windows, of
  the winning class's probability. Every entry of `name_scores` is that
  same `score`, because a combo's names share one softmax class.

Worked example: a 60 s source is 600 canonical frames. A returned
segment `start_frame=120, end_frame=175` spans seconds 12.0–17.5 of the
canonical timeline; the client converts to its own frame/timestamp
space from there.

### 3.3 Client postprocessing (normative)

1. **Threshold:** drop segments with `score < 0.5`. (Judgment call: a
   fixed default for now; a future manifest `client_config` may
   override it per model version.)
2. **NMS:** class-agnostic non-maximum suppression — for overlapping
   segments with temporal IoU > 0.5, keep the higher `score`.
   (Judgment call: single class-agnostic pass; per-name NMS only if
   false-positive analysis demands it.)
3. **Merge:** merge segments with identical `trick_names` separated by
   fewer than 5 frames (0.5 s) into one segment spanning the union.
   (Judgment call: covers flicker at trick boundaries.)
4. **Pre-fill only confident names.** A name with `name_score ≥ 0.5`
   becomes a pre-filled chip, `origin: model` (`BACKEND_DESIGN.md`
   §1.2.18); a name below that is not filled in and may only rank higher
   among the naming screen's typed-ahead suggestions (picking it there
   is `origin: vocabulary`). A segment whose names all fall below the
   bar is still proposed, as a window with no names. (Judgment call:
   0.5, the segment threshold's default, until holdout calibration says
   otherwise. Fewer, surer pre-fills leave less to rubber-stamp.)
5. **Blind sample.** At creation, each model-proposed clip is drawn
   into the blind sample with probability 0.1, once, and the draw is
   persisted with the clip: a blind clip gets its window but no
   pre-filled names and no model ranking in the suggestions
   (`suggestions_withheld: true`). Its names are then independent of
   the model, which is what `TRAINING_DESIGN.md` §6.4 measures
   rubber-stamping against. The app never treats a blind clip
   differently anywhere else — the Profile counts it like any other
   clip (`turnip-ios/docs/PROFILE_DESIGN.md` §6) — and never tells the
   user which clips are blind. (Judgment call: 0.1 trades unbiased
   data against the pre-fill convenience a tenth of clips lose; a
   future manifest `client_config` may set it per model version.)
6. **Record provenance.** For every model-proposed clip the client keeps
   `proposal_model_version` (the active `version`) and `suggested_names`
   — the names step 4 pre-filled, or on a blind clip would have, whether
   the user kept or removed them — and sends all three, with step 5's
   `suggestions_withheld`, with the clip if it is contributed (`POST
   /api/clips`). Names below the pre-fill bar are not recorded: they
   were never offered as an answer, only ranked.
7. The surviving segments feed the app as proposals — windows to the
   clip list, names to the naming step after it — and the user confirms
   or edits them per the confirmed contribution flow. Proposals are
   never contributed without explicit user confirmation.

### 3.4 Preprocessing checklist (client)

- Resample pose to canonical 10 Hz (gap-safe, §3.1).
- Do **not** re-normalize x/y per sequence — values are already 0–1 in
  source-frame coordinates, and absolute position in frame matters for
  tricks.
- Append the gap-indicator channel → `T × 52` float32.
- Inference is fully on-device; no data leaves the device for
  inference. The manifest/vocabulary download (§4) is the only network
  touch in this contract.

## 4. OTA update flow

- The client polls `GET /api/models/current` on app launch and on
  foreground, signed in or not. (Per the iOS side, this polling is
  already wired into launch/foreground — the contract it speaks is the
  §1 manifest.)
- Update iff `version` differs from the active model AND the
  client version ≥ `min_client_version`.
- Download the artifact from the manifest's presigned `url`
  streamed to a temp file; verify `sha256` (and `byte_size`) **before**
  touching the active model.
- Unzip the verified artifact and compile the package
  (`MLModel.compileModel(at:)`) into a temp location. A compile failure
  is a load failure (below).
- **Atomic swap:** verify → compile → move into place → activate on the
  next inference call. Never half-replace the active model.
- **Failure fallback:** any failure — network error, checksum mismatch,
  Core ML compile or load error — keeps the previous model active and retries on
  the next poll. The client retains the previous artifact until the new
  one loads successfully; on load failure the temp file is discarded.
  (Judgment call: load-success is the acceptance check; no on-device
  accuracy smoke test for v1.)
- **First run / never downloaded:** fall back to the **bundled model**
  shipped in the app bundle (a frozen champion checked in at release
  time), or — if none is bundled — the heuristic detector. The app must
  never be left without a proposal path.
- Provenance: the client includes the proposing model's `version` in
  contribution uploads as each clip's `proposal_model_version`, with
  its `suggested_names` and `suggestions_withheld` (§3.3), so training
  can tell which model proposed the clips the user confirmed and which
  names the user kept from it.

## 5. Compatibility

- **Taxonomy mismatch:** the client always reads the model's output
  names against the `vocabulary` embedded in the active model's
  manifest. A new model with a new `taxonomy_version` carries its own
  names, so an old client displays them correctly with no app update.
  Never merge vocabularies across versions.
- **Naming vocabulary (separate from the model's):** the label
  editor's suggestions and the Profile's grouping by trick use the
  newest vocabulary the client holds — `GET /api/taxonomy/current`
  (`BACKEND_DESIGN.md` §7.3), the active manifest's, or the seed
  bundled with the app, whichever has the greatest `taxonomy_version`
  — whole, never merged with another. A name the model output under an
  older version resolves to the newer vocabulary through its aliases:
  a canonical rename lists the old name as an alias of the new one
  (`BACKEND_DESIGN.md` §3, "Renames resolve forward").
- **Model absent:** heuristic detector fallback (master plan §5.4) —
  proposals are window-only and names come from the user, with
  suggestions from the naming vocabulary above.
- **`min_client_version`:** clients below it skip the update silently
  (logged locally). Breaking input-format changes bump
  `min_client_version` and the model major version together.
- **Rollback (server side):** publish a new `models` row that
  re-promotes the previous champion — `r2_key`, `sha256`, `byte_size`,
  `input_desc`, `output_desc`, `taxonomy_version`, `val_metrics`,
  `name_metrics`, `holdout_report_r2_key`, and `training_run_id` copied
  from the previous champion's row (the artifact and the report stay at
  their original `models/` and `reports/` keys; nothing is re-uploaded),
  and its training source list
  re-sent as `training_source_ids` (the registry keeps it,
  `TRAINING_DESIGN.md` §9); a new unique `version`
  (`trick-v<major>.<minor>-rb<N>`, e.g. `trick-v1.3-rb1` — versions are
  never re-published, so the rollback needs its own); and
  `promoted_at = now()`. Because "current" is the greatest non-null
  `promoted_at`, the insert alone moves the manifest back to the old
  artifacts — no retraining, no re-point endpoint. Clients pick the
  rollback up on the next poll via the normal version-compare path. No
  client-side rollback protocol is needed beyond the atomic-swap
  failure fallback (§4).

## 6. Open questions

1. Score threshold (§3.3) — 0.5 default is a starting point; tune
   against precision/recall on the holdout set.
2. NMS/merge policy (§3.3) — class-agnostic NMS + 5-frame merge is the
   MVP; revisit with real model behavior.
3. Gap-indicator placement (§3.1) — client-side vs baked into the
   exported graph.
4. On-device acceptance (§4) — whether a forward-pass smoke test over a
   cached pose snippet is worth the complexity for v1.
5. Pre-fill threshold (§3.3) — 0.5 on `name_score` is a starting
   point; tune once holdout calibration of the name head exists.
6. Blind-sample rate (§3.3) — 0.1 is a starting point; lower it once
   the rubber-stamping measure (`TRAINING_DESIGN.md` §6.4) is stable,
   raise it if that measure is too noisy to read.
7. Package vs compiled artifact (§2) — the pipeline ships `.mlpackage`
   for the client to compile, because compiling needs macOS. An export
   job on a macOS runner could ship `.mlmodelc` and skip the on-device
   compile.
