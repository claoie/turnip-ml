#!/usr/bin/env bash
# The training chain shared by both modes of
# .github/workflows/training-pipeline.yml (docs/TRAINING_DESIGN.md §5).
# Run from the repository root once the dataset is in place:
#
#   data/dataset/samples/*.json, data/dataset/manifest.jsonl   (required)
#   data/champion/model.npz   the current champion, if any (fine-tune base)
#
# Steps: split -> train (fine-tune the champion unless FROM_SCRATCH=true)
# -> predict holdout -> evaluate holdout -> per-name metrics (#41). Then,
# in the design's order, fixture gate (§6.3) -> promotion (§8) -> publish
# (#16: Core ML export, R2 upload, POST /api/models). Everything lands in
# data/run/.
#
# Explicit stop: the fixture gate needs fixture scores from the fixture
# runner (#27), which doesn't exist yet, so the gate cannot run and
# nothing is promoted. Publishing is wired but runs only after a promotion,
# so an ungated model is never published. A run therefore trains,
# evaluates and records, and changes no champion.
#
# Environment:
#   HOLDOUT_IDS    file of admin-pinned holdout sample ids (optional)
#   SPLIT_RATIOS   "train val holdout" split targets (default 0.8 0.1 0.1)
#   FROM_SCRATCH   "true" ignores data/champion
#   PUBLISH        "true" publishes a promoted champion (the workflow's farm
#                  job; it also needs the farm and R2 variables of
#                  Training/publish_model.py). Otherwise a promotion is
#                  recorded but not published.
#
# Privacy: stdout and the job summary carry counts and aggregate metrics
# only. Per-sample outputs (split, predictions, the report's per-sample
# section, val segments) stay in data/run/.
set -euo pipefail

DATASET=data/dataset
RUN=data/run
CHAMPION=data/champion
read -r TRAIN_RATIO VAL_RATIO HOLDOUT_RATIO <<<"${SPLIT_RATIOS:-0.8 0.1 0.1}"
SUMMARY=${GITHUB_STEP_SUMMARY:-/dev/null}

mkdir -p "$RUN"

split_args=(
  --in "$DATASET/manifest.jsonl" --out "$RUN/split.json" --seed 0
  --train-ratio "$TRAIN_RATIO" --val-ratio "$VAL_RATIO"
  --holdout-ratio "$HOLDOUT_RATIO"
)
if [ -n "${HOLDOUT_IDS:-}" ]; then
  split_args+=(--holdout-ids "$HOLDOUT_IDS")
fi
python Training/split_dataset.py "${split_args[@]}"

config=Training/config/pipeline.toml
if [ -f "$CHAMPION/model.npz" ] && [ "${FROM_SCRATCH:-false}" != "true" ]; then
  config=Training/config/pipeline-finetune.toml
fi
echo "training with $config"
python Training/train.py --config "$config"

python Training/predict.py \
  --model "$RUN/challenger/model.npz" --split "$RUN/split.json" \
  --samples-dir "$DATASET/samples" \
  --out "$RUN/challenger/holdout_predictions.json"
python Training/evaluate.py \
  --predictions "$RUN/challenger/holdout_predictions.json" \
  --split "$RUN/split.json" --samples-dir "$DATASET/samples" \
  --out "$RUN/challenger/holdout_report.json"

python Training/name_metrics.py \
  --report "$RUN/challenger/holdout_report.json" \
  --split "$RUN/split.json" --samples-dir "$DATASET/samples" \
  --out "$RUN/challenger/name_metrics.json"

# Fixture gate (§6.3) -> promotion (§8). Not wired: the gate needs fixture
# scores from the fixture runner (#27). When it lands, this line becomes
# the gate (fixture_gate.py) and then promote.py, setting PROMOTED=true
# only when both exit 0; until then nothing is promoted.
PROMOTED=false

if [ "$PROMOTED" = true ] && [ "${PUBLISH:-false}" = true ]; then
  python Training/publish_model.py --run "$RUN" --dataset "$DATASET"
  STOP="Published the promoted champion (#16)."
elif [ "$PROMOTED" = true ]; then
  STOP="Promoted, not published: PUBLISH is not true (only the farm run \
publishes)."
else
  STOP="Fixture gate, promotion and publish: not run. The gate needs fixture \
scores from the fixture runner (#27); promotion runs only after the gate, \
and publishing (#16) only after promotion, so no ungated model is \
published. No champion changed."
fi
echo "$STOP"

python - "$RUN" "$config" "$STOP" >>"$SUMMARY" <<'EOF'
import json
import sys
from pathlib import Path

run, config, stop = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
metrics = json.loads((run / "challenger" / "metrics.json").read_text())
report = json.loads((run / "challenger" / "holdout_report.json").read_text())
split = json.loads((run / "split.json").read_text())
counts = {s: sum(1 for v in split.values() if v == s) for s in ("train", "val", "holdout")}
print("## Training pipeline run\n")
print(f"- Split (samples): train {counts['train']}, val {counts['val']}, "
      f"holdout {counts['holdout']}")
print(f"- Config: `{config}`; classes: {len(metrics['classes'])}")
print(f"- Val accuracy (windows): {metrics['val_accuracy']:.4f} "
      f"over {metrics['n_val_windows']} windows")
print("- Holdout: " + ", ".join(
    f"{key} {report[key]}" for key in (
        "detection_rate", "precision", "recall", "f1",
        "name_accuracy_given_detection", "n_ground_truth_segments",
        "n_predicted_segments",
    )
))
names = json.loads((run / "challenger" / "name_metrics.json").read_text())
scored = sum(1 for m in names.values() if m["precision"] is not None)
print(f"- Names: {len(names)}, {scored} with enough holdout examples to score")
publish = run / "challenger" / "publish.json"
if publish.exists():
    record = json.loads(publish.read_text())
    print(f"- Published `{record['version']}`: {record['byte_size']} bytes, "
          f"{record['training_sources']} training sources")
print(f"\n**Stops.** {stop}")
EOF
