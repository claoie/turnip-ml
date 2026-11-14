#!/usr/bin/env bash
# The training chain shared by both modes of
# .github/workflows/training-pipeline.yml (docs/TRAINING_DESIGN.md §5).
# Run from the repository root once the dataset is in place:
#
#   data/dataset/samples/*.json, data/dataset/manifest.jsonl   (required)
#   data/champion/model.npz                 the current champion, if any
#   data/champion/metrics.json              its val metrics (promotion)
#   data/champion/fixture_scores.json       its recorded fixture scores
#   $FIXTURE_POSES/<sha256>.tkp1.gz         the fixture set's pose blobs
#   $INGESTED       sources changed since the watermark: the fixture
#                   gate's quarantine candidates (needed once the gate runs)
#
# Steps: split -> train (fine-tune the champion unless FROM_SCRATCH=true)
# -> predict holdout -> evaluate holdout -> membership-inference check
# (§6.2, written into the holdout report) -> per-name metrics (#41) ->
# score the challenger on the trick-labeled fixture set (fixture_runner.py)
# -> fixture gate (§6.3) against the champion's recorded fixture scores ->
# promotion gate (§8) if the fixture gate passed -> publish (#16: Core ML
# export, R2 upload, POST /api/models) if it promoted and PUBLISH=true.
# Everything lands in data/run/.
#
# The champion's fixture scores are recorded when it is promoted and kept
# next to it, so the gate never re-runs the champion. When they are
# missing or were taken on another fixture set, the champion is re-scored
# once and the new scores are staged to be kept instead.
#
# Explicit stops:
#   - the fixture set has nothing to score yet (no fixture with a labeled
#     segment): no gate, no promotion;
#   - no champion yet (first run): the gates have nothing to compare
#     against, so nothing is promoted unless BOOTSTRAP_CHAMPION=true, the
#     maintainer's explicit choice to make this challenger the first
#     champion (§6.3: the nightly path never promotes ungated);
#   - a challenger that fails the membership-inference check: the
#     promotion gate archives it (promote.py reads the holdout report);
#   - a fixture regression: promotion is skipped, quarantine.json written
#     (nothing here sends it: the workflow's farm job does, with
#     Training/report_quarantine.py, before it advances the watermark);
#   - a champion without metrics.json: the fixture gate runs, the
#     promotion gate has nothing to compare and does not;
#   - publishing runs only after a gated promotion, never after a
#     bootstrap: an ungated model is never published.
# A promoted (or bootstrapped) champion is staged in data/run/
# champion_update/ for the workflow to store as the next fine-tune base
# and gate baseline.
#
# Environment:
#   HOLDOUT_IDS    file of admin-pinned holdout sample ids (optional)
#   SPLIT_RATIOS   "train val holdout" split targets (default 0.8 0.1 0.1)
#   FROM_SCRATCH   "true" trains without data/champion as the base (the
#                  gates still compare against it)
#   REFRESH        "true": a scheduled retrain (Training/retrain_schedule.py,
#                  TRAINING_DESIGN.md §2.1). Needs FROM_SCRATCH=true; the
#                  promotion gate runs with --refresh, so the challenger
#                  replaces the champion unless it is clearly worse.
#   FIXTURES       fixture set (default Training/fixtures/tricks/fixtures.json)
#   FIXTURE_POSES  its pose blobs (default data/fixture-poses)
#   INGESTED       JSON array of source ids (default data/run/ingested.json)
#   REGISTRY       decision log both gates append to (default
#                  Training/registry.json; the farm job keeps it in R2)
#   BOOTSTRAP_CHAMPION  "true" with no champion: promote this challenger
#                  as the first champion (refused when one exists or the
#                  fixture set has nothing to score)
#   PUBLISH        "true" publishes a promoted champion (the workflow's farm
#                  job; it also needs the farm and R2 variables of
#                  Training/publish_model.py). Otherwise a promotion is
#                  recorded but not published.
#
# Outputs besides data/run/challenger/ (fixture_scores.json included):
#   outcome.json          what ran and what it decided (the workflow reads
#                         it; retrain_schedule.py record reads from_scratch
#                         and promotion)
#   fixture_gate/         the gate's decision.json and quarantine.json
#   promotion.json        the promotion gate's decision
#   champion/             the champion's re-scored fixture scores, if any
#   champion_update/      the whole champion slot as it should become, with
#                         champion.json (champion_slot.py manifest): the
#                         challenger's model, metrics and fixture scores on
#                         promotion, the champion's own files with its new
#                         fixture scores after a re-score
#
# Privacy: stdout and the job summary carry counts and aggregate metrics
# only. Per-sample outputs (split, predictions, the report's per-sample
# section, val segments, the quarantine list) stay in data/run/.
set -euo pipefail

DATASET=data/dataset
RUN=data/run
CHAMPION=data/champion
FIXTURES=${FIXTURES:-Training/fixtures/tricks/fixtures.json}
FIXTURE_POSES=${FIXTURE_POSES:-data/fixture-poses}
INGESTED=${INGESTED:-$RUN/ingested.json}
REGISTRY=${REGISTRY:-Training/registry.json}
BOOTSTRAP=${BOOTSTRAP_CHAMPION:-false}
REFRESH=${REFRESH:-false}
read -r TRAIN_RATIO VAL_RATIO HOLDOUT_RATIO <<<"${SPLIT_RATIOS:-0.8 0.1 0.1}"
SUMMARY=${GITHUB_STEP_SUMMARY:-/dev/null}
UPDATE=$RUN/champion_update
GATE_OUT=$RUN/fixture_gate

die() {
  echo "pipeline: error: $*" >&2
  exit 2
}

# Refuse an inconsistent start before spending a training run on it. The
# fixture runner alone decides whether the set can be scored.
scoreable=true
status=0
python Training/fixture_runner.py check --fixtures "$FIXTURES" || status=$?
case $status in
  0) ;;
  3) scoreable=false ;;
  *) die "the fixture set $FIXTURES is malformed (fixture_runner check: exit $status)" ;;
esac
has_champion=false
if [ -f "$CHAMPION/model.npz" ]; then
  has_champion=true
fi
if [ "$BOOTSTRAP" = "true" ]; then
  [ "$has_champion" = false ] ||
    die "BOOTSTRAP_CHAMPION=true, but a champion exists; bootstrap only seeds the first one"
  [ "$scoreable" = true ] ||
    die "BOOTSTRAP_CHAMPION=true, but $FIXTURES has nothing to score: the first champion needs fixture scores; add fixtures first"
fi
if [ "$REFRESH" = true ] && [ "${FROM_SCRATCH:-false}" != true ]; then
  die "REFRESH=true needs FROM_SCRATCH=true: a scheduled retrain starts from scratch"
fi
if [ "$has_champion" = true ] && [ "$scoreable" = true ] && [ ! -f "$INGESTED" ]; then
  die "the fixture gate needs the changed-source list at $INGESTED"
fi

mkdir -p "$RUN"
# A fresh slate for this run's decisions (a local data/run is reused, and
# train.py leaves files it doesn't write in the challenger directory). The
# summary reports a fixture score or a publish only when this run made one.
rm -rf "$UPDATE" "$GATE_OUT" "$RUN/champion" "$RUN/promotion.json" \
  "$RUN/outcome.json" "$RUN/challenger/fixture_scores.json" \
  "$RUN/challenger/publish.json" "$RUN/challenger/publish_request.json" \
  "$RUN/challenger/TrickDetector.mlpackage.zip"

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
if [ "$has_champion" = true ] && [ "${FROM_SCRATCH:-false}" != "true" ]; then
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

# Membership inference (§6.2): exit 3 is a failed check, not an error;
# it is recorded in the report, and promote.py archives on it.
membership=pass
status=0
python Training/membership_inference.py \
  --model "$RUN/challenger/model.npz" --split "$RUN/split.json" \
  --samples-dir "$DATASET/samples" \
  --report "$RUN/challenger/holdout_report.json" || status=$?
case $status in
  0) ;;
  3) membership=fail ;;
  *) die "membership-inference check failed to run (exit $status)" ;;
esac

python Training/name_metrics.py \
  --report "$RUN/challenger/holdout_report.json" \
  --metrics "$RUN/challenger/metrics.json" --split "$RUN/split.json" \
  --out "$RUN/challenger/name_metrics.json"

score_fixtures() { # score_fixtures <model.npz> <out scores>
  python Training/fixture_runner.py score --model "$1" --fixtures "$FIXTURES" \
    --poses "$FIXTURE_POSES" --out "$2"
}

stage_challenger() { # the challenger, with its fixture scores, as the champion
  mkdir -p "$UPDATE"
  cp "$RUN/challenger/model.npz" "$RUN/challenger/metrics.json" \
    "$RUN/challenger/fixture_scores.json" "$UPDATE/"
}

stage_rescored_champion() { # the champion as it is, with its new scores
  mkdir -p "$UPDATE"
  cp "$CHAMPION/model.npz" "$UPDATE/"
  if [ -f "$CHAMPION/metrics.json" ]; then
    cp "$CHAMPION/metrics.json" "$UPDATE/"
  fi
  cp "$RUN/champion/fixture_scores.json" "$UPDATE/fixture_scores.json"
}

gate=not-run
promotion=not-run
rescored=false
champion_scores=""

if [ "$scoreable" = false ]; then
  STOP="Fixture gate and promotion: not run. The trick-labeled fixture set \
($FIXTURES) has nothing to score yet (no fixture with a labeled segment); \
the maintainer adds fixtures (Training/fixtures/tricks/README.md)."
else
  score_fixtures "$RUN/challenger/model.npz" "$RUN/challenger/fixture_scores.json"
  if [ "$has_champion" = false ] && [ "$BOOTSTRAP" = "true" ]; then
    stage_challenger
    promotion=bootstrap
    STOP="Fixture gate and promotion: not run (bootstrap). There was no \
champion; BOOTSTRAP_CHAMPION made this challenger the first one, staged \
with its fixture scores in $UPDATE."
  elif [ "$has_champion" = false ]; then
    STOP="Fixture gate and promotion: not run. There is no champion yet, \
so nothing to compare against; the challenger's fixture scores are in the \
run. A maintainer dispatch with bootstrap_champion makes a run's \
challenger the first champion."
  else
    champion_scores=$CHAMPION/fixture_scores.json
    status=0
    python Training/fixture_runner.py compare --fixtures "$FIXTURES" \
      --scores "$champion_scores" || status=$?
    case $status in
      0) ;;
      3)
        echo "re-scoring the champion on the current fixture set"
        champion_scores=$RUN/champion/fixture_scores.json
        score_fixtures "$CHAMPION/model.npz" "$champion_scores"
        stage_rescored_champion
        rescored=true
        ;;
      *) die "cannot compare the champion's fixture scores (exit $status)" ;;
    esac

    # Decision records go to files, not the log: promote.py's embeds both
    # metrics.json files, val segments (per-sample) included.
    mkdir -p "$GATE_OUT"
    status=0
    python Training/fixture_gate.py --champion "$champion_scores" \
      --challenger "$RUN/challenger/fixture_scores.json" \
      --ingested "$INGESTED" --out "$GATE_OUT" \
      --registry "$REGISTRY" >"$GATE_OUT/decision.json" || status=$?
    case $status in
      0) gate=pass ;;
      3) gate=regression ;;
      *) die "fixture gate failed (exit $status)" ;;
    esac
    echo "fixture gate: $gate"

    if [ "$gate" = pass ] && [ -f "$CHAMPION/metrics.json" ]; then
      refresh_args=()
      if [ "$REFRESH" = true ]; then
        refresh_args=(--refresh)
      fi
      status=0
      python Training/promote.py --champion "$CHAMPION/metrics.json" \
        --challenger "$RUN/challenger/metrics.json" "${refresh_args[@]}" \
        --holdout-report "$RUN/challenger/holdout_report.json" \
        --registry "$REGISTRY" >"$RUN/promotion.json" || status=$?
      case $status in
        0) promotion=promote ;;
        3) promotion=archive ;;
        *) die "promotion gate failed (exit $status)" ;;
      esac
      echo "promotion gate: $promotion"
    fi

    case $gate/$promotion in
      regression/*)
        STOP="Fixture gate: regression, so promotion was skipped; \
$GATE_OUT/quarantine.json lists the changed sources. Only the \
workflow's farm job sends it to the farm (Training/report_quarantine.py), \
before it advances the watermark." ;;
      pass/promote)
        stage_challenger
        STOP="Fixture gate: pass. Promotion: promote; the challenger is \
staged with its fixture scores as the champion in $UPDATE." ;;
      pass/archive)
        STOP="Fixture gate: pass. Promotion: archive; the champion stays."
        if [ "$membership" = fail ]; then
          STOP="$STOP The challenger did not pass the membership-inference \
check (§6.2), so it was archived whatever its metric."
        fi
        ;;
      pass/not-run)
        STOP="Fixture gate: pass. Promotion: not run; the champion has no \
metrics.json to compare against. Store its val metrics with it \
(training/champion/) to let the promotion gate decide." ;;
    esac
  fi
fi
case $promotion in
  promote | bootstrap) ;;
  *)
    STOP="$STOP No champion changed."
    if [ "$rescored" = true ]; then
      STOP="$STOP The champion was re-scored on the current fixture \
set; its new fixture scores are staged in $UPDATE."
    fi
    ;;
esac
# Person-leakage gap (§2.6, #42): val-minus-holdout for the promotion
# metric (window accuracy, computed on holdout as train.py computes it on
# val) and segment F1, written into the challenger's holdout report,
# with the streak tracked in the registry per run (TRAINING_RUN_ID, else
# a hash of the run's inputs). Runs before publish so the published
# report carries the gap; a corrupt input fails the run rather than
# publishing a model whose leakage went unmeasured.
python Training/leakage_gap.py \
  --metrics "$RUN/challenger/metrics.json" \
  --model "$RUN/challenger/model.npz" \
  --report "$RUN/challenger/holdout_report.json" \
  --split "$RUN/split.json" --samples-dir "$DATASET/samples" \
  --registry "$REGISTRY" --promotion "$promotion" \
  || die "leakage gap computation failed"
# Publish (#16) only what both gates passed.
if [ "$promotion" = promote ] && [ "${PUBLISH:-false}" = true ]; then
  python Training/publish_model.py --run "$RUN" --dataset "$DATASET"
  STOP="$STOP Published the promoted champion (#16)."
elif [ "$promotion" = promote ]; then
  STOP="$STOP Publishing: not run; PUBLISH is not true (only the farm \
run publishes)."
elif [ "$promotion" = bootstrap ]; then
  STOP="$STOP Publishing: not run; a bootstrapped champion passed no \
fixture gate, so it is never published (the first gated promotion is)."
else
  STOP="$STOP Publishing: not run (nothing was promoted)."
fi
if [ "$REFRESH" = true ]; then
  case $promotion in
    promote) STOP="$STOP Scheduled retrain: the from-scratch challenger \
replaced the champion." ;;
    *) STOP="$STOP Scheduled retrain: the champion was not replaced, so \
the next run retrains from scratch again." ;;
  esac
fi
if [ -d "$UPDATE" ]; then
  python Training/champion_slot.py manifest "$UPDATE"
fi
echo "$STOP"

from_scratch=false
if [ "$config" = Training/config/pipeline.toml ]; then
  from_scratch=true
fi
python - "$RUN" "$config" "$STOP" "$gate" "$promotion" "$rescored" \
  "$scoreable" "$champion_scores" "$membership" "$from_scratch" "$REFRESH" \
  >>"$SUMMARY" <<'EOF'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
(config, stop, gate, promotion, rescored, scoreable, champion_scores,
 membership, from_scratch, refresh) = sys.argv[2:]


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


metrics = load(run / "challenger" / "metrics.json")
report = load(run / "challenger" / "holdout_report.json")
split = load(run / "split.json")
gap = report.get("val_minus_holdout", {})

outcome = {
    "fixture_gate": gate,
    "promotion": promotion,
    "champion_rescored": rescored == "true",
    "fixtures_scoreable": scoreable == "true",
    "from_scratch": from_scratch == "true",
    "refresh": refresh == "true",
    "leakage_flagged": gap.get("flagged", False),
    "leakage_streak": gap.get("streak_promoted_runs_over_threshold", 0),
    "membership_inference": membership,
}
(run / "outcome.json").write_text(
    json.dumps(outcome, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

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
if gap:
    flag_note = (
        " — FLAGGED: three consecutive promoted runs over threshold; "
        "reopens how splits group sources (§2.6)"
        if gap["flagged"] else ""
    )
    unseen = gap["n_holdout_windows_unseen_label"]
    unseen_note = f", {unseen} with a label train never saw" if unseen else ""
    print(f"- Val-minus-holdout (§2.6): val_accuracy {gap['promotion_metric']:+.4f} "
          f"(window accuracy {gap['val_accuracy']:.4f} vs holdout "
          f"{gap['holdout_window_accuracy']:.4f} over "
          f"{gap['n_holdout_windows']} holdout windows{unseen_note}), "
          f"segment_f1 {gap['segment_f1']:+.4f} "
          f"(val_f1 {gap['val_f1']:.4f} vs holdout f1 {gap['holdout_f1']:.4f}); "
          f"streak {gap['streak_promoted_runs_over_threshold']} promoted runs "
          f"over 0.05{flag_note}")


mi = report["membership_inference"]
print(f"- Membership inference (§6.2): {membership}, auc {mi['auc']:.4f} "
      f"(max {mi['max_auc']}) over {mi['n_train_sources']} train and "
      f"{mi['n_holdout_sources']} holdout sources (min {mi['min_sources']})")


def scores_line(label, path):
    s = load(path)
    print(f"- {label}: segment_iou {s['segment_iou']}, "
          f"name_accuracy {s['name_accuracy']}")
    return s


challenger_scores = run / "challenger" / "fixture_scores.json"
if challenger_scores.exists():
    s = scores_line("Challenger fixture scores", challenger_scores)
    print(f"- Fixture set: `{s['fixture_set']}`")
if champion_scores:
    how = "re-scored on this fixture set" if rescored == "true" else "recorded"
    scores_line(f"Champion fixture scores ({how})", champion_scores)
if gate != "not-run":
    decision = load(run / "fixture_gate" / "decision.json")
    m = decision["metrics"]
    print(f"- Fixture gate: {gate} (composite {m['champion']['composite']} -> "
          f"{m['challenger']['composite']}, delta {m['delta']:+}, tolerance "
          f"{m['tolerance']}; {decision['quarantined_sources']} sources "
          "quarantined)")
if promotion in ("promote", "archive"):
    decision = load(run / "promotion.json")
    print(f"- Promotion gate: {promotion} on `{decision['metric']}` "
          f"({decision['champion_metric']} -> {decision['challenger_metric']}; "
          f"{decision['reason']})")
names = load(run / "challenger" / "name_metrics.json")
scored = sum(1 for m in names.values() if m["precision"] is not None)
print(f"- Names: {len(names)}, {scored} with enough holdout examples to score")
publish = run / "challenger" / "publish.json"
if publish.exists():
    record = load(publish)
    print(f"- Published `{record['version']}`: {record['byte_size']} bytes, "
          f"{record['training_sources']} training sources")
print(f"\n**Stops.** {stop}")
EOF
