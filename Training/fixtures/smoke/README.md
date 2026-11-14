# Smoke fixture

Tiny deterministic synthetic dataset for the `train.py` smoke test
(turnip-ml#13). Eight 32-frame @10Hz samples with trick segments, generated
as MoveNet-17 keypoints and stored after the segment transform
(`Training/segment_transform.py`): 13 TKS1 joints in torso lengths, as the
farm's segments arrive. Trick boundaries sit between window starts, so no
window is half trick. Motion signatures are deliberately separable —
the smoke run must reach 100% val accuracy; anything less is a
training-pipeline bug, not data noise.

- `manifest.jsonl` — one `{"sample_id", "source_id"}` per line (splitter
  input); `source_id` groups samples as if cut from one source video
- `split.json` — `{sample_id: train|val|holdout}`, produced by the real
  `split_dataset.py` (turnip-ml#12), `--seed 0`
- `samples/*.json` — one sample per file (see `train.py` docstring for format)

Regenerate byte-identically with:

```sh
python Training/make_smoke_fixture.py --out Training/fixtures/smoke --seed 0
python Training/split_dataset.py \
  --in Training/fixtures/smoke/manifest.jsonl \
  --out Training/fixtures/smoke/split.json --seed 0
```

This fixture is synthetic test data (a few KB of JSON), not a training
dataset — no real footage, no PII, safe to commit.
