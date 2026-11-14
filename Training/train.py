"""Train (or fine-tune) the trick-detection champion on the train split.

turnip-ml#13. The champion maps pose-keypoint sequences to trick segments +
names (see Training/README.md). This script is the training harness:

  - Hermetic TOML config: every hyperparameter lives in the config file,
    never in CLI folklore. Unknown or missing keys fail closed.
  - Trains on the "train" split only (split.json from split_dataset.py,
    turnip-ml#12). "val" is for evaluation; "holdout" samples are never
    loaded for training.
  - Deterministic: seeded RNG, sorted ids, no timestamps — the same config
    + data always yields byte-identical metrics.json. A training run that
    cannot be reproduced is a bug (Training/README.md).
  - ``compute = "cpu" | "gpu"`` is the one-variable GPU-worker seam: it is
    read in exactly one place (``_select_backend``). The full-run
    architecture and the GPU worker are documented in Training/README.md.

Usage:
  python Training/train.py --config Training/config/smoke.toml [--out DIR]

Artifacts in the out dir:
  config.toml   byte-exact snapshot of the effective config
  model.npz     weights (W, b), class list, feature stats, window params
  metrics.json  sorted-key metrics incl. val accuracy, decoded segments,
                per-name train counts, the split's sha256, and the
                class-balance sampling record (§2.7)

Sample JSON format (one file per sample in samples_dir):
  {"sample_id": ..., "fps": 10,
   "keypoints": [[[x, y, score] * 13] * n_frames],  # TKS1 joint order
   "tricks": [{"name": ..., "start_frame": ..., "end_frame": ...,
               "clip_id": ...?}]}
Keypoints are on the farm's standard body (TKS1, turnip-farm
POSE_FORMAT.md §12; Training/segment_transform.py), in torso lengths; a
missing joint is (0, 0, 0). fetch_dataset.py writes one sample per
segment, and also label provenance (tricks[].origins and a top-level
"clips") and the segment's split_bucket; training does not read them, and
samples without them (the smoke fixture) load the same.
The tricks become segments through evaluate.ground_truth_segments, the
rule the holdout evaluation scores against: one segment per distinct
window with every name labeled on it. Multi-name segments label the
window "a+b" (sorted); is_combo is derived.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import types
import zipfile
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11; CI pins 3.12
    sys.stderr.write("train.py needs Python 3.11+ (tomllib)\n")
    raise SystemExit(2)

import evaluate
import numpy as np
import numpy.typing as npt
import segment_transform

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# The full, exact set of accepted config keys. Anything else fails closed:
# a typo'd hyperparameter must never silently fall back to a default.
CONFIG_SCHEMA: dict[str, type | tuple[type, ...]] = {
    "seed": int,
    "compute": str,
    "split_path": str,
    "samples_dir": str,
    "out_dir": str,
    "window_frames": int,
    "window_stride": int,
    "learning_rate": (float, int),
    "epochs": int,
    "l2": (float, int),
    "class_balance_exponent": (float, int),
    "background_ratio": (float, int),
    "background_margin_seconds": (float, int),
    "init_checkpoint": str,
}

COMPUTE_CHOICES = ("cpu", "gpu")


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the TOML config; fail closed on any problem."""
    try:
        with open(path, "rb") as fh:
            cfg = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ValueError(f"cannot read config {path}: {e}") from e
    if not isinstance(cfg, dict):
        raise TypeError(f"config {path} must be a TOML table")
    unknown = sorted(set(cfg) - set(CONFIG_SCHEMA))
    if unknown:
        raise ValueError(f"config {path}: unknown keys (typo?): {unknown}")
    missing = sorted(set(CONFIG_SCHEMA) - set(cfg))
    if missing:
        raise ValueError(f"config {path}: missing keys: {missing}")
    for key, types_ in CONFIG_SCHEMA.items():
        if not isinstance(cfg[key], types_):
            raise TypeError(
                f"config {path}: {key!r} must be {types_}, "
                f"got {type(cfg[key]).__name__}"
            )
    cfg = copy.deepcopy(cfg)
    if cfg["compute"] not in COMPUTE_CHOICES:
        raise ValueError(
            f"config {path}: compute must be one of {COMPUTE_CHOICES}, "
            f"got {cfg['compute']!r}"
        )
    if cfg["window_frames"] <= 0 or cfg["window_stride"] <= 0:
        raise ValueError("window_frames and window_stride must be positive")
    if cfg["epochs"] <= 0:
        raise ValueError("epochs must be positive")
    if not cfg["learning_rate"] > 0:
        raise ValueError("learning_rate must be positive")
    if not cfg["l2"] >= 0:
        raise ValueError("l2 must be non-negative")
    if not cfg["class_balance_exponent"] >= 0:
        raise ValueError("class_balance_exponent must be non-negative")
    if not 0 <= cfg["background_ratio"] < 1:
        raise ValueError("background_ratio must be in [0, 1)")
    if not cfg["background_margin_seconds"] >= 0:
        raise ValueError("background_margin_seconds must be non-negative")
    return cfg


def _select_backend(compute: str) -> types.ModuleType:
    """The one-variable GPU-worker seam.

    ``compute`` (from the config) is the only switch: "cpu" runs on numpy
    here; "gpu" fails closed and points at the full-run docs, which describe
    the GPU worker (cupy-backed) used for real training runs.
    """
    if compute == "cpu":
        return np
    raise RuntimeError(
        "compute='gpu' needs the GPU worker (cupy-backed backend), which is "
        "only set up for full runs — see Training/README.md 'Full run'. "
        "Use compute='cpu' for the smoke test."
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

SPLITS = ("train", "val", "holdout")
BACKGROUND = "background"
N_KEYPOINTS = segment_transform.N_JOINTS  # 13, TKS1 order (MODEL_CONTRACT.md §3.6)
# The root is the midpoint of the TKS1 hips.
HIP_JOINTS = (
    segment_transform.JOINTS.index("left_hip"),
    segment_transform.JOINTS.index("right_hip"),
)
POSE_FPS = 10  # canonical pose rate; converts the §2.5 margin from seconds to frames


def load_split(path: Path) -> dict[str, str]:
    try:
        with open(path, encoding="utf-8") as fh:
            split = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"cannot read split {path}: {e}") from e
    if not isinstance(split, dict):
        raise TypeError(f"split {path} must be a JSON object")
    for sid, s in split.items():
        if s not in SPLITS:
            raise ValueError(f"split {path}: sample {sid!r} has bad split {s!r}")
    return {str(sid): str(s) for sid, s in split.items()}


def check_keypoints(kps: Any, sample_id: str) -> npt.NDArray[np.float64]:
    arr = np.asarray(kps, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != N_KEYPOINTS or arr.shape[2] != 3:
        raise ValueError(
            f"sample {sample_id!r}: keypoints must be "
            f"[frames][{N_KEYPOINTS}][x, y, score], got shape {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError(f"sample {sample_id!r}: no frames")
    return arr


def load_samples(
    samples_dir: Path, split: dict[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load samples, partitioned into train/val. Holdout is never loaded.

    Fail closed: every sample file must have a split entry, and every split
    entry must resolve to a file.
    """
    files = sorted(samples_dir.glob("*.json"))
    if not files:
        raise ValueError(f"no sample files in {samples_dir}")
    by_id: dict[str, dict[str, Any]] = {}
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                rec = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"cannot read sample {f}: {e}") from e
        sid = rec.get("sample_id")
        if not sid or not isinstance(sid, str):
            raise ValueError(f"sample {f} needs a string sample_id")
        if sid in by_id:
            raise ValueError(f"duplicate sample_id {sid!r}")
        if sid not in split:
            raise ValueError(
                f"sample {sid!r} has no entry in the split — every sample "
                "must be assigned train/val/holdout (see split_dataset.py)"
            )
        rec["keypoints"] = check_keypoints(rec.get("keypoints"), sid)
        tricks = rec.get("tricks", [])
        if not isinstance(tricks, list):
            raise TypeError(f"sample {sid!r}: tricks must be a list")
        evaluate.ground_truth_segments(tricks, f"sample {sid!r}")  # fail closed
        by_id[sid] = rec
    missing_files = sorted(set(split) - set(by_id))
    if missing_files:
        raise ValueError(f"split entries with no sample file: {missing_files}")
    # Poisoning defense, enforced at load time: only "train" samples are ever
    # returned for training. Holdout (and anything else) cannot leak in.
    train = [by_id[sid] for sid in sorted(by_id) if split[sid] == "train"]
    val = [by_id[sid] for sid in sorted(by_id) if split[sid] == "val"]
    if not train:
        raise ValueError("no train samples in the split — refusing to train")
    if not val:
        raise ValueError("no val samples in the split — refusing to run unevaluated")
    return train, val


def train_name_counts(train: list[dict[str, Any]]) -> dict[str, int]:
    """Labeled segments per trick name over the train split, sorted by name.

    A combo segment ("a+b") counts once for each of its names. These are
    the per-name train counts of TRAINING_DESIGN.md §2.7, published as
    `train_count` in the model's name_metrics (name_metrics.py).
    """
    counts: dict[str, int] = {}
    for sample in train:
        for trick in sample.get("tricks", []):
            for name in {n for n in str(trick["name"]).split("+") if n}:
                counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Windowing / features / labels
# ---------------------------------------------------------------------------


def label_for_window(
    segments: list[tuple[int, int, frozenset[str]]],
    start: int,
    end: int,
    margin_frames: int = 0,
) -> str | None:
    """Label for a window: the names of every segment covering >= half of it,
    else background — unless the window falls in the background margin.

    segments are evaluate.ground_truth_segments() of the sample's tricks, the
    segments the holdout evaluation scores against. Center-frame labeling
    mislabels edge windows (the window is mostly outside the segment while
    its center is just inside, or vice versa); majority overlap keeps the
    label consistent with what the window's features show. N names label the
    window "a+b" (sorted, each name once) — is_combo is derived from the "+"
    downstream, per the model output contract.

    Margin (TRAINING_DESIGN.md §2.5): a window that would be background but
    lies within margin_frames of a labeled segment is excluded from the
    background pool — returned as None, not "background". Boundaries are
    where labelers disagree, and training on them as hard negatives injects
    label noise. A window is in the margin when the gap between its interval
    and the nearest segment interval is under margin_frames (a touching
    window has gap 0); a window exactly margin_frames away is background.
    Trick windows are never demoted: >= half overlap labels the window even
    inside another segment's margin.
    """
    names = sorted(
        {
            name
            for seg_start, seg_end, seg_names in segments
            if max(0, min(seg_end, end) - max(seg_start, start)) >= (end - start) / 2
            for name in seg_names
        }
    )
    if names:
        return "+".join(names)
    if margin_frames > 0 and any(
        max(seg_start - end, start - seg_end, 0) < margin_frames
        for seg_start, seg_end, _ in segments
    ):
        return None
    return BACKGROUND


def featurize_window(
    xp: types.ModuleType, window: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """52-dim feature: per-joint mean position about the window's root, then
    per-joint mean speed (MODEL_CONTRACT.md §3.6.2).

    The features don't depend on where the sequence was centered, so the
    farm's per-segment center and the client's per-take center give the
    same features. A joint is present iff its confidence is > 0 (missing
    joints are zeros).
    - Root: the window's mean hip midpoint over frames with both hips
      present; 0 when no frame has them.
    - Position: per joint, the mean over the window's W frames of
      (position - root) where the joint is present, 0 where it is missing.
    - Speed: per joint, the mean over the W - 1 frame pairs of the absolute
      x and y change, counting only pairs where the joint is present in
      both frames. Absolute: signed means cancel out over full oscillation
      periods, while speed separates "moving" from "static", and the x/y
      split separates horizontal tricks from vertical ones.
    Output layout: [joint 0 x, joint 0 y, ..., joint 12 y] positions, then
    the same for speeds, as Training/export_coreml.py computes them.
    """
    pos = window[:, :, :2]  # [W, J, 2]
    present = (window[:, :, 2] > 0).astype(pos.dtype)  # [W, J]
    left, right = HIP_JOINTS
    hips = present[:, left] * present[:, right]  # [W]
    root_sum = (hips[:, None] * (pos[:, left] + pos[:, right]) / 2).sum(axis=0)
    root = root_sum / max(float(hips.sum()), 1.0)  # [2]; 0 without hips
    centered = pos.mean(axis=0) - root[None, :] * present.mean(axis=0)[:, None]
    pairs = present[1:] * present[:-1]  # [W-1, J]
    speed = (xp.abs(pos[1:] - pos[:-1]) * pairs[:, :, None]).mean(axis=0)
    return xp.concatenate([centered.ravel(), speed.ravel()])


def build_dataset(
    xp: types.ModuleType,
    samples: list[dict[str, Any]],
    window_frames: int,
    window_stride: int,
    margin_frames: int = 0,
) -> tuple[npt.NDArray[np.float64], list[str], list[tuple[str, int, str]]]:
    """Window every sample -> (features, labels, provenance).

    Provenance is (sample_id, window_start, label) per row, used to decode
    predicted windows back into trick segments. Windows in the background
    margin (§2.5) are excluded entirely — no row, so they never enter the
    background pool — rather than labeled background.
    """
    feats: list[npt.NDArray[np.float64]] = []
    labels: list[str] = []
    prov: list[tuple[str, int, str]] = []
    for s in samples:
        kps: npt.NDArray[np.float64] = s["keypoints"]
        n = kps.shape[0]
        segments = evaluate.ground_truth_segments(
            s.get("tricks", []), f"sample {s['sample_id']!r}"
        )
        for start in range(0, n - window_frames + 1, window_stride):
            end = start + window_frames
            label = label_for_window(segments, start, end, margin_frames)
            if label is None:
                continue
            feats.append(featurize_window(xp, kps[start:end]))
            labels.append(label)
            prov.append((s["sample_id"], start, label))
    if not feats:
        raise ValueError("no windows: samples shorter than window_frames?")
    return xp.stack(feats), labels, prov


def decode_segments(
    prov: list[tuple[str, int, str]], pred: list[str], window_frames: int
) -> list[dict[str, Any]]:
    """Merge consecutive same-label windows into trick segments.

    Output contract (master plan): source-frame coordinates, one segment with
    N names, is_combo derived.
    """
    # Group windows per sample, in start order.
    by_sample: dict[str, list[tuple[int, str]]] = {}
    for (sid, start, _), p in zip(prov, pred):
        by_sample.setdefault(sid, []).append((start, p))
    segments: list[dict[str, Any]] = []
    for sid in sorted(by_sample):
        wins = sorted(by_sample[sid])
        cur_label: str | None = None
        cur_start = 0
        cur_end = 0
        for start, p in wins + [(wins[-1][0] + 1, BACKGROUND)]:  # sentinel
            if p == cur_label and p != BACKGROUND:
                cur_end = start + window_frames
                continue
            if cur_label is not None and cur_label != BACKGROUND:
                segments.append(
                    {
                        "sample_id": sid,
                        "name": cur_label,
                        "start_frame": cur_start,
                        "end_frame": cur_end,
                        "is_combo": "+" in cur_label,
                    }
                )
            cur_label = p
            cur_start = start
            cur_end = start + window_frames
    return segments


# ---------------------------------------------------------------------------
# Class-balance sampling (TRAINING_DESIGN.md §2.7, §2.5)
# ---------------------------------------------------------------------------


def window_sampling_weights(
    labels: list[str], name_counts: dict[str, int], exponent: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Per-window class-balance weights.

    Each labeled window weighs ``count ** -exponent`` for its rarest name —
    a combo window takes the weight of its rarest name — so frequent names
    don't drown the rest while rare names aren't repeated into
    memorization. The counts are the run's train-split counts (the
    ``train_count`` published per name in ``name_metrics``). Background
    windows get weight 0 here: they are sampled separately with their own
    ratio (§2.5), not by the name weights. Returns (weights, is_background).
    """
    weights = np.empty(len(labels), dtype=np.float64)
    is_bg = np.zeros(len(labels), dtype=bool)
    for i, label in enumerate(labels):
        if label == BACKGROUND:
            is_bg[i] = True
            weights[i] = 0.0
            continue
        names = set(label.split("+"))
        missing = sorted(names - set(name_counts))
        if missing:
            raise ValueError(
                f"window label {label!r} names names missing from the "
                f"train-split counts: {missing}"
            )
        rarest = min(name_counts[name] for name in names)
        weights[i] = rarest ** (-exponent)
    return weights, is_bg


def background_windows_per_epoch(n_name_windows: int, background_ratio: float) -> int:
    """Background windows per epoch, so the epoch sample lands at
    ~``background_ratio`` background (TRAINING_DESIGN.md §2.5)."""
    if background_ratio <= 0:
        return 0
    return round(background_ratio * n_name_windows / (1 - background_ratio))


def make_epoch_sampler(
    name_idx: npt.NDArray[np.int64],
    name_weights: npt.NDArray[np.float64],
    bg_idx: npt.NDArray[np.int64],
    background_ratio: float,
) -> Callable[[np.random.Generator], npt.NDArray[np.int64]]:
    """One epoch's window sample, deterministic given the rng.

    Name windows are drawn with replacement ∝ the §2.7 class-balance
    weights (the whole name set is redrawn each epoch, so rare names are
    upsampled, never subsampled); background windows are drawn uniformly
    with replacement to their own count (§2.5), unchanged by the name
    weights. The two draws are shuffled together so the epoch isn't
    name-then-background.
    """
    if len(name_idx) == 0:
        raise ValueError("no labeled windows to sample — refusing to train")
    probs = name_weights / name_weights.sum()
    n_bg = background_windows_per_epoch(len(name_idx), background_ratio)
    if n_bg > 0 and len(bg_idx) == 0:
        raise ValueError(
            "background_ratio > 0 but no background windows in the train "
            "split — refusing to train"
        )

    def sample_epoch(rng: np.random.Generator) -> npt.NDArray[np.int64]:
        name_draws = rng.choice(name_idx, size=len(name_idx), replace=True, p=probs)
        if n_bg:
            bg_draws = rng.choice(bg_idx, size=n_bg, replace=True)
            idx = np.concatenate([name_draws, bg_draws])
        else:
            idx = np.asarray(name_draws)
        rng.shuffle(idx)
        return idx.astype(np.int64)

    return sample_epoch


# ---------------------------------------------------------------------------
# Model: softmax linear classifier, gradient descent on resampled epochs
# ---------------------------------------------------------------------------


def _softmax(
    xp: types.ModuleType, z: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    z = z - z.max(axis=1, keepdims=True)
    e = xp.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_classifier(
    xp: types.ModuleType,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    n_classes: int,
    learning_rate: float,
    epochs: int,
    l2: float,
    seed: int,
    init: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
    epoch_sampler: Callable[[np.random.Generator], npt.NDArray[np.int64]] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, float]:
    """Gradient descent on softmax cross-entropy + L2. Returns (W, b, l0, l1).

    init=(W, b) continues from a checkpoint (fine-tune the champion);
    otherwise weights are seeded from scratch. epoch_sampler, when given,
    redraws the training windows every epoch (TRAINING_DESIGN.md §2.7
    class-balance sampling); None trains on every window, every epoch.
    l0/l1 are reported on the full train set either way, so runs stay
    comparable across sampler settings.
    """
    n, d = X.shape
    rng = np.random.default_rng(seed)
    if init is None:
        W = rng.standard_normal((d, n_classes)) * 0.01
        b = np.zeros(n_classes)
    else:
        W, b = init[0].copy(), init[1].copy()
    W = xp.asarray(W)
    b = xp.asarray(b)

    def loss_and_grad(
        Xb: npt.NDArray[np.float64], yb: npt.NDArray[np.int64], nb: int
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        logits = Xb @ W + b
        probs = _softmax(xp, logits)
        nll = -xp.log(probs[xp.arange(nb), yb] + 1e-12).mean()
        reg = 0.5 * l2 * float(xp.sum(W * W))
        dlogits = probs
        dlogits[xp.arange(nb), yb] -= 1
        dlogits /= nb
        dW = Xb.T @ dlogits + l2 * W
        db = dlogits.sum(axis=0)
        return float(nll + reg), dW, db

    l0, _, _ = loss_and_grad(X, y, n)
    for _ in range(epochs):
        if epoch_sampler is not None:
            idx = epoch_sampler(rng)
            Xb, yb = X[idx], y[idx]
        else:
            Xb, yb = X, y
        _, dW, db = loss_and_grad(Xb, yb, Xb.shape[0])
        W -= learning_rate * dW
        b -= learning_rate * db
    l1, _, _ = loss_and_grad(X, y, n)
    return np.asarray(W), np.asarray(b), l0, l1


def predict_proba(
    xp: types.ModuleType,
    X: npt.NDArray[np.float64],
    W: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Per-window class probabilities, one row per window of X, as an xp
    array."""
    return _softmax(xp, X @ xp.asarray(W) + xp.asarray(b))


def predict(
    xp: types.ModuleType,
    X: npt.NDArray[np.float64],
    W: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    return np.asarray(xp.argmax(predict_proba(xp, X, W, b), axis=1))


def load_checkpoint(path: Path) -> dict[str, Any]:
    """The checkpoint's arrays. Any unreadable or corrupt file is a
    ValueError, never another exception type."""
    try:
        z = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as e:
        raise ValueError(f"cannot read checkpoint {path}: {e}") from e
    # A bare .npy loads as an array, not an archive.
    if not isinstance(z, np.lib.npyio.NpzFile):
        # ValueError, not TypeError: callers treat every bad file alike.
        raise ValueError(f"checkpoint {path} is not an .npz archive")  # noqa: TRY004
    need = {
        "W",
        "b",
        "classes",
        "feat_mean",
        "feat_std",
        "window_frames",
        "window_stride",
    }
    if not need <= set(z.files):
        raise ValueError(f"checkpoint {path} missing keys: {need - set(z.files)}")
    try:
        return {k: z[k] for k in z.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile, zlib.error) as e:
        # Members are only decompressed here: a corrupt one fails late.
        raise ValueError(f"cannot read checkpoint {path}: {e}") from e


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(config_path: Path, out_override: str | None) -> dict[str, Any]:
    cfg = load_config(config_path)
    xp = _select_backend(cfg["compute"])
    # Deterministic RNGs: the seed is part of the artifact. All randomness
    # flows through np.random.default_rng(seed) below (no stdlib `random`).
    repo_root = Path(__file__).resolve().parent.parent

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else repo_root / path

    split_path = resolve(cfg["split_path"])
    split = load_split(split_path)
    split_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
    train_samples, val_samples = load_samples(resolve(cfg["samples_dir"]), split)
    name_counts = train_name_counts(train_samples)

    W, S = cfg["window_frames"], cfg["window_stride"]
    margin_frames = round(float(cfg["background_margin_seconds"]) * POSE_FPS)
    Xtr_raw, ytr_labels, _ = build_dataset(xp, train_samples, W, S, margin_frames)
    Xva_raw, yva_labels, va_prov = build_dataset(xp, val_samples, W, S, margin_frames)

    classes = sorted(set(ytr_labels))
    cls_index = {c: i for i, c in enumerate(classes)}
    # A val-only label is a data bug: the model cannot predict what it never
    # saw in training. Fail closed instead of silently mis-scoring.
    unseen = sorted(set(yva_labels) - set(classes))
    if unseen:
        raise ValueError(f"val labels never seen in train: {unseen}")

    # Standardize with train stats (stored in the checkpoint for inference).
    feat_mean = np.asarray(Xtr_raw).mean(axis=0)
    feat_std = np.asarray(Xtr_raw).std(axis=0)
    feat_std[feat_std < 1e-9] = 1.0
    Xtr = (np.asarray(Xtr_raw) - feat_mean) / feat_std
    Xva = (np.asarray(Xva_raw) - feat_mean) / feat_std
    ytr = np.array([cls_index[c] for c in ytr_labels], dtype=np.int64)
    yva = np.array([cls_index[c] for c in yva_labels], dtype=np.int64)

    # Class-balance sampling (TRAINING_DESIGN.md §2.7): each epoch redraws
    # its windows — name windows ∝ count**-exponent (combo = rarest name),
    # background windows separately at background_ratio (§2.5).
    exponent = float(cfg["class_balance_exponent"])
    background_ratio = float(cfg["background_ratio"])
    tr_weights, tr_is_bg = window_sampling_weights(ytr_labels, name_counts, exponent)
    name_idx = np.flatnonzero(~tr_is_bg).astype(np.int64)
    bg_idx = np.flatnonzero(tr_is_bg).astype(np.int64)
    epoch_sampler = make_epoch_sampler(
        name_idx, tr_weights[~tr_is_bg], bg_idx, background_ratio
    )
    n_bg_per_epoch = background_windows_per_epoch(len(name_idx), background_ratio)

    init = None
    if cfg["init_checkpoint"]:
        ckpt = load_checkpoint(resolve(cfg["init_checkpoint"]))
        ckpt_classes = [str(c) for c in ckpt["classes"]]
        if ckpt_classes != classes:
            raise ValueError(
                f"checkpoint classes {ckpt_classes} != train classes {classes}"
            )
        if int(ckpt["window_frames"]) != W or int(ckpt["window_stride"]) != S:
            raise ValueError("checkpoint window params != config window params")
        init = (
            np.asarray(ckpt["W"], dtype=np.float64),
            np.asarray(ckpt["b"], dtype=np.float64),
        )
        # Fine-tuning reuses the checkpoint's feature stats so the weight
        # space stays comparable.
        feat_mean = np.asarray(ckpt["feat_mean"], dtype=np.float64)
        feat_std = np.asarray(ckpt["feat_std"], dtype=np.float64)
        Xtr = (np.asarray(Xtr_raw) - feat_mean) / feat_std
        Xva = (np.asarray(Xva_raw) - feat_mean) / feat_std

    Wm, bm, loss0, loss1 = train_classifier(
        xp,
        xp.asarray(Xtr),
        ytr,
        len(classes),
        float(cfg["learning_rate"]),
        cfg["epochs"],
        float(cfg["l2"]),
        cfg["seed"],
        init,
        epoch_sampler,
    )

    va_pred_idx = predict(xp, xp.asarray(Xva), Wm, bm)
    va_pred = [classes[i] for i in va_pred_idx]
    val_acc = float(np.mean(va_pred_idx == yva))
    per_class: dict[str, float] = {}
    for c in classes:
        mask = np.array(yva_labels) == c
        if mask.any():
            per_class[c] = float(np.mean(np.array(va_pred)[mask] == c))
    segments = decode_segments(va_prov, va_pred, W)

    out_dir = Path(out_override) if out_override else resolve(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    # config.toml: byte-exact snapshot of the effective config.
    shutil.copyfile(config_path, out_dir / "config.toml")
    config_sha = hashlib.sha256((out_dir / "config.toml").read_bytes()).hexdigest()
    np.savez(
        out_dir / "model.npz",
        W=Wm,
        b=bm,
        classes=np.array(classes),
        feat_mean=feat_mean,
        feat_std=feat_std,
        window_frames=np.int64(W),
        window_stride=np.int64(S),
        background_margin_frames=np.int64(margin_frames),
    )
    metrics = {
        "seed": cfg["seed"],
        "compute": cfg["compute"],
        "classes": classes,
        "n_train_windows": int(Xtr.shape[0]),
        "n_val_windows": int(Xva.shape[0]),
        "train_loss_initial": loss0,
        "train_loss_final": loss1,
        "val_accuracy": val_acc,
        "val_accuracy_per_class": per_class,
        "val_segments": segments,
        "config_sha256": config_sha,
        "init_checkpoint": cfg["init_checkpoint"] or None,
        "split_sha256": split_sha,
        "train_name_counts": name_counts,
        # The effective sampling scheme, for reproducibility: the per-name
        # counts it read (train_count, §6.1) and the per-name window weights
        # derived from them (count**-exponent, combo = rarest name).
        "sampling": {
            "class_balance_exponent": exponent,
            "background_ratio": background_ratio,
            "window_weight_per_name": {
                name: name_counts[name] ** (-exponent) for name in sorted(name_counts)
            },
            "n_name_windows_per_epoch": len(name_idx),
            "n_background_windows_per_epoch": n_bg_per_epoch,
        },
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(
        f"trained {Xtr.shape[0]} windows -> val acc {val_acc:.3f} "
        f"({Xva.shape[0]} windows), loss {loss0:.4f} -> {loss1:.4f}; "
        f"artifacts in {out_dir}"
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="TOML config file")
    parser.add_argument("--out", default=None, help="override the config's out_dir")
    args = parser.parse_args(argv)
    try:
        run(Path(args.config), args.out)
    except (ValueError, TypeError, RuntimeError) as e:
        print(f"train.py: error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
