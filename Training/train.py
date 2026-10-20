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
                per-name train counts and the split's sha256

Sample JSON format (one file per sample in samples_dir):
  {"sample_id": ..., "fps": 10,
   "keypoints": [[[x, y, score] * 17] * n_frames],  # MoveNet-17 order
   "tricks": [{"name": ..., "start_frame": ..., "end_frame": ...}]}
Multi-name segments label the window "a+b" (sorted); is_combo is derived.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import types
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11; CI pins 3.12
    sys.stderr.write("train.py needs Python 3.11+ (tomllib)\n")
    raise SystemExit(2)

import numpy as np
import numpy.typing as npt

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
N_KEYPOINTS = 17  # MoveNet-17 order, 10 Hz canonical (master plan, TKP1)


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


def _check_keypoints(kps: Any, sample_id: str) -> npt.NDArray[np.float64]:
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
        rec["keypoints"] = _check_keypoints(rec.get("keypoints"), sid)
        tricks = rec.get("tricks", [])
        if not isinstance(tricks, list):
            raise TypeError(f"sample {sid!r}: tricks must be a list")
        for t in tricks:
            if not {"name", "start_frame", "end_frame"} <= set(t):
                raise ValueError(f"sample {sid!r}: trick needs name/start/end: {t!r}")
            if not t["start_frame"] < t["end_frame"]:
                raise ValueError(f"sample {sid!r}: trick has empty span: {t!r}")
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


def label_for_window(tricks: list[dict[str, Any]], start: int, end: int) -> str:
    """Label for a window: trick name(s) covering >= half of it, else background.

    Center-frame labeling mislabels edge windows (the window is mostly outside
    the segment while its center is just inside, or vice versa); majority
    overlap keeps the label consistent with what the window's features show.
    A segment carrying N names labels the window "a+b" (sorted) — is_combo is
    derived from the "+" downstream, per the model output contract.
    """
    names = sorted(
        t["name"]
        for t in tricks
        if max(0, min(t["end_frame"], end) - max(t["start_frame"], start))
        >= (end - start) / 2
    )
    return "+".join(names) if names else BACKGROUND


def featurize_window(
    xp: types.ModuleType, window: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """68-dim feature: per-joint mean position + per-joint mean speed.

    Speed is the mean *absolute* velocity per axis: signed means cancel out
    over full oscillation periods (a trick window then looks static), while
    speed separates "moving" from "static" and the x/y split separates
    horizontal tricks (corkscrew) from vertical ones (gainer).
    """
    pos = window[:, :, :2]  # [W, 17, 2]
    vel = pos[1:] - pos[:-1]  # [W-1, 17, 2]
    speed = xp.abs(vel).mean(axis=0)  # [17, 2]: mean |vx|, |vy| per joint
    return xp.concatenate([pos.mean(axis=0).ravel(), speed.ravel()])


def build_dataset(
    xp: types.ModuleType,
    samples: list[dict[str, Any]],
    window_frames: int,
    window_stride: int,
) -> tuple[npt.NDArray[np.float64], list[str], list[tuple[str, int, str]]]:
    """Window every sample -> (features, labels, provenance).

    Provenance is (sample_id, window_start, label) per row, used to decode
    predicted windows back into trick segments.
    """
    feats: list[npt.NDArray[np.float64]] = []
    labels: list[str] = []
    prov: list[tuple[str, int, str]] = []
    for s in samples:
        kps: npt.NDArray[np.float64] = s["keypoints"]
        n = kps.shape[0]
        for start in range(0, n - window_frames + 1, window_stride):
            end = start + window_frames
            label = label_for_window(s.get("tricks", []), start, end)
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
# Model: softmax linear classifier, full-batch gradient descent
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
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, float]:
    """Full-batch GD on softmax cross-entropy + L2. Returns (W, b, l0, l1).

    init=(W, b) continues from a checkpoint (fine-tune the champion);
    otherwise weights are seeded from scratch.
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

    def loss_and_grad() -> tuple[
        float, npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        logits = X @ W + b
        probs = _softmax(xp, logits)
        nll = -xp.log(probs[xp.arange(n), y] + 1e-12).mean()
        reg = 0.5 * l2 * float(xp.sum(W * W))
        dlogits = probs
        dlogits[xp.arange(n), y] -= 1
        dlogits /= n
        dW = X.T @ dlogits + l2 * W
        db = dlogits.sum(axis=0)
        return float(nll + reg), dW, db

    l0, _, _ = loss_and_grad()
    for _ in range(epochs):
        _, dW, db = loss_and_grad()
        W -= learning_rate * dW
        b -= learning_rate * db
    l1, _, _ = loss_and_grad()
    return np.asarray(W), np.asarray(b), l0, l1


def predict(
    xp: types.ModuleType,
    X: npt.NDArray[np.float64],
    W: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    return np.asarray(
        xp.argmax(_softmax(xp, X @ xp.asarray(W) + xp.asarray(b)), axis=1)
    )


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        z = np.load(path, allow_pickle=False)
    except OSError as e:
        raise ValueError(f"cannot read checkpoint {path}: {e}") from e
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
    return {k: z[k] for k in z.files}


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
    Xtr_raw, ytr_labels, _ = build_dataset(xp, train_samples, W, S)
    Xva_raw, yva_labels, va_prov = build_dataset(xp, val_samples, W, S)

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
