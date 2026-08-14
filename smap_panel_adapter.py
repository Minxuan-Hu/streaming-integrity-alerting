#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""smap_panel_adapter.py

Dataset adapter for NASA SMAP/MSL in Telemanom format.

Creates a deterministic panel CSV for the operational evaluation framework
runner (`run_contract_eval.py`):
- downsample train/test separately (boundary preserved)
- attacked-slice columns contain "|score|"; control-slice columns contain "|all|all|"

Run via `make_smap_panel.py`.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


def _load_npy_from_zip(z: zipfile.ZipFile, member: str) -> np.ndarray:
    with z.open(member) as f:
        b = f.read()
    return np.load(io.BytesIO(b), allow_pickle=False)


def list_ids(smap_zip: Path) -> list[str]:
    with zipfile.ZipFile(smap_zip, "r") as z:
        ids = sorted({Path(m).stem for m in z.namelist() if m.startswith("data/train/") and m.endswith(".npy")})
    return ids


def load_series(smap_zip: Path, series_id: str) -> tuple[np.ndarray, np.ndarray]:
    train_member = f"data/train/{series_id}.npy"
    test_member = f"data/test/{series_id}.npy"
    with zipfile.ZipFile(smap_zip, "r") as z:
        names = set(z.namelist())
        if train_member not in names:
            raise FileNotFoundError(f"Missing {train_member} in {smap_zip}")
        if test_member not in names:
            raise FileNotFoundError(f"Missing {test_member} in {smap_zip}")
        Xtr = _load_npy_from_zip(z, train_member)
        Xte = _load_npy_from_zip(z, test_member)

    if Xtr.ndim != 2 or Xte.ndim != 2:
        raise ValueError(f"Expected 2D arrays; got train {Xtr.shape}, test {Xte.shape}")
    if Xtr.shape[1] != Xte.shape[1]:
        raise ValueError(f"Train/test dim mismatch: {Xtr.shape} vs {Xte.shape}")
    if Xtr.shape[1] < 2:
        raise ValueError(f"Need at least D>=2 dims to form score/control slices; got D={Xtr.shape[1]}")
    return Xtr.astype(float), Xte.astype(float)


def _downsample(X: np.ndarray, k: int, mode: str) -> tuple[np.ndarray, dict]:
    """Downsample along time axis.

    Returns (X_ds, info). For mode="mean", we drop the tail remainder so
    all blocks are full length k (deterministic).
    """
    k = int(k)
    if k <= 1:
        return X, {"downsample_k": 1, "downsample_mode": "none", "dropped_tail": 0}

    mode = str(mode).lower().strip()
    if mode not in {"mean", "subsample"}:
        raise ValueError("--downsample_mode must be one of: mean, subsample")

    if mode == "subsample":
        return X[::k].copy(), {"downsample_k": k, "downsample_mode": "subsample", "dropped_tail": 0}

    T, D = X.shape
    T_full = (T // k) * k
    dropped = T - T_full
    if T_full <= 0:
        raise ValueError(f"Series too short for downsample_k={k} with mode=mean (T={T})")
    X0 = X[:T_full]
    Xr = X0.reshape(T_full // k, k, D).mean(axis=1)
    return Xr.copy(), {"downsample_k": k, "downsample_mode": "mean", "dropped_tail": int(dropped)}


def choose_score_dims(
    Xtr: np.ndarray,
    n_score_cols: int,
    min_std_eps: float,
    rule: str = "first_nonconstant",
) -> list[int]:
    """Choose attacked-surface dims deterministically.

    rule:
      - first_nonconstant (default): first K indices (in order) with std > eps (filter-only; non-optimizing)
      - top_std: top-K by std (auxiliary option; may change attacked subset)

    Always keeps at least one control dim (K <= D-1).
    """
    rule = str(rule).lower().strip()
    if rule not in {"first_nonconstant", "top_std"}:
        raise ValueError("--score_dim_rule must be one of: first_nonconstant, top_std")

    D = int(Xtr.shape[1])
    k_req = int(n_score_cols)
    if k_req < 1:
        raise ValueError("--n_score_cols must be >= 1")
    k_cap = min(k_req, D - 1)

    std = np.std(Xtr, axis=0)
    eps = float(min_std_eps)
    ok = [int(j) for j in range(D) if float(std[j]) > eps]
    if len(ok) == 0:
        ok = list(range(D))

    if rule == "first_nonconstant":
        chosen = ok[: min(k_cap, len(ok))]
        if len(chosen) == 0:
            chosen = [0]
        return [int(i) for i in chosen]

    ok_arr = np.array(ok, dtype=int)
    order = ok_arr[np.argsort(std[ok_arr])[::-1]]
    chosen = order[: min(k_cap, len(order))]
    if chosen.size == 0:
        chosen = np.array([0], dtype=int)
    return chosen.astype(int).tolist()


def build_panel(
    Xtr: np.ndarray,
    Xte: np.ndarray,
    series_id: str,
    score_dims: list[int],
    start_date: str,
    step_days: int,
    include_all_controls: bool,
) -> pd.DataFrame:
    X = np.concatenate([Xtr, Xte], axis=0)
    T, D = X.shape

    step_days = int(step_days)
    if step_days < 1:
        raise ValueError("step_days must be >= 1")

    dt = pd.date_range(start=pd.Timestamp(start_date), periods=T, freq=f"{step_days}D")
    df = pd.DataFrame({"date": dt})

    score_set = set(int(i) for i in score_dims)
    if len(score_set) == 0:
        raise ValueError("score_dims is empty")
    if len(score_set) >= D:
        raise ValueError("score_dims covers all dims; need at least one control dim")
    if not all(0 <= i < D for i in score_set):
        raise ValueError(f"score_dims out of range for D={D}: {sorted(score_set)[:10]}")

    for j in range(D):
        if j in score_set:
            col = f"telemetry|{series_id}|score|tier0|raw|value|c{j:02d}"
        else:
            if include_all_controls:
                col = f"telemetry|{series_id}|all|all|raw|value|c{j:02d}"
            else:
                col = f"telemetry|{series_id}|other|other|raw|value|c{j:02d}"
        df[col] = X[:, j]

    return df


def recommend_split_from_lengths(start_date: str, step_days: int, train_len: int) -> dict:
    train_len = int(train_len)
    if train_len < 2:
        raise ValueError("train_len too small")
    start = pd.Timestamp(start_date)
    train_end = start + pd.Timedelta(days=(train_len - 1) * int(step_days))
    earliest = start + pd.Timedelta(days=train_len * int(step_days))
    return {
        "split_mode": "by_provided_train_len",
        "train_len": int(train_len),
        "train_end": str(train_end.date()),
        "earliest": str(earliest.date()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smap_zip", type=str, required=True, help="Path to SMAP.zip (Telemanom-format)")
    ap.add_argument("--series_id", type=str, default="A-1", help="ID (e.g., A-1, D-15)")
    ap.add_argument("--out_dir", type=str, default="smap_refined", help="Output directory")

    ap.add_argument("--n_score_cols", type=int, default=10, help="How many dims to treat as attacked surface")
    ap.add_argument("--min_std_eps", type=float, default=1e-8, help="Exclude near-constant dims (train std <= eps)")
    ap.add_argument(
        "--score_dim_rule",
        type=str,
        default="first_nonconstant",
        help="Attacked dim selection rule: first_nonconstant (default) or top_std.",
    )
    ap.add_argument("--include_all_controls", type=int, default=1, help="If 1, put all non-score dims into |all|all|")

    ap.add_argument("--start_date", type=str, default="2000-01-01", help="Synthetic start date (YYYY-MM-DD)")
    ap.add_argument("--downsample_k", type=int, default=1, help="Downsample factor k>=1. Use 7 for weekly-ish.")
    ap.add_argument("--downsample_mode", type=str, default="mean", help="mean (recommended) or subsample")
    ap.add_argument("--steps_per_year", type=int, default=0, help="If 0, recommend round(365/k).")

    ap.add_argument("--list_ids", type=int, default=0, help="If 1, list available IDs and exit")
    args = ap.parse_args()

    smap_zip = Path(args.smap_zip)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if int(args.list_ids) == 1:
        ids = list_ids(smap_zip)
        print("Found", len(ids), "IDs. First 25:")
        print(ids[:25])
        with zipfile.ZipFile(smap_zip, "r") as z:
            for sid in ids[:10]:
                Xtr = _load_npy_from_zip(z, f"data/train/{sid}.npy")
                Xte = _load_npy_from_zip(z, f"data/test/{sid}.npy")
                print(f"{sid:>5}: train{tuple(Xtr.shape)} test{tuple(Xte.shape)}")
        return

    Xtr_raw, Xte_raw = load_series(smap_zip, args.series_id)

    k = int(args.downsample_k)
    if k < 1:
        raise ValueError("--downsample_k must be >= 1")

    # Downsample train/test separately to preserve the true boundary.
    Xtr, info_tr = _downsample(Xtr_raw, k, args.downsample_mode)
    Xte, info_te = _downsample(Xte_raw, k, args.downsample_mode)

    score_dims = choose_score_dims(
        Xtr=Xtr,
        n_score_cols=int(args.n_score_cols),
        min_std_eps=float(args.min_std_eps),
        rule=str(args.score_dim_rule),
    )

    df = build_panel(
        Xtr=Xtr,
        Xte=Xte,
        series_id=str(args.series_id),
        score_dims=score_dims,
        start_date=str(args.start_date),
        step_days=k,
        include_all_controls=bool(int(args.include_all_controls)),
    )

    feat = df.drop(columns=["date"])
    if feat.isna().any().any():
        raise ValueError("Panel has NaNs; unexpected for this dataset.")
    if not np.isfinite(feat.to_numpy()).all():
        raise ValueError("Panel has non-finite values; unexpected for this dataset.")

    split = recommend_split_from_lengths(str(args.start_date), k, train_len=int(Xtr.shape[0]))

    if int(args.steps_per_year) > 0:
        spy = int(args.steps_per_year)
    else:
        spy = int(round(365.0 / float(k)))

    D_raw = int(Xtr_raw.shape[1])
    score_set = set(int(i) for i in score_dims)
    control_dims = [int(i) for i in range(D_raw) if int(i) not in score_set]

    meta = {
        "series_id": str(args.series_id),
        "raw_train_shape": [int(Xtr_raw.shape[0]), int(Xtr_raw.shape[1])],
        "raw_test_shape": [int(Xte_raw.shape[0]), int(Xte_raw.shape[1])],
        "downsample": {
            "k": int(k),
            "mode": str(args.downsample_mode).lower().strip(),
            "train_info": info_tr,
            "test_info": info_te,
        },
        "effective_step_days": int(k),
        "start_date": str(args.start_date),
        "n_score_cols_requested": int(args.n_score_cols),
        "min_std_eps": float(args.min_std_eps),
        "score_dim_rule": str(args.score_dim_rule).lower().strip(),
        "n_score_cols_effective": int(len(score_dims)),
        "score_dim_indices": [int(i) for i in score_dims],
        "control_dim_indices": [int(i) for i in control_dims],
        "split_recommendation": split,
        "runner_recommendation": {
            "time_mode": "step",
            "steps_per_year": int(spy),
            "date_col": "date",
        },
    }

    tag = "daily" if k == 1 else f"k{k}_{meta['downsample']['mode']}"
    out_csv = out_dir / f"smap_{args.series_id}_panel_{tag}.csv"
    out_meta = out_dir / f"smap_{args.series_id}_meta_{tag}.json"
    df.to_csv(out_csv, index=False)
    out_meta.write_text(json.dumps(meta, indent=2))

    print("Wrote:")
    print(" ", out_csv)
    print(" ", out_meta)
    print("Split:", split)
    print("Runner recommendation:", meta["runner_recommendation"])
    print("Score dims (effective):", score_dims)


if __name__ == "__main__":
    main()
