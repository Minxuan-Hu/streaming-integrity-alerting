
"""
Robust standardization utilities.

We standardize per-feature using median and MAD computed on a training window.
This avoids leaking future information and is stable for heavy-tailed series.

Notation:
  z = (x - median) / (1.4826 * MAD + eps)

If MAD is zero, we fall back to std (or add eps).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def robust_fit(X: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0)
    scale = 1.4826 * mad
    scale = np.where(scale < eps, eps, scale)
    return med, scale

def robust_transform(X: np.ndarray, med: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (X - med) / scale

def robust_standardize_df(df: pd.DataFrame, train_mask: np.ndarray) -> tuple[pd.DataFrame, dict]:
    X = df.values.astype(float)
    med, scale = robust_fit(X[train_mask])
    Z = robust_transform(X, med, scale)
    meta = {"median": med.tolist(), "scale": scale.tolist(), "columns": df.columns.tolist()}
    return pd.DataFrame(Z, index=df.index, columns=df.columns), meta
