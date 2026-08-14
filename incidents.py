"""Incident and drift operators for synthetic corruption.

Given a standardized stream z_t and an incident window [t0, t1], operators construct a corrupted
replay \tilde z_t that:
  - matches the clean stream outside the window, and
  - differs from the clean stream only on the attacked-slice coordinates inside the window.

The paper's canonical incident families are:
  - freeze  : suppression / graded flatlining,
  - corr_mix: correlation mixing via an orthonormal transform.

Additional operators are retained for auxiliary experiments and are not used in the paper's
canonical runs.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Incident:
    kind: str
    start: pd.Timestamp
    end: pd.Timestamp
    target_cols: list[int]
    severity: float | None = None
    extra: dict | None = None


def _window_mask(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    return (index >= start) & (index <= end)


def inject_level_shift(df_z: pd.DataFrame, inc: Incident) -> tuple[pd.DataFrame, dict]:
    X = df_z.values.copy()
    m = _window_mask(df_z.index, inc.start, inc.end)
    X[np.ix_(m, inc.target_cols)] += float(inc.severity)
    return pd.DataFrame(X, index=df_z.index, columns=df_z.columns), {"kind": "level_shift", **inc.__dict__}


def inject_drift(df_z: pd.DataFrame, inc: Incident) -> tuple[pd.DataFrame, dict]:
    X = df_z.values.copy()
    m = _window_mask(df_z.index, inc.start, inc.end)
    t = np.linspace(0.0, 1.0, m.sum())
    sev = inc.severity
    if sev is None:
        sev = (inc.extra or {}).get('strength', None)
    if sev is None:
        raise ValueError("drift incident missing severity/strength; set inc.severity or inc.extra['strength']")
    ramp = float(sev) * t
    tgt = inc.target_cols
    if tgt is None:
        tgt_idx = list(range(X.shape[1]))
    else:
        tgt_list = list(tgt)
        if len(tgt_list) == 0:
            tgt_idx = []
        elif isinstance(tgt_list[0], str):
            tgt_idx = [df_z.columns.get_loc(c) for c in tgt_list]
        else:
            tgt_idx = [int(c) for c in tgt_list]
    if len(tgt_idx) > 0 and m.sum() > 0:
        X[np.ix_(m, tgt_idx)] += ramp[:, None]
    return pd.DataFrame(X, index=df_z.index, columns=df_z.columns), {"kind": "drift", **inc.__dict__}


def apply_variance_inflation(
    df_z: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    target_cols: list[int],
    delta: float,
    mu_mode: str = "zero",
    mu_win: int = 24,
) -> tuple[pd.DataFrame, dict]:
    """Auxiliary drift operator: variance inflation (not used in the paper's canonical runs).

    Implements: x_drift = mu + (1+delta) * (x - mu)

    Parameters
    - delta in [0,1] is an inflation factor (e.g., 0.2 => +20% deviations).
    - mu_mode:
        * 'zero'   : mu_t = 0 (appropriate in robust z-units; fast and stable)
        * 'rolling': mu_t = rolling mean with window mu_win
        * 'block'  : mu_t = mean over the drift block
    """
    X = df_z.values.copy()
    m = _window_mask(df_z.index, start, end)
    cols = list(int(c) for c in (target_cols or []))
    if len(cols) == 0 or m.sum() == 0:
        return df_z.copy(), {
            "kind": "variance_inflation",
            "start": start,
            "end": end,
            "target_cols": cols,
            "delta": float(delta),
            "mu_mode": str(mu_mode),
            "mu_win": int(mu_win),
        }

    d = float(delta)
    d = float(np.clip(d, 0.0, 1.0))
    a = 1.0 + d

    if str(mu_mode) == "rolling":
        # rolling mean per column (computed over full series for stability)
        mu_df = pd.DataFrame(X[:, cols], index=df_z.index).rolling(int(mu_win), min_periods=1).mean()
        mu = mu_df.values
        Xw = X[np.ix_(m, cols)]
        muw = mu[m, :]
        X[np.ix_(m, cols)] = muw + a * (Xw - muw)
    elif str(mu_mode) == "block":
        mu0 = np.nanmean(X[np.ix_(m, cols)], axis=0)
        mu0 = np.nan_to_num(mu0, nan=0.0)
        Xw = X[np.ix_(m, cols)]
        X[np.ix_(m, cols)] = mu0[None, :] + a * (Xw - mu0[None, :])
    else:
        # zero reference in z-units
        X[np.ix_(m, cols)] = a * X[np.ix_(m, cols)]

    return (
        pd.DataFrame(X, index=df_z.index, columns=df_z.columns),
        {
            "kind": "variance_inflation",
            "start": start,
            "end": end,
            "target_cols": cols,
            "delta": float(d),
            "mu_mode": str(mu_mode),
            "mu_win": int(mu_win),
        },
    )


def apply_benign_drift(
    df_z: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    target_cols: list[int],
    delta: float,
    mode: str = "variance_inflation",
    mu_mode: str = "zero",
    mu_win: int = 24,
) -> tuple[pd.DataFrame, dict]:
    """Convenience wrapper for benign drift slice perturbations."""
    mode = str(mode)
    if mode == "ramp":
        inc = Incident(kind="drift", start=start, end=end, target_cols=target_cols, severity=float(delta), extra={"strength": float(delta)})
        df2, info = inject_drift(df_z, inc)
        info.update({"mode": "ramp", "delta": float(delta)})
        return df2, info
    # default
    df2, info = apply_variance_inflation(df_z, start=start, end=end, target_cols=target_cols, delta=float(delta), mu_mode=mu_mode, mu_win=mu_win)
    info.update({"mode": "variance_inflation"})
    return df2, info



def _robust_sigma_ref(ref_values: np.ndarray | None, eps: float = 1e-6) -> np.ndarray:
    """Robust per-channel scale estimate.

    Uses MAD (scaled) when possible; falls back to std; then to 1.0.
    """
    if ref_values is None:
        return np.asarray([1.0], dtype=float)
    X = np.asarray(ref_values, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.size == 0:
        return np.ones((X.shape[1] if X.ndim == 2 else 1,), dtype=float)
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med[None, :]), axis=0)
    sigma = 1.4826 * mad
    # fallback to std where MAD is tiny / undefined
    std = np.nanstd(X, axis=0)
    sigma = np.where(np.isfinite(sigma) & (sigma > eps), sigma, std)
    sigma = np.where(np.isfinite(sigma) & (sigma > eps), sigma, 1.0)
    return sigma.astype(float)


def apply_noise_injection(
    df_z: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    target_cols: list[int],
    delta: float,
    rng: np.random.Generator,
    mode: str = "ar1",
    rho: float = 0.90,
    ref_values: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Benign drift via measurement-noise injection on selected channels.

    Intended as a drift-only compliance stress test:
      x_drift = x + delta * sigma * eps

    - Operates in robust z-units, but sigma is computed per channel from `ref_values` (typically training period).
    - mode:
        * 'iid' : eps_t ~ N(0,1) i.i.d.
        * 'ar1' : eps_t = rho*eps_{t-1} + sqrt(1-rho^2)*z_t (low-frequency noise)
    """
    X = df_z.values.copy()
    m = _window_mask(df_z.index, start, end)
    cols = [int(c) for c in (target_cols or [])]
    if len(cols) == 0 or m.sum() == 0:
        return df_z.copy(), {"kind": "noise_injection", "mode": str(mode), "start": start, "end": end, "target_cols": cols, "delta": float(delta), "rho": float(rho)}

    d = float(delta)
    d = float(np.clip(d, 0.0, 10.0))  # allow >1 for stronger stress if desired
    sig = _robust_sigma_ref(ref_values, eps=1e-6)
    if sig.ndim != 1 or sig.shape[0] != len(cols):
        # if mis-sized, fall back to ones
        sig = np.ones((len(cols),), dtype=float)

    T = int(m.sum())
    C = int(len(cols))
    mode = str(mode)
    if mode == "iid":
        eps = rng.normal(0.0, 1.0, size=(T, C))
    else:
        r = float(np.clip(float(rho), 0.0, 0.999))
        z = rng.normal(0.0, 1.0, size=(T, C))
        eps = np.zeros((T, C), dtype=float)
        if T > 0:
            eps[0, :] = z[0, :]
            a = float(np.sqrt(max(1.0 - r * r, 0.0)))
            for t in range(1, T):
                eps[t, :] = r * eps[t - 1, :] + a * z[t, :]

    Xw = X[np.ix_(m, cols)]
    X[np.ix_(m, cols)] = Xw + (d * sig[None, :] * eps)
    return (
        pd.DataFrame(X, index=df_z.index, columns=df_z.columns),
        {"kind": "noise_injection", "mode": ("noise_ar1" if mode != "iid" else "noise_iid"), "start": start, "end": end, "target_cols": cols, "delta": float(d), "rho": float(rho)},
    )


def inject_variance_jump(df_z: pd.DataFrame, inc: Incident, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    X = df_z.values.copy()
    m = _window_mask(df_z.index, inc.start, inc.end)
    noise = rng.normal(0.0, float(inc.severity), size=(m.sum(), len(inc.target_cols)))
    X[np.ix_(m, inc.target_cols)] += noise
    return pd.DataFrame(X, index=df_z.index, columns=df_z.columns), {"kind": "variance_jump", **inc.__dict__}


def inject_dropout(df_z: pd.DataFrame, inc: Incident) -> tuple[pd.DataFrame, dict]:
    X = df_z.values.copy()
    m = _window_mask(df_z.index, inc.start, inc.end)
    X[np.ix_(m, inc.target_cols)] = np.nan
    return pd.DataFrame(X, index=df_z.index, columns=df_z.columns), {"kind": "dropout", **inc.__dict__}


def inject_outlier_burst(
    df_z: pd.DataFrame,
    inc: Incident,
    rng: np.random.Generator,
    p: float = 0.25,
) -> tuple[pd.DataFrame, dict]:
    X = df_z.values.copy()
    m = _window_mask(df_z.index, inc.start, inc.end)
    idx = np.where(m)[0]
    if len(idx) == 0:
        return df_z.copy(), {"kind": "outlier_burst", **inc.__dict__}
    sev = float(inc.severity)
    for c in inc.target_cols:
        burst = rng.random(len(idx)) < p
        if burst.sum() == 0:
            continue
        spikes = rng.normal(loc=0.0, scale=sev, size=int(burst.sum()))
        X[idx[burst], c] += spikes
    return (
        pd.DataFrame(X, index=df_z.index, columns=df_z.columns),
        {"kind": "outlier_burst", **inc.__dict__, "p": float(p)},
    )


def inject_freeze(df_z: pd.DataFrame, inc: Incident) -> tuple[pd.DataFrame, dict]:
    X = df_z.values.copy()
    m = _window_mask(df_z.index, inc.start, inc.end)
    start_idx = np.where(df_z.index == inc.start)[0]
    if len(start_idx) == 0:
        return df_z.copy(), {"kind": "freeze", **inc.__dict__}
    t0 = int(start_idx[0])
    ref_idx = max(t0 - 1, 0)
    ref_vals = X[ref_idx, inc.target_cols]

    # Allow a continuous freeze strength in [0,1] for intensity sweeps.
    # a=1.0 is a full freeze (flatline). a=0.0 leaves the stream unchanged.
    extra = inc.extra or {}
    a = extra.get("strength", None)
    if a is None:
        a = inc.severity
    a = 1.0 if a is None else float(a)
    a = float(np.clip(a, 0.0, 1.0))

    if m.sum() > 0 and len(inc.target_cols) > 0:
        # Blend original values with a fixed reference segment to create a graded staleness incident.
        Xw = X[np.ix_(m, inc.target_cols)].copy()
        X[np.ix_(m, inc.target_cols)] = (1.0 - a) * Xw + a * ref_vals[None, :]

    return (
        pd.DataFrame(X, index=df_z.index, columns=df_z.columns),
        {"kind": "freeze", **inc.__dict__, "strength": float(a)},
    )


def inject_impute_constant(df_z: pd.DataFrame, inc: Incident, constant: float = 0.0) -> tuple[pd.DataFrame, dict]:
    X = df_z.values.copy()
    m = _window_mask(df_z.index, inc.start, inc.end)
    X[np.ix_(m, inc.target_cols)] = float(constant)
    return pd.DataFrame(X, index=df_z.index, columns=df_z.columns), {"kind": "impute_constant", **inc.__dict__, "constant": float(constant)}


def inject_swap(df_z: pd.DataFrame, inc: Incident, col_a: int, col_b: int) -> tuple[pd.DataFrame, dict]:
    X = df_z.values.copy()
    m = _window_mask(df_z.index, inc.start, inc.end)
    Xa = X[m, col_a].copy()
    X[m, col_a] = X[m, col_b]
    X[m, col_b] = Xa
    return pd.DataFrame(X, index=df_z.index, columns=df_z.columns), {"kind": "swap", **inc.__dict__, "swap_cols": [int(col_a), int(col_b)]}


def inject_corr_mix(
    df_z: pd.DataFrame,
    inc: Incident,
    rng: np.random.Generator,
    strength: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """Correlation-mixing incident on the selected columns.

    Goal: create a *second-order* shift (cross-series correlation / covariance) without an explicit
    mean shift.

    Supported modes (controlled by inc.extra['mode']):
      - 'pca_rotate'  : rotate within a provided PCA subspace basis U (k x r)
      - 'pair_rotate' : rotate a selected coordinate pair (i,j) by a fixed angle theta
      - (fallback)    : coordinate-space random orthonormal mix if U is not provided

    Mixing amplitude is controlled by a in [0,1]:
        v' = (1-a) v + a T(v)
    where T is the chosen transform (subspace rotation / pair rotation / coord mix).

    Notes:
      - We operate on robust-standardized signals (z-units).
      - If 'mu' is provided in inc.extra, we apply transforms to centered values (v-mu)
        and then add mu back.
    """
    X = df_z.values.copy()
    m = _window_mask(df_z.index, inc.start, inc.end)
    idx = np.where(m)[0]
    cols = inc.target_cols
    k = len(cols)

    extra = inc.extra or {}
    mode = str(extra.get('mode', 'pca_rotate'))
    a = float(np.clip(float(extra.get('strength', strength)), 0.0, 1.0))

    if len(idx) == 0 or k <= 1:
        return df_z.copy(), {"kind": "corr_mix", **inc.__dict__, "strength": float(a), "mode": mode}

    mu = extra.get('mu', None)
    if mu is not None:
        mu = np.asarray(mu, dtype=float)
        if mu.shape != (k,):
            raise ValueError(f"corr_mix: provided mu has shape {mu.shape}, expected (k,)={(k,)}")

    # ---- Mode B: coordinate pair rotation (localized second-order shift) ----
    if mode == 'pair_rotate':
        pair = extra.get('pair', None)
        if pair is None:
            i, j = rng.choice(k, size=2, replace=False)
            i, j = int(i), int(j)
        else:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                p0, p1 = int(pair[0]), int(pair[1])
            else:
                raise ValueError("corr_mix(pair_rotate): 'pair' must be a length-2 tuple/list")

            # Interpret pair as (a) positions within target_cols (0..k-1), or
            # (b) global column indices contained in target_cols.
            if 0 <= p0 < k and 0 <= p1 < k:
                i, j = p0, p1
            else:
                pos = {int(c): ii for ii, c in enumerate(cols)}
                if p0 in pos and p1 in pos:
                    i, j = int(pos[p0]), int(pos[p1])
                else:
                    raise ValueError(
                        f"corr_mix(pair_rotate): pair={pair} not interpretable as positions in [0,{k-1}] "
                        f"or global indices inside target_cols"
                    )

        theta = float(extra.get('theta', 1.57079632679))  # default pi/2
        c = float(np.cos(theta))
        s = float(np.sin(theta))

        for t in idx:
            v = np.nan_to_num(X[t, cols], nan=0.0)
            vc = (v - mu) if (mu is not None) else v
            x, y = float(vc[i]), float(vc[j])
            xr = c * x - s * y
            yr = s * x + c * y
            vc2 = vc.copy()
            vc2[i] = (1.0 - a) * x + a * xr
            vc2[j] = (1.0 - a) * y + a * yr
            v2 = (vc2 + mu) if (mu is not None) else vc2
            X[t, cols] = v2

        info = {
            "kind": "corr_mix",
            **inc.__dict__,
            "strength": float(a),
            "mode": "pair_rotate",
            "pair_pos": [int(i), int(j)],
            "theta": float(theta),
        }
        return pd.DataFrame(X, index=df_z.index, columns=df_z.columns), info

    # ---- Mode A: PCA-subspace rotation (preferred when U is supplied) ----
    U = extra.get('U', None)
    if mode == 'pca_rotate' and U is not None:
        U = np.asarray(U, dtype=float)
        if U.ndim != 2 or U.shape[0] != k:
            raise ValueError(f"corr_mix(pca_rotate): provided U has shape {U.shape}, expected (k,r)=({k},r)")
        r = U.shape[1]
        # Allow caller to provide a fixed rotation matrix for paired intensity sweeps
        # (common random numbers across strengths). If not provided, sample one.
        R = extra.get('R', None)
        if R is None:
            A = rng.normal(size=(r, r))
            Q, _ = np.linalg.qr(A)
            R = Q
        else:
            R = np.asarray(R, dtype=float)
            if R.shape != (r, r):
                raise ValueError(f"corr_mix(pca_rotate): provided R has shape {R.shape}, expected ({r},{r})")
        for t in idx:
            v = np.nan_to_num(X[t, cols], nan=0.0)
            vc = (v - mu) if (mu is not None) else v
            svec = U.T @ vc
            s2 = (1.0 - a) * svec + a * (R @ svec)
            vc_sub = U @ s2
            vc_res = vc - U @ (U.T @ vc)
            vc2 = vc_res + vc_sub
            v2 = (vc2 + mu) if (mu is not None) else vc2
            X[t, cols] = v2
        info = {"kind": "corr_mix", **inc.__dict__, "strength": float(a), "mode": "pca_rotate", "r": int(r)}
        return pd.DataFrame(X, index=df_z.index, columns=df_z.columns), info

    # ---- Fallback: coordinate-space orthonormal mix (global within target set) ----
    # Allow caller to provide a fixed coordinate-space orthonormal mix (paired sweeps)
    Q = extra.get('Q', None)
    if Q is None:
        A = rng.normal(size=(k, k))
        Q, _ = np.linalg.qr(A)
    else:
        Q = np.asarray(Q, dtype=float)
        if Q.shape != (k, k):
            raise ValueError(f"corr_mix(coord_mix): provided Q has shape {Q.shape}, expected ({k},{k})")
    for t in idx:
        v = np.nan_to_num(X[t, cols], nan=0.0)
        vc = (v - mu) if (mu is not None) else v
        vc2 = (1.0 - a) * vc + a * (Q @ vc)
        v2 = (vc2 + mu) if (mu is not None) else vc2
        X[t, cols] = v2

    return (
        pd.DataFrame(X, index=df_z.index, columns=df_z.columns),
        {"kind": "corr_mix", **inc.__dict__, "strength": float(a), "mode": "coord_mix", "mix_dim": int(k)},
    )


def inject_subspace_rotate(
    df_z: pd.DataFrame,
    inc: Incident,
    rng: np.random.Generator,
    U: np.ndarray,
    strength: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """Stealthy in-subspace covariance rotation.

    For each t in the incident window, we rotate the PCA score vector within the baseline subspace U:
        x = df_z[t]
        s = U^T x
        s' = R s, where R is a random orthogonal matrix (fixed across window)
        x' = x_residual + U s'

    This keeps the perturbation largely within the baseline subspace, so residual-energy detectors alone may miss it.
    """
    X = df_z.values.copy()
    m = _window_mask(df_z.index, inc.start, inc.end)
    idx = np.where(m)[0]
    if len(idx) == 0:
        return df_z.copy(), {"kind": "subspace_rotate", **inc.__dict__}

    r = U.shape[1]
    A = rng.normal(size=(r, r))
    Q, _ = np.linalg.qr(A)
    R = Q
    a = float(np.clip(strength, 0.0, 1.0))
    for t in idx:
        x = np.nan_to_num(X[t], nan=0.0)
        s = U.T @ x
        s2 = (R @ s) * a + s * (1.0 - a)
        x_sub = U @ s2
        x_res = x - U @ (U.T @ x)
        X[t] = x_res + x_sub
    return pd.DataFrame(X, index=df_z.index, columns=df_z.columns), {"kind": "subspace_rotate", **inc.__dict__, "strength": float(a)}


def inject(df_z: pd.DataFrame, inc: Incident, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Dispatch injection by incident kind."""
    kind = inc.kind
    if kind == "level_shift":
        return inject_level_shift(df_z, inc)
    if kind == "drift":
        return inject_drift(df_z, inc)
    if kind == "variance_jump":
        return inject_variance_jump(df_z, inc, rng)
    if kind == "dropout":
        return inject_dropout(df_z, inc)
    if kind == "outlier_burst":
        return inject_outlier_burst(df_z, inc, rng)
    if kind == "freeze":
        return inject_freeze(df_z, inc)
    if kind == "impute_constant":
        constant = 0.0 if (inc.extra is None) else float(inc.extra.get("constant", 0.0))
        return inject_impute_constant(df_z, inc, constant=constant)
    if kind == "swap":
        if inc.extra is None or "swap_cols" not in inc.extra:
            raise ValueError("swap incident requires inc.extra['swap_cols'] = (a,b)")
        a, b = inc.extra["swap_cols"]
        return inject_swap(df_z, inc, int(a), int(b))
    if kind == "corr_mix":
        strength = 1.0 if (inc.extra is None) else float(inc.extra.get("strength", 1.0))
        return inject_corr_mix(df_z, inc, rng=rng, strength=strength)

    raise ValueError(f"Unknown incident kind: {kind}")
