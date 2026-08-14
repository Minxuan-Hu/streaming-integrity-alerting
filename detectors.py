
"""Score generators (detectors) used by the Streaming Integrity Monitoring benchmark.

Each detector maps a multivariate input stream to a scalar score stream s_t.
The contract evaluates the alert stream induced downstream by a fixed policy
(thresholding + episode semantics), so detectors themselves are intentionally
lightweight and dependency-minimal for reproducibility.

This module implements a suite of classical and multivariate monitoring baselines
(e.g., Hotelling T², EWMA/CUSUM variants, residual-energy, factor-covariance change,
and simple fused-test constructions).
"""
from __future__ import annotations
import numpy as np
import evaluation as eval_mod
import pandas as pd

# Operational episode accounting helpers (no circular dependency: evaluation.py does not import detectors.py)
from evaluation import count_alert_events, apply_episode_policy_array

def score_max_abs(Z: pd.DataFrame) -> pd.Series:
    return pd.Series(np.nanmax(np.abs(Z.values), axis=1), index=Z.index, name="max_abs")

def fit_cov(Z_train: np.ndarray, eps: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu, inv_cov) with diagonal regularization."""
    mu = np.nanmean(Z_train, axis=0)
    X = Z_train - mu
    # Replace NaNs with 0 after centering (dropout is treated as 0 deviation; alternative is impute)
    X = np.nan_to_num(X, nan=0.0)
    cov = (X.T @ X) / max(1, X.shape[0] - 1)
    cov = cov + eps * np.eye(cov.shape[0])
    inv = np.linalg.inv(cov)
    return mu, inv

def score_hotelling_t2(Z: pd.DataFrame, mu: np.ndarray, inv_cov: np.ndarray) -> pd.Series:
    X = Z.values - mu[None, :]
    X = np.nan_to_num(X, nan=0.0)
    t2 = np.einsum("bi,ij,bj->b", X, inv_cov, X)
    return pd.Series(t2, index=Z.index, name="T2")

def fit_pca(Z_train: np.ndarray, r: int = 6, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit PCA on Z_train (assumed already standardized).
    Returns (mu, U, eigvals) where U is (d,r) orthonormal, eigvals are top-r eigenvalues.
    """
    mu = np.nanmean(Z_train, axis=0)
    X = Z_train - mu
    X = np.nan_to_num(X, nan=0.0)
    cov = (X.T @ X) / max(1, X.shape[0] - 1)
    # Eigen-decomp
    w, V = np.linalg.eigh(cov)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    w_r = np.maximum(w[:r], eps)
    U = V[:, :r]
    return mu, U, w_r

def score_subspace(Z: pd.DataFrame, mu: np.ndarray, U: np.ndarray, eigvals: np.ndarray) -> tuple[pd.Series, pd.Series]:
    """
    Returns (residual_energy, pc_t2) score streams.

    - residual_energy = ||x||^2 - ||U^T x||^2
    - pc_t2 = sum_k ( (u_k^T x)^2 / eigval_k )
    """
    X = Z.values - mu[None, :]
    X = np.nan_to_num(X, nan=0.0)
    proj = X @ U  # (n,r)
    proj_energy = np.sum(proj * proj, axis=1)
    total_energy = np.sum(X * X, axis=1)
    res_energy = np.maximum(total_energy - proj_energy, 0.0)
    pc_t2 = np.sum((proj * proj) / eigvals[None, :], axis=1)

    return (
        pd.Series(res_energy, index=Z.index, name="res_energy"),
        pd.Series(pc_t2, index=Z.index, name="pc_t2"),
    )

def attribution_residual(Z_t: np.ndarray, mu: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Per-feature squared residual contributions at a single time t."""
    x = np.nan_to_num(Z_t - mu, nan=0.0)
    xhat = U @ (U.T @ x)
    r = x - xhat
    return r * r

def attribution_t2_diag(Z_t: np.ndarray, mu: np.ndarray, var_diag: np.ndarray) -> np.ndarray:
    """Diagonal approximation contributions for Hotelling T2."""
    x = np.nan_to_num(Z_t - mu, nan=0.0)
    return (x * x) / np.maximum(var_diag, 1e-6)


# =========================
# Additional lightweight baselines and diagnostics (e.g., staleness) are included for completeness.

def fit_staleness_params(Z_train: pd.DataFrame, q: float = 0.2, eps_floor: float = 0.05) -> np.ndarray:
    """Per-feature epsilon thresholds for 'near-zero' month-to-month innovations.

    We compute abs(ΔZ) on training and set eps_i as the q-quantile, floored by eps_floor.
    """
    X = Z_train.values
    X = np.nan_to_num(X, nan=0.0)
    dX = np.abs(np.diff(X, axis=0))
    if dX.shape[0] < 5:
        return np.full((X.shape[1],), eps_floor, dtype=float)
    eps = np.quantile(dX, q, axis=0)
    eps = np.maximum(eps, eps_floor)
    return eps.astype(float)

def score_staleness_runlength(Z: pd.DataFrame, eps: np.ndarray, agg: str = "max") -> pd.Series:
    """Run-length score for staleness (freeze / impute-to-constant) in a multivariate panel.

    For each feature i, define I_{t,i} = 1[|ΔZ_{t,i}| < eps_i]. The run length r_{t,i} increments while I=1.
    We aggregate across features with either:
      - agg='max': max_i r_{t,i}  (high sensitivity to targeted staleness)
      - agg='p95': 95th percentile across i
    Returns a score per timestamp t (first timestamp score is 0).
    """
    X = np.nan_to_num(Z.values, nan=0.0)
    dX = np.abs(np.diff(X, axis=0))
    I = (dX < eps[None, :]).astype(np.int32)  # (n-1,d)
    r = np.zeros_like(I, dtype=np.int32)
    for t in range(I.shape[0]):
        if t == 0:
            r[t] = I[t]
        else:
            r[t] = (r[t-1] + 1) * I[t]
    # align to timestamps: first month has no diff
    if agg == "max":
        s = np.concatenate([[0.0], r.max(axis=1).astype(float)])
    elif agg == "p95":
        s = np.concatenate([[0.0], np.quantile(r.astype(float), 0.95, axis=1)])
    else:
        raise ValueError("agg must be 'max' or 'p95'")
    return pd.Series(s, index=Z.index, name=f"stale_{agg}")

def score_subspace_change(
    Z: pd.DataFrame,
    mu: np.ndarray,
    r: int,
    window: int = 24,
    U0: np.ndarray | None = None,
    eps: float = 1e-8,
) -> pd.Series:
    """Rolling subspace change score: ||P_t - P0||_F where P is the rank-r projection matrix.

    - P0: baseline projection from the *training* PCA basis is not passed here; instead we estimate it
          from the first `window` samples in Z (after centering by mu), which should be a stable 'clean' period.
      In practice, call this on deployment data where the early segment is clean, or replace P0 externally.
    - Pt: projection from PCA on the trailing `window` samples.

    Score range: [0, sqrt(2r)].
    """
    X = np.nan_to_num(Z.values - mu[None, :], nan=0.0)
    n, d = X.shape
    r = int(max(1, min(r, d)))
    if n < window + 2:
        return pd.Series(np.zeros((n,), dtype=float), index=Z.index, name="subspace_change")

    def top_r_basis(Xw: np.ndarray) -> np.ndarray:
        # covariance eigvecs via SVD
        C = (Xw.T @ Xw) / max(Xw.shape[0], 1)
        # symmetric
        Uc, S, _ = np.linalg.svd(C, full_matrices=False)
        return Uc[:, :r]

    U0 = U0 if U0 is not None else top_r_basis(X[:window, :])
    P0 = U0 @ U0.T

    scores = np.zeros((n,), dtype=float)
    scores[:window] = 0.0
    for t in range(window, n):
        Xw = X[t-window+1:t+1, :]
        Ut = top_r_basis(Xw)
        Pt = Ut @ Ut.T
        D = P0 - Pt
        # Frobenius norm
        scores[t] = float(np.sqrt(np.sum(D * D) + eps))
    return pd.Series(scores, index=Z.index, name="subspace_change")

def robust_z(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Robust z-score using median and IQR."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    q1 = np.nanquantile(x, 0.25)
    q3 = np.nanquantile(x, 0.75)
    iqr = max(q3 - q1, eps)
    return (x - med) / iqr


def robust_loc_scale_train(x: np.ndarray, train_mask: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
    """Return (median, IQR) computed on the training window (finite values only)."""
    x = np.asarray(x, dtype=float)
    tm = np.asarray(train_mask, dtype=bool)
    base = x[tm]
    base = base[np.isfinite(base)]
    if base.size < 10:
        base = x[np.isfinite(x)]
    if base.size == 0:
        return 0.0, 1.0
    med = float(np.nanmedian(base))
    q1 = float(np.nanquantile(base, 0.25))
    q3 = float(np.nanquantile(base, 0.75))
    iqr = float(max(q3 - q1, eps))
    return med, iqr

def robust_standardize_train(x: np.ndarray, train_mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Robust-standardize x using (median, IQR) from the training window."""
    med, iqr = robust_loc_scale_train(x, train_mask=train_mask, eps=eps)
    return (np.asarray(x, dtype=float) - med) / iqr

def score_ewma_from_base(
    s: pd.Series,
    train_mask: np.ndarray,
    lam: float = 0.20,
    eps: float = 1e-6,
    one_sided: bool = True,
) -> pd.Series:
    """EWMA chart statistic on a scalar base score.

    We robust-standardize the base score using the training window, then apply:
        z_t = lam * x_t + (1-lam) * z_{t-1}

    If one_sided, we clamp z_t = max(z_t, 0) to focus on upward shifts (typical for anomaly scores).
    """
    x = robust_standardize_train(s.values, train_mask=train_mask, eps=eps)
    lam = float(np.clip(lam, 1e-3, 1.0))
    z = np.zeros_like(x, dtype=float)
    for t in range(x.shape[0]):
        if t == 0:
            z[t] = lam * x[t]
        else:
            z[t] = lam * x[t] + (1.0 - lam) * z[t - 1]
        if one_sided and np.isfinite(z[t]):
            z[t] = max(z[t], 0.0)
    name = f"ewma_{s.name}" if s.name else "ewma"
    return pd.Series(z, index=s.index, name=name)

def score_cusum_from_base(
    s: pd.Series,
    train_mask: np.ndarray,
    k: float = 0.50,
    eps: float = 1e-6,
) -> pd.Series:
    """One-sided CUSUM statistic on a scalar base score (upward shifts).

    We robust-standardize the base score using the training window (median/IQR), then:
        g_t = max(0, g_{t-1} + x_t - k)

    k is the reference value (drift allowance); larger k makes the CUSUM less sensitive to small shifts.
    """
    x = robust_standardize_train(s.values, train_mask=train_mask, eps=eps)
    k = float(max(k, 0.0))
    g = np.zeros_like(x, dtype=float)
    for t in range(x.shape[0]):
        prev = g[t - 1] if t > 0 else 0.0
        val = prev + x[t] - k
        g[t] = max(0.0, float(val)) if np.isfinite(val) else prev
    name = f"cusum_{s.name}" if s.name else "cusum"
    return pd.Series(g, index=s.index, name=name)

def _cusum_chart_sim(
    x: np.ndarray,
    k: float,
    h: float,
    hold_months: int = 1,
) -> tuple[np.ndarray, int]:
    """Simulate a one-sided CUSUM *chart* with reset-on-alarm.

    State:
      g_t = max(0, g_{t-1} + x_t - k)

    Alarm logic:
      - If g_t > h at time t, emit an alarm event at t, reset g_t = 0, and enter a hold period.
      - During hold, alarms remain 1 (warning state) and g is held at 0.

    Returns:
      alarms: int array in {0,1} (warning state)
      n_events: number of alarm events (0->1 episode starts) generated by the chart
    """
    x = np.asarray(x, dtype=float)
    T = int(x.shape[0])
    alarms = np.zeros(T, dtype=np.int32)

    k = float(max(k, 0.0))
    h = float(max(h, 0.0))
    hold = int(max(hold_months, 1))

    g = 0.0
    hold_left = 0
    n_events = 0

    for t in range(T):
        if hold_left > 0:
            alarms[t] = 1
            hold_left -= 1
            continue

        xt = float(x[t])
        if np.isfinite(xt):
            g = max(0.0, g + (xt - k))
        # else: keep g as-is (no update)

        if g > h and np.isfinite(g):
            alarms[t] = 1
            n_events += 1
            g = 0.0
            hold_left = hold - 1

    return alarms, int(n_events)


def calibrate_cusum_h_from_base(
    s: pd.Series,
    train_mask: np.ndarray,
    target_events_per_year: float,
    k: float = 0.50,
    hold_months: int = 1,
    eps: float = 1e-6,
    max_iter: int = 30,
) -> float:
    """Calibrate a CUSUM chart threshold h on benign training history.

    We robust-standardize s using (median, IQR) from the training window, then choose h such that
    the chart generates approximately `target_events_per_year` alarm events on the *training* window.

    Notes:
      - This is a *chart* calibration (reset-on-alarm), not a generic score-threshold.
      - The achieved event rate in deployment may deviate under drift; this is addressed by
        achieved-frontier reporting (FAE on the x-axis).
    """
    x = robust_standardize_train(s.values, train_mask=train_mask, eps=eps)
    x_tr = np.asarray(x[train_mask], dtype=float)
    tr_months = int(np.sum(train_mask))
    years = max(tr_months / float(getattr(eval_mod, 'STEPS_PER_YEAR', 12.0)), 1e-6)
    target_events = float(target_events_per_year) * years

    if target_events_per_year <= 0 or tr_months < 2:
        return float("inf")

    # Upper bound heuristic from the *no-reset* CUSUM statistic.
    g_raw = np.zeros_like(x_tr, dtype=float)
    g = 0.0
    kk = float(max(k, 0.0))
    for i in range(x_tr.shape[0]):
        xi = float(x_tr[i])
        if np.isfinite(xi):
            g = max(0.0, g + (xi - kk))
        g_raw[i] = g
    hi = float(np.nanquantile(g_raw, 0.999)) if np.isfinite(np.nanmax(g_raw)) else 1.0
    if not np.isfinite(hi) or hi <= 0:
        hi = float(np.nanmax(g_raw)) if np.isfinite(np.nanmax(g_raw)) else 1.0
    hi = float(max(hi, 1e-3))

    def n_events_for(hh: float) -> int:
        _a, ne = _cusum_chart_sim(x_tr, k=kk, h=float(hh), hold_months=int(hold_months))
        return int(ne)

    # Ensure hi yields <= target_events (monotone: higher h -> fewer events).
    ev_hi = n_events_for(hi)
    while ev_hi > target_events and hi < 1e6:
        hi *= 2.0
        ev_hi = n_events_for(hi)

    lo = 0.0
    ev_lo = n_events_for(lo)
    if ev_lo < target_events:
        # Even at h=0, we cannot reach the target; return 0 as the most aggressive chart.
        return 0.0

    # Binary search
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        ev_mid = n_events_for(mid)
        if ev_mid > target_events:
            lo = mid
        else:
            hi = mid

    return float(hi)




def cusum_chart_train_stats_from_base(
    s: pd.Series,
    train_mask: np.ndarray,
    h: float,
    k: float = 0.50,
    hold_months: int = 1,
    eps: float = 1e-6,
) -> dict:
    """Compute CUSUM chart training statistics on a benign window.

    Returns:
      - tr_months: number of months in the calibration window
      - years: window length in years
      - n_events: number of alarm events produced by the chart
      - fae: false alert events/year achieved on the window
      - tiw: time-in-warning fraction achieved on the window
    """
    train_mask = np.asarray(train_mask, dtype=bool)
    tr_months = int(np.sum(train_mask))
    years = max(tr_months / float(getattr(eval_mod, 'STEPS_PER_YEAR', 12.0)), 1e-6)
    if tr_months < 2:
        return {
            "tr_months": tr_months,
            "years": years,
            "n_events": 0,
            "fae": 0.0,
            "tiw": 0.0,
        }
    x = robust_standardize_train(s.values, train_mask=train_mask, eps=eps)
    x_tr = np.asarray(x[train_mask], dtype=float)
    alarms, n_events = _cusum_chart_sim(
        x_tr, k=float(max(k, 0.0)), h=float(max(h, 0.0)), hold_months=int(hold_months)
    )
    alarms = np.asarray(alarms, dtype=float)
    tiw = float(np.nanmean(alarms)) if alarms.size else 0.0
    fae = float(n_events) / years
    return {
        "tr_months": tr_months,
        "years": years,
        "n_events": int(n_events),
        "fae": float(fae),
        "tiw": float(tiw),
    }


def calibrate_cusum_h_occupancy_from_base(
    s: pd.Series,
    train_mask: np.ndarray,
    target_events_per_year: float,
    tiw_cap: float,
    k: float = 0.50,
    hold_months: int = 1,
    event_tol: float = 0.0,
    eps: float = 1e-6,
    max_iter: int = 30,
) -> tuple[float, dict]:
    """Occupancy-aware calibration of the CUSUM chart threshold h.

    Chooses the *most sensitive* threshold h that satisfies BOTH on the benign calibration window:
      1) events/year <= (1+event_tol) * target_events_per_year
      2) time-in-warning (TIW) <= tiw_cap

    This prevents the degenerate baseline failure mode where a detector enters an almost-continuous
    warning state (TIW≈1) while still showing a low event rate.

    Returns:
      (h, stats_dict) where stats_dict includes achieved fae/tiw/n_events/tr_months.
    """
    train_mask = np.asarray(train_mask, dtype=bool)
    tr_months = int(np.sum(train_mask))
    if target_events_per_year <= 0 or tr_months < 2:
        st = cusum_chart_train_stats_from_base(
            s, train_mask=train_mask, h=float("inf"), k=k, hold_months=hold_months, eps=eps
        )
        return float("inf"), st

    # If no occupancy cap is provided, fall back to the event-only calibration.
    if (tiw_cap is None) or (not np.isfinite(float(tiw_cap))) or float(tiw_cap) <= 0:
        h = calibrate_cusum_h_from_base(
            s,
            train_mask=train_mask,
            target_events_per_year=float(target_events_per_year),
            k=float(k),
            hold_months=int(hold_months),
            eps=float(eps),
            max_iter=int(max_iter),
        )
        st = cusum_chart_train_stats_from_base(
            s, train_mask=train_mask, h=float(h), k=float(k), hold_months=int(hold_months), eps=float(eps)
        )
        return float(h), st

    cap = float(np.clip(float(tiw_cap), 0.0, 1.0))
    tol = float(max(event_tol, 0.0))
    target_fae = float(target_events_per_year) * (1.0 + tol)

    # Standardize once and operate on the training slice for deterministic monotonic search.
    x = robust_standardize_train(s.values, train_mask=train_mask, eps=eps)
    x_tr = np.asarray(x[train_mask], dtype=float)
    kk = float(max(k, 0.0))
    years = max(tr_months / float(getattr(eval_mod, 'STEPS_PER_YEAR', 12.0)), 1e-6)

    # Upper bound heuristic from the *no-reset* CUSUM statistic.
    g_raw = np.zeros_like(x_tr, dtype=float)
    g = 0.0
    for i in range(x_tr.shape[0]):
        xi = float(x_tr[i])
        if np.isfinite(xi):
            g = max(0.0, g + (xi - kk))
        g_raw[i] = g
    hi = float(np.nanquantile(g_raw, 0.999)) if np.isfinite(np.nanmax(g_raw)) else 1.0
    if not np.isfinite(hi) or hi <= 0:
        hi = float(np.nanmax(g_raw)) if np.isfinite(np.nanmax(g_raw)) else 1.0
    hi = float(max(hi, 1e-3))

    def stats_for(hh: float) -> dict:
        alarms, n_events = _cusum_chart_sim(x_tr, k=kk, h=float(hh), hold_months=int(hold_months))
        alarms = np.asarray(alarms, dtype=float)
        tiw = float(np.nanmean(alarms)) if alarms.size else 0.0
        fae = float(n_events) / years
        return {
            "tr_months": tr_months,
            "years": years,
            "n_events": int(n_events),
            "fae": float(fae),
            "tiw": float(tiw),
        }

    def passes(st: dict) -> bool:
        return (st["fae"] <= target_fae + 1e-12) and (st["tiw"] <= cap + 1e-12)

    st_hi = stats_for(hi)
    while (not passes(st_hi)) and hi < 1e6:
        hi *= 2.0
        st_hi = stats_for(hi)

    # If even huge h fails (unexpected; huge h -> no alarms), return hi.
    if not passes(st_hi):
        return float(hi), st_hi

    lo = 0.0
    st_lo = stats_for(lo)
    if passes(st_lo):
        return float(lo), st_lo

    # Binary search for minimal passing h.
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        st_mid = stats_for(mid)
        if passes(st_mid):
            hi = mid
            st_hi = st_mid
        else:
            lo = mid

    return float(hi), st_hi


def cusum_chart_train_stats_operational_from_base(
    s: pd.Series,
    train_mask: np.ndarray,
    h: float,
    k: float = 0.50,
    hold_months: int = 1,
    cooldown_months: int = 0,
    max_episode_months: int = 0,
    merge_gap_months: int = 0,
    eps: float = 1e-6,
) -> dict:
    """Compute *operational* train-window stats for a CUSUM chart.

    Unlike `cusum_chart_train_stats_from_base` (which counts raw chart alarm events),
    this function applies the same episode policy used in evaluation (cooldown/max
    episode length) and then counts alert *events* using `count_alert_events`.

    This aligns CUSUM calibration with Task-B evaluation, preventing the failure mode
    where TIW is high (near-continuous warning) but the effective event rate appears low
    due to episode merging.
    """
    train_mask = np.asarray(train_mask, dtype=bool)
    tr_months = int(np.sum(train_mask))
    years = max(tr_months / float(getattr(eval_mod, 'STEPS_PER_YEAR', 12.0)), 1e-6)
    if tr_months < 2:
        return {"tr_months": tr_months, "years": years, "n_events": 0, "fae": 0.0, "tiw": 0.0}

    x = robust_standardize_train(s.values, train_mask=train_mask, eps=float(eps))
    x_tr = np.asarray(x[train_mask], dtype=float)
    alarms_raw, _ne = _cusum_chart_sim(x_tr, k=float(max(k, 0.0)), h=float(max(h, 0.0)), hold_months=int(max(hold_months, 1)))
    alarms_man = apply_episode_policy_array(
        alarms_raw,
        cooldown_months=int(max(cooldown_months, 0)),
        max_episode_months=int(max(max_episode_months, 0)),
    )
    tiw = float(np.mean(alarms_man)) if alarms_man.size else 0.0
    mask = np.ones_like(alarms_man, dtype=bool)
    n_events = int(count_alert_events(alarms_man.astype(int), mask=mask, merge_gap_months=int(max(merge_gap_months, 0))))
    fae = float(n_events) / years
    return {"tr_months": tr_months, "years": years, "n_events": int(n_events), "fae": float(fae), "tiw": float(tiw)}


def calibrate_cusum_h_operational_occupancy_from_base(
    s: pd.Series,
    train_mask: np.ndarray,
    target_events_per_year: float,
    tiw_cap: float,
    k: float = 0.50,
    hold_months: int = 1,
    cooldown_months: int = 0,
    max_episode_months: int = 0,
    merge_gap_months: int = 0,
    event_tol: float = 0.0,
    eps: float = 1e-6,
    max_iter: int = 30,
) -> tuple[float, dict]:
    """Occupancy-aware CUSUM calibration using *operational* event accounting.

    This calibrates the CUSUM threshold h on a benign calibration window using the same
    operational accounting as the contract: after applying episode semantics, we enforce
    (i) a target false-alert event rate (events/year) within tolerance, and (ii) an optional
    time-in-warning (TIW) cap. This keeps score calibration aligned with burden accounting.
    Returns (h, stats_dict) where stats_dict contains achieved fae/tiw etc.
    """
    train_mask = np.asarray(train_mask, dtype=bool)
    tr_months = int(np.sum(train_mask))
    if target_events_per_year <= 0 or tr_months < 2:
        st = cusum_chart_train_stats_operational_from_base(
            s,
            train_mask=train_mask,
            h=float("inf"),
            k=float(k),
            hold_months=int(hold_months),
            cooldown_months=int(cooldown_months),
            max_episode_months=int(max_episode_months),
            merge_gap_months=int(merge_gap_months),
            eps=float(eps),
        )
        return float("inf"), st

    if (tiw_cap is None) or (not np.isfinite(float(tiw_cap))) or float(tiw_cap) <= 0:
        # Still calibrate against operational event rate (no TIW cap)
        tiw_cap = 1.0

    cap = float(np.clip(float(tiw_cap), 0.0, 1.0))
    tol = float(max(event_tol, 0.0))
    target_fae = float(target_events_per_year) * (1.0 + tol)

    # Standardize once and operate on training slice for deterministic search.
    x = robust_standardize_train(s.values, train_mask=train_mask, eps=float(eps))
    x_tr = np.asarray(x[train_mask], dtype=float)
    kk = float(max(k, 0.0))

    # Upper bound heuristic from the *no-reset* CUSUM statistic.
    g_raw = np.zeros_like(x_tr, dtype=float)
    g = 0.0
    for i in range(x_tr.shape[0]):
        xi = float(x_tr[i])
        if np.isfinite(xi):
            g = max(0.0, g + (xi - kk))
        g_raw[i] = g
    hi = float(np.nanquantile(g_raw, 0.999)) if np.isfinite(np.nanmax(g_raw)) else 1.0
    if not np.isfinite(hi) or hi <= 0:
        hi = float(np.nanmax(g_raw)) if np.isfinite(np.nanmax(g_raw)) else 1.0
    hi = float(max(hi, 1e-3))

    def stats_for(hh: float) -> dict:
        alarms_raw, _ne = _cusum_chart_sim(x_tr, k=kk, h=float(hh), hold_months=int(max(hold_months, 1)))
        alarms_man = apply_episode_policy_array(
            alarms_raw,
            cooldown_months=int(max(cooldown_months, 0)),
            max_episode_months=int(max(max_episode_months, 0)),
        )
        tiw = float(np.mean(alarms_man)) if alarms_man.size else 0.0
        mask = np.ones_like(alarms_man, dtype=bool)
        n_events = int(count_alert_events(alarms_man.astype(int), mask=mask, merge_gap_months=int(max(merge_gap_months, 0))))
        years = max(tr_months / float(getattr(eval_mod, 'STEPS_PER_YEAR', 12.0)), 1e-6)
        fae = float(n_events) / years
        return {"tr_months": tr_months, "years": years, "n_events": int(n_events), "fae": float(fae), "tiw": float(tiw)}

    def passes(st: dict) -> bool:
        return (st["fae"] <= target_fae + 1e-12) and (st["tiw"] <= cap + 1e-12)

    st_hi = stats_for(hi)
    while (not passes(st_hi)) and hi < 1e6:
        hi *= 2.0
        st_hi = stats_for(hi)

    if not passes(st_hi):
        return float(hi), st_hi

    lo = 0.0
    st_lo = stats_for(lo)
    if passes(st_lo):
        return float(lo), st_lo

    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        st_mid = stats_for(mid)
        if passes(st_mid):
            hi = mid
            st_hi = st_mid
        else:
            lo = mid

    return float(hi), st_hi

def cusum_chart_alarms_from_base(
    s: pd.Series,
    train_mask: np.ndarray,
    h: float,
    k: float = 0.50,
    hold_months: int = 1,
    eps: float = 1e-6,
) -> pd.Series:
    """Run a calibrated one-sided CUSUM chart and return a warning-state alarm Series."""
    x = robust_standardize_train(s.values, train_mask=train_mask, eps=eps)
    alarms, _ne = _cusum_chart_sim(x, k=float(k), h=float(h), hold_months=int(hold_months))
    name = f"alarm_cusum_{s.name}" if s.name else "alarm_cusum"
    return pd.Series(alarms.astype(int), index=s.index, name=name)


def fuse_scores_max_robust(scores: list[pd.Series]) -> pd.Series:
    """Fuse heterogeneous score streams by robust-standardizing each, then taking max."""
    Zs = []
    for s in scores:
        z = robust_z(s.values)
        Zs.append(z)
    fused = np.max(np.stack(Zs, axis=1), axis=1)
    return pd.Series(fused, index=scores[0].index, name="fused_score")



def score_factor_cov_change(
    Z: pd.DataFrame,
    mu: np.ndarray,
    U: np.ndarray,
    eigvals: np.ndarray,
    window: int = 24,
    eps: float = 1e-8,
) -> pd.Series:
    """Rolling change in covariance of PCA scores within the fixed baseline subspace.

    Let s_t = U^T (x_t - mu). Baseline covariance is diag(eigvals).
    For each t, compute covariance of {s_{t-window+1},...,s_t} and score ||C_t - C0||_F.

    This is a lightweight alternative to refitting PCA each step, and is effective for 'stealthy'
    in-subspace manipulations that reallocate variance/correlation across factors.
    """
    X = np.nan_to_num(Z.values - mu[None, :], nan=0.0)
    S = X @ U  # (n,r)
    n, r = S.shape
    C0 = np.diag(np.maximum(eigvals, eps))
    scores = np.zeros((n,), dtype=float)
    if n < window + 2:
        return pd.Series(scores, index=Z.index, name="factor_cov_change")
    for t in range(window-1, n):
        Sw = S[t-window+1:t+1, :]
        # center within window
        Sw = Sw - Sw.mean(axis=0, keepdims=True)
        C = (Sw.T @ Sw) / max(Sw.shape[0], 1)
        D = C - C0
        # Focus on correlation / cross-factor covariance shifts: off-diagonal energy
        D_off = D - np.diag(np.diag(D))
        scores[t] = float(np.sqrt(np.sum(D_off * D_off) + eps))
    return pd.Series(scores, index=Z.index, name="factor_cov_change")


def score_factor_cov_delta(
    Z: pd.DataFrame,
    mu: np.ndarray,
    U: np.ndarray,
    window: int = 24,
    eps: float = 1e-8,
) -> pd.Series:
    """Change-onset covariance score: ||C_t - C_{t-1}||_F (off-diagonal only) in PCA-score space."""
    X = np.nan_to_num(Z.values - mu[None, :], nan=0.0)
    S = X @ U
    n, r = S.shape
    scores = np.zeros((n,), dtype=float)
    if n < window + 3:
        return pd.Series(scores, index=Z.index, name="factor_cov_delta")
    C_prev = None
    for t in range(window - 1, n):
        Sw = S[t - window + 1 : t + 1, :]
        Sw = Sw - Sw.mean(axis=0, keepdims=True)
        C = (Sw.T @ Sw) / max(Sw.shape[0], 1)
        if C_prev is None:
            C_prev = C
            continue
        D = C - C_prev
        D_off = D - np.diag(np.diag(D))
        scores[t] = float(np.sqrt(np.sum(D_off * D_off) + eps))
        C_prev = C
    return pd.Series(scores, index=Z.index, name="factor_cov_delta")


def _cov_lrt_distance(C0: np.ndarray, C: np.ndarray, eps: float = 1e-8) -> float:
    """Gaussian covariance distance: tr(C0^{-1}C) - logdet(C0^{-1}C) - r."""
    r = C0.shape[0]
    C0r = C0 + eps * np.eye(r)
    Cr = C + eps * np.eye(r)
    try:
        L0 = np.linalg.cholesky(C0r)
        Y = np.linalg.solve(L0, Cr)
        M = np.linalg.solve(L0, Y.T).T
    except np.linalg.LinAlgError:
        M = np.linalg.pinv(C0r) @ Cr
    tr = float(np.trace(M))
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        logdet = np.log(max(np.abs(np.linalg.det(M)), eps))
    return float(tr - logdet - r)


def score_factor_cov_lrt(
    Z: pd.DataFrame,
    mu: np.ndarray,
    U: np.ndarray,
    eigvals: np.ndarray,
    window: int = 24,
    eps: float = 1e-8,
) -> pd.Series:
    """Covariance LRT distance to baseline diag(eigvals) in PCA-score space."""
    X = np.nan_to_num(Z.values - mu[None, :], nan=0.0)
    S = X @ U
    n, r = S.shape
    C0 = np.diag(np.maximum(eigvals[:r], eps))
    scores = np.zeros((n,), dtype=float)
    if n < window + 2:
        return pd.Series(scores, index=Z.index, name="factor_cov_lrt")
    for t in range(window - 1, n):
        Sw = S[t - window + 1 : t + 1, :]
        Sw = Sw - Sw.mean(axis=0, keepdims=True)
        C = (Sw.T @ Sw) / max(Sw.shape[0], 1)
        scores[t] = _cov_lrt_distance(C0, C, eps=eps)
    return pd.Series(scores, index=Z.index, name="factor_cov_lrt")


def score_factor_cov_lrt_delta(
    Z: pd.DataFrame,
    mu: np.ndarray,
    U: np.ndarray,
    window: int = 24,
    eps: float = 1e-8,
) -> pd.Series:
    """Change-onset covariance LRT: distance between consecutive window covariances."""
    X = np.nan_to_num(Z.values - mu[None, :], nan=0.0)
    S = X @ U
    n, r = S.shape
    scores = np.zeros((n,), dtype=float)
    if n < window + 3:
        return pd.Series(scores, index=Z.index, name="factor_cov_lrt_delta")
    C_prev = None
    for t in range(window - 1, n):
        Sw = S[t - window + 1 : t + 1, :]
        Sw = Sw - Sw.mean(axis=0, keepdims=True)
        C = (Sw.T @ Sw) / max(Sw.shape[0], 1)
        if C_prev is None:
            C_prev = C
            continue
        scores[t] = _cov_lrt_distance(C_prev, C, eps=eps)
        C_prev = C
    return pd.Series(scores, index=Z.index, name="factor_cov_lrt_delta")



def robust_z_train(x: np.ndarray, train_mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Robust z-score using median/IQR computed on training indices only."""
    x = np.asarray(x, dtype=float)
    tm = np.asarray(train_mask, dtype=bool)
    base = x[tm]
    base = base[np.isfinite(base)]
    if base.size < 10:
        return robust_z(x, eps=eps)
    med = float(np.nanmedian(base))
    q1 = float(np.nanquantile(base, 0.25))
    q3 = float(np.nanquantile(base, 0.75))
    iqr = max(q3 - q1, eps)
    return (x - med) / iqr

def fuse_scores_max_robust_train(scores: list[pd.Series], train_mask: np.ndarray) -> pd.Series:
    """Fuse score streams via robust standardization on training window, then max."""
    Zs = []
    for s in scores:
        z = robust_z_train(s.values, train_mask=train_mask)
        Zs.append(z)
    fused = np.max(np.stack(Zs, axis=1), axis=1)
    return pd.Series(fused, index=scores[0].index, name="fused_score")

# =========================
# P-value fusion utilities (Fisher-style combinations).

def survival_pvals_from_train(x: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """Empirical survival p-values using the training distribution.

    For scores where "larger = more anomalous", we define:
        p(x) = ( # {s_train >= x} + 1 ) / (n_train + 1)

    This yields p in (0,1], robust to score scaling.
    """
    x = np.asarray(x, dtype=float)
    tm = np.asarray(train_mask, dtype=bool)
    base = x[tm]
    base = base[np.isfinite(base)]
    if base.size < 25:
        # fall back to global finite values
        base = x[np.isfinite(x)]
    base = np.sort(base)
    n = int(base.size)
    if n == 0:
        return np.ones_like(x, dtype=float)

    # For each x_i, find first index where base[idx] >= x_i
    # Then count >= x_i is n - idx
    idx = np.searchsorted(base, x, side="left")
    ge = (n - idx).astype(float)
    p = (ge + 1.0) / (n + 1.0)
    # numerical safety
    p = np.clip(p, 1.0 / (n + 1.0), 1.0)
    return p


def fuse_scores_fisher_train(scores: list[pd.Series], train_mask: np.ndarray, eps: float = 1e-12) -> pd.Series:
    """Fuse score streams using Fisher's method on empirical training-calibrated p-values.

    For each component score s_j(t), compute empirical survival p-values p_j(t) using training distribution,
    then fuse:
        S(t) = -2 * sum_j log(p_j(t))

    Larger S(t) indicates stronger evidence of anomaly.
    """
    if len(scores) == 0:
        raise ValueError("scores must be a non-empty list")
    idx = scores[0].index
    P = []
    for s in scores:
        if not s.index.equals(idx):
            raise ValueError("All score series must share the same index")
        p = survival_pvals_from_train(s.values, train_mask=train_mask)
        P.append(p)
    Pm = np.stack(P, axis=1)
    Pm = np.clip(Pm, eps, 1.0)
    fused = -2.0 * np.sum(np.log(Pm), axis=1)
    return pd.Series(fused, index=idx, name="fused_fisher")


# -----------------------------------------------------------------------------
# Cross-panel coherence (all_only -> score_only) detector
# -----------------------------------------------------------------------------

def fit_coherence_ridge(
    X_all: np.ndarray,
    Y_score: np.ndarray,
    alpha: float = 1e-2,
    fit_intercept: bool = True,
    cov_shrink: float = 0.10,
    cov_eps: float = 1e-6,
) -> dict:
    """Fit a cross-panel coherence baseline: predict score-panel signals from all-only signals.

    This supports integrity monitoring of *relationships* across redundant panels. Under a
    relationship-tampering attack (e.g., corr_mix), marginals may remain plausible while
    cross-panel consistency breaks.

    Args:
        X_all: (T, d_all) standardized all-only signals.
        Y_score: (T, d_score) standardized score-only signals.
        alpha: ridge regularization strength.
        fit_intercept: whether to fit an intercept term.
        cov_shrink: covariance shrinkage toward diagonal for residual T^2 stability in small T.
        cov_eps: diagonal jitter for numerical stability.

    Returns:
        model dict containing W, b, residual mean mu_r, and inv residual covariance.
    """
    X = np.asarray(X_all, dtype=float)
    Y = np.asarray(Y_score, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X_all and Y_score must be 2D arrays")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Row mismatch: X has {X.shape[0]} rows, Y has {Y.shape[0]} rows")
    T, d_all = X.shape
    _, d_score = Y.shape

    good = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X = X[good]
    Y = Y[good]
    if X.shape[0] < max(20, d_all + 2):
        raise ValueError(f"Not enough finite training rows for coherence fit: {X.shape[0]}")

    if fit_intercept:
        x_mu = X.mean(axis=0, keepdims=True)
        y_mu = Y.mean(axis=0, keepdims=True)
        Xc = X - x_mu
        Yc = Y - y_mu
    else:
        x_mu = np.zeros((1, d_all), dtype=float)
        y_mu = np.zeros((1, d_score), dtype=float)
        Xc = X
        Yc = Y

    # Ridge: W = (X^T X + alpha I)^-1 X^T Y
    XtX = Xc.T @ Xc
    XtY = Xc.T @ Yc
    A = XtX + float(alpha) * np.eye(d_all)
    W = np.linalg.solve(A, XtY)  # (d_all, d_score)

    # Intercept in original coordinates
    b = (y_mu - x_mu @ W).reshape(-1)  # (d_score,)

    # Residuals on training rows (after fit)
    Y_hat = X @ W + b
    R = Y - Y_hat
    mu_r = R.mean(axis=0)

    # Residual covariance (shrinkage + jitter), then invert
    if d_score == 1:
        var = float(np.var(R[:, 0], ddof=1)) if R.shape[0] > 1 else 1.0
        cov = np.array([[max(var, 1e-12)]], dtype=float)
    else:
        cov = np.cov(R, rowvar=False, ddof=1)
        cov = np.asarray(cov, dtype=float)
        diag = np.diag(np.diag(cov))
        lam = float(np.clip(cov_shrink, 0.0, 1.0))
        cov = (1.0 - lam) * cov + lam * diag

    # Scale-aware jitter
    tr = float(np.trace(cov))
    scale = tr / max(d_score, 1)
    cov = cov + (float(cov_eps) * max(scale, 1e-12)) * np.eye(d_score)

    inv_cov = np.linalg.inv(cov)

    return {
        "W": W,
        "b": b,
        "mu_r": mu_r,
        "inv_cov": inv_cov,
        "fit_intercept": bool(fit_intercept),
        "alpha": float(alpha),
        "cov_shrink": float(cov_shrink),
        "cov_eps": float(cov_eps),
        "d_all": int(d_all),
        "d_score": int(d_score),
        "n_fit": int(X.shape[0]),
    }


def score_coherence_resid_t2(
    X_all: pd.DataFrame,
    Y_score: pd.DataFrame,
    model: dict,
) -> pd.Series:
    """Residual Hotelling T^2 for cross-panel coherence: r_t^T Σ_r^{-1} r_t."""
    if not X_all.index.equals(Y_score.index):
        # align on intersection to be safe
        idx = X_all.index.intersection(Y_score.index)
        X = X_all.loc[idx].values
        Y = Y_score.loc[idx].values
        out_idx = idx
    else:
        X = X_all.values
        Y = Y_score.values
        out_idx = X_all.index

    W = model["W"]
    b = model["b"]
    mu_r = model["mu_r"]
    inv = model["inv_cov"]

    Y_hat = X @ W + b
    R = (Y - Y_hat) - mu_r  # center residuals
    # quadratic form per row
    t2 = np.einsum("ij,jk,ik->i", R, inv, R)
    return pd.Series(t2, index=out_idx, name="coh_resid_t2")


def score_coherence_resid_energy(
    X_all: pd.DataFrame,
    Y_score: pd.DataFrame,
    model: dict,
) -> pd.Series:
    """L2 residual energy for cross-panel coherence: ||r_t||_2^2."""
    if not X_all.index.equals(Y_score.index):
        idx = X_all.index.intersection(Y_score.index)
        X = X_all.loc[idx].values
        Y = Y_score.loc[idx].values
        out_idx = idx
    else:
        X = X_all.values
        Y = Y_score.values
        out_idx = X_all.index

    W = model["W"]
    b = model["b"]
    mu_r = model["mu_r"]

    Y_hat = X @ W + b
    R = (Y - Y_hat) - mu_r
    e = np.sum(R * R, axis=1)
    return pd.Series(e, index=out_idx, name="coh_resid_energy")