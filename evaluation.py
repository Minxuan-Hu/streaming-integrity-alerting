
"""Metric computation for the Streaming Integrity Monitoring operational evaluation framework.

This module computes framework-facing quantities from the score and warning streams, including:
  - window-scoped counterfactual divergence (CAD) between clean and corrupted warning streams,
  - event-level timeliness / delay and onset-saturation diagnostics (AIW),
  - chance-overlap baselines on the realized clean alert stream, and
  - the NC-2 episode-start alignment check used in the validity-controls appendix.

Terminology note: some function names retain the historical ``task_A`` / ``task_B`` prefixes for
backward compatibility. In paper terms:
  - ``task_A``: incident-window evaluation on paired clean/corrupted replays (CAD, timeliness,
    delays, AIW, and related event-level quantities).
  - ``task_B``: operating-point and regime-style summaries derived from the realized warning stream
    (coverage/feasibility, compliance ledgers, drift stress tests, overlap baselines).

All results are written as paper-facing CSV artifacts by the main evaluation runner.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

# ---- Time-axis configuration (defaults preserve monthly semantics) ----
STEPS_PER_YEAR: float = 12.0
TIME_MODE: str = "calendar_month"  # {"calendar_month", "step"}

def set_time_config(time_mode: str = "calendar_month", steps_per_year: float = 12.0) -> None:
    """Configure how the evaluation interprets time units.

    - calendar_month: monthly-series behavior (12 steps/year; delays measured in calendar months).
    - step: generic behavior (each row is one time step; delays measured in index steps).

    This is a global module setting so the runner can switch modes without
    threading an extra parameter through every function.
    """
    global TIME_MODE, STEPS_PER_YEAR
    TIME_MODE = str(time_mode)
    STEPS_PER_YEAR = float(steps_per_year)


def infer_regular_steps_per_year(dt_index: pd.DatetimeIndex) -> float:
    """Infer an approximate steps-per-year from a regular DatetimeIndex.

    Used only when the runner is invoked with `--time_mode auto`.
    Falls back to 12.0 if inference fails.
    """
    try:
        if not isinstance(dt_index, pd.DatetimeIndex) or len(dt_index) < 3:
            return 12.0
        freq = pd.infer_freq(dt_index)
        if freq is not None:
            freq = str(freq).upper()
            # Month start/end
            if freq in {"MS", "M"}:
                return 12.0
            # Quarterly
            if freq in {"QS", "Q"}:
                return 4.0
            # Weekly
            if freq in {"W", "W-SUN", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT"}:
                return 52.0
            # Daily / hourly / minute / second
            if freq in {"D"}:
                return 365.25
            if freq.endswith("H"):
                return 365.25 * 24.0
            if freq.endswith("T") or freq.endswith("MIN"):
                return 365.25 * 24.0 * 60.0
            if freq.endswith("S"):
                return 365.25 * 24.0 * 3600.0
        # Generic fallback: median delta in seconds
        dsecs = np.median(np.diff(dt_index.values).astype("timedelta64[s]").astype(float))
        if not np.isfinite(dsecs) or dsecs <= 0:
            return 12.0
        return float((365.25 * 24.0 * 3600.0) / dsecs)
    except Exception:
        return 12.0

def _steps_per_year() -> float:
    return float(STEPS_PER_YEAR) if float(STEPS_PER_YEAR) > 0 else 12.0

def _idx_pos(index: pd.DatetimeIndex, ts: pd.Timestamp) -> int:
    """Return integer position of ts in index (nearest if not exact)."""
    ix = int(index.get_indexer([ts])[0])
    if ix < 0:
        ix = int(index.get_indexer([ts], method="nearest")[0])
    return int(ix)

def _time_diff_units(index: pd.DatetimeIndex, a: pd.Timestamp, b: pd.Timestamp) -> int:
    """Difference between a and b in the contract's time units."""
    if TIME_MODE == "step":
        return int(_idx_pos(index, b) - _idx_pos(index, a))
    return int(months_diff(a, b))

def _shift_back(index: pd.DatetimeIndex, ts: pd.Timestamp, units: int) -> pd.Timestamp:
    """Shift ts backward by `units` months (calendar) or steps (step mode)."""
    units = int(max(units, 0))
    if units == 0:
        return ts
    if TIME_MODE == "step":
        pos = _idx_pos(index, ts)
        return pd.Timestamp(index[max(0, pos - units)])
    return (ts.to_period("M") - units).to_timestamp()


@dataclass
class Window:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp

def calibrate_threshold(scores: pd.Series, calib_mask: np.ndarray, fa_per_year: float) -> float:
    p = min(max(fa_per_year / _steps_per_year(), 1e-6), 0.25)
    vals = scores.values[calib_mask]
    vals = vals[np.isfinite(vals)]
    if len(vals) < 10:
        raise ValueError("Not enough calibration scores to set a threshold.")
    return float(np.quantile(vals, 1.0 - p))

def apply_threshold(scores: pd.Series, thr: float) -> pd.Series:
    return pd.Series((scores.values > thr).astype(int), index=scores.index, name="alarm")

def mask_window(index: pd.DatetimeIndex, w: Window) -> np.ndarray:
    return (index >= w.start) & (index <= w.end)

def first_alarm_date(index: pd.DatetimeIndex, alarms_bool: np.ndarray) -> pd.Timestamp | None:
    idx = np.where(alarms_bool)[0]
    if len(idx) == 0:
        return None
    return index[idx[0]]

def months_diff(a: pd.Timestamp, b: pd.Timestamp) -> int:
    pa = a.to_period("M")
    pb = b.to_period("M")
    return (pb.year - pa.year) * 12 + (pb.month - pa.month)

def _fill_short_gaps(x: np.ndarray, gap: int) -> np.ndarray:
    """Fill gaps of zeros of length <= gap between ones (1D binary)."""
    if gap <= 0:
        return x
    x = x.astype(np.int32)
    n = x.size
    i = 0
    while i < n:
        if x[i] == 1:
            j = i + 1
            # find next 1
            while j < n and x[j] == 0:
                j += 1
            if j < n and (j - i - 1) <= gap:
                # fill zeros between i and j
                x[i+1:j] = 1
            i = j
        else:
            i += 1
    return x

def count_alert_events(alarms: np.ndarray, mask: np.ndarray, merge_gap_months: int = 0) -> int:
    """Count alert *episodes* (0->1 transitions) inside `mask`.

    We treat the first masked point as if previous state were 0 (deployment start).
    If merge_gap_months > 0, short 0-gaps between 1s are merged into a single episode.
    """
    a = np.asarray(alarms, dtype=np.int32).copy()
    m = np.asarray(mask, dtype=bool)
    if a.size == 0 or m.sum() == 0:
        return 0
    # Restrict to masked region but keep continuity by zeroing outside mask
    a = a * m.astype(np.int32)
    if merge_gap_months > 0:
        a = _fill_short_gaps(a, gap=int(merge_gap_months))
        a = a * m.astype(np.int32)
    # Count 0->1 transitions within mask
    idx = np.where(m)[0]
    if idx.size == 0:
        return 0
    # previous state is 0 at deployment start
    prev = 0
    events = 0
    for t in idx:
        cur = int(a[t])
        if cur == 1 and prev == 0:
            events += 1
        prev = cur
    return int(events)



def count_events(
    alarms: np.ndarray,
    merge_gap_months: int = 0,
    mask: np.ndarray | None = None,
) -> int:
    """Count alert *episodes* (0->1 transitions) under an optional evaluation mask.

    - If `mask` is None, this is equivalent to counting episodes on the full array,
      treating the slice start as a fresh deployment start (previous state = 0).
    - If `mask` is provided, episodes are counted *within* the masked region, treating
      the first masked point as deployment start (previous state = 0).

    This function exists for backward compatibility with earlier codepaths that used
    `count_events(...)` and for newer code that needs masked event accounting.
    """
    a = np.asarray(alarms, dtype=np.int32)
    if mask is None:
        m = np.ones_like(a, dtype=bool)
    else:
        m = np.asarray(mask, dtype=bool)
        if m.shape != a.shape:
            raise ValueError("count_events: mask shape mismatch")
    return int(count_alert_events(a, m, merge_gap_months=int(merge_gap_months)))

def episode_starts(alarms: np.ndarray, merge_gap_months: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (starts, merged_alarms) for alert episodes on a full binary alarm stream.

    An episode start is defined as a 0->1 transition.

    If `merge_gap_months > 0`, short 0-gaps (length <= merge_gap_months) between 1s are filled
    before computing episode starts. This matches the semantics used for event counting.

    Returns:
      starts: boolean array, True at episode start indices
      merged_alarms: int array in {0,1} after optional gap filling
    """
    a = np.asarray(alarms, dtype=np.int32).copy()
    if a.size == 0:
        return np.zeros((0,), dtype=bool), a
    if merge_gap_months > 0:
        a = _fill_short_gaps(a, gap=int(merge_gap_months))
    starts = np.zeros_like(a, dtype=bool)
    prev = 0
    for t in range(a.size):
        cur = int(a[t])
        if cur == 1 and prev == 0:
            starts[t] = True
        prev = cur
    return starts, a


def apply_merge_gap_array(alarms: np.ndarray, merge_gap_months: int = 0) -> np.ndarray:
    """Return a merged-gap warning-state array.

    This converts a binary alarm stream into an operational "in-warning" state by optionally
    filling short 0-gaps of length <= merge_gap_months between 1s.

    Note: cooldown / max-episode policy is assumed to have already been applied upstream.
    """
    a = np.asarray(alarms, dtype=np.int32).copy()
    if a.size == 0:
        return a
    if merge_gap_months > 0:
        a = _fill_short_gaps(a, gap=int(merge_gap_months))
    return a.astype(np.int32)

def apply_event_budgeted_threshold(
    scores: pd.Series,
    target_events_per_year: float,
    train_mask: np.ndarray,
    window: int = 60,
    min_periods: int = 24,
    merge_gap_months: int = 1,
    max_time_in_warning: float | None = None,
    guardband: float = 0.90,
    p_lo: float = 1e-6,
    p_hi: float = 0.999,
    cooldown_months: int = 0,
    max_episode_months: int = 0,
    event_tol: float = 0.0,
) -> tuple[pd.Series, pd.Series, float]:
    """Tune a constant p* on benign history to hit an operational event budget.

    Important semantics (paper-facing):
      - Event rate is computed on *warning state* w_t formed by:
            raw alarms -> episode policy (cooldown / max_episode) -> merge-gap
      - TIW is mean(w_t) over the benign calibration region.
      - Rolling thresholds are lagged by 1 month (no lookahead).

    guardband is a conservative factor in (0,1]; calibration target = guardband * target_events_per_year.
    For paper "matched-burden" artifacts, set guardband≈1.0 to avoid systematic under-shoot.
    """
    idx = scores.index
    s = scores.astype(float).copy()
    valid = np.asarray(train_mask, dtype=bool) & np.isfinite(s.values)

    target = float(target_events_per_year)
    if target <= 0:
        # Degenerate: no alerts desired.
        alarms = pd.Series(np.zeros(len(idx), dtype=int), index=idx, name="alarm")
        thr = pd.Series(np.inf, index=idx, name="thr")
        return alarms, thr, float(p_lo)

    gb = float(np.clip(float(guardband), 1e-6, 1.0))
    target_adj = target * gb

    p_lo = float(max(p_lo, 1e-12))
    p_hi = float(max(p_hi, p_lo * 1.01))
    p_grid = np.unique(np.geomspace(p_lo, p_hi, num=40))

    def _metrics_for(p: float) -> tuple[float, float]:
        # rolling, lagged quantile threshold
        thr = s.rolling(int(window), min_periods=int(min_periods)).quantile(1.0 - float(p)).shift(1)
        raw = (s > thr).fillna(False).astype(int).values
        # apply episode policy first (operational alarms)
        if int(cooldown_months) > 0 or int(max_episode_months) > 0:
            raw = apply_episode_policy_array(
                raw,
                cooldown_months=int(cooldown_months),
                max_episode_months=int(max_episode_months),
            ).astype(int)
        # warning state used for event accounting and TIW (merge-gap)
        # IMPORTANT: do not allow merge-gap filling across masked-out months.
        raw_masked = raw.copy()
        raw_masked[~valid] = 0
        w = apply_merge_gap_array(raw_masked, merge_gap_months=int(merge_gap_months)).astype(int)
        n_events = count_events(w, mask=valid)
        months = float(np.sum(valid))
        years = months / _steps_per_year() if months > 0 else 0.0
        ev_per_year = float(n_events / max(years, 1e-9))
        tiw = float(np.mean(w[valid])) if np.any(valid) else float("nan")
        return ev_per_year, tiw

    best_p = float(p_grid[0])
    best_obj = float("inf")

    # Soft objective: fit event rate, with hard penalty on exceeding target*(1+tol)
    for p in p_grid:
        evpy, tiw = _metrics_for(float(p))
        if not np.isfinite(evpy):
            continue
        obj = (evpy - target_adj) ** 2
        if max_time_in_warning is not None and np.isfinite(float(max_time_in_warning)):
            cap = float(max_time_in_warning) * gb
            if np.isfinite(tiw) and tiw > cap:
                obj += 100.0 * (tiw - cap) ** 2
        # hard-ish penalty for overshoot
        if evpy > target_adj * (1.0 + float(event_tol)):
            obj += 1e3 * (evpy - target_adj) ** 2
        if obj < best_obj:
            best_obj = float(obj)
            best_p = float(p)

    # Produce full-series alarms and thresholds with chosen p*
    thr = s.rolling(int(window), min_periods=int(min_periods)).quantile(1.0 - best_p).shift(1)
    raw = (s > thr).fillna(False).astype(int).values
    if int(cooldown_months) > 0 or int(max_episode_months) > 0:
        raw = apply_episode_policy_array(
            raw,
            cooldown_months=int(cooldown_months),
            max_episode_months=int(max_episode_months),
        ).astype(int)
    alarms = pd.Series(raw.astype(int), index=idx, name="alarm")

    return alarms, thr, float(best_p)


def _effective_cooldown(cooldown_months: int, merge_gap_months: int) -> int:
    """Ensure cooldown produces a real episode break under merge-gap accounting.

    If cooldown <= merge_gap, the post-episode gap can be filled by merge-gap, collapsing
    distinct episodes into one and making high event budgets unattainable. We enforce:
        effective_cooldown >= merge_gap + 1  (when cooldown > 0 and merge_gap > 0).
    """
    cd = int(max(cooldown_months, 0))
    mg = int(max(merge_gap_months, 0))
    if cd > 0 and mg > 0 and cd <= mg:
        return mg + 1
    return cd

def dynamic_quantile_threshold(
    scores: pd.Series,
    p_series: pd.Series,
    window: int = 60,
    min_periods: int = 24,
    cooldown_months: int = 0,
    max_episode_months: int = 0,
    p_lo: float = 1e-6,
    p_hi: float = 0.999,
) -> tuple[pd.Series, pd.Series]:
    """Budgeted alerting via rolling quantile threshold with optional episode policy.

    scores: scalar score series (higher = more anomalous)
    p_series: per-time tail probability p*(t). Larger p => lower threshold => more alarms.
    window/min_periods: rolling quantile window on score history (strictly lagged by 1 month)
    cooldown_months/max_episode_months: episode policy applied to the raw alarm stream
    p_lo/p_hi: numeric clamps for p*(t)
    """
    idx = scores.index
    p = p_series.reindex(idx).ffill().bfill()
    thr = pd.Series(np.nan, index=idx, name="thr")
    alarms = np.zeros(len(idx), dtype=int)

    for t in range(len(idx)):
        pt = float(p.iloc[t])
        pt = float(np.clip(pt, float(p_lo), float(p_hi)))
        # lagged rolling quantile (no lookahead)
        start = max(0, t - int(window))
        hist = scores.iloc[start:t]
        hist = hist[np.isfinite(hist.values)]
        if len(hist) >= int(min_periods):
            thr_t = float(np.nanquantile(hist.values, 1.0 - pt))
            thr.iloc[t] = thr_t
            alarms[t] = int(float(scores.iloc[t]) > thr_t) if np.isfinite(scores.iloc[t]) else 0
        else:
            alarms[t] = 0

    # Apply episode policy (consistent with Task-B evaluation).
    if int(cooldown_months) > 0 or int(max_episode_months) > 0:
        alarms = apply_episode_policy_array(
            alarms,
            cooldown_months=int(cooldown_months),
            max_episode_months=int(max_episode_months),
        ).astype(int)

    return pd.Series(alarms, index=idx, name="alarm"), thr

def apply_episode_policy(
    alarm: pd.Series,
    cooldown_months: int = 0,
    max_episode_months: int = 0,
) -> pd.Series:
    """Apply causal episode-management to a precomputed binary alarm stream.

    This mirrors the episode logic used inside `dynamic_quantile_threshold`, but operates
    on an already-thresholded alarm stream (e.g., chart-based CUSUM alarms).

    The policy is purely causal (no lookahead): it can be applied consistently to *any*
    detector output so that downstream evaluation (episode starts / alert-event counting)
    is comparable across detectors.

    Args:
      alarm: binary alarm series (0/1)
      cooldown_months: after an alert episode ends, suppress new alarms for this many months
      max_episode_months: cap the maximum length of a continuous alert episode

    Returns:
      managed_alarm: binary alarm series after applying the episode policy
    """
    a = alarm.to_numpy(dtype=int)
    n = int(a.shape[0])

    cd = int(max(cooldown_months, 0))
    max_ep = int(max(max_episode_months, 0))
    cooldown_left = 0
    in_episode = False
    ep_len = 0

    out = np.zeros(n, dtype=int)
    for t in range(n):
        raw = int(a[t] > 0)

        # cooldown suppression
        if cooldown_left > 0:
            raw = 0
            cooldown_left -= 1

        if raw == 1:
            if not in_episode:
                in_episode = True
                ep_len = 1
            else:
                ep_len += 1
                if max_ep > 0 and ep_len > max_ep:
                    # force-clear and start cooldown
                    raw = 0
                    in_episode = False
                    ep_len = 0
                    if cd > 0:
                        cooldown_left = max(cooldown_left, cd)
        else:
            if in_episode:
                # natural episode end
                in_episode = False
                ep_len = 0
                if cd > 0:
                    cooldown_left = max(cooldown_left, cd)

        out[t] = int(raw)

    return pd.Series(out, index=alarm.index, name=alarm.name)


def apply_episode_policy_array(
    alarm: np.ndarray,
    cooldown_months: int = 0,
    max_episode_months: int = 0,
) -> np.ndarray:
    """Array version of apply_episode_policy (for fast inner-loop use)."""
    a = np.asarray(alarm, dtype=int)
    n = int(a.shape[0])
    cd = int(max(cooldown_months, 0))
    max_ep = int(max(max_episode_months, 0))
    cooldown_left = 0
    in_episode = False
    ep_len = 0
    out = np.zeros(n, dtype=np.int32)
    for t in range(n):
        raw = 1 if a[t] > 0 else 0
        if cooldown_left > 0:
            raw = 0
            cooldown_left -= 1
        if raw == 1:
            if not in_episode:
                in_episode = True
                ep_len = 1
            else:
                ep_len += 1
                if max_ep > 0 and ep_len > max_ep:
                    raw = 0
                    in_episode = False
                    ep_len = 0
                    if cd > 0:
                        cooldown_left = max(cooldown_left, cd)
        else:
            if in_episode:
                in_episode = False
                ep_len = 0
                if cd > 0:
                    cooldown_left = max(cooldown_left, cd)
        out[t] = raw
    return out


def apply_event_budgeted_threshold_online(
    scores: pd.Series,
    target_events_per_year: float,
    window: int = 60,
    min_periods: int = 24,
    merge_gap_months: int = 1,
    cooldown_months: int = 1,
    max_episode_months: int = 6,
    max_time_in_warning: Optional[float] = None,
    safety_eps: float = 0.10,
    eta: float = 0.30,
    setpoint_tol: float = 0.05,
    update_every: int = 1,
    min_control_months: int = 12,
    allow_loosen: bool = True,
    init_p: Optional[float] = None,
    guardband: float = 1.00,
    control_window: Optional[int] = None,
    p_lo: Optional[float] = None,
    p_hi: Optional[float] = None,
    max_up: float = 1.25,
    max_down: float = 0.60,
    eval_mask: Optional[np.ndarray] = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Online controller for event budgets with TIW treated as safety (lexicographic).

    Key contract points:
      - Uses the SAME dynamic-quantile thresholding primitive as the offline pipeline
        (quantile window + min_periods).
      - TIW is handled as a safety constraint: if TIW violates, we tighten regardless
        of the event budget error.

    Args:
      scores: score stream (higher => more anomalous)
      target_events_per_year: target alert episodes per year
      window: quantile recalibration window (months)
      min_periods: minimum months required before thresholds are defined
      control_window: window (months) for burden measurement used by the controller
      init_p: initial p* (tail probability). If None, defaults to 0.02.
      guardband: multiplicative guardband applied to init_p (>=1 loosens, <=1 tightens)
      p_lo/p_hi: explicit controller p* range. If None, uses conservative heuristics.

    Returns:
      alarms: episode-managed alarm state (0/1)
      thr: threshold per month (NaN before defined)
      p*: p* per month (NaN before defined)
    """
    x = scores.values.astype(float)
    n = len(x)

    if eval_mask is None:
        eval_mask = np.ones(n, dtype=bool)
    else:
        eval_mask = np.asarray(eval_mask, dtype=bool)

    window = int(window)
    min_periods = int(min_periods)

    if control_window is None:
        control_window = window
    control_window = int(control_window)

    # Initialize p
    if init_p is None:
        init_p = 0.02
    p0 = float(init_p) * float(guardband)

    if p_lo is None or p_hi is None:
        # Heuristic bounds, but allow a wide top-end so high budgets can be feasible.
        _p_lo = max(1e-6, float(np.clip(p0 / 50.0, 1e-6, 0.05)))
        _p_hi = max(0.50, float(np.clip(p0 * 20.0, 0.05, 0.95)))
        if p_lo is None:
            p_lo = _p_lo
        if p_hi is None:
            p_hi = _p_hi
    p_lo = float(p_lo)
    p_hi = float(p_hi)
    p = float(np.clip(p0, p_lo, p_hi))

    alarms = np.zeros(n, dtype=int)
    thr = np.full(n, np.nan, dtype=float)
    p_star = np.full(n, np.nan, dtype=float)
    def _burden_metrics(a_prefix: np.ndarray, window_len: int) -> tuple[float, float, int, float]:
        """Estimate achieved events/year and TIW using policy-consistent rules.
    
        We apply the same episode rules (cooldown, max-episode, merge-gap) and then
        compute the metrics on the *tail* window. Starts are counted on the full
        processed prefix so ongoing episodes crossing the window boundary are not
        double-counted as new alerts.
        """
        a_prefix = a_prefix.astype(int)
        if len(a_prefix) == 0:
            return 0.0, 0.0, 0, 1.0
        # Episode management (causal)
        w_full = apply_episode_policy_array(a_prefix, cooldown_months=int(cooldown_months), max_episode_months=max_episode_months)
        # Merge-gap and episode starts (policy-consistent)
        starts_full, w_m_full = episode_starts(w_full, merge_gap_months=int(merge_gap_months))
        window_len = int(window_len) if window_len is not None else len(w_full)
        window_len = max(1, min(window_len, len(w_full)))
        starts_tail = starts_full[-window_len:]
        w_m_tail = w_m_full[-window_len:]
        tiw = float(np.mean(w_m_tail)) if len(w_m_tail) else 0.0
        n_events = int(np.sum(starts_tail))
        years = (len(w_m_tail) / _steps_per_year()) if len(w_m_tail) else 1.0
        events_per_year = float(n_events / years)
        return events_per_year, tiw, n_events, years

    for i in range(n):
        # Quantile threshold from trailing window (STRICTLY lagged: excludes current month)
        j0 = max(0, i - window)
        hist = x[j0:i]
        hist = hist[np.isfinite(hist)]
        if len(hist) < min_periods:
            continue

        q = float(np.clip(1.0 - p, 1e-6, 1.0 - 1e-6))
        thr[i] = float(np.quantile(hist, q))
        alarms[i] = int(np.isfinite(x[i]) and (x[i] > thr[i]))
        p_star[i] = p

        if not eval_mask[i]:
            continue

        if update_every > 1 and ((i + 1) % int(update_every) != 0):
            continue
        a_in = alarms[:i + 1]
        if len(a_in) < int(min_control_months):
            continue
        tail_len = min(int(control_window), len(a_in))
        achieved_ev, achieved_tiw, achieved_cnt, achieved_years = _burden_metrics(a_in, tail_len)

        # Lexicographic safety: TIW violation forces tightening (no loosening).
        tiw_violation = False
        if max_time_in_warning is not None:
            tiw_violation = achieved_tiw > float(max_time_in_warning) * (1.0 + float(safety_eps))

        if tiw_violation:
            err = math.log(max(float(max_time_in_warning), 1e-12) / max(achieved_tiw, 1e-12))
            can_loosen = False
        else:
            _tol = 0.0 if (setpoint_tol is None) else float(setpoint_tol)
            # Count-based setpoint tracking: compare achieved episode *counts* in the control window
            # to the nearest feasible integer target implied by target_events_per_year.
            target_cnt = int(round(float(target_events_per_year) * float(achieved_years)))
            if target_cnt < 0:
                target_cnt = 0
            # Convert tolerance to a count band (at least +/-1 when target_cnt>0).
            band = 0
            if _tol > 0:
                band = int(max(1, math.ceil(_tol * max(target_cnt, 1))))
            if abs(int(achieved_cnt) - int(target_cnt)) <= band:
                continue
            err = math.log(max(float(target_cnt), 1e-9) / max(float(achieved_cnt), 1e-9))
            # Only loosen if TIW is comfortably below the cap.
            tiw_safe_to_loosen = True
            if max_time_in_warning is not None:
                tiw_safe_to_loosen = achieved_tiw < float(max_time_in_warning) * (1.0 - float(safety_eps))
            can_loosen = bool(allow_loosen) and (err > 0) and tiw_safe_to_loosen

        if (err > 0) and (not can_loosen):
            continue

        p_target = float(p * math.exp(float(eta) * float(err)))
        p_target = float(np.clip(p_target, p_lo, p_hi))

        ratio = p_target / max(p, 1e-12)
        ratio = float(np.clip(ratio, float(max_down), float(max_up)))
        p = float(np.clip(p * ratio, p_lo, p_hi))

    alarms = apply_episode_policy_array(
        alarms, cooldown_months=int(cooldown_months), max_episode_months=int(max_episode_months)
    )
    if int(merge_gap_months) > 0:
        alarms = apply_merge_gap_array(alarms.astype(int), int(merge_gap_months))
    alarms_ser = pd.Series(alarms, index=scores.index, name="alarm")
    thr_ser = pd.Series(thr, index=scores.index, name="thr")
    pser = pd.Series(p_star, index=scores.index, name="p_star")
    return alarms_ser, thr_ser, pser


# ``task_B`` is retained as a compatibility alias. This function computes per-trial quantities derived from the
# realized warning stream (after merge-gap), including event semantics and non-incident
# Burden accounting used for operating-point and regime summaries.
def eval_task_B_integrity(
    scores: pd.Series,
    alarms: pd.Series,
    incident: Window,
    eval_mask: np.ndarray,
    merge_gap_months: int = 0,
    sla_months: int = 1,
) -> dict:
    """Compute paper-facing per-trial metrics from the realized warning stream.

    Inputs are the score stream and a binary alarm/crossing stream for a single configuration.
    The function applies merge-gap bridging to construct the operator-facing warning stream,
    then returns:
      * incident-window event semantics (success, delay, Timely@K, AIW), and
      * non-incident burden quantities (episode/event counts and time-in-warning) used for
        matched-burden acceptance and TIW evaluation.

    The incident window is excluded from burden accounting to avoid mechanically inflating
    burden due to the injected event.
    """
    idx = scores.index
    inc_m = mask_window(idx, incident)
    non_inc_eval = eval_mask & (~inc_m)

# Event semantics (paper-facing):
#   - any-warning success: the incident window contains at least one warning (w_t = 1).
#   - episode-start success: the incident window contains at least one episode start (u_t = 1); used by NC-2.
#   - delay: first-warning delay within the incident window, censored at the incident duration.
    starts, a_merged = episode_starts(alarms.values.astype(int), merge_gap_months=int(merge_gap_months))

    # Identify whether we are already alarming at incident start.
    inc_idx = np.where(inc_m)[0]
    if inc_idx.size == 0:
        already_in_warning_at_start = False
    else:
        t0 = int(inc_idx[0])
        already_in_warning_at_start = bool(a_merged[t0] == 1 and (not starts[t0]))



    # Incident duration in contract units (months in calendar mode; steps in step mode)
    inc_duration_months = int(_time_diff_units(idx, incident.start, incident.end)) + 1

    # Any-warning overlap with the incident (includes preexisting warning).
    anywarn_m = (a_merged.astype(int) == 1) & inc_m
    anywarn_success = bool(np.any(anywarn_m))
    if anywarn_success:
        first_warn_idx = int(np.where(anywarn_m)[0][0])
        first_warn_date = idx[first_warn_idx]
        event_delay_months = float(_time_diff_units(idx, incident.start, pd.Timestamp(first_warn_date)))
    else:
        event_delay_months = float('nan')

    # Censor to full incident duration if not detected by overlap.
    event_delay_months_cens = float(event_delay_months) if (anywarn_success and np.isfinite(event_delay_months)) else float(inc_duration_months)
    event_timely_sla = int(anywarn_success and (event_delay_months <= sla_months))

    # Back-compat aliases (kept to avoid breaking older analysis code).
    event_anywarn_delay_months = float(event_delay_months) if np.isfinite(event_delay_months) else float('nan')
    event_anywarn_delay_months_cens = float(event_delay_months_cens)
    event_anywarn_timely_sla = int(event_timely_sla)
    first_start = first_alarm_date(idx, (starts & inc_m))
    triggered = first_start is not None
    time_to_trigger = _time_diff_units(idx, incident.start, pd.Timestamp(first_start)) if triggered else np.nan
    # Censored delay: if no trigger during incident window, charge full incident duration.
    time_to_trigger_cens = float(time_to_trigger) if np.isfinite(time_to_trigger) else float(inc_duration_months)
    # Burden on benign eval months is computed on the operator-facing warning state,
    # i.e., after merge-gap filling (a_merged).
    fa_count = int(np.sum(a_merged.astype(bool) & non_inc_eval))
    years = max((np.sum(non_inc_eval) / _steps_per_year()), 1e-6)
    fa_per_year = fa_count / years
    time_in_warning = float(np.mean(a_merged.astype(bool)[non_inc_eval])) if np.sum(non_inc_eval) > 0 else 0.0

    fa_events = count_alert_events(
        alarms.values.astype(int),
        mask=non_inc_eval,
        merge_gap_months=int(merge_gap_months),
    )
    fa_events_per_year = fa_events / years

    return {
        # Backwards-compatible keys (used by existing runners/summaries)
        "detected": float(triggered),
        "delay_months": float(time_to_trigger) if np.isfinite(time_to_trigger) else np.nan,
        "delay_months_cens": float(time_to_trigger_cens),
        # Explicit operational metrics
        "triggered": float(triggered),
        # Primary "one alert per incident" success = ANY overlap.
        "event_success": int(anywarn_success),
        "event_sla_months": int(sla_months),
        # Primary SLA uses overlap-delay.
        "event_timely_sla": int(event_timely_sla),
        "event_delay_months": float(event_delay_months) if np.isfinite(event_delay_months) else np.nan,
        "event_delay_months_cens": float(event_delay_months_cens),
        # Secondary (episode-start) timing for diagnostics.
        "event_time_to_trigger_months": float(time_to_trigger) if np.isfinite(time_to_trigger) else np.nan,
        "event_time_to_trigger_months_cens": float(time_to_trigger_cens),
        "event_anywarn_success": int(anywarn_success),
        "event_anywarn_delay_months": float(event_anywarn_delay_months) if (event_anywarn_delay_months == event_anywarn_delay_months) else float('nan'),
        "event_anywarn_delay_months_cens": float(event_anywarn_delay_months_cens),
        "event_anywarn_timely_sla": int(event_anywarn_timely_sla),
        "time_to_trigger_months": float(time_to_trigger) if np.isfinite(time_to_trigger) else np.nan,
        "time_to_trigger_months_cens": float(time_to_trigger_cens),
        "incident_duration_months": float(inc_duration_months),
        "already_in_warning_at_start": float(already_in_warning_at_start),
        "false_alarms_per_year": float(fa_per_year),
        "false_alert_months_per_year": float(fa_per_year),
        "false_alert_events_per_year": float(fa_events_per_year),
        "time_in_warning": float(time_in_warning),
        # This is the first *episode start* (not merely any alarm month)
        "first_episode_start_in_incident": str(first_start.date()) if first_start is not None else None,
        # Back-compat: older code may look for this key name
        "first_alarm_in_incident": str(first_start.date()) if first_start is not None else None,
        # Count-based burden fields used for feasibility and matched-burden acceptance.
        "false_alert_events_count": int(fa_events),
        "false_alert_months_count": int(fa_count),
        "false_alert_years": float(years),
        "false_alert_eval_months": int(np.sum(non_inc_eval)),

    }


# Compatibility note: ``task_A`` is retained for reproducibility with older experiments.
def eval_task_A_real(scores: pd.Series, alarms: pd.Series, event: Window, eval_mask: np.ndarray, early_months: int = 12) -> dict:
    """Compatibility event-window evaluation helper.

    This helper evaluates detection within a window around the event (including an optional
    look-back period) and reports simple burden summaries. The main paper pipeline uses
    ``eval_task_B_integrity`` for incident-window event semantics and burden accounting.
    """
    idx = scores.index
    ev_m = mask_window(idx, event)
    non_ev_eval = eval_mask & (~ev_m)

    early_start = _shift_back(idx, event.start, int(early_months))
    det_m = (idx >= early_start) & (idx <= event.end)

    first = first_alarm_date(idx, (alarms.values.astype(bool) & det_m))
    detected = first is not None
    delay = _time_diff_units(idx, event.start, pd.Timestamp(first)) if detected else np.nan

    fa_count = int(np.sum(alarms.values.astype(bool) & non_ev_eval))
    years = max((np.sum(non_ev_eval) / _steps_per_year()), 1e-6)
    fa_per_year = fa_count / years
    time_in_warning = float(np.mean(alarms.values.astype(bool)[non_ev_eval])) if np.sum(non_ev_eval) > 0 else 0.0
    fa_events = count_alert_events(alarms.values.astype(int), mask=non_ev_eval, merge_gap_months=0)
    fa_events_per_year = fa_events / years

    return {
        "false_alert_events_per_year": float(fa_events_per_year),
        "false_alert_months_per_year": float(fa_per_year),

        "detected": float(detected),
        "delay_months": float(delay) if np.isfinite(delay) else np.nan,
        "false_alarms_per_year": float(fa_per_year),
        "false_alert_months_per_year": float(fa_per_year),
        "time_in_warning": float(time_in_warning),
        "first_alarm_in_det_window": str(first.date()) if first is not None else None,
    }


def apply_budgeted_threshold(scores: pd.Series, fa_per_year: float, window: int = 60, min_periods: int = 24) -> tuple[pd.Series, pd.Series]:
    """
    Rolling quantile thresholding to approximately maintain a target false alarm rate under nonstationarity.

    Returns:
      alarms: binary Series
      thr_series: per-timestamp threshold (NaN for warm-up)
    """
    p = min(max(fa_per_year / _steps_per_year(), 1e-6), 0.25)
    # rolling quantile of *past* values (shift by 1)
    thr = scores.rolling(window=window, min_periods=min_periods).quantile(1.0 - p).shift(1)
    alarms = (scores > thr).fillna(False).astype(int)
    return alarms.rename("alarm"), thr.rename("thr")


def apply_budgeted_topk(scores: pd.Series, fa_per_year: float, eval_mask: np.ndarray, block_months: int | None = None) -> pd.Series:
    """
    Deterministic budget enforcement via top-k selection in contiguous blocks.

    Idea: if the operator can handle `fa_per_year` alerts per year on monthly data, we can enforce the budget
    exactly by selecting the top-k score months within each block.

    - If fa_per_year >= 1: use 12-month blocks (calendar-like); k = round(fa_per_year)
    - If fa_per_year < 1: use block_months = round(_steps_per_year() / fa_per_year) and k = 1

    Returns:
      alarms: binary Series with alarms only within eval_mask.
    """
    if block_months is None:
        if fa_per_year >= 1.0:
            block_months = int(round(_steps_per_year()))
            k = int(round(fa_per_year))
        else:
            block_months = int(round(_steps_per_year() / fa_per_year))
            k = 1
    else:
        k = max(1, int(round(fa_per_year * block_months / _steps_per_year())))

    idx = scores.index
    alarms = np.zeros(len(idx), dtype=int)

    # work only on eval months
    eval_idx = np.where(eval_mask)[0]
    if len(eval_idx) == 0:
        return pd.Series(alarms, index=idx, name="alarm")

    # create contiguous blocks in index order over eval months
    start_pos = eval_idx[0]
    pos = start_pos
    while pos <= eval_idx[-1]:
        block_mask = np.zeros(len(idx), dtype=bool)
        # block is [pos, pos+block_months)
        block_slice = slice(pos, min(pos + block_months, len(idx)))
        block_mask[block_slice] = True
        block_mask &= eval_mask

        block_inds = np.where(block_mask)[0]
        if len(block_inds) > 0:
            block_scores = scores.values[block_inds]
            # choose top-k finite
            finite = np.isfinite(block_scores)
            bi = block_inds[finite]
            bs = block_scores[finite]
            if len(bi) > 0:
                topk = min(k, len(bi))
                order = np.argsort(bs)[::-1][:topk]
                alarms[bi[order]] = 1
        pos += block_months

    return pd.Series(alarms, index=idx, name="alarm")


# =========================
# Rolling quantile alarm thresholding (used by adaptive threshold schedules)
# =========================

def rolling_quantile_alarms(
    scores: pd.Series,
    window_months: int,
    fa_per_year: float,
    warmup_months: int | None = None,
    calib_mask: np.ndarray | None = None,
) -> pd.Series:
    """Generate alarms using a rolling quantile threshold over past `window_months`.

    - For each t, threshold is computed from scores in (t-window_months, t-1) intersected with calib_mask if provided.
    - This is more robust to nonstationarity than a single global threshold.
    - warmup_months: months at start to force alarms=0; default=window_months.
    """
    p = min(max(fa_per_year / _steps_per_year(), 1e-6), 0.25)
    idx = scores.index
    n = len(scores)
    w = int(window_months)
    warm = int(warmup_months if warmup_months is not None else w)
    svals = scores.values
    alarms = np.zeros((n,), dtype=np.int32)
    for t in range(n):
        if t < warm:
            continue
        lo = max(0, t - w)
        hist = np.arange(lo, t)
        if calib_mask is not None:
            hist = hist[calib_mask[hist]]
        vals = svals[hist]
        vals = vals[np.isfinite(vals)]
        if len(vals) < max(24, w//2):
            continue
        thr = float(np.quantile(vals, 1.0 - p))
        if np.isfinite(svals[t]) and svals[t] > thr:
            alarms[t] = 1
    return pd.Series(alarms, index=idx, name="alarm")
