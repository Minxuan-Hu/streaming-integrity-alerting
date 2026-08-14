
"""Main experiment runner for the Streaming Integrity Monitoring operational evaluation framework.

The module name is retained for backward compatibility.

The evaluation object is the alert stream induced by:
  scores -> threshold crossings -> warning stream -> episodes
under an operational policy (thresholding + episode semantics: merge-gap, cooldown, max-episode-length)
and burden constraints (episode budget and optional TIW cap).

The runner produces:
  - counterfactual ops divergence (clean vs corrupted) in incident and post windows,
  - threat-surface event-level timeliness vs intensity at matched burden with explicit coverage/infeasibility,
  - regime-sliced compliance (pass/fail) and overshoot diagnostics,
  - monotonicity diagnostics for threat-surface curves,
  - label-free benign drift compliance stress tests, and
  - a uniform-placement overlap baseline on clean warning streams (baseline only; separate from NC-2).

Most outputs are exported as paper-facing CSV artifacts; see docs/artifacts.md.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import hashlib
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from standardize import robust_standardize_df
from detectors import (
    fit_coherence_ridge,
    score_coherence_resid_t2,
    score_coherence_resid_energy,
    score_ewma_from_base,
    score_cusum_from_base,
    calibrate_cusum_h_from_base,
    calibrate_cusum_h_occupancy_from_base,
    calibrate_cusum_h_operational_occupancy_from_base,
    cusum_chart_train_stats_from_base,
    cusum_chart_alarms_from_base,
    score_max_abs,
    fit_cov,
    score_hotelling_t2,
    fit_pca,
    score_subspace,
    fit_staleness_params,
    score_staleness_runlength,
    score_factor_cov_change,
    score_factor_cov_delta,
    score_factor_cov_lrt,
    score_factor_cov_lrt_delta,
    fuse_scores_max_robust_train,
    fuse_scores_fisher_train,
)
from incidents import Incident, inject, apply_benign_drift, apply_noise_injection
import evaluation as eval_mod
from evaluation import (
    Window,
    apply_budgeted_topk,
    apply_budgeted_threshold,
    apply_event_budgeted_threshold,
    apply_event_budgeted_threshold_online,
    dynamic_quantile_threshold,
    apply_episode_policy_array,
    apply_merge_gap_array,
    count_alert_events,
    count_events,
    eval_task_B_integrity,
)


def effective_cooldown(cooldown_months: int, merge_gap_months: int) -> int:
    """Ensure cooldown is long enough to create a *real* break under merge-gap accounting.

    If cooldown <= merge_gap, the cooldown gap can be filled by merge-gap, collapsing
    distinct episodes into one and causing chronic budget undershoot at higher budgets.
    """
    cd = int(max(cooldown_months, 0))
    mg = int(max(merge_gap_months, 0))
    if cd > 0 and mg > 0 and cd <= mg:
        return mg + 1
    return cd

def make_windows(
    dt: pd.DatetimeIndex,
    rng: np.random.Generator,
    n: int,
    duration: int,
    earliest: pd.Timestamp,
) -> list[Window]:
    """Sample incident windows of fixed length.

    In calendar_month mode, windows are aligned on calendar month boundaries.
    In step mode, each row is treated as one time step; duration is in steps.
    """
    dt = pd.DatetimeIndex(dt)
    if len(dt) == 0:
        return []

    duration = int(duration)
    if duration <= 0:
        raise ValueError("duration must be positive")

    if getattr(eval_mod, "TIME_MODE", "calendar_month") == "step":
        # Start positions such that [pos, pos+duration-1] is valid.
        e = pd.Timestamp(earliest)
        start0 = int(dt.searchsorted(e, side="left"))
        start0 = min(max(start0, 0), max(0, len(dt) - duration))
        feasible = np.arange(start0, max(start0 + 1, len(dt) - duration + 1))
        starts = rng.choice(feasible, size=int(n), replace=True) if len(feasible) > 0 else np.array([start0] * int(n))
        wins: list[Window] = []
        for pos in starts:
            pos = int(pos)
            s = pd.Timestamp(dt[pos])
            e = pd.Timestamp(dt[pos + duration - 1])
            wins.append(Window(name="incident", start=s, end=e))
        return wins

    # calendar_month mode: choose only starts where the full window fits
    earliest_ts = pd.Timestamp(earliest)
    feasible_starts: list[pd.Timestamp] = []
    for s in pd.DatetimeIndex(dt):
        if s < earliest_ts:
            continue
        e = (pd.Timestamp(s).to_period("M") + (duration - 1)).to_timestamp()
        if e <= dt.max():
            feasible_starts.append(pd.Timestamp(s))
    if len(feasible_starts) == 0:
        raise ValueError("No feasible incident windows after earliest.")
    starts = rng.choice(np.array(feasible_starts, dtype="datetime64[ns]"), size=int(n), replace=True)
    wins: list[Window] = []
    for s in starts:
        s = pd.Timestamp(s)
        e = (s.to_period("M") + (duration - 1)).to_timestamp()
        wins.append(Window(name="incident", start=s, end=e))
    return wins

def effect_size(scores: pd.Series, inc: Window, eval_mask: np.ndarray) -> float:
    idx = scores.index
    inc_m = (idx >= inc.start) & (idx <= inc.end)
    non_inc = eval_mask & (~inc_m)
    base = scores.values[non_inc]
    base = base[np.isfinite(base)]
    incv = scores.values[inc_m]
    incv = incv[np.isfinite(incv)]
    if base.size < 10 or incv.size < 1:
        return float("nan")
    med = float(np.nanmedian(base))
    q1 = float(np.nanquantile(base, 0.25))
    q3 = float(np.nanquantile(base, 0.75))
    iqr = max(q3 - q1, 1e-6)
    return float((np.nanmedian(incv) - med) / iqr)


def _window_mask(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    return (index >= start) & (index <= end)


def intensity_proxy_from_z(kind: str, Z_clean: pd.DataFrame, Z_att: pd.DataFrame, inc: Window, target_cols: list[int], dur_months: int) -> tuple[float, str]:
    """Compute a transparent, incident-consistent "attack intensity" proxy.

    This is used for stealth/intensity curves (diagnostic only). It is intentionally simple and
    tied to the attacked subset.

    - freeze: 1 - (std_win / std_base) averaged over attacked cols (staleness/variance suppression)
    - corr_mix: Frobenius norm of corr-change between win and pre-win baseline (attacked cols)
    """
    idx = Z_clean.index
    m_win = _window_mask(idx, inc.start, inc.end)
    if not np.any(m_win) or len(target_cols) < 2:
        return (float("nan"), "na")

    # baseline window immediately preceding the incident (same duration)
    if getattr(eval_mod, "TIME_MODE", "calendar_month") == "step":
        s_pos = int(idx.searchsorted(inc.start, side="left"))
        base_end_pos = s_pos - 1
        base_start_pos = s_pos - int(max(dur_months, 1))
        if base_end_pos >= 0:
            base_start = pd.Timestamp(idx[max(0, base_start_pos)])
            base_end = pd.Timestamp(idx[base_end_pos])
            m_base = _window_mask(idx, base_start, base_end)
        else:
            m_base = np.zeros(len(idx), dtype=bool)
    else:
        base_end = inc.start - pd.DateOffset(months=1)
        base_start = (inc.start.to_period("M") - dur_months).to_timestamp()
        m_base = _window_mask(idx, base_start, base_end)

    if not np.any(m_base):
        m_base = (~m_win)

    Xc_win = np.asarray(Z_clean.values[np.ix_(m_win, target_cols)], dtype=float)
    Xa_win = np.asarray(Z_att.values[np.ix_(m_win, target_cols)], dtype=float)
    Xc_base = np.asarray(Z_clean.values[np.ix_(m_base, target_cols)], dtype=float)

    if kind == "freeze":
        # variance suppression proxy
        std_base = np.nanstd(Xc_base, axis=0)
        std_win = np.nanstd(Xa_win, axis=0)
        ratio = np.nanmedian(std_win / (std_base + 1e-6))
        proxy = float(np.clip(1.0 - ratio, 0.0, 1.0))
        return proxy, "1-std_ratio"

    if kind == "corr_mix":
        # correlation-change proxy (stable estimator; guard against tiny windows)
        if Xc_base.shape[0] < 6 or Xa_win.shape[0] < 6:
            return (float("nan"), "corr_fro")
        C0 = np.corrcoef(Xc_base, rowvar=False)
        C1 = np.corrcoef(Xa_win, rowvar=False)
        if not (np.all(np.isfinite(C0)) and np.all(np.isfinite(C1))):
            return (float("nan"), "corr_fro")
        fro = float(np.linalg.norm(C1 - C0, ord="fro"))
        return fro, "corr_fro"

    if kind in ("drift", "benign_drift", "gradual_bias"):
        # mean-shift proxy (benign drift): standardized absolute shift of window mean vs baseline
        mu0 = np.nanmean(Xc_base, axis=0)
        mu1 = np.nanmean(Xa_win, axis=0)
        sd0 = np.nanstd(Xc_base, axis=0) + 1e-6
        zshift = float(np.nanmedian(np.abs((mu1 - mu0) / sd0)))
        return zshift, "mean_zshift"

    return (float("nan"), "na")


def cad_metrics_from_alarms(
    alarms_att: pd.Series,
    alarms_clean: pd.Series,
    inc: Window,
    merge_gap_months: int,
    post_months: int,
) -> dict:
    """Compute counterfactual alert divergence (CAD) and ΔTIW under identical policy.

    CAD is computed on merged-gap warning states w_t (operator-facing "in-warning" status).

    Returns:
      cad_win, cad_post, cad_all
      dtiw_win, dtiw_post, dtiw_all
    """
    idx = alarms_att.index
    a_att = np.asarray(alarms_att.values, dtype=int)
    a_cln = np.asarray(alarms_clean.values, dtype=int)
    w_att = apply_merge_gap_array(a_att, merge_gap_months=int(merge_gap_months)).astype(int)
    w_cln = apply_merge_gap_array(a_cln, merge_gap_months=int(merge_gap_months)).astype(int)

    m_win = _window_mask(idx, inc.start, inc.end)
    # post window: (end+1) ... (end+H)
    if getattr(eval_mod, "TIME_MODE", "calendar_month") == "step":
        end_pos = int(idx.searchsorted(inc.end, side="left"))
        post_h = int(max(post_months, 0))
        if post_h <= 0 or end_pos + 1 >= len(idx):
            m_post = np.zeros(len(idx), dtype=bool)
        else:
            ps = end_pos + 1
            pe = min(len(idx) - 1, end_pos + post_h)
            post_start = pd.Timestamp(idx[ps])
            post_end = pd.Timestamp(idx[pe])
            m_post = _window_mask(idx, post_start, post_end)
    else:
        post_start = (inc.end.to_period("M") + 1).to_timestamp()
        post_end = (inc.end.to_period("M") + int(max(post_months, 0))).to_timestamp()
        m_post = _window_mask(idx, post_start, post_end) if post_months > 0 else np.zeros(len(idx), dtype=bool)
    m_all = np.ones(len(idx), dtype=bool)

    def _cad(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.mean((w_att[mask] != w_cln[mask]).astype(float)))

    def _dtiw(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.mean(w_att[mask]) - np.mean(w_cln[mask]))

    return {
        "cad_win": _cad(m_win),
        "cad_post": _cad(m_post),
        "cad_all": _cad(m_all),
        "dtiw_win": _dtiw(m_win),
        "dtiw_post": _dtiw(m_post),
        "dtiw_all": _dtiw(m_all),
    }

def summarize(ms: list[dict]) -> dict:
    """Summarize a list of per-trial metric dicts into an operating-point row.

    Conventions:
      - "median_delay" is *censored* delay (misses charged full incident duration).
      - Clean achieved burden is reported via mean_fae_null / mean_tiw_null.
      - CAD / ΔTIW are reported as window-level means (and |ΔTIW|).
    """
    n_trials = len(ms)

    det = np.array([m.get("detected", np.nan) for m in ms], dtype=float)
    det0 = np.array([m.get("detected_null", np.nan) for m in ms], dtype=float)

    # Delay semantics:
    #   - delay_months (uncensored): NaN when not triggered in-window
    #   - delay_months_cens (censored): full incident duration when not triggered
    delays_uncens = np.array([m.get("delay_months", np.nan) for m in ms], dtype=float)
    delays_cens = np.array([m.get("delay_months_cens", m.get("delay_months", np.nan)) for m in ms], dtype=float)

    delay0_uncens = np.array([m.get("delay_months_null", np.nan) for m in ms], dtype=float)
    delay0_cens = np.array([m.get("delay_months_null_cens", m.get("delay_months_null", np.nan)) for m in ms], dtype=float)

    fa_month = np.array([m.get("false_alarms_per_year", np.nan) for m in ms], dtype=float)
    fa_evt = np.array([m.get("false_alert_events_per_year", np.nan) for m in ms], dtype=float)
    tiw = np.array([m.get("time_in_warning", np.nan) for m in ms], dtype=float)
    eff = np.array([m.get("effect_size", np.nan) for m in ms], dtype=float)
    pstar = np.array([m.get("p_star", np.nan) for m in ms], dtype=float)

    # Clean achieved burden (counterfactual)
    fae0 = np.array([m.get("fae_null", np.nan) for m in ms], dtype=float)
    tiw0 = np.array([m.get("tiw_null", np.nan) for m in ms], dtype=float)

    # CAD / ΔTIW probes
    cad_win = np.array([m.get("cad_win", np.nan) for m in ms], dtype=float)
    cad_post = np.array([m.get("cad_post", np.nan) for m in ms], dtype=float)
    dtiw_win = np.array([m.get("dtiw_win", np.nan) for m in ms], dtype=float)
    dtiw_post = np.array([m.get("dtiw_post", np.nan) for m in ms], dtype=float)
    abs_dtiw_win = np.abs(dtiw_win)

    detect_rate = float(np.nanmean(det)) if np.any(np.isfinite(det)) else float("nan")
    detect_rate_null = float(np.nanmean(det0)) if np.any(np.isfinite(det0)) else float("nan")
    uplift_detect = float(detect_rate - detect_rate_null) if np.isfinite(detect_rate_null) else float("nan")

    # Event-level success + timeliness SLA (one alert per incident).
    ev_success = np.array([m.get("event_success", np.nan) for m in ms], dtype=float)
    ev_timely = np.array([m.get("event_timely_sla", np.nan) for m in ms], dtype=float)
    ev_sla_months = ms[0].get("event_sla_months", np.nan) if ms else float("nan")
    ev_success_rate = float(np.nanmean(ev_success)) if np.any(np.isfinite(ev_success)) else float("nan")
    ev_timely_rate = float(np.nanmean(ev_timely)) if np.any(np.isfinite(ev_timely)) else float("nan")

    # Event-level delay / timeliness quantiles (censored).
    ev_delay_cens = np.array([m.get("event_delay_months_cens", np.nan) for m in ms], dtype=float)
    ev_ttt_cens = np.array([m.get("event_time_to_trigger_months_cens", np.nan) for m in ms], dtype=float)

    def _q(arr: np.ndarray, q: float) -> float:
        arr = arr[np.isfinite(arr)]
        return float(np.quantile(arr, q)) if arr.size else float("nan")

    ev_delay_p50 = _q(ev_delay_cens, 0.50)
    ev_delay_p90 = _q(ev_delay_cens, 0.90)
    ev_ttt_p50 = _q(ev_ttt_cens, 0.50)
    ev_ttt_p90 = _q(ev_ttt_cens, 0.90)

    delay_p90_cens = _q(delays_cens, 0.90)
    delay0_p90_cens = _q(delay0_cens, 0.90)

    # Primary paper metric: censored delay; undetected events are charged the full incident duration.
    median_delay_cens = float(np.nanmedian(delays_cens)) if np.any(np.isfinite(delays_cens)) else float("nan")
    median_delay_null_cens = float(np.nanmedian(delay0_cens)) if np.any(np.isfinite(delay0_cens)) else float("nan")

    # Back-compat diagnostic: uncensored conditional-on-detection delay.
    median_delay_uncens = float(np.nanmedian(delays_uncens)) if np.any(np.isfinite(delays_uncens)) else float("nan")
    median_delay_null_uncens = float(np.nanmedian(delay0_uncens)) if np.any(np.isfinite(delay0_uncens)) else float("nan")

    return {
        "n_trials": int(n_trials),

        "detect_rate": detect_rate,
        "detect_rate_null": detect_rate_null,
        "uplift_detect_rate": uplift_detect,

        # Event-level (one alert per incident) and timeliness SLA
        "event_success_rate": ev_success_rate,        "event_timely_sla_rate": ev_timely_rate,
        "event_anywarn_success_rate": float(np.nanmean([m.get("event_anywarn_success", float("nan")) for m in ms])),
        "event_anywarn_timely_sla_rate": float(np.nanmean([m.get("event_anywarn_timely_sla", float("nan")) for m in ms])),
        "event_preexisting_warning_rate": float(np.nanmean([m.get("already_in_warning_at_start", float("nan")) for m in ms])),
        "contract_pass_full_rate": float(np.nanmean([m.get("contract_pass_full", float("nan")) for m in ms])),
        "contract_pass_full_anywarn_rate": float(np.nanmean([m.get("contract_pass_full_anywarn", float("nan")) for m in ms])),
        "event_sla_months": int(ev_sla_months) if (ev_sla_months == ev_sla_months) else float("nan"),

        # NOTE: median_delay is censored (paper-facing default).
        "median_delay": median_delay_cens,
        "median_delay_null": median_delay_null_cens,

        # Delay quantiles (censored; paper-facing)
        "delay_p50": median_delay_cens,
        "delay_p90": delay_p90_cens,
        "delay_null_p50": median_delay_null_cens,
        "delay_null_p90": delay0_p90_cens,

        # Event-level delay/timeliness quantiles (censored; one alert per incident)
        "event_delay_p50": ev_delay_p50,
        "event_delay_p90": ev_delay_p90,
        "event_time_to_trigger_p50": ev_ttt_p50,
        "event_time_to_trigger_p90": ev_ttt_p90,

        # Diagnostics: uncensored delay (NaN for misses).
        "median_delay_uncens": median_delay_uncens,
        "median_delay_null_uncens": median_delay_null_uncens,

        "mean_false_alarm_months_per_year": float(np.nanmean(fa_month)) if np.any(np.isfinite(fa_month)) else float("nan"),
        "mean_false_alert_events_per_year": float(np.nanmean(fa_evt)) if np.any(np.isfinite(fa_evt)) else float("nan"),
        "mean_time_in_warning": float(np.nanmean(tiw)) if np.any(np.isfinite(tiw)) else float("nan"),
        "mean_p_star": float(np.nanmean(pstar)) if np.any(np.isfinite(pstar)) else float("nan"),
        "mean_effect_size": float(np.nanmean(eff)) if np.any(np.isfinite(eff)) else float("nan"),

        # Clean achieved burden (matched-burden axis)
        "mean_fae_null": float(np.nanmean(fae0)) if np.any(np.isfinite(fae0)) else float("nan"),
        "mean_tiw_null": float(np.nanmean(tiw0)) if np.any(np.isfinite(tiw0)) else float("nan"),

        # CAD / ΔTIW (counterfactual divergence)
        "mean_cad_win": float(np.nanmean(cad_win)) if np.any(np.isfinite(cad_win)) else float("nan"),
        "mean_cad_post": float(np.nanmean(cad_post)) if np.any(np.isfinite(cad_post)) else float("nan"),
        "mean_dtiw_win": float(np.nanmean(dtiw_win)) if np.any(np.isfinite(dtiw_win)) else float("nan"),
        "mean_abs_dtiw_win": float(np.nanmean(abs_dtiw_win)) if np.any(np.isfinite(abs_dtiw_win)) else float("nan"),
        "mean_dtiw_post": float(np.nanmean(dtiw_post)) if np.any(np.isfinite(dtiw_post)) else float("nan"),
    }


def _circular_block_bootstrap_indices(
    n: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Circular moving-block bootstrap indices.

    Returns an index array of length n formed by concatenating i.i.d. sampled
    contiguous blocks of size `block_size` from {0,...,n-1}, with wrap-around.

    This is the standard circular moving-block bootstrap used to preserve local
    dependence when the observations have an inherent ordering (here: trials
    ordered by incident window start date).
    """
    n = int(n)
    bs = int(max(1, block_size))
    if n <= 0:
        return np.zeros((0,), dtype=int)

    out = np.empty(n, dtype=int)
    t = 0
    while t < n:
        start = int(rng.integers(0, n))
        take = min(bs, n - t)
        out[t : t + take] = (start + np.arange(take)) % n
        t += take
    return out




def _bootstrap_group_ci(
    g: pd.DataFrame,
    n_boot: int,
    block_size: int,
    alpha: float,
    rng: np.random.Generator,
) -> dict:
    """Block-bootstrap percentile CIs for key operating-point summary metrics for one group.

    Bootstrap units are *trials*, ordered by window start date. We use a circular moving-block
    bootstrap to preserve local temporal dependence among neighboring trials.
    """
    g2 = g.copy()
    g2["_start_dt"] = pd.to_datetime(g2["start"])
    g2 = g2.sort_values(["_start_dt", "trial"]).reset_index(drop=True)
    n = len(g2)
    if n == 0 or int(n_boot) <= 0:
        return {}

    det = g2["detected"].to_numpy(dtype=float)
    det0 = g2["detected_null"].to_numpy(dtype=float)
    # Delay semantics:
    #   - delay_months (uncensored): NaN when not triggered
    #   - delay_months_cens (censored): full incident duration when not triggered
    delay_uncens = g2["delay_months"].to_numpy(dtype=float)
    delay0_uncens = g2["delay_months_null"].to_numpy(dtype=float)
    if "delay_months_cens" in g2.columns:
        delay = g2["delay_months_cens"].to_numpy(dtype=float)
    else:
        delay = delay_uncens
    if "delay_months_null_cens" in g2.columns:
        delay0 = g2["delay_months_null_cens"].to_numpy(dtype=float)
    else:
        delay0 = delay0_uncens
    fae = g2["false_alert_events_per_year"].to_numpy(dtype=float)
    tiw = g2["time_in_warning"].to_numpy(dtype=float)
    fam = g2["false_alarms_per_year"].to_numpy(dtype=float)  # matches summarize() naming

    B = int(n_boot)
    s_detect = np.empty(B, dtype=float)
    s_detect0 = np.empty(B, dtype=float)
    s_uplift = np.empty(B, dtype=float)
    s_mdelay = np.empty(B, dtype=float)
    s_mdelay0 = np.empty(B, dtype=float)
    s_mdelay_uncens = np.empty(B, dtype=float)
    s_mdelay0_uncens = np.empty(B, dtype=float)
    s_fae = np.empty(B, dtype=float)
    s_tiw = np.empty(B, dtype=float)
    s_fam = np.empty(B, dtype=float)

    for b in range(B):
        idx = _circular_block_bootstrap_indices(n, block_size=block_size, rng=rng)
        d = float(np.mean(det[idx]))
        d0 = float(np.mean(det0[idx]))
        s_detect[b] = d
        s_detect0[b] = d0
        s_uplift[b] = d - d0

        # Same definition as summarize(): nanmedian over delay arrays (undetected -> NaN)
        di = delay[idx]
        d0i = delay0[idx]
        s_mdelay[b] = float(np.nanmedian(di)) if np.any(np.isfinite(di)) else float("nan")
        s_mdelay0[b] = float(np.nanmedian(d0i)) if np.any(np.isfinite(d0i)) else float("nan")

        # Diagnostics: uncensored (NaN for misses)
        diu = delay_uncens[idx]
        d0iu = delay0_uncens[idx]
        s_mdelay_uncens[b] = float(np.nanmedian(diu)) if np.any(np.isfinite(diu)) else float("nan")
        s_mdelay0_uncens[b] = float(np.nanmedian(d0iu)) if np.any(np.isfinite(d0iu)) else float("nan")

        fi = fae[idx]
        ti = tiw[idx]
        mi = fam[idx]
        s_fae[b] = float(np.nanmean(fi)) if np.any(np.isfinite(fi)) else float("nan")
        s_tiw[b] = float(np.nanmean(ti)) if np.any(np.isfinite(ti)) else float("nan")
        s_fam[b] = float(np.nanmean(mi)) if np.any(np.isfinite(mi)) else float("nan")

    lo = 100.0 * (float(alpha) / 2.0)
    hi = 100.0 * (1.0 - float(alpha) / 2.0)

    def q(a: np.ndarray, p: float) -> float:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return float("nan")
        return float(np.nanpercentile(a, p))

    return {
        "detect_rate_ci_lo": q(s_detect, lo),
        "detect_rate_ci_hi": q(s_detect, hi),
        "detect_rate_null_ci_lo": q(s_detect0, lo),
        "detect_rate_null_ci_hi": q(s_detect0, hi),
        "uplift_detect_rate_ci_lo": q(s_uplift, lo),
        "uplift_detect_rate_ci_hi": q(s_uplift, hi),
        "median_delay_ci_lo": q(s_mdelay, lo),
        "median_delay_ci_hi": q(s_mdelay, hi),
        "median_delay_null_ci_lo": q(s_mdelay0, lo),
        "median_delay_null_ci_hi": q(s_mdelay0, hi),

        "median_delay_uncens_ci_lo": q(s_mdelay_uncens, lo),
        "median_delay_uncens_ci_hi": q(s_mdelay_uncens, hi),
        "median_delay_null_uncens_ci_lo": q(s_mdelay0_uncens, lo),
        "median_delay_null_uncens_ci_hi": q(s_mdelay0_uncens, hi),
        "mean_false_alert_events_per_year_ci_lo": q(s_fae, lo),
        "mean_false_alert_events_per_year_ci_hi": q(s_fae, hi),
        "mean_time_in_warning_ci_lo": q(s_tiw, lo),
        "mean_time_in_warning_ci_hi": q(s_tiw, hi),
        "mean_false_alarm_months_per_year_ci_lo": q(s_fam, lo),
        "mean_false_alarm_months_per_year_ci_hi": q(s_fam, hi),
    }

def compute_block_bootstrap_cis(
    df_trials: pd.DataFrame,
    group_cols: list[str],
    n_boot: int,
    block_size: int,
    alpha: float,
    seed: int,
) -> pd.DataFrame:
    """Compute block-bootstrap CIs for each (kind, panel, detector, budget) group."""
    rng = np.random.default_rng(int(seed))
    out_rows = []
    for key, g in df_trials.groupby(group_cols):
        ci = _bootstrap_group_ci(g, n_boot=n_boot, block_size=block_size, alpha=alpha, rng=rng)
        row = {c: v for c, v in zip(group_cols, key)}
        row.update(ci)
        out_rows.append(row)
    return pd.DataFrame(out_rows)

def compute_scores(Z: pd.DataFrame, baselines: dict, train_mask: np.ndarray, cov_win: int) -> dict[str, pd.Series]:
    mu_cov, inv_cov = baselines["cov"]
    mu_pca, U, eig = baselines["pca"]
    eps_stale = baselines["stale_eps"]

    s: dict[str, pd.Series] = {}
    s["max_abs"] = score_max_abs(Z)
    s["T2"] = score_hotelling_t2(Z, mu_cov, inv_cov)

    s_res, s_pc = score_subspace(Z, mu_pca, U, eig)
    s["res_energy"], s["pc_t2"] = s_res, s_pc

    s["stale"] = score_staleness_runlength(Z, eps_stale, agg="max")
    s_stale_log = pd.Series(np.log1p(s["stale"].values), index=s["stale"].index, name="stale_log")

    # Covariance monitors in fixed PCA-score space
    s["factor_cov_change"] = score_factor_cov_change(Z, mu=mu_pca, U=U, eigvals=eig, window=cov_win)
    s["factor_cov_delta"] = score_factor_cov_delta(Z, mu=mu_pca, U=U, window=cov_win)
    s["factor_cov_lrt"] = score_factor_cov_lrt(Z, mu=mu_pca, U=U, eigvals=eig, window=cov_win)
    s["factor_cov_lrt_delta"] = score_factor_cov_lrt_delta(Z, mu=mu_pca, U=U, window=cov_win)

    # Fusions (train-standardized)
    s["fused_sp"] = fuse_scores_max_robust_train([s["res_energy"], s["pc_t2"], s_stale_log, s["factor_cov_change"]], train_mask=train_mask)
    s["fused_fisher"] = fuse_scores_fisher_train([s["res_energy"], s["pc_t2"], s_stale_log, s["factor_cov_change"]], train_mask=train_mask)

    s["fused_sp_lrt_delta"] = fuse_scores_max_robust_train([s["res_energy"], s["pc_t2"], s_stale_log, s["factor_cov_lrt_delta"]], train_mask=train_mask)
    s["fused_fisher_lrt_delta"] = fuse_scores_fisher_train([s["res_energy"], s["pc_t2"], s_stale_log, s["factor_cov_lrt_delta"]], train_mask=train_mask)
    return s


def add_sequential_baselines(
    scores: dict[str, pd.Series],
    train_mask: np.ndarray,
    ewma_lambda: float,
    cusum_k: float,
    bases: list[str],
) -> None:
    """Add classic sequential detection baselines (EWMA/CUSUM) on top of selected scalar scores.

    We keep these as *score transforms* so they can be evaluated under the same alert-budget policy.
    """
    for base in bases:
        if base not in scores:
            continue
        s0 = scores[base]
        # Ensure stable naming
        s0 = s0.rename(base)
        ew = score_ewma_from_base(s0, train_mask=train_mask, lam=float(ewma_lambda))
        cu = score_cusum_from_base(s0, train_mask=train_mask, k=float(cusum_k))
        scores[ew.name] = ew
        scores[cu.name] = cu


def fit_panel_baselines(Z: pd.DataFrame, train_mask: np.ndarray, r: int) -> dict:
    d = Z.shape[1]
    r = min(int(r), max(1, d - 1))
    mu_cov, inv_cov = fit_cov(Z.values[train_mask], eps=1e-3)
    mu_pca, U, eig = fit_pca(Z.values[train_mask], r=r)
    eps_stale = fit_staleness_params(Z.loc[Z.index[train_mask]])
    return {"cov": (mu_cov, inv_cov), "pca": (mu_pca, U, eig), "stale_eps": eps_stale, "r": r}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--train_end", type=str, default="2019-12-01")
    ap.add_argument("--earliest", type=str, default="2020-01-01")
    ap.add_argument("--train_frac", type=float, default=0.60,
                    help="If --train_end=auto, use this fraction of data as training prefix.")
    ap.add_argument("--earliest_after_train", type=int, default=1,
                    help="If --earliest=auto, start incidents this many steps after train_end.")
    ap.add_argument("--date_col", type=str, default="date",
                    help="Timestamp column name in panel_csv. Default: date.")
    ap.add_argument("--time_mode", type=str, default="calendar_month",
                    choices=["calendar_month", "step"],
                    help="Time mode: calendar_month uses calendar months; step uses row-index steps (e.g., weekly panels).")
    ap.add_argument("--steps_per_year", type=str, default="auto",
                    help="Steps per year for burden conversions. Use 'auto' to infer from index for step mode; defaults to 12 in calendar_month mode.")
    ap.add_argument("--score_cols_regex", type=str, default=r"\|score\|",
                    help="Regex selecting attacked-panel columns. Default matches CFPB score tier columns.")
    ap.add_argument("--all_cols_regex", type=str, default=r"\|all\|all\|",
                    help="Regex selecting negative-control columns. Default matches CFPB all/all columns.")
    ap.add_argument("--seed", type=int, default=20260107,
                    help="Base RNG seed for trial-window sampling and paired/common-random-number choices. Set explicitly for paper runs.")
    ap.add_argument("--seed_all", type=int, default=0,
                    help="If 1, set ci_seed, drift_seed, and drift_noise_seed to --seed (one-seed mode).")
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--duration", type=int, default=6)
    ap.add_argument("--r_pca", type=int, default=6)
    ap.add_argument("--r_cov", type=int, default=12)
    ap.add_argument("--cov_win", type=int, default=6)
    ap.add_argument("--budgets", type=str, default="0.5,1.0,2.0")
    ap.add_argument("--alert_policy", type=str, default="event_budgeted", choices=["topk","rolling_quantile","event_budgeted"])
    ap.add_argument("--rq_window", type=int, default=60)
    ap.add_argument("--rq_min_periods", type=int, default=24)
    ap.add_argument("--merge_gap", type=int, default=1)
    ap.add_argument("--tiw_budget", type=float, default=0.15,
                    help="Max time-in-warning fraction during benign calibration (<=1). Set <=0 to disable.")
    ap.add_argument("--guardband", type=float, default=1.00,
                    help="Conservative calibration factor in (0,1]. Calibration targets are guardband*budget and guardband*tiw_budget.")

    ap.add_argument("--online_recal", type=int, default=1,
                    help="If 1, use rolling/online recalibration of p* in deployment/eval period to stabilize event/TIW budgets. If 0, use fixed p* from benign training.")
    ap.add_argument("--p_ctrl_window", type=int, default=60,
                    help="Trailing control window (months) used for online p* budget control.")
    ap.add_argument("--p_update_every", type=int, default=1,
                    help="Update p* every N months during deployment/eval.")
    ap.add_argument("--p_eta", type=float, default=0.30,
                    help="Controller step size for p* updates (higher = faster adaptation).")
    ap.add_argument("--p_max_up", type=float, default=1.25, help="Max multiplicative loosen step for p* updates.")
    ap.add_argument("--p_max_down", type=float, default=0.60, help="Min multiplicative tighten step for p* updates (smaller => more aggressive tightening).")
    ap.add_argument("--p_min_control_months", type=int, default=12,
                    help="Minimum months in the control window before p* is updated.")
    ap.add_argument("--allow_loosen", type=int, default=1,
                    help="If 1, allow online p* to loosen when under budget (symmetric update). If 0, tighten-only.")
    ap.add_argument("--p_star_floor", type=float, default=1e-6, help="Floor for p* (tail prob) search/controller.")
    ap.add_argument("--p_star_ceil",  type=float, default=0.98, help="Ceiling for p* (tail prob) search/controller.")
    ap.add_argument("--cooldown_months", type=int, default=1,
                    help="Episode policy: after an alert episode ends, suppress new alarms for this many months. Set 0 to disable.")
    ap.add_argument("--max_episode_months", type=int, default=6,
                    help="Episode policy: cap continuous alert episode length (months). Set 0 to disable.")

    ap.add_argument("--coh_alpha", type=float, default=1e-2,
                    help="Coherence detector: ridge alpha for mapping all_only -> score_only.")
    ap.add_argument("--coh_cov_shrink", type=float, default=0.10,
                    help="Coherence detector: residual covariance shrinkage toward diagonal (0..1).")
    ap.add_argument("--coh_cov_eps", type=float, default=1e-6,
                    help="Coherence detector: residual covariance jitter (scale-aware).")
    ap.add_argument("--ewma_lambda", type=float, default=0.20,
                help="Sequential baseline: EWMA lambda on selected scalar scores (robust-z standardized on training).")
    ap.add_argument("--cusum_k", type=float, default=0.50,
                help="Sequential baseline: one-sided CUSUM reference value k on selected scalar scores (robust-z standardized on training).")
    ap.add_argument("--cusum_hold_months", type=int, default=1,
                help="CUSUM chart: warning hold length (months) after an alarm event; chart resets on alarm and holds warnings for this many months.")
    ap.add_argument("--cusum_local_train_months", type=int, default=60,
                help="CUSUM chart: calibrate h using the most recent N months prior to each trial window start (no lookahead).")

    ap.add_argument("--cusum_tiw_cap", type=float, default=-1.0,
                help="CUSUM calibration: absolute TIW cap on benign calibration window. If <=0, use auto cap based on budget and hold.")
    ap.add_argument("--cusum_tiw_cap_mult", type=float, default=3.0,
                help="CUSUM calibration (auto cap): cap = mult * (budget_events_per_year * hold_months / 12).")
    ap.add_argument("--cusum_tiw_cap_abs", type=float, default=0.50,
                help="CUSUM calibration (auto cap): absolute upper bound for TIW cap.")
    ap.add_argument("--cusum_tiw_cap_min", type=float, default=0.10,
                help="CUSUM calibration (auto cap): minimum TIW cap.")
    ap.add_argument("--cusum_event_tol", type=float, default=0.00,
                help="CUSUM calibration: allow events/year to exceed target by this fractional tolerance during calibration (default 0).")
    # Step 2: Uncertainty via block bootstrap confidence intervals (CIs)
    ap.add_argument("--ci_n_boot", type=int, default=500,
                help="Number of circular moving-block bootstrap resamples per group. Set 0 to disable.")
    ap.add_argument("--ci_block_size", type=int, default=5,
                help="Circular moving-block bootstrap block size (in trials ordered by window start date).")
    ap.add_argument("--ci_alpha", type=float, default=0.05,
                help="Two-sided CI level: alpha=0.05 gives 95%% CIs.")
    ap.add_argument("--ci_seed", type=int, default=20260109,
                help="RNG seed for block bootstrap CIs.")
    ap.add_argument("--corr_mix_strength", type=float, default=0.35)
    ap.add_argument("--corr_mix_strength_grid", type=str, default="0.15,0.25,0.35,0.50",
                    help="Comma-separated corr_mix strengths in [0,1] for threat-surface sweeps.")
    ap.add_argument("--freeze_strength_grid", type=str, default="0.25,0.50,0.75,1.00",
                    help="Comma-separated freeze strengths in [0,1] for threat-surface sweeps (1.0=full flatline).")
    ap.add_argument(
        "--do_benign_drift",
        type=int,
        default=0,
        help=(
            "If 1, run a benign drift slice  as a *drift-only compliance test* (default). "
            "This evaluates budget/TIW safety under nonstationarity *without* incident labels. "
            "Use --benign_drift_eval_mode=incident_like only if you explicitly want drift treated as an incident kind."
        ),
    )
    ap.add_argument(
        "--benign_drift_eval_mode",
        type=str,
        default="compliance",
        choices=["compliance", "incident_like"],
        help=(
            "Benign drift evaluation mode. compliance=drift-only regime compliance (paper-default). "
            "incident_like=inject drift inside trial windows and evaluate like other incident kinds (optional ablation mode)."
        ),
    )
    ap.add_argument(
        "--benign_drift_strength_grid",
        type=str,
        default="",
        help="Comma-separated benign drift severities (delta) in [0,1]. If empty, reuse --freeze_strength_grid.",
    )
    ap.add_argument(
        "--drift_mode",
        type=str,
        default="variance_inflation",
        choices=["variance_inflation", "ramp", "noise_iid", "noise_ar1"],
        help=(
            "Drift type for the drift-only slice. "
            "variance_inflation matches the proposal: x_drift = mu + (1+delta)(x-mu). "
            "ramp is an optional mean-ramp setting. "
            "noise_iid injects benign per-channel measurement noise: x_drift = x + delta*sigma*eps (eps~N(0,1)). "
            "noise_ar1 is a low-frequency (AR(1)) noise version (more 'drift-like')."
        ),
    )
    ap.add_argument("--drift_noise_rho", type=float, default=0.90,
                    help="If --drift_mode=noise_ar1, AR(1) coefficient rho in [0,1). Higher => lower-frequency drift-like noise (default 0.90).")
    ap.add_argument("--drift_noise_seed", type=int, default=20260208,
                    help="Seed for benign drift noise injection (noise_iid / noise_ar1).")
    ap.add_argument("--drift_noise_ref", type=str, default="train", choices=["train","preblock","block"],
                    help="Reference segment for per-channel sigma in noise injection. train=use training window (default). preblock=use data prior to drift_start (no lookahead). block=use drift block (not recommended for strictness).")
    ap.add_argument(
        "--drift_scope",
        type=str,
        default="score_all",
        choices=["score_all", "score_k", "full"],
        help=(
            "Surface area affected by drift in the drift-only slice. score_all=all score-tier columns (default). "
            "score_k=K randomly-chosen score-tier columns. full=all columns."
        ),
    )
    ap.add_argument(
        "--drift_k",
        type=int,
        default=0,
        help="If --drift_scope=score_k, drift only K score-tier columns (0 => reuse --attack_k).",
    )
    ap.add_argument(
        "--drift_seed",
        type=int,
        default=20260207,
        help="Seed for selecting drifted columns when --drift_scope=score_k.",
    )
    ap.add_argument(
        "--drift_start",
        type=str,
        default="",
        help="Start date (YYYY-MM-DD) for drift block in drift-only slice. Default: --earliest.",
    )
    ap.add_argument(
        "--drift_end",
        type=str,
        default="",
        help="End date (YYYY-MM-DD) for drift block in drift-only slice (inclusive). Default: end of panel.",
    )
    ap.add_argument(
        "--drift_block_months",
        type=int,
        default=0,
        help="If >0 and --drift_end is empty, set drift_end = drift_start + drift_block_months - 1 (inclusive).",
    )
    ap.add_argument(
        "--drift_mu_mode",
        type=str,
        default="zero",
        choices=["zero", "rolling", "block"],
        help=(
            "Mean reference mu_t for variance inflation. zero=mu_t=0 (fast, stable in z-units). "
            "rolling=rolling mean (use --drift_mu_win). block=mean over the drift block."
        ),
    )
    ap.add_argument("--drift_mu_win", type=int, default=24, help="Rolling mean window (months) if --drift_mu_mode=rolling.")

    # monotonicity diagnostics
    ap.add_argument("--monotone_tau", type=float, default=0.02,
                    help="Monotonicity tolerance tau: flag p_{j+1} < p_j - tau (default 0.02).")
    ap.add_argument("--monotone_min_primary", type=int, default=20,
                    help="Require at least this many primary trials in-band per strength before testing monotonicity here.")
    ap.add_argument("--monotone_min_coverage", type=float, default=0.05,
                    help="Require coverage_primary >= this threshold per strength before testing monotonicity here.")
    ap.add_argument("--monotone_focus_metric", type=str, default="event_timely_sla_rate",
                    help="Metric used for the monotonicity check (default: event_timely_sla_rate = detected_within_k).")
    ap.add_argument("--paired_intensity_sweeps", type=int, default=1,
                    help="If 1 (default), use paired/common-random-numbers intensity sweeps: for each base trial window/target-columns selection, evaluate all strengths in the corresponding grid. This reduces variance and makes threat-surface curves more monotone.")
    ap.add_argument("--paper_strength_policy", type=str, default="max", choices=["max","median","min"],
                    help="Which strength in the grid defines the main paper operating point for Table 1 (achieved-band). max=max strength in the sweep grid (default).")
    ap.add_argument("--cad_post_months", type=int, default=3,
                    help="Post-incident horizon (months) for CAD_post / ΔTIW_post.")
    ap.add_argument(
        "--timeliness_sla_months",
        type=int,
        default=1,
        help=(
            "Event-level timeliness SLA (months). A trial is timely if the first WARNING timestamp within the incident "
            "window occurs within K months of incident onset (after merge/cooldown rules). This matches the 'episode intersects' "
            "event contract; onset-based trigger timing is reported separately."
        ),
    )
    ap.add_argument("--matched_tol", type=float, default=0.10,
                    help="Matched-burden tolerance: keep trials with |achieved-target|/target <= tol.")
    ap.add_argument("--matched_mode", type=str, default="budget_only", choices=["budget_only","joint"],
                    help="Define matched-burden: budget_only=match event budget only (main text); joint=also require TIW in-band.")
    ap.add_argument("--matched_tiw_tol", type=float, default=None,
                    help="Relative tolerance for TIW matching when --matched_mode=joint (default: matched_tol).")
    ap.add_argument("--matched_budget_band", type=str, default="count", choices=["count","rate"],
                    help="Definition of the event-budget matching band. count=match nearest feasible integer episode count within a tolerance-derived band (robust to quantization); rate=match events/year rate directly (can be infeasible at low budgets).")
    ap.add_argument("--intensity_bins", type=int, default=6,
                    help="Number of bins for stealth/intensity curves.")
    ap.add_argument("--regime_window_months", type=int, default=36,
                    help="Safety-compliance slicing: regime block length in months.")
    ap.add_argument("--regime_stride_months", type=int, default=12,
                    help="Safety-compliance slicing: regime stride in months.")
    ap.add_argument("--safety_eps", type=float, default=0.10,
                    help="Safety violation threshold: overshoot if achieved > (1+eps)*target.")
    ap.add_argument("--do_perm_control", type=int, default=1,
                    help="Write uniform-placement overlap diagnostics on clean streams.")
    ap.add_argument("--perm_control_n", type=int, default=200,
                    help="DEPRECATED (unused). Overlap diagnostics are computed exactly over all valid placements; kept for backwards compatibility.")
    ap.add_argument("--corr_mix_mode", type=str, default="pair_rotate", choices=["pca_rotate","pair_rotate"])
    ap.add_argument("--corr_pair", type=str, default="0,1")
    ap.add_argument("--corr_theta", type=float, default=1.57079632679)  # pi/2
    ap.add_argument("--attack_k", type=int, default=-1,
                    help="Attack scope. If >0 and < #score_cols, randomly select K score columns to attack per trial. If <=0, attack all score columns." )
    args = ap.parse_args()

    # One-seed mode (optional): simplify third-party reproduction by using a single seed
    # for all stochastic components. Default is 0 to preserve the historical one-seed default.
    if int(getattr(args, "seed_all", 0)) == 1:
        s = int(getattr(args, "seed", 20260107))
        args.ci_seed = s
        args.drift_seed = s
        args.drift_noise_seed = s


    def cusum_tiw_cap_for_budget(budget_events_per_year: float) -> float:
        """Compute the benign-window TIW cap used for occupancy-aware CUSUM calibration."""
        if float(getattr(args, "cusum_tiw_cap", -1.0)) > 0:
            return float(np.clip(float(args.cusum_tiw_cap), 0.0, 1.0))
        hold = int(max(getattr(args, "cusum_hold_months", 1), 1))
        expected = float(budget_events_per_year) * (float(hold) / steps_per_year)
        cap = float(getattr(args, "cusum_tiw_cap_mult", 3.0)) * expected
        cap = max(cap, float(getattr(args, "cusum_tiw_cap_min", 0.10)))
        cap = min(cap, float(getattr(args, "cusum_tiw_cap_abs", 0.50)))
        return float(np.clip(cap, 0.0, 1.0))

    # Cache for local (per-trial) CUSUM calibrations: (panel, budget, detector, window_start) -> dict
    cusum_local_cache: dict[tuple, dict] = {}

    base_seed = int(getattr(args, "seed", 20260107))

    rng = np.random.default_rng(base_seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write the run arguments and random seeds.
    try:
        import json as _json
        import sys as _sys
        manifest = {
            "panel_csv": str(args.panel_csv),
            "out_dir": str(args.out_dir),
            "train_end": str(getattr(args, "train_end", "")),
            "earliest": str(getattr(args, "earliest", "")),
            "time_mode": str(getattr(args, "time_mode", "calendar_month")),
            "steps_per_year": str(getattr(args, "steps_per_year", "auto")),
            "seed": int(getattr(args, "seed", 20260107)),
            "seed_all": int(getattr(args, "seed_all", 0)),
            "ci_seed": int(getattr(args, "ci_seed", 20260109)),
            "drift_seed": int(getattr(args, "drift_seed", 20260207)),
            "drift_noise_seed": int(getattr(args, "drift_noise_seed", 20260208)),
            "cmd": " ".join(["python"] + _sys.argv),
        }
        (out_dir / "run_manifest.json").write_text(_json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        pass


    # --- Load panel (flexible time axis + column selection) ---
    date_col = str(getattr(args, "date_col", "date"))
    df_panel = pd.read_csv(args.panel_csv)
    if date_col in df_panel.columns:
        df_panel[date_col] = pd.to_datetime(df_panel[date_col], errors="raise")
        panel = df_panel.set_index(date_col)
    else:
        # Fallback: treat the first column as timestamp
        first = df_panel.columns[0]
        df_panel[first] = pd.to_datetime(df_panel[first], errors="raise")
        panel = df_panel.set_index(first)

    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()

    # Choose train_end / eval_start (allow 'auto' to avoid hard-coding a calendar window)
    if str(args.train_end).lower() == "auto":
        te_pos = max(0, int(round(float(getattr(args, "train_frac", 0.60)) * len(panel))) - 1)
        train_end = pd.Timestamp(panel.index[te_pos])
    else:
        train_end = pd.Timestamp(args.train_end)
        # Align to last index <= train_end (prevents empty train masks if exact timestamp not present)
        te_pos = int(panel.index.searchsorted(train_end, side="right") - 1)
        te_pos = max(0, te_pos)
        train_end = pd.Timestamp(panel.index[te_pos])

    if str(args.earliest).lower() == "auto":
        te_pos = int(panel.index.searchsorted(train_end, side="right") - 1)
        te_pos = max(0, te_pos)
        epos = min(len(panel.index) - 1, te_pos + int(getattr(args, "earliest_after_train", 1)))
        eval_start = pd.Timestamp(panel.index[epos])
    else:
        eval_start = pd.Timestamp(args.earliest)
        epos = int(panel.index.searchsorted(eval_start, side="left"))
        epos = min(max(epos, 0), len(panel.index) - 1)
        eval_start = pd.Timestamp(panel.index[epos])

    # Time configuration (calendar-month mode vs generic step mode)
    time_mode = str(getattr(args, "time_mode", "calendar_month"))
    steps_per_year_arg = str(getattr(args, "steps_per_year", "auto")).lower()

    if steps_per_year_arg == "auto":
        if time_mode == "calendar_month":
            steps_per_year = 12.0
        else:
            # Infer from median index spacing (assumes near-regular sampling)
            if len(panel.index) >= 3:
                d_ns = np.diff(panel.index.values.astype("datetime64[ns]").astype("int64"))
                med_ns = float(np.median(d_ns))
                if med_ns <= 0:
                    steps_per_year = 12.0
                else:
                    sec_year = 365.25 * 24.0 * 3600.0
                    steps_per_year = float(sec_year / (med_ns / 1e9))
            else:
                steps_per_year = 12.0
    else:
        steps_per_year = float(steps_per_year_arg)

    eval_mod.set_time_config(time_mode=time_mode, steps_per_year=steps_per_year)

    train_mask_full = np.asarray(panel.index <= train_end)
    Z_full, _ = robust_standardize_df(panel, train_mask=train_mask_full)

    # Flexible column selection
    score_pat = re.compile(str(getattr(args, "score_cols_regex", r"\|score\|")))
    all_pat = re.compile(str(getattr(args, "all_cols_regex", r"\|all\|all\|")))

    score_cols = [c for c in Z_full.columns if score_pat.search(c)]
    if len(score_cols) == 0:
        raise ValueError(f"No attacked-panel columns matched score_cols_regex={getattr(args,'score_cols_regex',None)!r}.")
    score_set = set(score_cols)
    score_idx = [i for i, c in enumerate(Z_full.columns) if c in score_set]

    # score_pos maps global column index in Z_full -> position in the score-only matrix used for PCA injection stats.
    score_pos = {int(gi): int(j) for j, gi in enumerate(score_idx)}
    attack_k = int(getattr(args, "attack_k", -1))
    if attack_k <= 0 or attack_k >= len(score_idx):
        attack_k = int(len(score_idx))

    all_cols = [c for c in Z_full.columns if all_pat.search(c)]
    if len(all_cols) == 0:
        # Fallback: use the complement as a "negative-control" slice (not attacked), if available.
        all_cols = [c for c in Z_full.columns if c not in score_set]
        if len(all_cols) == 0:
            # Degenerate fallback: no meaningful negative control; mirror score panel.
            all_cols = list(score_cols)

    Z_all = Z_full[all_cols].copy()
    Z_score = Z_full[score_cols].copy()
    Z_panels_clean = {"full": Z_full, "score_only": Z_score, "all_only": Z_all}

    # Baseline statistics used to build scalar detectors on top of the (standardized) panels.
    # IMPORTANT: keep behavior stable in calendar_month mode.
    baselines = {
        "full": fit_panel_baselines(Z_full, train_mask_full, r=int(args.r_pca)),
        "all_only": fit_panel_baselines(Z_all, train_mask_full, r=min(int(args.r_pca), max(1, Z_all.shape[1] - 1))),
        "score_only": fit_panel_baselines(Z_score, train_mask_full, r=min(int(args.r_cov), max(1, Z_score.shape[1] - 1))),
    }

    # Cross-panel coherence baseline: predict score-only signals from all-only signals (benign history).
    coh_model = fit_coherence_ridge(
        Z_all.values[train_mask_full],
        Z_score.values[train_mask_full],
        alpha=float(args.coh_alpha),
        cov_shrink=float(args.coh_cov_shrink),
        cov_eps=float(args.coh_cov_eps),
    )

    # PCA basis for corr_mix injection (score-only panel)
    r_cov = min(int(args.r_cov), max(1, Z_score.shape[1] - 1))
    mu_score_inj, U_score_inj, _eig_score_inj = fit_pca(Z_score.values[train_mask_full], r=r_cov)

    eval_mask = np.asarray(Z_full.index >= eval_start)

    budgets = [float(x) for x in args.budgets.split(",")]
    dets_common = [
        "max_abs", "T2", "res_energy", "pc_t2", "stale",
        "factor_cov_change", "factor_cov_delta", "factor_cov_lrt", "factor_cov_lrt_delta",
        "fused_sp", "fused_fisher", "fused_sp_lrt_delta", "fused_fisher_lrt_delta",
        "ewma_factor_cov_lrt", "cusum_factor_cov_lrt",
    ]
    # Coherence detectors are only meaningful for the score-only panel (they depend on all-only).
    dets_by_panel = {
        "full": list(dets_common),
        "all_only": list(dets_common),
        "score_only": list(dets_common) + ["coh_resid_t2", "coh_resid_energy", "ewma_coh_resid_t2", "cusum_coh_resid_t2"],
    }

    # Panels actually constructed above (e.g., full / score_only / all_only).
    panels_to_run = list(Z_panels_clean.keys())

    # Clean scores
    scores_clean = {p: compute_scores(Z_panels_clean[p], baselines[p], train_mask_full, cov_win=int(args.cov_win)) for p in Z_panels_clean}
    # Cross-panel coherence scores (score-only panel only)
    scores_clean["score_only"]["coh_resid_t2"] = score_coherence_resid_t2(Z_panels_clean["all_only"], Z_panels_clean["score_only"], coh_model)
    scores_clean["score_only"]["coh_resid_energy"] = score_coherence_resid_energy(Z_panels_clean["all_only"], Z_panels_clean["score_only"], coh_model)
    # Sequential baselines (EWMA/CUSUM) on selected scalar scores (SP-friendly citations).
    for p_name in scores_clean:
        add_sequential_baselines(
            scores_clean[p_name],
            train_mask=train_mask_full,
            ewma_lambda=float(args.ewma_lambda),
            cusum_k=float(args.cusum_k),
            bases=["factor_cov_lrt"],
        )

    add_sequential_baselines(
        scores_clean["score_only"],
        train_mask=train_mask_full,
        ewma_lambda=float(args.ewma_lambda),
        cusum_k=float(args.cusum_k),
        bases=["coh_resid_t2"],
    )

    # Occupancy-aware CUSUM chart calibration (episode events + TIW).
    # Calibrate h on benign history to satisfy BOTH:
    #   - false alert events/year <= target
    #   - time-in-warning (TIW) <= cap (prevents TIW≈1 degenerate baselines)
    cusum_h: dict[str, dict[float, dict[str, float]]] = {p: {b: {} for b in budgets} for p in Z_panels_clean}
    cusum_rows = []
    for p in Z_panels_clean:
        for b in budgets:
            tiw_cap = cusum_tiw_cap_for_budget(float(b))
            for d in dets_by_panel[p]:
                if not d.startswith("cusum_"):
                    continue
                base = d[len("cusum_"):]
                if base not in scores_clean[p]:
                    continue
                h, st = calibrate_cusum_h_operational_occupancy_from_base(
                    scores_clean[p][base],
                    train_mask=train_mask_full,
                    target_events_per_year=float(b),
                    tiw_cap=float(tiw_cap),
                    k=float(args.cusum_k),
                    hold_months=int(args.cusum_hold_months),
                    cooldown_months=int(getattr(args, "cooldown_eff", getattr(args, "cooldown_months", 0))),
                    max_episode_months=int(getattr(args, "max_episode_months", 0)),
                    merge_gap_months=int(getattr(args, "merge_gap", 0)),
                    event_tol=float(getattr(args, "cusum_event_tol", 0.0)),
                )
                cusum_h[p][b][d] = float(h)
                cusum_rows.append({
                    "panel": p,
                    "budget": float(b),
                    "detector": d,
                    "base": base,
                    "cusum_k": float(args.cusum_k),
                    "hold_months": int(args.cusum_hold_months),
                    "tiw_cap": float(tiw_cap),
                    "h": float(h),
                    "train_fae": float(st.get("fae", np.nan)),
                    "train_tiw": float(st.get("tiw", np.nan)),
                    "train_events": float(st.get("n_events", np.nan)),
                    "train_months": float(st.get("tr_months", np.nan)),
                })
    if len(cusum_rows) > 0:
        pd.DataFrame(cusum_rows).to_csv(out_dir / "cusum_h.csv", index=False)

    # Policy: pre-tune p for event_budgeted (benign history only)
    tuned_p: dict[str, dict[float, dict[str, float]]] = {p: {b: {} for b in budgets} for p in Z_panels_clean}
    if args.alert_policy == "event_budgeted":
        for p in Z_panels_clean:
            for b in budgets:
                for d in dets_by_panel[p]:
                    _a, _thr, pstar = apply_event_budgeted_threshold(
                        scores_clean[p][d],
                        target_events_per_year=b,
                        train_mask=train_mask_full,
                        window=int(args.rq_window),
                        min_periods=int(args.rq_min_periods),
                        merge_gap_months=int(args.merge_gap),
                        max_time_in_warning=(None if float(args.tiw_budget) <= 0 else float(args.tiw_budget)),
                        guardband=float(args.guardband),
                        # IMPORTANT: ensure the pre-tuned p* is computed under the SAME alert policy
                        # constraints as the deployed alarms (cooldown/max-episode) and within the
                        # same p-range as the online controller.
                        cooldown_months=int(args.cooldown_months),
                        max_episode_months=int(args.max_episode_months),
                    )
                    tuned_p[p][b][d] = float(pstar)

    
    # Optional: rolling/online recalibration of p* in the deployment/eval period .
    # We compute p*(t) on *clean* scores only once per panel/budget/det, then re-use p*(t)
    # when thresholding injected trials (thresholds still depend on the observed score history).
    pstar_series: dict[str, dict[float, dict[str, pd.Series]]] = {p: {b: {} for b in budgets} for p in Z_panels_clean}
    alarms_online: dict[str, dict[float, dict[str, pd.Series]]] = {p: {B: {} for B in budgets} for p in panels_to_run}
    thr_online: dict[str, dict[float, dict[str, pd.Series]]] = {p: {B: {} for B in budgets} for p in panels_to_run}
    if args.alert_policy == "event_budgeted" and int(getattr(args, "online_recal", 1)) == 1:
        for p in Z_panels_clean:
            for b in budgets:
                for d in dets_by_panel[p]:
                    init_p = tuned_p[p][b].get(d, float(np.clip(b / steps_per_year, float(getattr(args, 'p_star_floor', 1e-6)), float(getattr(args, 'p_star_ceil', 0.98)))))
                    _a, _thr, pser = apply_event_budgeted_threshold_online(
                        scores_clean[p][d],
                        init_p=float(init_p),
                        target_events_per_year=float(b),
                        eval_mask=eval_mask,
                        window=int(args.rq_window),
                        min_periods=int(args.rq_min_periods),
                        merge_gap_months=int(args.merge_gap),
                        max_time_in_warning=(None if float(args.tiw_budget) <= 0 else float(args.tiw_budget)),
                        guardband=float(args.guardband),
                        control_window=int(getattr(args, "p_ctrl_window", 60)),
                        update_every=int(getattr(args, "p_update_every", 1)),
                        eta=float(getattr(args, "p_eta", 0.30)),
                        p_lo=float(getattr(args, "p_star_floor", 1e-6)),
                        p_hi=float(getattr(args, "p_star_ceil", 0.98)),
                        max_up=float(getattr(args, "p_max_up", 1.25)),
                        max_down=float(getattr(args, "p_max_down", 0.60)),
                        min_control_months=int(getattr(args, "p_min_control_months", 12)),
                        allow_loosen=int(getattr(args, "allow_loosen", 1)),
                        cooldown_months=int(getattr(args, "cooldown_eff", getattr(args, "cooldown_months", 0))),
                        max_episode_months=int(getattr(args, "max_episode_months", 0)),
                    )
                    pstar_series[p][b][d] = pser
                    alarms_online[p][b][d] = _a
                    thr_online[p][b][d] = _thr
    else:
        # Fixed p*: represent as a constant series for uniform downstream handling.
        for p in Z_panels_clean:
            for b in budgets:
                for d in dets_by_panel[p]:
                    p0 = tuned_p[p][b].get(d, float(np.clip(b / steps_per_year, float(getattr(args, 'p_star_floor', 1e-6)), float(getattr(args, 'p_star_ceil', 0.98)))))
                    pstar_series[p][b][d] = pd.Series(float(p0), index=panel.index, name="p_star")


    # =========================
    # Dev-only invariant: online p*(t) replay parity
    #
    # For alert_policy=event_budgeted with online_recal=1, we expect the clean-stream alarms produced by
    # apply_event_budgeted_threshold_online(...) to exactly match a replay that applies the SAME dynamic quantile
    # thresholding to the clean scores, using the emitted p*(t) series, and the SAME p* bounds.
    #
    # This is critical for counterfactual parity: injected/drifted streams are thresholded via dynamic_quantile_threshold
    # with the clean p*(t) series. Any mismatch here can create non-zero CAD even on negative controls.
    # =========================
    try:
        if args.alert_policy == "event_budgeted" and int(getattr(args, "online_recal", 1)) == 1:
            inv_rows = []
            for p_name in Z_panels_clean:
                for b in budgets:
                    for d_name in dets_by_panel[p_name]:
                        if d_name not in scores_clean[p_name]:
                            continue
                        if p_name not in alarms_online or b not in alarms_online[p_name] or d_name not in alarms_online[p_name][b]:
                            continue
                        a_on = alarms_online[p_name][b][d_name].values.astype(int)
                        pser = pstar_series[p_name][b][d_name]
                        a_rep, _thr_rep = dynamic_quantile_threshold(
                            scores_clean[p_name][d_name],
                            pser,
                            window=int(args.rq_window),
                            min_periods=int(args.rq_min_periods),
                            cooldown_months=int(getattr(args, "cooldown_eff", getattr(args, "cooldown_months", 0))),
                            max_episode_months=int(getattr(args, "max_episode_months", 0)),
                            p_lo=float(getattr(args, "p_star_floor", 1e-6)),
                            p_hi=float(getattr(args, "p_star_ceil", 0.98)),
                        )
                        a_rep = a_rep.values.astype(int)
                        if int(getattr(args, "merge_gap", 0)) > 0:
                            a_rep = apply_merge_gap_array(a_rep, int(getattr(args, "merge_gap", 0)))
                        m = (np.asarray(eval_mask).astype(bool)) & np.isfinite(scores_clean[p_name][d_name].values)
                        n_eval = int(m.sum())
                        if n_eval <= 0:
                            continue
                        mismatch = float(np.mean(a_on[m] != a_rep[m]))
                        ps = pser.values.astype(float)
                        p_hi = float(getattr(args, "p_star_ceil", 0.98))
                        p_lo = float(getattr(args, "p_star_floor", 1e-6))
                        clip_hi = float(np.mean((ps > p_hi + 1e-12) & m))
                        clip_lo = float(np.mean((ps < p_lo - 1e-12) & m))
                        inv_rows.append({
                            "panel": p_name,
                            "budget": float(b),
                            "detector": d_name,
                            "alarm_mismatch_rate": mismatch,
                            "p_star_clip_hi_rate": clip_hi,
                            "p_star_clip_lo_rate": clip_lo,
                            "n_eval_months": n_eval,
                        })
            if len(inv_rows) > 0:
                df_inv = pd.DataFrame(inv_rows)
                df_inv.to_csv(out_dir / "dev_invariants_pstar_parity.csv", index=False)
                worst = float(df_inv["alarm_mismatch_rate"].max())
                if worst > 1e-9:
                    msg = f"[WARN] Invariant failed: online p*(t) replay parity mismatch (max mismatch={worst:.4g})."
                    print(msg)
                    if int(os.environ.get("NV_DEV_CHECKS", "0")) == 1:
                        raise AssertionError(msg)
    except Exception as _e:
        print(f"[WARN] dev invariant check (p*(t) parity) encountered an exception: {_e}")

    def make_alarms(scores_dict: dict[str, pd.Series], budget: float, panel_name: str, det_name: str) -> pd.Series:
        # CUSUM uses a calibrated reset-on-alarm chart (not the generic budgeted threshold).
        if det_name.startswith('cusum_'):
            # Calibrated reset-on-alarm CUSUM chart.
            base = det_name[len('cusum_'):]
            s_base = scores_dict.get(base)
            if s_base is None:
                # Defensive fallback: no base score => no alarms.
                return pd.Series(np.zeros(len(eval_mask), dtype=int), index=Z_full.index, name=f"alarm_{det_name}")
            h = cusum_h.get(panel_name, {}).get(budget, {}).get(det_name)
            if h is None or (not np.isfinite(float(h))):
                tiw_cap = cusum_tiw_cap_for_budget(float(budget))
                h, _st = calibrate_cusum_h_operational_occupancy_from_base(
                    s_base,
                    train_mask=train_mask_full,
                    target_events_per_year=float(budget),
                    tiw_cap=float(tiw_cap),
                    k=float(args.cusum_k),
                    hold_months=int(args.cusum_hold_months),
                    cooldown_months=int(getattr(args, "cooldown_eff", getattr(args, "cooldown_months", 0))),
                    max_episode_months=int(getattr(args, "max_episode_months", 0)),
                    merge_gap_months=int(getattr(args, "merge_gap", 0)),
                    event_tol=float(getattr(args, "cusum_event_tol", 0.0)),
                )
            a_raw = cusum_chart_alarms_from_base(
                s_base, train_mask=train_mask_full, h=float(h), k=float(args.cusum_k), hold_months=int(args.cusum_hold_months)
            )
            # Enforce the same episode policy used elsewhere before evaluation.
            a_man = apply_episode_policy_array(
                a_raw.values.astype(int),
                cooldown_months=int(getattr(args, "cooldown_eff", getattr(args, "cooldown_months", 0))),
                max_episode_months=int(getattr(args, "max_episode_months", 0)),
            )
            return pd.Series(a_man.astype(int), index=a_raw.index, name=a_raw.name)
        if args.alert_policy == "topk":
            return apply_budgeted_topk(scores_dict[det_name], fa_per_year=budget, eval_mask=eval_mask)
        if args.alert_policy == "rolling_quantile":
            alarms, _thr = apply_budgeted_threshold(scores_dict[det_name], fa_per_year=budget, window=int(args.rq_window), min_periods=int(args.rq_min_periods))
            return alarms
        # event_budgeted: apply (possibly time-varying) p*(t)
        if bool(getattr(args, 'online_recal', 0)) and (scores_dict is scores_clean.get(panel_name, None)):
            try:
                if panel_name in alarms_online and budget in alarms_online[panel_name] and det_name in alarms_online[panel_name][budget]:
                    return alarms_online[panel_name][budget][det_name]
            except Exception:
                pass
        pser = pstar_series[panel_name][budget].get(det_name)
        alarms, _thr = dynamic_quantile_threshold(
            scores_dict[det_name],
            pser,
            window=int(args.rq_window),
            min_periods=int(args.rq_min_periods),
            cooldown_months=int(getattr(args, "cooldown_eff", getattr(args, "cooldown_months", 0))),
            max_episode_months=int(getattr(args, "max_episode_months", 0)),
        )
        return alarms

    # Clean alarms (null) per panel/budget/det
    alarms_clean = {
        p: {
            b: {d: make_alarms(scores_clean[p], b, p, d) for d in dets_by_panel[p]}
            for b in budgets
        }
        for p in Z_panels_clean
    }

    # -------------------------------------------------------------------------
    # Budget compliance exports (time-resolved traces + deterministic regime slicing)
    # Initialize to safe empties (used later by regime contract v2)
    df_comp = pd.DataFrame()
    df_comp_sum = pd.DataFrame()
    # -------------------------------------------------------------------------
    try:
        trace_rows: list[pd.DataFrame] = []
        mask_ser = pd.Series(eval_mask, index=Z_full.index)

        ctrl_win = int(getattr(args, "p_ctrl_window", 60))
        ctrl_min = int(getattr(args, "p_min_control_months", 12))
        tiw_target = float(getattr(args, "tiw_budget", 0.0) or 0.0)

        def _warning_from_alarm(a: pd.Series) -> pd.Series:
            w = apply_merge_gap_array(a.values.astype(int), int(getattr(args, "merge_gap", 1)))
            return pd.Series(w, index=a.index, name="warning")

        def _rolling_burden(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            n = len(w)
            ev = np.full(n, np.nan, dtype=float)
            tiw = np.full(n, np.nan, dtype=float)
            for i in range(n):
                if not eval_mask[i]:
                    continue
                j0 = max(0, i + 1 - ctrl_win)
                sub = w[j0:i + 1]
                if len(sub) < ctrl_min:
                    continue
                tiw[i] = float(np.mean(sub))
                ev[i] = float(count_alert_events(sub, np.ones(len(sub), dtype=bool), merge_gap_months=0) / (len(sub) / steps_per_year))
            return ev, tiw

        for p in Z_panels_clean:
            for b in budgets:
                for d in dets_by_panel[p]:
                    s = scores_clean[p][d]
                    a = alarms_clean[p][b][d]
                    w_ser = _warning_from_alarm(a)
                    w = w_ser.values.astype(int)

                    ev_roll, tiw_roll = _rolling_burden(w)

                    pser = pstar_series.get(p, {}).get(b, {}).get(d, pd.Series(np.nan, index=s.index, name="p_star"))
                    thr_ser = thr_online.get(p, {}).get(b, {}).get(d, pd.Series(np.nan, index=s.index, name="thr"))

                    df_t = pd.DataFrame(
                        {
                            "date": s.index,
                            "panel": p,
                            "detector": d,
                            "budget": float(b),
                            "score": s.values,
                            "thr": thr_ser.values,
                            "p_star": pser.values,
                            "alarm": a.values.astype(int),
                            "warning": w,
                            "ctrl_events_per_year": ev_roll,
                            "ctrl_tiw": tiw_roll,
                        }
                    )
                    df_t = df_t.loc[mask_ser.values].copy()
                    trace_rows.append(df_t)

        df_traces = pd.concat(trace_rows, ignore_index=True) if len(trace_rows) else pd.DataFrame()
        df_traces.to_csv(out_dir / "budget_traces.csv", index=False)

        # Deterministic regime slicing compliance (R4)
        regime_W = int(getattr(args, "regime_window_months", 36))
        regime_stride = int(getattr(args, "regime_stride_months", 12))
        eval_pos = np.where(eval_mask)[0]
        L = len(eval_pos)
        starts = list(range(0, max(0, L - regime_W + 1), regime_stride))
        rows_comp = []
        for s0 in starts:
            i0 = int(eval_pos[s0])
            i1 = int(eval_pos[s0 + regime_W - 1])
            start_date = str(Z_full.index[i0].date())
            end_date = str(Z_full.index[i1].date())

            for p in Z_panels_clean:
                for b in budgets:
                    for d in dets_by_panel[p]:
                        a = alarms_clean[p][b][d].values.astype(int)
                        w = apply_merge_gap_array(a, int(getattr(args, "merge_gap", 1)))
                        w_sub = w[i0:i1 + 1]
                        months = len(w_sub)
                        if months <= 0:
                            continue
                        tiw = float(np.mean(w_sub))
                        ev = float(count_alert_events(w_sub, np.ones(len(w_sub), dtype=bool), merge_gap_months=0) / (months / steps_per_year))

                        ev_ratio = ev / max(float(b), 1e-9)
                        tiw_ratio = tiw / max(tiw_target, 1e-9) if tiw_target > 0 else float("nan")

                        rows_comp.append(
                            {
                                "panel": p,
                                "detector": d,
                                "budget": float(b),
                                "window_start": start_date,
                                "window_end": end_date,
                                "months": int(months),
                                "events_per_year": ev,
                                "tiw": tiw,
                                "target_events_per_year": float(b),
                                "target_tiw": float(tiw_target) if tiw_target > 0 else float("nan"),
                                "events_ratio": ev_ratio,
                                "tiw_ratio": tiw_ratio,
                                "violate_events": int(ev > float(b) * (1.0 + float(getattr(args, "safety_eps", 0.10)))),
                                "violate_tiw": int((tiw_target > 0) and (tiw > tiw_target * (1.0 + float(getattr(args, "safety_eps", 0.10))))),
                            }
                        )

        df_comp = pd.DataFrame(rows_comp)
        # Contract-style pass/fail per regime 
        if len(df_comp):
            df_comp["viol_any"] = (df_comp["violate_events"].astype(int) | df_comp["violate_tiw"].astype(int)).astype(int)
            df_comp["contract_pass"] = (1 - df_comp["viol_any"]).astype(int)
            conds = [
                (df_comp["violate_events"].astype(int) == 1) & (df_comp["violate_tiw"].astype(int) == 1),
                (df_comp["violate_events"].astype(int) == 1) & (df_comp["violate_tiw"].astype(int) == 0),
                (df_comp["violate_events"].astype(int) == 0) & (df_comp["violate_tiw"].astype(int) == 1),
            ]
            choices = ["events+tiw", "events", "tiw"]
            df_comp["contract_fail_reason"] = np.select(conds, choices, default="")
        df_comp.to_csv(out_dir / "safety_compliance.csv", index=False)

        if len(df_comp):
            df_comp_sum = (
                df_comp.groupby(["panel", "detector", "budget"], as_index=False)
                .agg(
                    max_events_ratio=("events_ratio", "max"),
                    max_tiw_ratio=("tiw_ratio", "max"),
                    frac_violate_events=("violate_events", "mean"),
                    frac_violate_tiw=("violate_tiw", "mean"),
                    pass_rate=("contract_pass", "mean"),
                    frac_violate_any=("viol_any", "mean"),
                    n_windows=("months", "size"),
                )
            )
            df_comp_sum.to_csv(out_dir / "safety_compliance_summary.csv", index=False)

        # Uniform-placement overlap diagnostic on the clean warning stream (NOT used for NC-2).
        if bool(getattr(args, "do_perm_control", 0)):
            dur = int(getattr(args, "duration", 6))
            rows_ctl = []
            for p in Z_panels_clean:
                for b in budgets:
                    for d in dets_by_panel[p]:
                        a = alarms_clean[p][b][d].values.astype(int)
                        w = apply_merge_gap_array(a, int(getattr(args, "merge_gap", 1)))
                        w_eval = w[eval_mask].astype(int)
                        if len(w_eval) < dur:
                            continue
                        # exact (deterministic) probability over all valid placements
                        win_sum = np.convolve(w_eval, np.ones(dur, dtype=int), mode="valid")
                        perm_detect = float(np.mean(win_sum > 0))
                        perm_tiw = float(np.mean(win_sum / float(dur)))  # mean TIW inside the window
                        rows_ctl.append(
                            {
                                "panel": p,
                                "detector": d,
                                "budget": float(b),
                                "duration_months": int(dur),
                                "tiw_eval": float(w_eval.mean()) if len(w_eval) else float('nan'),
                                "p_overlap_warning_uniform": perm_detect,
                                "tiw_in_window_uniform": perm_tiw,
                                # Independence baseline (not a claim; just a reference point)
                                "p_overlap_indep": float(1.0 - (1.0 - float(w_eval.mean())) ** dur) if len(w_eval) else float('nan'),
                                "p_overlap_minus_indep": float(perm_detect - (1.0 - (1.0 - float(w_eval.mean())) ** dur)) if len(w_eval) else float('nan'),
                                # Counterfactual metrics on a single stream are identically zero (clean vs clean).
                                "cad_win_zero": 0.0,
                                "dtiw_win_zero": 0.0,
                                "dfae_win_zero": 0.0,
                            }
                        )
            pd.DataFrame(rows_ctl).to_csv(out_dir / "overlap_uniform_warning.csv", index=False)

    except Exception as _e:
        print(f"[WARN] Budget trace/compliance exports failed: {_e}")
    # Incident windows
    wins = make_windows(Z_full.index, rng, n=int(args.n_trials), duration=int(args.duration), earliest=eval_start)
    # Incident scenarios (operating-point summaries)
    pair = tuple(int(x) for x in args.corr_pair.split(","))
    base_corr_extra = {
        # strength may be overridden per trial from --corr_mix_strength_grid
        "strength": float(args.corr_mix_strength),
        "mode": str(args.corr_mix_mode),
        "pair": pair,
        "theta": float(args.corr_theta),
    }

    def _parse_grid(s: str, default: list[float] | None = None) -> list[float]:
        """Parse a comma-separated [0,1] grid.

        If parsing yields no values, return `default` if provided, else [] (caller decides fallback).
        """
        out: list[float] = []
        for tok in str(s).split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.append(float(tok))
            except Exception:
                pass
        out = [float(np.clip(x, 0.0, 1.0)) for x in out]
        if len(out) == 0:
            return list(default) if default is not None else []
        return out

    # Parse attack/drift strength grids once, then store the parsed lists back onto args.
    # This makes downstream code (tables, reports, metadata) self-consistent and avoids
    # string-vs-list mismatches that can silently produce NaNs.
    corr_strength_grid = _parse_grid(getattr(args, "corr_mix_strength_grid", ""), default=[float(args.corr_mix_strength)])
    freeze_strength_grid = _parse_grid(getattr(args, "freeze_strength_grid", ""), default=[0.25, 0.50, 0.75, 1.00])
    drift_strength_grid = _parse_grid(getattr(args, "benign_drift_strength_grid", ""), default=list(freeze_strength_grid))
    # Parse CLI strength-grid strings into float lists for downstream use.
    args.corr_mix_strength_grid = list(corr_strength_grid)
    args.freeze_strength_grid = list(freeze_strength_grid)
    args.benign_drift_strength_grid = list(drift_strength_grid)
    if len(drift_strength_grid) == 0:
        drift_strength_grid = list(freeze_strength_grid) if len(freeze_strength_grid) else [0.5]
    # ---------------------------------------------------------------------
    # benign drift slice as a drift-only compliance test
    # ---------------------------------------------------------------------
    if int(getattr(args, "do_benign_drift", 0)) != 0 and str(getattr(args, "benign_drift_eval_mode", "compliance")) == "compliance":
        try:
            # Drift block placement (persistent enough to affect regime windows)
            drift_start = pd.Timestamp(args.drift_start) if str(getattr(args, "drift_start", "")).strip() else pd.Timestamp(eval_start)
            drift_end = pd.Timestamp(args.drift_end) if str(getattr(args, "drift_end", "")).strip() else pd.Timestamp(Z_full.index.max())
            if int(getattr(args, "drift_block_months", 0)) > 0 and not str(getattr(args, "drift_end", "")).strip():
                if getattr(eval_mod, "TIME_MODE", "calendar_month") == "step":
                    s_pos = int(Z_full.index.searchsorted(drift_start, side="left"))
                    e_pos = min(len(Z_full.index) - 1, s_pos + int(args.drift_block_months) - 1)
                    drift_end = pd.Timestamp(Z_full.index[e_pos])
                else:
                    drift_end = drift_start + pd.DateOffset(months=int(args.drift_block_months) - 1)
            # Clamp to available range
            drift_start = max(drift_start, pd.Timestamp(Z_full.index.min()))
            drift_end = min(drift_end, pd.Timestamp(Z_full.index.max()))
            if drift_end < drift_start:
                drift_end = drift_start

            # Drift scope (surface area)
            drift_scope = str(getattr(args, "drift_scope", "score_all"))
            if drift_scope == "full":
                drift_cols = list(range(int(Z_full.shape[1])))
            elif drift_scope == "score_k":
                k0 = int(getattr(args, "drift_k", 0))
                if k0 <= 0:
                    k0 = int(attack_k)
                k0 = max(1, min(int(k0), len(score_idx)))
                rng_d = np.random.default_rng(int(getattr(args, "drift_seed", 20260207)))
                drift_cols = sorted(int(x) for x in rng_d.choice(score_idx, size=int(k0), replace=False))
            else:
                drift_cols = list(score_idx)
            drift_cols_hash = hashlib.sha1(",".join(map(str, drift_cols)).encode("utf-8")).hexdigest()[:12]

            # Per-delta evaluation (kept small by default; reuse freeze grid if no explicit drift grid)
            deltas = list(drift_strength_grid)
            if len(deltas) == 0:
                deltas = list(freeze_strength_grid) if len(freeze_strength_grid) else [0.2]

            drift_comp_rows = []
            drift_comp_sum_rows = []
            drift_slice_rows = []
            dfc_clean_ref = None  # computed once for flip diagnostics (clean vs drift)

            ctrl_win = int(getattr(args, "p_ctrl_window", 60))
            ctrl_min = int(getattr(args, "p_min_control_months", 12))
            tiw_target = float(getattr(args, "tiw_budget", 0.0) or 0.0)
            eps = float(getattr(args, "safety_eps", 0.10))
            drift_mask = np.asarray((Z_full.index >= drift_start) & (Z_full.index <= drift_end))

            def _warning_from_alarm(a: pd.Series) -> np.ndarray:
                return apply_merge_gap_array(a.values.astype(int), int(getattr(args, "merge_gap", 1))).astype(int)

            def _rolling_burden(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                n = len(w)
                ev = np.full(n, np.nan, dtype=float)
                tiw = np.full(n, np.nan, dtype=float)
                for i in range(n):
                    if not eval_mask[i]:
                        continue
                    j0 = max(0, i + 1 - ctrl_win)
                    sub = w[j0:i + 1]
                    if len(sub) < ctrl_min:
                        continue
                    tiw[i] = float(np.mean(sub))
                    ev[i] = float(count_alert_events(sub, np.ones(len(sub), dtype=bool), merge_gap_months=0) / (len(sub) / steps_per_year))
                return ev, tiw

            # Helper: compute drift alarms for a given drifted score dict
            def _compute_online_pstar(scores_dict: dict[str, dict[str, pd.Series]]) -> tuple[dict, dict, dict]:
                pser_map: dict[str, dict[float, dict[str, pd.Series]]] = {p: {b: {} for b in budgets} for p in panels_to_run}
                alarms_on: dict[str, dict[float, dict[str, pd.Series]]] = {p: {b: {} for b in budgets} for p in panels_to_run}
                thr_on: dict[str, dict[float, dict[str, pd.Series]]] = {p: {b: {} for b in budgets} for p in panels_to_run}
                if args.alert_policy == "event_budgeted" and int(getattr(args, "online_recal", 1)) == 1:
                    for p in panels_to_run:
                        for b in budgets:
                            for d in dets_by_panel[p]:
                                init_p = tuned_p[p][b].get(
                                    d,
                                    float(
                                        np.clip(
                                            float(b) / steps_per_year,
                                            float(getattr(args, "p_star_floor", 1e-6)),
                                            float(getattr(args, "p_star_ceil", 0.98)),
                                        )
                                    ),
                                )
                                _a, _thr, pser = apply_event_budgeted_threshold_online(
                                    scores_dict[p][d],
                                    init_p=float(init_p),
                                    target_events_per_year=float(b),
                                    eval_mask=eval_mask,
                                    window=int(args.rq_window),
                                    min_periods=int(args.rq_min_periods),
                                    merge_gap_months=int(args.merge_gap),
                                    max_time_in_warning=(None if float(args.tiw_budget) <= 0 else float(args.tiw_budget)),
                                    guardband=float(args.guardband),
                                    control_window=int(getattr(args, "p_ctrl_window", 60)),
                                    update_every=int(getattr(args, "p_update_every", 1)),
                                    eta=float(getattr(args, "p_eta", 0.30)),
                                    p_lo=float(getattr(args, "p_star_floor", 1e-6)),
                                    p_hi=float(getattr(args, "p_star_ceil", 0.98)),
                                    max_up=float(getattr(args, "p_max_up", 1.25)),
                                    max_down=float(getattr(args, "p_max_down", 0.60)),
                                    min_control_months=int(getattr(args, "p_min_control_months", 12)),
                                    allow_loosen=int(getattr(args, "allow_loosen", 1)),
                                    cooldown_months=int(getattr(args, "cooldown_eff", getattr(args, "cooldown_months", 0))),
                                    max_episode_months=int(getattr(args, "max_episode_months", 0)),
                                )
                                pser_map[p][b][d] = pser
                                alarms_on[p][b][d] = _a
                                thr_on[p][b][d] = _thr
                else:
                    for p in panels_to_run:
                        for b in budgets:
                            for d in dets_by_panel[p]:
                                p0 = tuned_p[p][b].get(
                                    d,
                                    float(
                                        np.clip(
                                            float(b) / steps_per_year,
                                            float(getattr(args, "p_star_floor", 1e-6)),
                                            float(getattr(args, "p_star_ceil", 0.98)),
                                        )
                                    ),
                                )
                                pser_map[p][b][d] = pd.Series(float(p0), index=Z_full.index, name="p_star")
                return pser_map, alarms_on, thr_on

            def _make_alarms_generic(scores_panel: dict[str, pd.Series], panel_name: str, budget: float, det_name: str,
                                      pser_map: dict, alarms_on: dict) -> pd.Series:
                # CUSUM uses chart alarms on the base score (same as clean)
                if det_name.startswith('cusum_'):
                    base = det_name[len('cusum_'):]
                    s_base = scores_panel.get(base)
                    if s_base is None:
                        return pd.Series(np.zeros(len(eval_mask), dtype=int), index=Z_full.index, name=f"alarm_{det_name}")
                    h = cusum_h.get(panel_name, {}).get(budget, {}).get(det_name)
                    if h is None or (not np.isfinite(float(h))):
                        tiw_cap = cusum_tiw_cap_for_budget(float(budget))
                        h, _st = calibrate_cusum_h_operational_occupancy_from_base(
                            s_base,
                            train_mask=train_mask_full,
                            target_events_per_year=float(budget),
                            tiw_cap=float(tiw_cap),
                            k=float(args.cusum_k),
                            hold_months=int(args.cusum_hold_months),
                            cooldown_months=int(getattr(args, "cooldown_eff", getattr(args, "cooldown_months", 0))),
                            max_episode_months=int(getattr(args, "max_episode_months", 0)),
                            merge_gap_months=int(getattr(args, "merge_gap", 0)),
                            event_tol=float(getattr(args, "cusum_event_tol", 0.0)),
                        )
                    a_raw = cusum_chart_alarms_from_base(
                        s_base, train_mask=train_mask_full, h=float(h), k=float(args.cusum_k), hold_months=int(args.cusum_hold_months)
                    )
                    a_man = apply_episode_policy_array(
                        a_raw.values.astype(int),
                        cooldown_months=int(getattr(args, "cooldown_eff", getattr(args, "cooldown_months", 0))),
                        max_episode_months=int(getattr(args, "max_episode_months", 0)),
                    )
                    return pd.Series(a_man.astype(int), index=a_raw.index, name=a_raw.name)

                if args.alert_policy == "topk":
                    return apply_budgeted_topk(scores_panel[det_name], fa_per_year=budget, eval_mask=eval_mask)
                if args.alert_policy == "rolling_quantile":
                    alarms, _thr = apply_budgeted_threshold(scores_panel[det_name], fa_per_year=budget, window=int(args.rq_window), min_periods=int(args.rq_min_periods))
                    return alarms
                # event_budgeted
                if panel_name in alarms_on and budget in alarms_on[panel_name] and det_name in alarms_on[panel_name][budget]:
                    return alarms_on[panel_name][budget][det_name]
                pser = pser_map[panel_name][budget].get(det_name)
                alarms, _thr = dynamic_quantile_threshold(
                    scores_panel[det_name],
                    pser,
                    window=int(args.rq_window),
                    min_periods=int(args.rq_min_periods),
                    cooldown_months=int(getattr(args, "cooldown_eff", getattr(args, "cooldown_months", 0))),
                    max_episode_months=int(getattr(args, "max_episode_months", 0)),
                )
                return alarms

            for delta in deltas:
                drift_mode = str(getattr(args, "drift_mode", "variance_inflation"))
                if drift_mode in ("noise_iid", "noise_ar1"):
                    # Benign noise injection drift (highest-confidence for episode-budget fragility).
                    # Inject into raw score channels (score_all / selected drift_cols) before detector transforms.
                    ref_mode = str(getattr(args, "drift_noise_ref", "train"))
                    cols_names = [Z_full.columns[int(c)] for c in drift_cols]
                    if ref_mode == "preblock":
                        ref_mask = (Z_full.index < drift_start)
                    elif ref_mode == "block":
                        ref_mask = (Z_full.index >= drift_start) & (Z_full.index <= drift_end)
                    else:
                        ref_mask = train_mask_full
                    ref_vals = Z_full.loc[ref_mask, cols_names].values if len(cols_names) else None
                    # Deterministic per-delta seed so comparisons are reproducible.
                    seed0 = int(getattr(args, "drift_noise_seed", 20260208))
                    seed = int(seed0 + 1000003 * int(round(float(delta) * 1000.0)))
                    rng_noise = np.random.default_rng(seed)
                    Z_drift_full, drift_info = apply_noise_injection(
                        Z_full,
                        start=drift_start,
                        end=drift_end,
                        target_cols=drift_cols,
                        delta=float(delta),
                        rng=rng_noise,
                        mode=("ar1" if drift_mode == "noise_ar1" else "iid"),
                        rho=float(getattr(args, "drift_noise_rho", 0.90)),
                        ref_values=ref_vals,
                    )
                    drift_info.update({"mode": drift_mode, "drift_noise_ref": ref_mode, "drift_noise_seed": int(seed), "drift_noise_rho": float(getattr(args, "drift_noise_rho", 0.90))})
                else:
                    Z_drift_full, drift_info = apply_benign_drift(
                        Z_full,
                        start=drift_start,
                        end=drift_end,
                        target_cols=drift_cols,
                        delta=float(delta),
                        mode=drift_mode,
                        mu_mode=str(getattr(args, "drift_mu_mode", "zero")),
                        mu_win=int(getattr(args, "drift_mu_win", 24)),
                    )

                Z_panels_drift = {
                    "full": Z_drift_full,
                    "score_only": Z_drift_full[score_cols].copy(),
                    "all_only": Z_drift_full[all_cols].copy(),
                }

                scores_drift = {p: compute_scores(Z_panels_drift[p], baselines[p], train_mask_full, cov_win=int(args.cov_win)) for p in Z_panels_drift}
                scores_drift["score_only"]["coh_resid_t2"] = score_coherence_resid_t2(Z_panels_drift["all_only"], Z_panels_drift["score_only"], coh_model)
                scores_drift["score_only"]["coh_resid_energy"] = score_coherence_resid_energy(Z_panels_drift["all_only"], Z_panels_drift["score_only"], coh_model)
                for p_name in scores_drift:
                    add_sequential_baselines(
                        scores_drift[p_name],
                        train_mask=train_mask_full,
                        ewma_lambda=float(args.ewma_lambda),
                        cusum_k=float(args.cusum_k),
                        bases=["factor_cov_lrt"],
                    )
                add_sequential_baselines(
                    scores_drift["score_only"],
                    train_mask=train_mask_full,
                    ewma_lambda=float(args.ewma_lambda),
                    cusum_k=float(args.cusum_k),
                    bases=["coh_resid_t2"],
                )

                pstar_drift, alarms_on_drift, _thr_drift = _compute_online_pstar(scores_drift)

                alarms_drift = {
                    p: {
                        b: {d: _make_alarms_generic(scores_drift[p], p, b, d, pstar_drift, alarms_on_drift) for d in dets_by_panel[p]}
                        for b in budgets
                    }
                    for p in panels_to_run
                }

                # Regime slicing compliance on the drift-only (no incidents) stream
                regime_W = int(getattr(args, "regime_window_months", 36))
                regime_stride = int(getattr(args, "regime_stride_months", 12))
                eval_pos = np.where(eval_mask)[0]
                L = len(eval_pos)
                starts = list(range(0, max(0, L - regime_W + 1), regime_stride))
                # Compute clean regime compliance once (for flip diagnostics).
                if dfc_clean_ref is None:
                    rows_clean = []
                    for s0 in starts:
                        i0 = int(eval_pos[s0])
                        i1 = int(eval_pos[s0 + regime_W - 1])
                        start_date = str(Z_full.index[i0].date())
                        end_date = str(Z_full.index[i1].date())
                        for p in panels_to_run:
                            for b in budgets:
                                for d in dets_by_panel[p]:
                                    try:
                                        w0 = _warning_from_alarm(alarms_clean[p][b][d])
                                    except Exception:
                                        continue
                                    w_sub0 = w0[i0:i1 + 1]
                                    months0 = len(w_sub0)
                                    if months0 <= 0:
                                        continue
                                    tiw0 = float(np.mean(w_sub0))
                                    ev0 = float(count_alert_events(w_sub0, np.ones(len(w_sub0), dtype=bool), merge_gap_months=0) / (months0 / steps_per_year))
                                    rows_clean.append(
                                        {
                                            "panel": p,
                                            "detector": d,
                                            "budget": float(b),
                                            "window_start": start_date,
                                            "window_end": end_date,
                                            "events_per_year_clean": ev0,
                                            "tiw_clean": tiw0,
                                            "events_ratio_clean": ev0 / max(float(b), 1e-9),
                                            "tiw_ratio_clean": (tiw0 / max(tiw_target, 1e-9) if tiw_target > 0 else float("nan")),
                                            "violate_events_clean": int(ev0 > float(b) * (1.0 + eps)),
                                            "violate_tiw_clean": int((tiw_target > 0) and (tiw0 > tiw_target * (1.0 + eps))),
                                        }
                                    )
                    dfc_clean_ref = pd.DataFrame(rows_clean)
                    if len(dfc_clean_ref):
                        dfc_clean_ref["viol_any_clean"] = (dfc_clean_ref["violate_events_clean"].astype(int) | dfc_clean_ref["violate_tiw_clean"].astype(int)).astype(int)
                        dfc_clean_ref["contract_pass_clean"] = (1 - dfc_clean_ref["viol_any_clean"]).astype(int)

                rows_comp = []
                for s0 in starts:
                    i0 = int(eval_pos[s0])
                    i1 = int(eval_pos[s0 + regime_W - 1])
                    start_date = str(Z_full.index[i0].date())
                    end_date = str(Z_full.index[i1].date())
                    for p in panels_to_run:
                        for b in budgets:
                            for d in dets_by_panel[p]:
                                w = _warning_from_alarm(alarms_drift[p][b][d])
                                w_sub = w[i0:i1 + 1]
                                months = len(w_sub)
                                if months <= 0:
                                    continue
                                tiw = float(np.mean(w_sub))
                                ev = float(count_alert_events(w_sub, np.ones(len(w_sub), dtype=bool), merge_gap_months=0) / (months / steps_per_year))
                                ev_ratio = ev / max(float(b), 1e-9)
                                tiw_ratio = tiw / max(tiw_target, 1e-9) if tiw_target > 0 else float("nan")
                                rows_comp.append(
                                    {
                                        "drift_delta": float(delta),
                                        "drift_mode": str(getattr(args, "drift_mode", "variance_inflation")),
                                        "drift_scope": drift_scope,
                                        "drift_start": str(drift_start.date()),
                                        "drift_end": str(drift_end.date()),
                                        "drift_cols_hash": drift_cols_hash,
                                        "panel": p,
                                        "detector": d,
                                        "budget": float(b),
                                        "window_start": start_date,
                                        "window_end": end_date,
                                        "months": int(months),
                                        "events_per_year": ev,
                                        "tiw": tiw,
                                        "target_events_per_year": float(b),
                                        "target_tiw": float(tiw_target) if tiw_target > 0 else float("nan"),
                                        "events_ratio": ev_ratio,
                                        "tiw_ratio": tiw_ratio,
                                        "violate_events": int(ev > float(b) * (1.0 + eps)),
                                        "violate_tiw": int((tiw_target > 0) and (tiw > tiw_target * (1.0 + eps))),
                                    }
                                )
                dfc = pd.DataFrame(rows_comp)
                if len(dfc):
                    dfc["viol_any"] = (dfc["violate_events"].astype(int) | dfc["violate_tiw"].astype(int)).astype(int)
                    dfc["contract_pass"] = (1 - dfc["viol_any"]).astype(int)
                drift_comp_rows.append(dfc)

                if len(dfc):
                    dfcs = (
                        dfc.groupby(["drift_delta", "drift_mode", "drift_scope", "panel", "detector", "budget"], as_index=False)
                        .agg(
                            max_events_ratio=("events_ratio", "max"),
                            max_tiw_ratio=("tiw_ratio", "max"),
                            frac_violate_events=("violate_events", "mean"),
                            frac_violate_tiw=("violate_tiw", "mean"),
                            pass_rate=("contract_pass", "mean"),
                            frac_violate_any=("viol_any", "mean"),
                            n_windows=("months", "size"),
                        )
                    )
                    drift_comp_sum_rows.append(dfcs)

                    # Recovery diagnostics from rolling controller burdens (informative, not a theorem)
                    rec_months = []
                    peak_ratios = []
                    fp_rates = []
                    for p in panels_to_run:
                        for b in budgets:
                            for d in dets_by_panel[p]:
                                w = _warning_from_alarm(alarms_drift[p][b][d])
                                # overall FP event rate over eval horizon
                                w_eval = w[eval_mask].astype(int)
                                fp_ev = float(count_alert_events(w_eval, np.ones(len(w_eval), dtype=bool), merge_gap_months=0) / (len(w_eval) / steps_per_year))
                                fp_rates.append(fp_ev)
                                ev_roll, _tiw_roll = _rolling_burden(w)
                                ratio = ev_roll / max(float(b), 1e-9)
                                # peak ratio during drift
                                mask_peak = drift_mask & eval_mask & np.isfinite(ratio)
                                if mask_peak.any():
                                    peak = float(np.nanmax(ratio[mask_peak]))
                                    peak_ratios.append(peak)
                                    # recovery: first month after drift_start where ratio <= (1+eps) after a violation
                                    idx = np.where((Z_full.index >= drift_start) & eval_mask & np.isfinite(ratio))[0]
                                    if len(idx) > 0:
                                        violated = ratio[idx] > (1.0 + eps)
                                        if violated.any():
                                            first_viol = idx[int(np.argmax(violated))]
                                            cand = idx[idx >= first_viol]
                                            ok = ratio[cand] <= (1.0 + eps)
                                            if ok.any():
                                                rec = int(cand[int(np.argmax(ok))] - first_viol)
                                                rec_months.append(rec)
                    # Tail-oriented safety summaries (TMLR-friendly): focus on worst-case overshoot + recovery, not mean pass-rate.
                    events_ratio_p95 = float(np.nanquantile(dfc["events_ratio"].values, 0.95)) if len(dfc) else float("nan")
                    events_ratio_p99 = float(np.nanquantile(dfc["events_ratio"].values, 0.99)) if len(dfc) else float("nan")
                    tiw_ratio_p95 = float(np.nanquantile(dfc["tiw_ratio"].values, 0.95)) if len(dfc) else float("nan")
                    tiw_ratio_p99 = float(np.nanquantile(dfc["tiw_ratio"].values, 0.99)) if len(dfc) else float("nan")
                    max_events_ratio = float(np.nanmax(dfc["events_ratio"].values)) if len(dfc) else float("nan")
                    max_tiw_ratio = float(np.nanmax(dfc["tiw_ratio"].values)) if len(dfc) else float("nan")

                    # Violation streaks across regime windows (overlapping windows, but diagnostic).
                    streaks = []
                    try:
                        dfc_sorted = dfc.sort_values(["panel", "detector", "budget", "window_start"])
                        for _k, gg in dfc_sorted.groupby(["panel", "detector", "budget"]):
                            arr = gg["violate_events"].astype(int).values
                            best = 0
                            cur = 0
                            for v in arr:
                                if int(v) == 1:
                                    cur += 1
                                    if cur > best:
                                        best = cur
                                else:
                                    cur = 0
                            streaks.append(best)
                    except Exception:
                        pass
                    viol_streak_p95 = float(np.nanquantile(np.asarray(streaks, dtype=float), 0.95)) if len(streaks) else float("nan")
                    viol_streak_max = float(np.nanmax(np.asarray(streaks, dtype=float))) if len(streaks) else float("nan")
                    viol_streak_months_p95 = (viol_streak_p95 * float(regime_stride)) if np.isfinite(viol_streak_p95) else float("nan")
                    viol_streak_months_max = (viol_streak_max * float(regime_stride)) if np.isfinite(viol_streak_max) else float("nan")

                    # Flip diagnostics vs clean (how often a previously-compliant window becomes noncompliant under drift).
                    clean_pass_rate_mean = float("nan")
                    flip_1to0_rate = float("nan")
                    flip_0to1_rate = float("nan")
                    pass_rate_delta_vs_clean = float("nan")
                    max_events_ratio_delta_vs_clean = float("nan")
                    if dfc_clean_ref is not None and len(dfc_clean_ref):
                        dfm = dfc.merge(
                            dfc_clean_ref[["panel", "detector", "budget", "window_start", "window_end", "contract_pass_clean", "events_ratio_clean", "tiw_ratio_clean"]],
                            on=["panel", "detector", "budget", "window_start", "window_end"],
                            how="left",
                        )
                        m_ok = dfm["contract_pass_clean"].notna()
                        if bool(m_ok.any()):
                            clean_pass_rate_mean = float(dfm.loc[m_ok, "contract_pass_clean"].mean())
                            drift_pass_rate_mean0 = float(dfm.loc[m_ok, "contract_pass"].mean())
                            pass_rate_delta_vs_clean = float(drift_pass_rate_mean0 - clean_pass_rate_mean)
                            flip_1to0_rate = float(np.mean((dfm.loc[m_ok, "contract_pass_clean"].astype(int) == 1) & (dfm.loc[m_ok, "contract_pass"].astype(int) == 0)))
                            flip_0to1_rate = float(np.mean((dfm.loc[m_ok, "contract_pass_clean"].astype(int) == 0) & (dfm.loc[m_ok, "contract_pass"].astype(int) == 1)))
                            try:
                                max_events_ratio_delta_vs_clean = float(np.nanmax(dfm.loc[m_ok, "events_ratio"].values - dfm.loc[m_ok, "events_ratio_clean"].values))
                            except Exception:
                                pass

                    drift_slice_rows.append(
                        {
                            "drift_delta": float(delta),
                            "drift_mode": str(getattr(args, "drift_mode", "variance_inflation")),
                            "drift_scope": drift_scope,
                            "drift_start": str(drift_start.date()),
                            "drift_end": str(drift_end.date()),
                            "drift_cols_hash": drift_cols_hash,
                            "online_recal": int(getattr(args, "online_recal", 1)),
                            "pass_rate_mean": float(dfcs["pass_rate"].mean()),
                            "clean_pass_rate_mean": float(clean_pass_rate_mean),
                            "pass_rate_delta_vs_clean": float(pass_rate_delta_vs_clean),
                            "flip_1to0_rate": float(flip_1to0_rate),
                            "flip_0to1_rate": float(flip_0to1_rate),
                            # worst-case & tail overshoot (episode budget)
                            "events_ratio_p95": float(events_ratio_p95),
                            "events_ratio_p99": float(events_ratio_p99),
                            "max_events_ratio_overall": float(max_events_ratio),
                            "max_events_ratio_delta_vs_clean": float(max_events_ratio_delta_vs_clean),
                            # TIW tails (secondary)
                            "tiw_ratio_p95": float(tiw_ratio_p95),
                            "tiw_ratio_p99": float(tiw_ratio_p99),
                            "max_tiw_ratio_overall": float(max_tiw_ratio),
                            # violation streaks
                            "viol_streak_windows_p95": float(viol_streak_p95),
                            "viol_streak_windows_max": float(viol_streak_max),
                            "viol_streak_months_p95": float(viol_streak_months_p95),
                            "viol_streak_months_max": float(viol_streak_months_max),
                            # FP burden + controller recovery diagnostics
                            "fp_events_per_year_mean": float(np.nanmean(fp_rates)) if len(fp_rates) else float("nan"),
                            "peak_events_ratio_roll_median": float(np.nanmedian(peak_ratios)) if len(peak_ratios) else float("nan"),
                            "peak_events_ratio_roll_p95": float(np.nanquantile(np.asarray(peak_ratios, dtype=float), 0.95)) if len(peak_ratios) else float("nan"),
                            "recovery_months_median": float(np.nanmedian(rec_months)) if len(rec_months) else float("nan"),
                            "recovery_months_p90": float(np.nanquantile(np.asarray(rec_months, dtype=float), 0.90)) if len(rec_months) else float("nan"),
                            "recovery_months_max": float(np.nanmax(np.asarray(rec_months, dtype=float))) if len(rec_months) else float("nan"),
                        }
                    )

            # Write combined drift-only outputs
            if len(drift_comp_rows) > 0:
                pd.concat(drift_comp_rows, ignore_index=True).to_csv(out_dir / "drift_safety_compliance.csv", index=False)
            if len(drift_comp_sum_rows) > 0:
                pd.concat(drift_comp_sum_rows, ignore_index=True).to_csv(out_dir / "drift_safety_compliance_summary.csv", index=False)
            if len(drift_slice_rows) > 0:
                pd.DataFrame(drift_slice_rows).to_csv(out_dir / "drift_slice_summary.csv", index=False)

                # Paper-facing drift slice summary (focused slice to avoid dilution):
                #   panel == score_only AND budget >= 0.75
                #
                # Implementation note: this summary is computed directly from the
                # drift-only regime compliance table, with optional clean references
                # used only for flip diagnostics.
                try:
                    if len(drift_comp_rows) > 0:
                        _df_all = pd.concat(drift_comp_rows, ignore_index=True)
                    else:
                        _df_all = pd.DataFrame()

                    _mask_paper = (
                        (_df_all.get("panel", "").astype(str) == "score_only")
                        & (_df_all.get("budget", np.nan).astype(float) >= 0.75)
                    ) if len(_df_all) else np.asarray([], dtype=bool)

                    _dfp = _df_all.loc[_mask_paper].copy() if len(_df_all) else pd.DataFrame()
                    if len(_dfp) > 0:
                        # attach clean reference columns for flip diagnostics (if available)
                        if dfc_clean_ref is not None and len(dfc_clean_ref):
                            _dfp = _dfp.merge(
                                dfc_clean_ref[[
                                    "panel", "detector", "budget", "window_start", "window_end",
                                    "contract_pass_clean", "events_ratio_clean", "tiw_ratio_clean",
                                ]],
                                on=["panel", "detector", "budget", "window_start", "window_end"],
                                how="left",
                            )

                        _out_rows = []

                        # local quantile helper (numpy API differs across versions)
                        def drift_quant(a, q):
                            a = np.asarray(a, dtype=float)
                            a = a[~np.isnan(a)]
                            if a.size == 0:
                                return float("nan")
                            try:
                                return float(np.quantile(a, q, method="linear"))
                            except TypeError:
                                return float(np.quantile(a, q, interpolation="linear"))

                        for (dd, dm, ds), gg in _dfp.groupby(["drift_delta", "drift_mode", "drift_scope"], dropna=False):
                            evr = gg["events_ratio"].astype(float).values
                            evr = evr[~np.isnan(evr)]
                            tir = gg["tiw_ratio"].astype(float).values
                            tir = tir[~np.isnan(tir)]

                            row = {
                                "slice": "score_only_budget_ge_0p75",
                                "drift_delta": float(dd),
                                "drift_mode": str(dm),
                                "drift_scope": str(ds),
                                "online_recal": int(getattr(args, "online_recal", 1)),
                                "n_rows": int(len(gg)),
                                "pass_rate_mean": float(np.nanmean(gg.get("contract_pass", np.nan).astype(float).values)),
                            }

                            if len(evr) > 0:
                                row.update({
                                    "events_ratio_p95": float(drift_quant(evr, 0.95)),
                                    "events_ratio_p99": float(drift_quant(evr, 0.99)),
                                    "max_events_ratio_overall": float(np.max(evr)),
                                })
                            if len(tir) > 0:
                                row.update({
                                    "tiw_ratio_p95": float(drift_quant(tir, 0.95)),
                                    "tiw_ratio_p99": float(drift_quant(tir, 0.99)),
                                    "max_tiw_ratio_overall": float(np.max(tir)),
                                })

                            if "contract_pass_clean" in gg.columns:
                                m_ok = gg["contract_pass_clean"].notna()
                                if bool(m_ok.any()):
                                    clean_pass = gg.loc[m_ok, "contract_pass_clean"].astype(int)
                                    drift_pass = gg.loc[m_ok, "contract_pass"].astype(int)
                                    row["clean_pass_rate_mean"] = float(np.mean(clean_pass))
                                    row["pass_rate_delta_vs_clean"] = float(np.mean(drift_pass) - np.mean(clean_pass))
                                    row["flip_1to0_rate"] = float(np.mean((clean_pass == 1) & (drift_pass == 0)))
                                    row["flip_0to1_rate"] = float(np.mean((clean_pass == 0) & (drift_pass == 1)))
                                    try:
                                        row["max_events_ratio_delta_vs_clean"] = float(
                                            np.nanmax(
                                                gg.loc[m_ok, "events_ratio"].astype(float).values
                                                - gg.loc[m_ok, "events_ratio_clean"].astype(float).values
                                            )
                                        )
                                    except Exception:
                                        pass

                            _out_rows.append(row)

                        if _out_rows:
                            pd.DataFrame(_out_rows).to_csv(out_dir / "drift_slice_summary_paper.csv", index=False)
                except Exception as _e:
                    print(f"[WARN] Drift slice paper-summary failed: {_e}")
        except Exception as _e:
            print(f"[WARN] Drift slice exports failed: {_e}")

    # ---------------------------------------------------------------------
    # Incident scenarios (freeze/corr_mix) + optional drift-as-incident ablation
    # ---------------------------------------------------------------------
    scenarios = []
    # Drift-compliance runs historically retain zero-strength incident rows as
    # the reference population used by their contract exports.  Other runs may
    # still use a singleton zero grid to disable an incident family.
    retain_drift_reference_rows = (
        int(getattr(args, "do_benign_drift", 0)) != 0
        and str(getattr(args, "benign_drift_eval_mode", "compliance")) == "compliance"
    )
    if retain_drift_reference_rows or not (
        len(freeze_strength_grid) == 1 and float(freeze_strength_grid[0]) == 0.0
    ):
        scenarios.append(("freeze", None))
    if retain_drift_reference_rows or not (
        len(corr_strength_grid) == 1 and float(corr_strength_grid[0]) == 0.0
    ):
        scenarios.append(("corr_mix", base_corr_extra))

    if int(getattr(args, "do_benign_drift", 0)) != 0 and str(getattr(args, "benign_drift_eval_mode", "compliance")) == "incident_like":
        scenarios.append(("gradual_bias", {"mode": "ramp", "sigma_win": 24}))

    trial_rows: list[dict] = []
    nc1_rows: list[dict] = []
    paired_sweeps = int(getattr(args, "paired_intensity_sweeps", 1)) != 0

    for kind, extra in scenarios:
        # Paired intensity sweeps (common random numbers): for each base trial/window, we evaluate *all* strengths
        # under the same (window, attacked columns) to reduce variance and make the threat-surface monotone.
        if kind == "freeze":
            strength_grid = list(freeze_strength_grid)
        elif kind == "corr_mix":
            strength_grid = list(corr_strength_grid)
        elif kind in ("benign_drift", "gradual_bias"):
            strength_grid = list(drift_strength_grid)
        else:
            strength_grid = [1.0]

        if len(strength_grid) == 0:
            strength_grid = [0.35]

        n_base = int(args.n_trials)
        n_strength = int(len(strength_grid))
        n_eff = int(n_base * n_strength) if paired_sweeps else int(n_base)

        for t in range(n_eff):
            base_t = int(t // n_strength) if paired_sweeps else int(t)
            s_idx = int(t % n_strength)
            win = wins[base_t]

            # In paired sweeps, the attacked subset is *paired across strengths* (depends only on base_t).
            if int(attack_k) >= int(len(score_idx)):
                target_cols = list(score_idx)
            else:
                rng_cols = np.random.default_rng(base_seed + 104729 * base_t + (0 if kind == "freeze" else 7))
                target_cols = sorted(int(x) for x in rng_cols.choice(score_idx, size=int(attack_k), replace=False))
            attack_cols_hash = hashlib.sha1(",".join(map(str, target_cols)).encode("utf-8")).hexdigest()[:12]

            if kind == "freeze":
                a_strength = float(strength_grid[int(s_idx)])
                extra_t = {"strength": float(a_strength)}
            elif kind == "corr_mix" and extra is not None:
                a_strength = float(strength_grid[int(s_idx)])
                extra_t = dict(extra)
                extra_t["strength"] = float(a_strength)
                # Specialize mu/U to the chosen target columns so corr_mix is shape-consistent under localization.
                tpos = [score_pos[int(c)] for c in target_cols]
                mu_sub = mu_score_inj[tpos]
                extra_t["mu"] = mu_sub
                mode = str(args.corr_mix_mode)
                if mode == "pca_rotate":
                    U_sub = U_score_inj[tpos, :]
                    extra_t["U"] = U_sub
                    # Paired sweeps: fix the PCA-space rotation per base trial so curves are monotone.
                    rng_rot = np.random.default_rng(base_seed + 104729 * int(base_t) + 17)
                    A = rng_rot.normal(size=(int(U_sub.shape[1]), int(U_sub.shape[1])))
                    Qr, _ = np.linalg.qr(A)
                    extra_t["R"] = Qr
                elif mode == "coord_mix":
                    kk = int(len(tpos))
                    rng_rot = np.random.default_rng(base_seed + 104729 * int(base_t) + 29)
                    A = rng_rot.normal(size=(kk, kk))
                    Qc, _ = np.linalg.qr(A)
                    extra_t["Q"] = Qc
            elif kind in ("benign_drift", "gradual_bias"):
                a_strength = float(strength_grid[int(s_idx)])
                extra_t = dict(extra) if extra is not None else {}
                extra_t["strength"] = float(a_strength)
            else:
                a_strength = float("nan")
                extra_t = extra

            inj_kind = ("drift" if kind in ("benign_drift", "gradual_bias") else kind)
            inc = Incident(
                kind=inj_kind,
                start=win.start,
                end=win.end,
                target_cols=target_cols,
                severity=(float(a_strength) if inj_kind == "drift" else None),
                extra=extra_t,
            )

            # For paired intensity sweeps, keep injection randomness fixed across strengths.
            rng_inj = np.random.default_rng(base_seed + 17 * int(base_t) + 999)
            Z_cor_full, _info = inject(Z_full, inc, rng=rng_inj)

            # NC-1 invariant: corruption does not modify non-target columns,
            # and does not modify target columns outside the incident window.
            mask_win = np.asarray((Z_full.index >= pd.Timestamp(win.start)) & (Z_full.index <= pd.Timestamp(win.end)))
            Z0 = Z_full.values
            Za = Z_cor_full.values
            tgt = set(int(c) for c in target_cols)
            all_idx = list(range(int(Z_full.shape[1])))
            non_tgt_idx = [i for i in all_idx if i not in tgt]
            if len(non_tgt_idx) == 0:
                nc1_max_abs_non_target_in_win = 0.0
                nc1_max_abs_non_target_all = 0.0
            else:
                nc1_max_abs_non_target_in_win = float(np.max(np.abs(Za[mask_win][:, non_tgt_idx] - Z0[mask_win][:, non_tgt_idx]))) if mask_win.any() else 0.0
                nc1_max_abs_non_target_all = float(np.max(np.abs(Za[:, non_tgt_idx] - Z0[:, non_tgt_idx])))
            mask_out = ~mask_win
            nc1_max_abs_target_outside_win = float(np.max(np.abs(Za[mask_out][:, list(target_cols)] - Z0[mask_out][:, list(target_cols)]))) if len(target_cols) > 0 else 0.0
            nc1_pass = (nc1_max_abs_non_target_all <= 1e-12) and (nc1_max_abs_target_outside_win <= 1e-12)
            if not nc1_pass:
                raise RuntimeError(
                    f"NC-1 invariance violated (trial={t}, kind={kind}, strength={a_strength}): "
                    f"non_target_all={nc1_max_abs_non_target_all:.3e}, target_outside={nc1_max_abs_target_outside_win:.3e}"
                )

            Z_panels_cor = {
                "full": Z_cor_full,
                "score_only": Z_cor_full[score_cols].copy(),
                "all_only": Z_cor_full[all_cols].copy(),
            }

            scores_cor = {p: compute_scores(Z_panels_cor[p], baselines[p], train_mask_full, cov_win=int(args.cov_win)) for p in Z_panels_cor}
            # Cross-panel coherence scores (score-only panel only) on the (possibly) corrupted stream
            scores_cor["score_only"]["coh_resid_t2"] = score_coherence_resid_t2(Z_panels_cor["all_only"], Z_panels_cor["score_only"], coh_model)
            scores_cor["score_only"]["coh_resid_energy"] = score_coherence_resid_energy(Z_panels_cor["all_only"], Z_panels_cor["score_only"], coh_model)
            # Sequential baselines (EWMA/CUSUM) on corrupted stream (same transforms).
            for p_name in scores_cor:
                add_sequential_baselines(
                    scores_cor[p_name],
                    train_mask=train_mask_full,
                    ewma_lambda=float(args.ewma_lambda),
                    cusum_k=float(args.cusum_k),
                    bases=["factor_cov_lrt"],
                )

            add_sequential_baselines(
                scores_cor["score_only"],
                train_mask=train_mask_full,
                ewma_lambda=float(args.ewma_lambda),
                cusum_k=float(args.cusum_k),
                bases=["coh_resid_t2"],
            )

            inc_win = Window(name="incident", start=win.start, end=win.end)

            for panel_name in Z_panels_cor:
                for budget in budgets:
                    for det_name in dets_by_panel[panel_name]:
                        cusum_h_used = float("nan")
                        cusum_fae_train = float("nan")
                        cusum_tiw_train = float("nan")
                        cusum_tiw_cap = float("nan")
                        cusum_train_months = float("nan")
                        if det_name.startswith("cusum_"):
                            base = det_name[len("cusum_"):]
                            scores = scores_cor[panel_name][base]
                            scores0 = scores_clean[panel_name][base]
                            # Local (recent) calibration to avoid long-horizon drift from the global pre-2020 baseline.
                            # Uses ONLY months strictly before the trial window start (no lookahead).
                            start_ts = pd.Timestamp(win.start)
                            if getattr(eval_mod, "TIME_MODE", "calendar_month") == "step":
                                s_pos = int(Z_full.index.searchsorted(start_ts, side="left"))
                                lb_pos = max(0, s_pos - int(args.cusum_local_train_months))
                                lb = pd.Timestamp(Z_full.index[lb_pos])
                            else:
                                lb = start_ts - pd.DateOffset(months=int(args.cusum_local_train_months))
                            train_mask_cu = np.asarray((Z_full.index < start_ts) & (Z_full.index >= lb))
                            if int(train_mask_cu.sum()) < 12:
                                train_mask_cu = train_mask_full
                            # Cache occupancy-aware (events+TIW) calibration by (panel, budget, detector, window_start).
                            _k = (panel_name, float(budget), det_name, str(start_ts.date()))
                            if _k in cusum_local_cache:
                                _rec = cusum_local_cache[_k]
                                h_local = float(_rec["h"])
                                cusum_fae_train = float(_rec.get("train_fae", np.nan))
                                cusum_tiw_train = float(_rec.get("train_tiw", np.nan))
                                cusum_tiw_cap = float(_rec.get("tiw_cap", np.nan))
                                cusum_train_months = float(_rec.get("train_months", np.nan))
                            else:
                                tiw_cap = cusum_tiw_cap_for_budget(float(budget))
                                h_local, st = calibrate_cusum_h_operational_occupancy_from_base(
                                    scores0,
                                    train_mask=train_mask_cu,
                                    target_events_per_year=float(budget),
                                    tiw_cap=float(tiw_cap),
                                    k=float(args.cusum_k),
                                    hold_months=int(args.cusum_hold_months),
                                    cooldown_months=int(getattr(args, "cooldown_eff", getattr(args, "cooldown_months", 0))),
                                    max_episode_months=int(getattr(args, "max_episode_months", 0)),
                                    merge_gap_months=int(getattr(args, "merge_gap", 0)),
                                    event_tol=float(getattr(args, "cusum_event_tol", 0.0)),
                                )
                                cusum_fae_train = float(st.get("fae", np.nan))
                                cusum_tiw_train = float(st.get("tiw", np.nan))
                                cusum_tiw_cap = float(tiw_cap)
                                cusum_train_months = float(st.get("tr_months", np.nan))
                                cusum_local_cache[_k] = {
                                    "panel": panel_name,
                                    "budget": float(budget),
                                    "detector": det_name,
                                    "base": base,
                                    "window_start": str(start_ts.date()),
                                    "train_start": str(pd.Timestamp(lb).date()),
                                    "train_end": (str(pd.Timestamp(Z_full.index[max(0, int(Z_full.index.searchsorted(start_ts, side="left") - 1))]).date()) if getattr(eval_mod, "TIME_MODE", "calendar_month") == "step" else str((start_ts - pd.DateOffset(months=1)).date())),
                                    "cusum_k": float(args.cusum_k),
                                    "hold_months": int(args.cusum_hold_months),
                                    "tiw_cap": float(tiw_cap),
                                    "h": float(h_local),
                                    "train_fae": float(cusum_fae_train),
                                    "train_tiw": float(cusum_tiw_train),
                                    "train_events": float(st.get("n_events", np.nan)),
                                    "train_months": float(cusum_train_months),
                                }
                            cusum_h_used = float(h_local)

                            alarms = cusum_chart_alarms_from_base(
                                scores,
                                train_mask=train_mask_cu,
                                h=float(h_local),
                                k=float(args.cusum_k),
                                hold_months=int(args.cusum_hold_months),
                            )
                            alarms0 = cusum_chart_alarms_from_base(
                                scores0,
                                train_mask=train_mask_cu,
                                h=float(h_local),
                                k=float(args.cusum_k),
                                hold_months=int(args.cusum_hold_months),
                            )
                            # Enforce episode policy on chart alarms (no lookahead).
                            _cd = int(getattr(args, "cooldown_months", 0))
                            _mx = int(getattr(args, "max_episode_months", 0))
                            alarms = pd.Series(
                                apply_episode_policy_array(alarms.values.astype(int), cooldown_months=_cd, max_episode_months=_mx),
                                index=alarms.index,
                                name=alarms.name,
                            )
                            alarms0 = pd.Series(
                                apply_episode_policy_array(alarms0.values.astype(int), cooldown_months=_cd, max_episode_months=_mx),
                                index=alarms0.index,
                                name=alarms0.name,
                            )
                        else:
                            scores = scores_cor[panel_name][det_name]
                            scores0 = scores_clean[panel_name][det_name]
                            alarms = make_alarms(scores_cor[panel_name], budget, panel_name, det_name)
                            alarms0 = alarms_clean[panel_name][budget][det_name]

                        # evaluation on corrupted stream
                        m = eval_task_B_integrity(
                            scores,
                            alarms,
                            incident=inc_win,
                            eval_mask=eval_mask,
                            merge_gap_months=int(args.merge_gap),
                            sla_months=int(args.timeliness_sla_months),
                        )
                        # null baseline: clean alarms measured on same window
                        m0 = eval_task_B_integrity(
                            scores0,
                            alarms0,
                            incident=inc_win,
                            eval_mask=eval_mask,
                            merge_gap_months=int(args.merge_gap),
                            sla_months=int(args.timeliness_sla_months),
                        )

                        row = {
                            "kind": kind,
                            "panel": panel_name,
                            "detector": det_name,
                            "budget": float(budget),
                            "trial": int(t),
                            "base_trial": int(base_t),
                            "strength_idx": int(s_idx),
                                "paired_sweeps": int(paired_sweeps),
                                "nc1_pass": int(nc1_pass),
                                "nc1_max_abs_non_target_all": float(nc1_max_abs_non_target_all),
                                "nc1_max_abs_target_outside_win": float(nc1_max_abs_target_outside_win),
                                "attack_k": int(attack_k),
                            "attack_cols_hash": str(attack_cols_hash),
                            "start": str(win.start.date()),
                            "end": str(win.end.date()),
                            "incident_start": str(inc_win.start.date()),
                            "incident_end": str(inc_win.end.date()),
                            "effect_size": effect_size(scores, inc_win, eval_mask),
                            "cusum_h": float(cusum_h_used),
                            "cusum_train_fae": float(cusum_fae_train),
                            "cusum_train_tiw": float(cusum_tiw_train),
                            "cusum_tiw_cap": float(cusum_tiw_cap),
                            "cusum_train_months": float(cusum_train_months),
                        }
                        row.update(m)

                        row.update(
                            cad_metrics_from_alarms(
                                alarms_att=alarms,
                                alarms_clean=alarms0,
                                inc=inc_win,
                                merge_gap_months=int(args.merge_gap),
                                post_months=int(getattr(args, "cad_post_months", 3)),
                            )
                        )

                        ip, ip_name = intensity_proxy_from_z(
                            kind=kind,
                            Z_clean=Z_full,
                            Z_att=Z_cor_full,
                            inc=inc_win,
                            target_cols=target_cols,
                            dur_months=int(args.duration),
                        )
                        # attack intensity parameter used for threat-surface curves
                        row["attack_strength"] = float(a_strength)

                        # Store the data-derived proxy for diagnostics (can be noisy for corr_mix).
                        row["intensity_proxy_data"] = float(ip)
                        row["intensity_proxy_data_name"] = str(ip_name)

                        # Primary axis for stealth/intensity curves:
                        #   - freeze: variance-suppression proxy (1-std_ratio)
                        #   - corr_mix: use the explicit attack-strength parameter.
                        if kind == "corr_mix":
                            row["intensity_proxy"] = float(a_strength)
                            row["intensity_proxy_name"] = "corr_mix_strength"
                        else:
                            row["intensity_proxy"] = float(ip)
                            row["intensity_proxy_name"] = str(ip_name)
                        
                        if args.alert_policy == "event_budgeted":
                            if det_name.startswith("cusum_"):
                                # Chart-based CUSUM does not use p*(t).
                                row["p_star"] = float("nan")
                                row["p_star_mean"] = float("nan")
                                row["p_star_last"] = float("nan")
                                row["p_star_init"] = float("nan")
                            else:
                                pser = pstar_series[panel_name][budget].get(det_name)
                                if pser is not None:
                                    p_eval = np.asarray(pser.to_numpy(dtype=float)[eval_mask], dtype=float)
                                    row["p_star"] = float(np.nanmedian(p_eval)) if p_eval.size else float("nan")
                                    row["p_star_mean"] = float(np.nanmean(p_eval)) if p_eval.size else float("nan")
                                    row["p_star_last"] = float(p_eval[-1]) if p_eval.size else float("nan")
                                else:
                                    row["p_star"] = float("nan")
                                    row["p_star_mean"] = float("nan")
                                    row["p_star_last"] = float("nan")
                                row["p_star_init"] = float(tuned_p[panel_name][budget].get(det_name, float("nan")))
                        else:
                            row["p_star"] = float("nan")
                            row["p_star_mean"] = float("nan")
                            row["p_star_last"] = float("nan")
                            row["p_star_init"] = float("nan")
                        row["detected_null"] = m0["detected"]
                        row["delay_months_null"] = m0["delay_months"]
                        row["delay_months_null_cens"] = m0.get("delay_months_cens", m0["delay_months"])
                        row["fae_null"] = float(m0.get("false_alert_events_per_year", np.nan))
                        row["false_alert_events_count"] = int(m0.get("false_alert_events_count", 0)) if ("false_alert_events_count" in m0) else int(round(row["fae_null"] * float(m0.get("false_alert_years", 1.0))))
                        row["false_alert_years"] = float(m0.get("false_alert_years", m0.get("eval_years", 1.0)))
                        row["false_alert_eval_months"] = int(m0.get("false_alert_eval_months", m0.get("eval_months", 0)))
                        # Count-based burden matching (derived on null/benign stream).
                        row["tiw_null"] = float(m0.get("time_in_warning", np.nan))
                        row["delta_fae"] = float(m.get("false_alert_events_per_year", np.nan) - m0.get("false_alert_events_per_year", np.nan))
                        row["delta_tiw"] = float(m.get("time_in_warning", np.nan) - m0.get("time_in_warning", np.nan))

                        # Full contract pass/fail (includes timeliness SLA)
                        eps = float(args.safety_eps)
                        budget_target = float(row.get('budget', float('nan')))
                        fae_null = float(row.get('fae_null', float('nan')))
                        budget_ok = (not np.isnan(fae_null)) and (fae_null <= budget_target * (1.0 + eps))
                        tiw_cap = float(args.tiw_budget) if (args.tiw_budget is not None and float(args.tiw_budget) > 0) else float('nan')
                        tiw_null = float(row.get('tiw_null', float('nan')))
                        tiw_ok = True if np.isnan(tiw_cap) else ((not np.isnan(tiw_null)) and (tiw_null <= tiw_cap * (1.0 + eps)))
                        sla_ok = int(row.get('event_timely_sla', 0)) == 1
                        row['contract_pass_full'] = int(budget_ok and tiw_ok and sla_ok)
                        row['contract_fail_budget'] = int(not budget_ok)
                        row['contract_fail_tiw'] = int(not tiw_ok)
                        row['contract_fail_sla'] = int(not sla_ok)
                        any_sla_ok = int(row.get('event_anywarn_timely_sla', 0)) == 1
                        row['contract_pass_full_anywarn'] = int(budget_ok and tiw_ok and any_sla_ok)
                        trial_rows.append(row)

    df_trials = pd.DataFrame(trial_rows)
    df_trials.to_csv(out_dir / "trial_level_metrics.csv", index=False)

    # Contract-style summaries (full contract includes timeliness SLA)
    try:
        df_contract = (df_trials.groupby(["panel", "detector", "kind", "attack_strength", "budget"], dropna=False)
                            .agg(n_trials=("contract_pass_full", "size"),
                                 contract_pass_full_rate=("contract_pass_full", "mean"),
                                 contract_pass_full_anywarn_rate=("contract_pass_full_anywarn", "mean"),
                                 event_success_rate=("event_success", "mean"),
                                 event_timely_sla_rate=("event_timely_sla", "mean"),
                                 event_anywarn_success_rate=("event_anywarn_success", "mean"),
                                 event_anywarn_timely_sla_rate=("event_anywarn_timely_sla", "mean"),
                                 budget_overshoot_rate=("contract_fail_budget", "mean"),
                                 tiw_overshoot_rate=("contract_fail_tiw", "mean"),
                                 sla_fail_rate=("contract_fail_sla", "mean"))
                            .reset_index())
        df_contract.to_csv(out_dir / "contract_full_summary.csv", index=False)


        # Regime contract v2: compliance-aware event summaries.
        # Anchor each incident to the regime window containing its onset on the clean stream, and
        # compute compliant rates by restricting to incidents whose onset regime passes the clean burden contract.
        if "df_comp" in locals() and isinstance(df_comp, pd.DataFrame) and len(df_comp):
            try:
                df_windows = df_comp.copy()
                df_windows["window_start_dt"] = pd.to_datetime(df_windows["window_start"], errors="coerce")
                df_windows["window_end_dt"] = pd.to_datetime(df_windows["window_end"], errors="coerce")
                df_windows = df_windows.sort_values(["panel", "detector", "budget", "window_start_dt"]).reset_index(drop=True)

                df_tr = df_trials.copy()
                df_tr["incident_start_dt"] = pd.to_datetime(df_tr["incident_start"], errors="coerce")
                for col in [
                    "regime_window_start", "regime_window_end", "regime_budget_ok", "regime_tiw_ok",
                    "regime_contract_pass", "regime_contract_fail_reason"
                ]:
                    df_tr[col] = np.nan

                # Build a lookup of windows by (panel, detector, budget)
                win_lookup = {}
                for (p, d, b), subw in df_windows.groupby(["panel", "detector", "budget"], dropna=False):
                    win_lookup[(p, d, b)] = subw

                for i, r in df_tr.iterrows():
                    key = (r.get("panel"), r.get("detector"), r.get("budget"))
                    subw = win_lookup.get(key)
                    ts = r.get("incident_start_dt")
                    if subw is None or pd.isna(ts):
                        continue
                    cand = subw[(subw["window_start_dt"] <= ts) & (subw["window_end_dt"] >= ts)]
                    if len(cand) == 0:
                        continue
                    w = cand.iloc[-1]
                    df_tr.at[i, "regime_window_start"] = w.get("window_start")
                    df_tr.at[i, "regime_window_end"] = w.get("window_end")
                    df_tr.at[i, "regime_budget_ok"] = w.get("budget_ok")
                    df_tr.at[i, "regime_tiw_ok"] = w.get("tiw_ok")
                    df_tr.at[i, "regime_contract_pass"] = w.get("contract_pass")
                    df_tr.at[i, "regime_contract_fail_reason"] = w.get("contract_fail_reason")

                # Derive pass/fail that includes SLA (timeliness) at the incident level
                df_tr["regime_contract_pass"] = pd.to_numeric(df_tr["regime_contract_pass"], errors="coerce").fillna(0).astype(int)
                df_tr["regime_pass_with_sla"] = ((df_tr["regime_contract_pass"] == 1) & (df_tr["event_timely_sla"] == 1)).astype(int)
                df_tr["regime_pass_with_sla_anywarn"] = ((df_tr["regime_contract_pass"] == 1) & (df_tr["event_anywarn_timely_sla"] == 1)).astype(int)

                def _q(series, q):
                    s = pd.to_numeric(series, errors="coerce")
                    s = s[np.isfinite(s)]
                    return float(s.quantile(q)) if len(s) else float("nan")

                grp = ["panel", "detector", "kind", "attack_strength", "budget"]
                rows = []
                for key, g in df_tr.groupby(grp, dropna=False):
                    g_comp = g[g["regime_contract_pass"] == 1]
                    rows.append({
                        "panel": key[0], "detector": key[1], "kind": key[2], "attack_strength": key[3], "budget": key[4],
                        "n_trials": int(len(g)),
                        "event_success_rate": float(pd.to_numeric(g["event_success"], errors="coerce").mean()),
                        "event_timely_sla_rate": float(pd.to_numeric(g["event_timely_sla"], errors="coerce").mean()),
                        "event_delay_p50": _q(g["event_delay_months_cens"], 0.50),
                        "event_delay_p90": _q(g["event_delay_months_cens"], 0.90),
                        "event_ttt_p50": _q(g["event_time_to_trigger_months_cens"], 0.50),
                        "event_ttt_p90": _q(g["event_time_to_trigger_months_cens"], 0.90),
                        "regime_pass_rate": float(pd.to_numeric(g["regime_contract_pass"], errors="coerce").mean()),
                        "compliant_event_success_rate": float(pd.to_numeric(g_comp["event_success"], errors="coerce").mean()) if len(g_comp) else float("nan"),
                        "compliant_event_timely_sla_rate": float(pd.to_numeric(g_comp["event_timely_sla"], errors="coerce").mean()) if len(g_comp) else float("nan"),
                        "full_pass_rate_regime_plus_sla": float(pd.to_numeric(g["regime_pass_with_sla"], errors="coerce").mean()),
                        "full_pass_rate_regime_plus_sla_anywarn": float(pd.to_numeric(g["regime_pass_with_sla_anywarn"], errors="coerce").mean()),
                    })
                pd.DataFrame(rows).to_csv(out_dir / "regime_contract_v2_summary.csv", index=False)

                # Per-regime (window) view: attach detection/SLA rates to each anchored regime window.
                # Use group sizes (not a specific column like `seed`) so this is robust to different df_tr schemas.
                _keys = [
                    "panel", "detector", "budget", "kind", "attack_strength", "regime_window_start", "regime_window_end"
                ]
                _gb = df_tr.groupby(_keys, dropna=False)
                df_win_rates = _gb.agg(
                    event_success_rate=("event_success", "mean"),
                    event_timely_sla_rate=("event_timely_sla", "mean"),
                    event_delay_p50=("event_delay_months_cens", lambda s: _q(s, 0.50)),
                    event_delay_p90=("event_delay_months_cens", lambda s: _q(s, 0.90)),
                    event_ttt_p50=("event_time_to_trigger_months_cens", lambda s: _q(s, 0.50)),
                    event_ttt_p90=("event_time_to_trigger_months_cens", lambda s: _q(s, 0.90)),
                    regime_contract_pass=("regime_contract_pass", "mean"),
                ).reset_index()
                df_win_rates["n_trials"] = _gb.size().values
                df_win_rates.to_csv(out_dir / "regime_contract_v2_by_window.csv", index=False)

            except Exception as e:
                print(f"[WARN] Failed to write regime contract v2 CSVs: {e}")

        # Per-regime view: map each incident to deterministic regimes/windows, then summarize pass rate
        if "incident_start" in df_trials.columns and args.regime_window_months and args.regime_stride_months:
            try:
                inc_ts = pd.to_datetime(df_trials["incident_start"])
                eval_dates = pd.to_datetime(Z_full.index[eval_mask])
                W = int(args.regime_window_months)
                S = int(args.regime_stride_months)
                reg_starts = list(range(0, max(0, len(eval_dates) - W + 1), S))
                reg_windows = [(int(s), eval_dates[int(s)], eval_dates[int(s + W - 1)]) for s in reg_starts]

                def assign_regime(ts):
                    # Deterministic anchor regime: choose the *latest* window_start that still contains ts
                    for rid in range(len(reg_windows) - 1, -1, -1):
                        s_idx, s_dt, e_dt = reg_windows[rid][0], reg_windows[rid][1], reg_windows[rid][2]
                        if s_dt <= ts <= e_dt:
                            return rid
                    return -1

                df_trials["incident_regime_id"] = [assign_regime(t) for t in inc_ts]
                df_reg = (df_trials[df_trials["incident_regime_id"] >= 0]
                          .groupby(["incident_regime_id", "panel", "detector", "kind", "attack_strength", "budget"], dropna=False)
                          .agg(n_trials=("contract_pass_full", "size"),
                               contract_pass_full_rate=("contract_pass_full", "mean"),
                               event_timely_sla_rate=("event_timely_sla", "mean"),
                               budget_overshoot_rate=("contract_fail_budget", "mean"),
                               tiw_overshoot_rate=("contract_fail_tiw", "mean"))
                          .reset_index())
                # Add window dates for the regime id
                df_reg["regime_start"] = df_reg["incident_regime_id"].apply(lambda rid: str(reg_windows[int(rid)][1].date()))
                df_reg["regime_end"] = df_reg["incident_regime_id"].apply(lambda rid: str(reg_windows[int(rid)][2].date()))
                df_reg.to_csv(out_dir / "contract_full_by_regime.csv", index=False)
            except Exception as e:
                print(f"[WARN] Failed to compute contract_full_by_regime.csv: {e}")
            
    except Exception as e:
        print(f"[WARN] Failed to write contract summary CSVs: {e}")

    # Write local CUSUM calibration records (one row per unique window_start).
    if len(cusum_local_cache) > 0:
        pd.DataFrame(list(cusum_local_cache.values())).to_csv(out_dir / "cusum_calibration.csv", index=False)

    # Summaries
    # Define summaries per attack intensity to avoid averaging across strengths.
    # Averaging strengths can blur intensity trends and monotonicity diagnostics.
    group_cols = ["kind", "panel", "detector", "budget", "attack_strength"]
    rows = []
    for (kind, panel_name, det_name, budget, attack_strength), g in df_trials.groupby(group_cols):
        s = summarize(g.to_dict(orient="records"))
        rows.append(
            {
                "kind": kind,
                "panel": panel_name,
                "detector": det_name,
                "budget": float(budget),
                "attack_strength": float(attack_strength),
                **s,
            }
        )
    df_sum = pd.DataFrame(rows)

    # Step 2: Block-bootstrap confidence intervals (CIs) over trials (circular moving blocks).
    # This provides uncertainty bands for uplift/delay/TIW/FAE at each operating point.
    df_ci = None  # may remain None if bootstrap disabled
    if int(getattr(args, "ci_n_boot", 0)) > 0:
        df_ci = compute_block_bootstrap_cis(
            df_trials=df_trials,
            group_cols=group_cols,
            n_boot=int(args.ci_n_boot),
            block_size=int(args.ci_block_size),
            alpha=float(args.ci_alpha),
            seed=int(args.ci_seed),
        )
        df_ci.to_csv(out_dir / "trial_summary_ci.csv", index=False)
        df_sum = df_sum.merge(df_ci, on=group_cols, how="left")

    df_sum = df_sum.sort_values(["kind", "panel", "budget", "uplift_detect_rate"], ascending=[True, True, True, False])
    df_sum.to_csv(out_dir / "trial_summary.csv", index=False)

    # Budget tracking diagnostics (achieved vs nominal)
    try:
        df_track = df_sum.copy()
        if "mean_fae_null" in df_track.columns:
            df_track["fae_null_error"] = df_track["mean_fae_null"] - df_track["budget"]
            df_track["fae_null_rel_error"] = df_track["fae_null_error"] / df_track["budget"].replace({0.0: float("nan")})
        if ("mean_tiw_null" in df_track.columns) and (getattr(args, "tiw_budget", None) is not None):
            tiw_t = float(args.tiw_budget)
            df_track["tiw_null_error"] = df_track["mean_tiw_null"] - tiw_t
        cols = [c for c in [
            "kind", "panel", "detector", "budget",
            "mean_fae_null", "fae_null_error", "fae_null_rel_error",
            "mean_tiw_null", "tiw_null_error",
        ] if c in df_track.columns]
        df_track[cols].to_csv(out_dir / "budget_tracking.csv", index=False)
    except Exception as e:
        print(f"[warn] budget tracking export failed: {e}")

    
    # Paper-facing artifacts (tables + curves + frontiers)
    write_paper_artifacts(df_trials=df_trials, df_sum=df_sum, df_ci=df_ci, out_dir=out_dir, args=args)



# =========================
# Plotting + envelope helpers
# =========================

def _slug(s: str, max_len: int = 120) -> str:
    """Filesystem-safe slug."""
    s = str(s)
    s = s.replace("|", "_").replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("_")
    if not s:
        s = "x"
    if len(s) > int(max_len):
        s = s[: int(max_len)]
    return s


def _fig_dir(out_dir: Path) -> Path:
    d = Path(out_dir) / "figs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_curve(sub: pd.DataFrame, out_dir: Path, kind: str, panel: str, detector: str, y: str) -> None:
    """Plot metric y vs nominal budget for one (kind,panel,detector)."""
    if sub is None or len(sub) == 0 or (y not in sub.columns) or ("budget" not in sub.columns):
        return
    df = sub.copy()
    df = df[np.isfinite(df["budget"].values.astype(float))]
    df = df[np.isfinite(df[y].values.astype(float))]
    if len(df) == 0:
        return

    df = df.sort_values("budget")
    x = df["budget"].to_numpy(dtype=float)
    v = df[y].to_numpy(dtype=float)

    lo_col = f"{y}_ci_lo"
    hi_col = f"{y}_ci_hi"
    have_ci = (lo_col in df.columns) and (hi_col in df.columns)

    fig_dir = _fig_dir(out_dir)
    fn = fig_dir / f"budget_curve_{_slug(y)}_{_slug(kind)}_{_slug(panel)}_{_slug(detector)}.png"

    plt.figure(figsize=(6.0, 4.0))
    plt.plot(x, v, marker="o", linewidth=1.5)
    if have_ci:
        lo = df[lo_col].to_numpy(dtype=float)
        hi = df[hi_col].to_numpy(dtype=float)
        ok = np.isfinite(lo) & np.isfinite(hi)
        if np.any(ok):
            plt.fill_between(x[ok], lo[ok], hi[ok], alpha=0.15)
    plt.xlabel("Nominal budget (events/year)")
    plt.ylabel(y)
    plt.title(f"{kind} | {panel} | {detector}")
    plt.tight_layout()
    plt.savefig(fn, dpi=160)
    plt.close()


def plot_frontier(sub: pd.DataFrame, out_dir: Path, kind: str, panel: str, y: str) -> None:
    """Scatter metric y vs achieved clean burden (or nominal if unavailable) for one (kind,panel)."""
    if sub is None or len(sub) == 0 or (y not in sub.columns):
        return
    xcol = "mean_fae_null" if "mean_fae_null" in sub.columns else "mean_false_alert_events_per_year"
    if xcol not in sub.columns:
        return
    df = sub.copy()
    df = df[np.isfinite(df[xcol].values.astype(float))]
    df = df[np.isfinite(df[y].values.astype(float))]
    if len(df) == 0:
        return
    x = df[xcol].to_numpy(dtype=float)
    v = df[y].to_numpy(dtype=float)

    fig_dir = _fig_dir(out_dir)
    fn = fig_dir / f"frontier_{_slug(y)}_{_slug(kind)}_{_slug(panel)}.png"

    plt.figure(figsize=(6.0, 4.0))
    plt.scatter(x, v, s=18)
    plt.xlabel(xcol)
    plt.ylabel(y)
    plt.title(f"{kind} | {panel}")
    plt.tight_layout()
    plt.savefig(fn, dpi=160)
    plt.close()


def envelope_points(df: pd.DataFrame, xcol: str, ycol: str, mode: str = "max", eps: float = 1e-12) -> pd.DataFrame:
    """Compute a 1D envelope of points as x increases.

    For mode='max': keep points that strictly improve the best-seen y.
    For mode='min': keep points that strictly improve (decrease) the best-seen y.
    """
    if df is None or len(df) == 0:
        return df.iloc[0:0].copy()
    if xcol not in df.columns or ycol not in df.columns:
        return df.iloc[0:0].copy()
    sub = df.copy()
    sub = sub[np.isfinite(sub[xcol].values.astype(float))]
    sub = sub[np.isfinite(sub[ycol].values.astype(float))]
    if len(sub) == 0:
        return sub
    sub = sub.sort_values([xcol, ycol], ascending=[True, (mode != "max")]).reset_index(drop=True)

    keep = []
    if mode == "max":
        best = -np.inf
        for i in range(len(sub)):
            yv = float(sub.loc[i, ycol])
            if yv > best + float(eps):
                keep.append(i)
                best = yv
    else:
        best = np.inf
        for i in range(len(sub)):
            yv = float(sub.loc[i, ycol])
            if yv < best - float(eps):
                keep.append(i)
                best = yv
    out = sub.loc[keep].copy().reset_index(drop=True)
    return out


def plot_paper_frontier(env: pd.DataFrame, out_dir: Path, kind: str, panel: str, ycol: str) -> None:
    """Plot an envelope (frontier) curve."""
    if env is None or len(env) == 0 or (ycol not in env.columns):
        return
    xcol = "mean_fae_null" if "mean_fae_null" in env.columns else "mean_false_alert_events_per_year"
    if xcol not in env.columns:
        # fall back to the first numeric column not ycol
        for c in env.columns:
            if c != ycol and np.issubdtype(env[c].dtype, np.number):
                xcol = c
                break
        if xcol not in env.columns:
            return

    df = env.copy()
    df = df[np.isfinite(df[xcol].values.astype(float))]
    df = df[np.isfinite(df[ycol].values.astype(float))]
    if len(df) == 0:
        return
    df = df.sort_values(xcol)

    x = df[xcol].to_numpy(dtype=float)
    v = df[ycol].to_numpy(dtype=float)

    fig_dir = _fig_dir(out_dir)
    fn = fig_dir / f"paper_frontier_{_slug(ycol)}_{_slug(kind)}_{_slug(panel)}.png"

    plt.figure(figsize=(6.0, 4.0))
    plt.plot(x, v, marker="o", linewidth=1.8)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(f"{kind} | {panel}")
    plt.tight_layout()
    plt.savefig(fn, dpi=180)
    plt.close()



def write_paper_artifacts(df_trials: pd.DataFrame, df_sum: pd.DataFrame, df_ci, out_dir: Path, args):
    """Create paper-facing artifacts (tables + curves) from trial- and op-point summaries."""
    # -------------------------
    # 1) Nominal budget ~ 1.0 table (paper Table 1 anchor)
    # -------------------------
    budget_table_target = 1.0
    budgets_avail = np.sort(pd.unique(df_sum["budget"].values))
    if budgets_avail.size == 0:
        return

    budget_table_selected = float(budgets_avail[np.argmin(np.abs(budgets_avail - budget_table_target))])

    # Paper-facing operating-point tables select a single attack strength per family.
    # Default: use the strongest intensity for each attack family.
    strength_policy = getattr(args, "paper_strength_policy", "max")
    freeze_grid = getattr(args, "freeze_strength_grid", [])
    corr_grid = getattr(args, "corr_mix_strength_grid", [])

    def _select_strength(kind: str):
        """Select a single paper operating-point strength per incident kind.

        This keeps the achieved-band summary deterministic.
        """
        drift_grid = getattr(args, 'benign_drift_strength_grid', [])
        if kind == 'freeze':
            grid = freeze_grid
        elif kind == 'corr_mix':
            grid = corr_grid
        elif kind in ('benign_drift','gradual_bias','drift'):
            grid = drift_grid
        else:
            grid = []
        try:
            g = [float(x) for x in (grid or [])]
        except Exception:
            g = []
        if not g:
            return None
        g = sorted(g)
        pol = str(strength_policy).lower()
        if pol == 'min':
            return g[0]
        if pol == 'median':
            return g[len(g)//2]
        return g[-1]

    # This runner's contract experiments populate `kind` with the *attack type*
    # `kind` encodes the incident family (e.g., freeze, corr_mix); keep those rows.
    # for a non-existent "integrity" label.
    bmask = (df_sum["budget"] == budget_table_selected)
    df_b1 = df_sum.loc[bmask].copy()
    df_b1["budget_target"] = float(budget_table_target)
    df_b1["budget_selected"] = float(budget_table_selected)
    df_b1.to_csv(out_dir / "table_budget1.csv", index=False)

    if df_ci is not None and "budget" in df_ci.columns:
        df_b1_ci = df_ci.loc[(df_ci["budget"] == budget_table_selected)].copy()
        df_b1_ci["budget_target"] = float(budget_table_target)
        df_b1_ci["budget_selected"] = float(budget_table_selected)
        df_b1_ci.to_csv(out_dir / "table_budget1_ci.csv", index=False)
    # -------------------------
    # 2) Achieved-budget (matched-burden) tables
    # Matched-burden tables follow --matched_mode (budget_only vs joint) and an explicit band definition.
    #
    #   - Default band: **count-based** (nearest feasible integer episode count under the eval horizon),
    #     with tolerance-derived integer width. This avoids "mean looks close but every trial is OOB"
    #     failures caused by quantization of events/year.
    #   - Optional alternative: rate-based band (events/year), which can be mathematically infeasible at low budgets.
    # -------------------------
    tol_budget = float(getattr(args, "matched_tol", 0.10) or 0.10)
    _tol_tiw_raw = getattr(args, "matched_tiw_tol", None)
    tol_tiw = float(_tol_tiw_raw) if (_tol_tiw_raw is not None) else tol_budget
    tiw_cap = float(getattr(args, "tiw_budget", 0.0) or 0.0)
    matched_mode = str(getattr(args, "matched_mode", "budget_only") or "budget_only").lower()
    matched_budget_band = str(getattr(args, "matched_budget_band", "count") or "count").lower()
    if matched_budget_band not in ("count", "rate"):
        matched_budget_band = "count"

    def _years_col(df: pd.DataFrame):
        for c in ("false_alert_years", "false_alert_years_clean", "years_eval", "years"):
            if c in df.columns:
                return c, 1.0
        if "false_alert_eval_months" in df.columns:
            return "false_alert_eval_months", (1.0 / steps_per_year)
        return None, 1.0

    def _ok_budget_band(df: pd.DataFrame, B: float) -> np.ndarray:
        """Matched-burden membership for event budget (null stream)."""
        B = float(B)
        if matched_budget_band == "count" and "false_alert_events_count" in df.columns:
            ycol, yscale = _years_col(df)
            if ycol is not None:
                years = df[ycol].astype(float).values * float(yscale)
                cnt = df["false_alert_events_count"].astype(float).values
                tgt = np.rint(years * B)  # nearest feasible integer count per-trial
                bw = np.maximum(1.0, np.ceil(tol_budget * np.maximum(tgt, 1.0)))  # integer band width (>=1)
                return np.isfinite(cnt) & np.isfinite(tgt) & (np.abs(cnt - tgt) <= bw)

        # Fallback: rate-based band (events/year)
        if "fae_null" in df.columns:
            fae = df["fae_null"].astype(float).values
        elif "false_alert_events_per_year" in df.columns:
            fae = df["false_alert_events_per_year"].astype(float).values
        else:
            return np.zeros(len(df), dtype=bool)
        return np.isfinite(fae) & (np.abs(fae - B) <= tol_budget * max(B, 1e-12))

    def _ok_tiw_safe(df: pd.DataFrame) -> np.ndarray:
        """TIW safety membership (null stream): one-sided cap."""
        if (tiw_cap <= 0.0) or ("tiw_null" not in df.columns):
            return np.ones(len(df), dtype=bool)
        tiw = df["tiw_null"].astype(float).values
        return np.isfinite(tiw) & (tiw <= (1.0 + tol_tiw) * tiw_cap)

    use_joint = (matched_mode == "joint") and (tiw_cap > 0.0)

    # --- 2A) Achieved-band table at the paper anchor budget (Table 1 companion) ---
    df_int = df_trials.loc[df_trials["budget"].astype(float) == float(budget_table_selected)].copy()

    rows_ach = []
    if len(df_int) > 0 and {"kind", "panel", "detector"}.issubset(df_int.columns):
        for (kind, panel, detector), g_all in df_int.groupby(["kind", "panel", "detector"]):
            # Select a single intensity for paper-facing table (avoid averaging across strengths).
            strength_sel = _select_strength(str(kind))
            if strength_sel is not None and "attack_strength" in g_all.columns:
                v = g_all["attack_strength"].astype(float)
                g_all = g_all[np.isclose(v.values, float(strength_sel), atol=1e-12, rtol=0.0)]

            # If filtering leaves no rows (e.g., missing strength), still emit an explicit infeasible row.
            if len(g_all) == 0:
                rows_ach.append({
                    "kind": kind,
                    "panel": panel,
                    "detector": detector,
                    "budget_target_paper": float(budget_table_target),
                    "budget_nominal_selected": float(budget_table_selected),
                    "matched_mode": matched_mode,
                    "matched_budget_band": matched_budget_band,
                    "matched_tol_budget": float(tol_budget),
                    "tiw_cap": float(tiw_cap) if (tiw_cap > 0) else float("nan"),
                    "matched_tol_tiw": float(tol_tiw) if (tiw_cap > 0) else float("nan"),
                    "paper_strength_policy": strength_policy,
                    # Always materialize a numeric value (no NaNs) so Table 1 is self-contained.
                    # For kinds without a meaningful intensity (e.g., controls), we set 0.0 and mark it as not-applicable.
                    "attack_strength_selected": float(strength_sel) if (strength_sel is not None) else 0.0,
                    "attack_strength_selected_is_applicable": 1 if (strength_sel is not None) else 0,
                    "n_trials_total": 0,
                    "n_trials_budget_in_band": 0,
                    "n_trials_joint_in_band": 0,
                    "n_trials_primary": 0,
                    "coverage_primary": 0.0,
                    "primary_status": "INFEASIBLE",
                })
                continue
            okB = _ok_budget_band(g_all, budget_table_target)
            okT = _ok_tiw_safe(g_all)
            g_budget = g_all.loc[okB].copy()
            g_joint = g_all.loc[okB & okT].copy()
            g_primary = g_joint if use_joint else g_budget

            ycol, yscale = _years_col(g_all)
            years_med = float(np.nanmedian(g_all[ycol].astype(float).values * float(yscale))) if ycol is not None else float("nan")
            tgt_cnt = int(round(float(budget_table_target) * years_med)) if (np.isfinite(years_med) and years_med > 0) else -1
            band_w = int(max(1, int(np.ceil(tol_budget * max(tgt_cnt, 1))))) if (matched_budget_band == "count" and tgt_cnt >= 0) else -1
            implied_rate = (tgt_cnt / years_med) if (tgt_cnt >= 0 and np.isfinite(years_med) and years_med > 0) else float("nan")

            n_total = int(len(g_all))
            n_budget = int(len(g_budget))
            n_joint = int(len(g_joint))
            n_primary = int(len(g_primary))

            row = {
                "kind": kind,
                "panel": panel,
                "detector": detector,

                "paper_strength_policy": str(strength_policy),
                "attack_strength_selected": float(strength_sel) if (strength_sel is not None) else 0.0,
                "attack_strength_selected_is_applicable": 1 if (strength_sel is not None) else 0,

                "budget_target_paper": float(budget_table_target),
                "budget_nominal_selected": float(budget_table_selected),

                "matched_mode": matched_mode,
                "matched_budget_band": matched_budget_band,
                "matched_tol_budget": float(tol_budget),
                "tiw_cap": float(tiw_cap) if (tiw_cap > 0) else float("nan"),
                "matched_tol_tiw": float(tol_tiw) if (tiw_cap > 0) else float("nan"),

                "years_eval_median": float(years_med),
                "target_count_nearest": int(tgt_cnt),
                "count_band_width": int(band_w),
                "implied_rate_nearest": float(implied_rate),

                "n_trials_total": n_total,
                "n_trials_budget_in_band": n_budget,
                "n_trials_joint_in_band": n_joint,
                "n_trials_primary": n_primary,

                "coverage_budget": float(n_budget / max(n_total, 1)),
                "coverage_joint": float(n_joint / max(n_total, 1)),
                "coverage_primary": float(n_primary / max(n_total, 1)),
            }

            row["primary_status"] = "OK" if (n_primary > 0) else "INFEASIBLE"
            if n_primary > 0:
                row.update(summarize(g_primary.to_dict("records")))
            rows_ach.append(row)

    df_ach = pd.DataFrame(rows_ach)
    if len(df_ach) > 0:
        df_ach = df_ach.sort_values(["kind", "panel", "detector"]).reset_index(drop=True)

    # Compute NC-2 stats inline so Table 1 is self-contained.
    # This avoids brittle post-joins / join-key drift in paper-facing CSVs.
    #
    # Definitions (paper-facing):
    #   nc2_p0   = null detection probability on the clean replay.
    #   p_hat    = observed detection rate in the achieved-band operating point.
    #   p_binom  = one-sided binomial tail p-value P[X>=k | n, nc2_p0].
    #   nc2_pass = (p_binom < alpha).
    #
    # NOTE: We compute this on the same detection definition used by event-level evaluation (trial-level 'detected').
    import math as _math

    def _binom_sf_ge(n: int, k: int, p0: float) -> float:
        """Survival function P[X>=k] for Binomial(n, p0) computed stably."""
        try:
            n = int(n)
            k = int(k)
            p0 = float(p0)
        except Exception:
            return float('nan')
        if n < 0 or k < 0:
            return float('nan')
        if k <= 0:
            return 1.0
        if k > n:
            return 0.0
        if not (0.0 <= p0 <= 1.0):
            return float('nan')
        # log-sum-exp for stability
        log_terms = []
        log_p = -1e100 if p0 == 0.0 else _math.log(p0)
        log_q = -1e100 if p0 == 1.0 else _math.log(1.0 - p0)
        for i in range(k, n + 1):
            # log C(n,i) + i log p + (n-i) log q
            log_c = _math.lgamma(n + 1) - _math.lgamma(i + 1) - _math.lgamma(n - i + 1)
            log_terms.append(log_c + i * log_p + (n - i) * log_q)
        m = max(log_terms)
        if m <= -1e99:
            return 0.0
        s = sum(_math.exp(t - m) for t in log_terms)
        return float(_math.exp(m) * s)

    nc2_alpha = 0.05
    # Start with empty/default cols so the table always has the NC-2 fields.
    for _c in ["nc2_p0", "p_hat", "p_binom", "nc2_pass", "nc2_alpha"]:
        if _c not in df_ach.columns:
            df_ach[_c] = float('nan')

    # Only compute if null stats are present in the summaries.
    if "detect_rate_null" in df_ach.columns and "detect_rate" in df_ach.columns:
        df_ach["nc2_alpha"] = float(nc2_alpha)

        # p0 from the null detection rate; p_hat from observed detection rate.
        _p0 = pd.to_numeric(df_ach["detect_rate_null"], errors="coerce")
        _ph = pd.to_numeric(df_ach["detect_rate"], errors="coerce")
        _n = pd.to_numeric(df_ach.get("n_trials", df_ach.get("n_trials_primary", pd.Series([float('nan')]*len(df_ach)))), errors="coerce")

        df_ach["nc2_p0"] = _p0
        df_ach["p_hat"] = _ph

        # Compute p-values row-wise (n is small in practice; log-sum-exp keeps this stable).
        pvals = []
        for p0, ph, n in zip(_p0.tolist(), _ph.tolist(), _n.tolist()):
            if not (p0 is not None and ph is not None and n is not None):
                pvals.append(float('nan'))
                continue
            if not (_math.isfinite(p0) and _math.isfinite(ph) and _math.isfinite(n) and n > 0):
                pvals.append(float('nan'))
                continue
            # Conservative: round to nearest integer count
            k = int(round(float(ph) * int(n)))
            pvals.append(_binom_sf_ge(int(n), k, float(p0)))
        df_ach["p_binom"] = pvals
        df_ach["nc2_pass"] = (pd.to_numeric(df_ach["p_binom"], errors="coerce") < float(nc2_alpha)).astype(float)

    # Keep strength metadata consistent across exported tables.
    if "attack_strength" in df_ach.columns and "attack_strength_selected" in df_ach.columns:
        _as = pd.to_numeric(df_ach["attack_strength"], errors="coerce")
        _ass = pd.to_numeric(df_ach["attack_strength_selected"], errors="coerce")
        _fix = _as.notna() & (_ass.isna() | ((_as - _ass).abs() > 1e-9))
        if _fix.any():
            df_ach.loc[_fix, "attack_strength_selected"] = _as[_fix].values

    # Write a numeric-raw version and a paper-friendly version with structured infeasibility markers.
    df_ach_raw = df_ach.copy()
    df_ach_raw.to_csv(out_dir / "table_achieved_band1_raw.csv", index=False)

    df_ach_pretty = df_ach.copy()
    # Rows with zero matched coverage are *not missing*: they are INFEASIBLE under the contract/band.
    cov = df_ach_pretty.get("coverage_primary", pd.Series([np.nan]*len(df_ach_pretty)))
    infeas = (~cov.astype(float).fillna(0).astype(float).gt(0))
    df_ach_pretty["row_status"] = np.where(infeas, "INFEASIBLE", "OK")
    df_ach_pretty["infeasible_reason"] = np.where(infeas, "no_trials_in_matched_band", "")
    id_cols = set(["kind","panel","detector","budget_target_paper","attack_strength_selected","matched_mode","matched_budget_band","matched_tol_budget","matched_tol_rate","count_band_width","target_count_nearest","implied_rate_nearest","n_trials_total","n_trials_primary","coverage_primary","primary_status","row_status","infeasible_reason"])
    metric_cols = [c for c in df_ach_pretty.columns if c not in id_cols]
    # Mark all metric fields explicitly (avoids NaNs in paper-facing tables).
    for c in metric_cols:
        if c in df_ach_pretty.columns:
            df_ach_pretty[c] = df_ach_pretty[c].astype(object)
            df_ach_pretty.loc[infeas, c] = "INFEASIBLE"
            df_ach_pretty[c] = df_ach_pretty[c].where(df_ach_pretty[c].notna(), "NA")

    df_ach_pretty.to_csv(out_dir / "table_achieved_band1.csv", index=False)
    # DEV-ONLY: Table 1 integrity checks (set NV_DEV_CHECKS=1 to hard-fail).
    try:
        import os as _os
        _dev = int(_os.environ.get("NV_DEV_CHECKS", "0")) == 1
        _errs = []
        _req = ["nc2_p0", "p_hat", "p_binom", "nc2_pass"]
        for _c in _req:
            if _c not in df_ach_pretty.columns:
                _errs.append(f"missing_col:{_c}")
        if int(getattr(args, "do_perm_control", 0)) == 1 and all(_c in df_ach_pretty.columns for _c in _req):
            # If merge keys drift, these show up as NaNs.
            if df_ach_pretty["nc2_p0"].isna().any():
                _errs.append("nc2_p0_has_NaN")
            if df_ach_pretty["p_hat"].isna().any():
                _errs.append("p_hat_has_NaN")
            if df_ach_pretty["p_binom"].isna().any():
                _errs.append("p_binom_has_NaN")
            if df_ach_pretty["nc2_pass"].isna().any():
                _errs.append("nc2_pass_has_NaN")
        if "attack_strength_selected" in df_ach_pretty.columns and df_ach_pretty["attack_strength_selected"].isna().any():
            _errs.append("attack_strength_selected_has_NaN")
        if _errs:
            _msg = "; ".join(_errs)
            if _dev:
                raise AssertionError(f"Table1 integrity check failed: {_msg}")
            else:
                print(f"[WARN] Table1 integrity check: {_msg}")
    except Exception as _e:
        print(f"[WARN] Table1 integrity check exception: {_e}")

    # --- 2B) Determine the budget grid to report (for intensity surfaces / infeasibility reporting) ---
    raw_budgets = getattr(args, "budgets", None)
    if raw_budgets is None or raw_budgets == []:
        grid_budgets = sorted(df_trials["budget"].dropna().astype(float).unique().tolist())
    elif isinstance(raw_budgets, str):
        toks = [t.strip() for t in raw_budgets.replace(";", ",").split(",") if t.strip()]
        grid_budgets = [float(t) for t in toks]
    else:
        grid_budgets = [float(b) for b in list(raw_budgets)]

    # This is a *rate-band* feasibility diagnostic: given a finite horizon (years) and integer episode counts,
    # some budgets cannot land within ±tol_budget even with an ideal controller.
    years_global = float(np.nanmedian(df_trials.get("false_alert_years", np.nan)))
    if not np.isfinite(years_global) or years_global <= 0:
        fa_eval_m = float(np.nanmedian(df_trials.get("false_alert_eval_months", np.nan)))
        years_global = (fa_eval_m / steps_per_year) if (np.isfinite(fa_eval_m) and fa_eval_m > 0) else float("nan")

    infeas_rows = []
    for b in grid_budgets:
        b = float(b)
        feasible_counts = []
        nearest_cnt = None
        nearest_rate = None
        if np.isfinite(years_global) and years_global > 0 and b > 0:
            target_cnt = b * years_global
            max_cnt = int(max(0, round(target_cnt + 10)))
            for k in range(0, max_cnt + 1):
                rate = k / years_global
                if abs(rate - b) <= tol_budget * b:
                    feasible_counts.append(k)
            nearest_cnt = int(round(target_cnt))
            nearest_rate = nearest_cnt / years_global
        infeas_rows.append({
            "budget": b,
            "years_eval": years_global,
            "matched_tol_budget": float(tol_budget),
            "rateband_math_infeasible": (len(feasible_counts) == 0) if (np.isfinite(years_global) and years_global > 0 and b > 0) else True,
            "rateband_feasible_counts": ";".join(str(k) for k in feasible_counts) if feasible_counts else "",
            "nearest_count": nearest_cnt,
            "nearest_implied_rate": nearest_rate,
        })

    try:
        pd.DataFrame(infeas_rows).to_csv(out_dir / "infeasible_budgets.csv", index=False)
    except Exception:
        pass

    # One row per (kind, panel, detector, nominal budget, attack_strength), with both budget-only and joint conditioning.
    grid_rows = []
    needed = {"kind", "panel", "detector", "budget", "attack_strength"}
    if needed.issubset(df_trials.columns):
        df_grid_src = df_trials.copy()
        df_grid_src = df_grid_src[df_grid_src["budget"].astype(float).isin([float(b) for b in grid_budgets])]
        for (kind, panel, detector, B, strength), sub in df_grid_src.groupby(["kind", "panel", "detector", "budget", "attack_strength"]):
            B = float(B)
            okB = _ok_budget_band(sub, B)
            okT = _ok_tiw_safe(sub)
            inB = sub.loc[okB].copy()
            inJ = sub.loc[okB & okT].copy()
            inP = inJ if use_joint else inB

            row = {
                "kind": kind,
                "panel": panel,
                "detector": detector,
                "budget": float(B),
                "attack_strength": float(strength),

                "matched_mode": matched_mode,
                "matched_budget_band": matched_budget_band,
                "matched_tol_budget": float(tol_budget),
                "tiw_cap": float(tiw_cap) if (tiw_cap > 0) else float("nan"),
                "matched_tol_tiw": float(tol_tiw) if (tiw_cap > 0) else float("nan"),

                "n_trials_total": int(len(sub)),
                "n_trials_budget": int(len(inB)),
                "n_trials_joint": int(len(inJ)),
                "n_trials_primary": int(len(inP)),

                "coverage_budget": float(len(inB) / max(len(sub), 1)),
                "coverage_joint": float(len(inJ) / max(len(sub), 1)),
                "coverage_primary": float(len(inP) / max(len(sub), 1)),

                "fae_null_mean_primary": float(np.nanmean(inP["fae_null"].values)) if ("fae_null" in inP.columns and len(inP)) else float("nan"),
                "tiw_null_mean_primary": float(np.nanmean(inP["tiw_null"].values)) if ("tiw_null" in inP.columns and len(inP)) else float("nan"),
                "false_alert_events_count_mean_primary": float(np.nanmean(inP["false_alert_events_count"].values)) if ("false_alert_events_count" in inP.columns and len(inP)) else float("nan"),
            }
            if len(inP) > 0:
                summ = summarize(inP.to_dict("records"))
                for k, v in summ.items():
                    if k == "n_trials":
                        continue
                    row[k] = v
            grid_rows.append(row)

    if grid_rows:
        df_grid = pd.DataFrame(grid_rows).sort_values(["kind", "panel", "detector", "budget", "attack_strength"])

        # Write raw + paper-friendly versions (avoid NaNs by marking infeasibility).
        df_grid_raw = df_grid.copy()
        df_grid_raw.to_csv(out_dir / "table_intensity_grid_raw.csv", index=False)
        df_grid_pretty = df_grid.copy()
        cov = pd.to_numeric(df_grid_pretty.get("coverage_primary", pd.Series([np.nan]*len(df_grid_pretty))), errors="coerce").fillna(0)
        infeasible = cov.le(0)
        df_grid_pretty["row_status"] = np.where(infeasible, "INFEASIBLE", "OK")
        df_grid_pretty["infeasible_reason"] = np.where(infeasible, "no_trials_in_band", "")
        id_cols = {"panel","detector","kind","attack_strength","budget_target","budget_target_paper","matched_mode","matched_budget_band","matched_tol","tiw_budget","safety_eps","event_sla_months","count_band_width","target_count_nearest","implied_rate_nearest","row_status","infeasible_reason"}
        metric_cols = [c for c in df_grid_pretty.columns if c not in id_cols]
        df_grid_pretty[metric_cols] = df_grid_pretty[metric_cols].astype(object)
        for c in metric_cols:
            df_grid_pretty.loc[infeasible, c] = "INFEASIBLE"
            df_grid_pretty[c] = df_grid_pretty[c].where(~df_grid_pretty[c].isna(), "NA")
        df_grid_pretty.to_csv(out_dir / "table_intensity_grid.csv", index=False)


        try:
            df_mono = df_grid_raw.copy()
            # Compatibility: some grids use `budget` rather than `budget_target`.
            # Compute monotonicity per budget to avoid pooling points from different operating requests.
            # operating requests are pooled.
            if "budget" in df_mono.columns and "budget_target" not in df_mono.columns:
                df_mono["budget_target"] = df_mono["budget"]

            # Exclude null-strength points (attack_strength == 0) from monotonicity checks.
            # Some datasets include a no-incident baseline in the intensity grid; including it can
            # create artificial monotonicity violations unrelated to incident strength.
            if "attack_strength" in df_mono.columns:
                df_mono = df_mono[df_mono["attack_strength"].fillna(0) > 0].copy()

            # Ensure numeric ordering keys
            for c in ("attack_strength", "budget_target"):
                if c in df_mono.columns:
                    df_mono[c] = pd.to_numeric(df_mono[c], errors="coerce")

            # Metric columns (direction of improvement with higher strength)
            # NOTE: tau (tolerance) is a diagnostic knob, not a theorem.
            tau = float(getattr(args, "monotone_tau", 0.02))
            inc_metrics = [
                ("event_success_rate", +1, tau),
                ("event_timely_sla_rate", +1, tau),
                ("event_anywarn_success_rate", +1, tau),
                ("event_anywarn_timely_sla_rate", +1, tau),
            ]
            dec_metrics = [
                ("event_delay_p50", -1, tau),
                ("event_delay_p90", -1, tau),
            ]

            # Keep only groups where we have >=2 strengths
            gcols = [c for c in ("kind", "panel", "detector", "budget_target", "matched_mode", "matched_budget_band", "matched_tol") if c in df_mono.columns]
            rows = []
            # Coverage gating: avoid over-flagging noise/coverage collapse segments.
            min_primary = int(getattr(args, "monotone_min_primary", 20))
            min_cov = float(getattr(args, "monotone_min_coverage", 0.05))

            for key, g in df_mono.groupby(gcols, dropna=False):
                g = g.sort_values("attack_strength")
                if g["attack_strength"].nunique(dropna=True) < 2:
                    continue

                # Gate on coverage and accepted-trial mass to avoid low-evidence monotonicity flags.
                nprim = pd.to_numeric(g.get("n_trials_primary", pd.Series([np.nan] * len(g))), errors="coerce").fillna(0)
                covp = pd.to_numeric(g.get("coverage_primary", pd.Series([np.nan] * len(g))), errors="coerce").fillna(0.0)
                ok_row = (nprim >= float(min_primary)) & (covp >= float(min_cov))
                g2 = g.loc[ok_row].copy()
                if g2["attack_strength"].nunique(dropna=True) < 2:
                    # Not enough stable points to assess monotonicity.
                    continue
                g2 = g2.sort_values("attack_strength")

                # Helper: count adjacent monotonicity violations + emit violating segments
                def _violations_segments(
                    strengths: pd.Series,
                    arr: pd.Series,
                    nprim_s: pd.Series,
                    cov_s: pd.Series,
                    direction: int,
                    tol: float,
                ):
                    strengths = pd.to_numeric(strengths, errors="coerce")
                    arr = pd.to_numeric(arr, errors="coerce")
                    nprim_s = pd.to_numeric(nprim_s, errors="coerce")
                    cov_s = pd.to_numeric(cov_s, errors="coerce")
                    m = strengths.notna() & arr.notna() & nprim_s.notna() & cov_s.notna()
                    if m.sum() < 2:
                        return None, "", "", "", "", "", 0
                    s = strengths[m].astype(float).values
                    y = arr[m].astype(float).values
                    n_used = nprim_s[m].astype(float).values
                    c_used = cov_s[m].astype(float).values
                    # deterministic ordering (already sorted, but keep safe)
                    order = np.argsort(s)
                    s = s[order]
                    y = y[order]
                    n_used = n_used[order]
                    c_used = c_used[order]

                    segs = []
                    for j in range(len(y) - 1):
                        if direction > 0:
                            if y[j + 1] < y[j] - tol:
                                segs.append((j, j + 1))
                        else:
                            if y[j + 1] > y[j] + tol:
                                segs.append((j, j + 1))
                    seg_str = ";".join([f"{s[i]:.4g}->{s[j]:.4g}" for i, j in segs])
                    strengths_csv = ",".join([f"{v:.6g}" for v in s])
                    values_csv = ",".join([f"{v:.6g}" for v in y])
                    nprim_csv = ",".join([f"{v:.6g}" for v in n_used])
                    cov_csv = ",".join([f"{v:.6g}" for v in c_used])
                    return int(len(segs)), seg_str, strengths_csv, values_csv, nprim_csv, cov_csv, int(len(s))

                # Spearman (robust monotonic trend signal)
                def _spearman(x, y):
                    x = pd.to_numeric(x, errors="coerce")
                    y = pd.to_numeric(y, errors="coerce")
                    m = x.notna() & y.notna()
                    if m.sum() < 3:
                        return None
                    xr = x[m].rank(method="average")
                    yr = y[m].rank(method="average")
                    return float(xr.corr(yr))

                for name, direction, tol in inc_metrics + dec_metrics:
                    if name not in g2.columns:
                        continue
                    vio, segs, s_csv, v_csv, n_csv, c_csv, npts = _violations_segments(
                        g2["attack_strength"],
                        g2[name],
                        g2.get("n_trials_primary", pd.Series([np.nan] * len(g2))),
                        g2.get("coverage_primary", pd.Series([np.nan] * len(g2))),
                        direction,
                        tol,
                    )
                    rho = _spearman(g2["attack_strength"], g2[name])
                    rows.append({
                        **{gc: (key[i] if isinstance(key, tuple) else key) for i, gc in enumerate(gcols)},
                        "metric": name,
                        "direction": "nondecreasing" if direction > 0 else "nonincreasing",
                        "tau": float(tol),
                        "min_primary": int(min_primary),
                        "min_coverage": float(min_cov),
                        "n_strengths": int(g2["attack_strength"].nunique(dropna=True)),
                        "n_points_used": int(npts),
                        "violations_adjacent": vio,
                        "violating_segments": segs,
                        "spearman_rho": rho,
                        "min_strength": float(g2["attack_strength"].min()),
                        "max_strength": float(g2["attack_strength"].max()),
                        "strength_points": s_csv,
                        "metric_points": v_csv,
                        "n_primary_points": n_csv,
                        "coverage_points": c_csv,
                    })

            df_mono_report = pd.DataFrame(rows)
            if len(df_mono_report) > 0:
                df_mono_report.to_csv(out_dir / "monotonicity_report.csv", index=False)

                # Paper-facing summary: per (kind, panel, detector, budget), whether the *plotted* metric is monotone
                keycols = [c for c in ("kind", "panel", "detector", "budget_target") if c in df_mono_report.columns]
                focus = str(getattr(args, "monotone_focus_metric", "event_timely_sla_rate"))

                def _pass_grp(gr):
                    sub = gr.set_index("metric")
                    if focus in sub.index and pd.notna(sub.loc[focus, "violations_adjacent"]):
                        ok = (float(sub.loc[focus, "violations_adjacent"]) == 0.0)
                    else:
                        ok = False
                    return pd.Series({"monotone_focus_metric": focus, "monotone_pass": int(ok)})

                df_mono_paper = df_mono_report.groupby(keycols, dropna=False).apply(_pass_grp).reset_index()
                df_mono_paper.to_csv(out_dir / "table_monotonicity_paper.csv", index=False)
        except Exception as e:
            warn(f"Failed monotonicity report: {e}")
        # Backward-compatible alias
        df_grid.to_csv(out_dir / "table_achieved_band_grid.csv", index=False)



    # -------------------------
    # 3) Curves + frontiers (including CAD / |ΔTIW|)
    # -------------------------
    y_curve = [
        "detect_rate",
        "uplift_detect_rate",
        "median_delay",
        "mean_time_in_warning",
        "mean_cad_win",
        "mean_abs_dtiw_win",
    ]
    for kind in sorted(df_sum["kind"].unique()):
        for panel in sorted(df_sum["panel"].unique()):
            for detector in sorted(df_sum["detector"].unique()):
                sub = df_sum[(df_sum["kind"] == kind) & (df_sum["panel"] == panel) & (df_sum["detector"] == detector)]
                if len(sub) == 0:
                    continue
                for y in y_curve:
                    if y in sub.columns:
                        plot_curve(sub, out_dir, kind, panel, detector, y)

    # frontiers per (kind,panel) — x is achieved clean burden when available
    y_frontier = [
        "detect_rate",
        "uplift_detect_rate",
        "median_delay",
        "mean_time_in_warning",
        "mean_cad_win",
        "mean_abs_dtiw_win",
    ]
    for kind in sorted(df_sum["kind"].unique()):
        for panel in sorted(df_sum["panel"].unique()):
            sub = df_sum[(df_sum["kind"] == kind) & (df_sum["panel"] == panel)]
            if len(sub) == 0:
                continue
            for y in y_frontier:
                if y in sub.columns:
                    plot_frontier(sub, out_dir, kind, panel, y)

    # -------------------------
    # 4) Paper points + envelopes (main pipeline)
    # -------------------------
    if "achieved_events_per_year_clean" in df_sum.columns:
        xcol = "achieved_events_per_year_clean"
    elif "mean_fae_null" in df_sum.columns:
        xcol = "mean_fae_null"
    else:
        xcol = "mean_false_alert_events_per_year"

    for kind, panel in df_sum.groupby(["kind", "panel"]).groups.keys():
        sub = df_sum[(df_sum["kind"] == kind) & (df_sum["panel"] == panel)].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values(xcol).reset_index(drop=True)
        sub.to_csv(out_dir / f"paper_points_{kind}_{panel}.csv", index=False)

        # Envelopes (maximize uplift, minimize burden-like y)
        env_rows = []
        env_cols = [xcol]

        # Uplift frontier (max)
        if "uplift_detect_rate" in sub.columns:
            env_uplift = envelope_points(sub, xcol=xcol, ycol="uplift_detect_rate", mode="max")
            env_uplift.to_csv(out_dir / f"paper_frontier_uplift_{kind}_{panel}.csv", index=False)
            plot_paper_frontier(env_uplift, out_dir, kind, panel, "uplift_detect_rate")
            env_rows.append(env_uplift[[xcol, "uplift_detect_rate"]].rename(columns={"uplift_detect_rate": "env_uplift_detect_rate"}))
            env_cols.append("env_uplift_detect_rate")

        # Delay frontier (min)
        if "median_delay" in sub.columns:
            env_delay = envelope_points(sub, xcol=xcol, ycol="median_delay", mode="min")
            env_delay.to_csv(out_dir / f"paper_frontier_delay_{kind}_{panel}.csv", index=False)
            plot_paper_frontier(env_delay, out_dir, kind, panel, "median_delay")
            env_rows.append(env_delay[[xcol, "median_delay"]].rename(columns={"median_delay": "env_median_delay"}))
            env_cols.append("env_median_delay")

        # TIW frontier (min)
        if "mean_time_in_warning" in sub.columns:
            env_tiw = envelope_points(sub, xcol=xcol, ycol="mean_time_in_warning", mode="min")
            env_tiw.to_csv(out_dir / f"paper_frontier_tiw_{kind}_{panel}.csv", index=False)
            plot_paper_frontier(env_tiw, out_dir, kind, panel, "mean_time_in_warning")
            env_rows.append(env_tiw[[xcol, "mean_time_in_warning"]].rename(columns={"mean_time_in_warning": "env_mean_time_in_warning"}))
            env_cols.append("env_mean_time_in_warning")

        # CAD frontier (min)
        if "mean_cad_win" in sub.columns:
            env_cad = envelope_points(sub, xcol=xcol, ycol="mean_cad_win", mode="min")
            env_cad.to_csv(out_dir / f"paper_frontier_cad_{kind}_{panel}.csv", index=False)
            plot_paper_frontier(env_cad, out_dir, kind, panel, "mean_cad_win")
            env_rows.append(env_cad[[xcol, "mean_cad_win"]].rename(columns={"mean_cad_win": "env_mean_cad_win"}))
            env_cols.append("env_mean_cad_win")

        # |ΔTIW| frontier (min)
        if "mean_abs_dtiw_win" in sub.columns:
            env_dtiw = envelope_points(sub, xcol=xcol, ycol="mean_abs_dtiw_win", mode="min")
            env_dtiw.to_csv(out_dir / f"paper_frontier_absdtiw_{kind}_{panel}.csv", index=False)
            plot_paper_frontier(env_dtiw, out_dir, kind, panel, "mean_abs_dtiw_win")
            env_rows.append(env_dtiw[[xcol, "mean_abs_dtiw_win"]].rename(columns={"mean_abs_dtiw_win": "env_mean_abs_dtiw_win"}))
            env_cols.append("env_mean_abs_dtiw_win")

        # Merge envelope columns by x
        if env_rows:
            # Each env_rows element is expected to contain [xcol, one env_* column] to avoid merge suffix collisions.
            env = env_rows[0].copy()
            for r in env_rows[1:]:
                r2 = r[[c for c in r.columns if c == xcol or c.startswith("env_")]].copy()
                env = env.merge(r2, on=xcol, how="outer")
            env = env.sort_values(xcol).reset_index(drop=True)
            env.to_csv(os.path.join(out_dir, f"paper_envelope_{kind}_{panel}.csv"), index=False)

    # -------------------------
    # 5) Counterfactual metrics & stealth curves (matched-burden conditioning by nominal budget)
    # -------------------------
    # Counterfactual metrics: keep window metadata + CAD / ΔTIW probes.
    cf_cols = [
        "kind", "panel", "detector", "budget", "trial", "start", "end",
        "fae_null", "tiw_null", "cad_win", "dtiw_win", "cad_post", "dtiw_post",
        "detected", "delay_months_cens", "false_alert_events_per_year", "time_in_warning",
    ]
    cf_cols = [c for c in cf_cols if c in df_trials.columns]
    df_trials[cf_cols].to_csv(out_dir / "counterfactual_metrics.csv", index=False)

    # Stealth curves: condition on matched-burden relative to each nominal budget.
    # Matched-burden primary = EVENT BUDGET (count-based when available). TIW is reported as safety/coverage.
    stealth_rows = []
    tol_budget = float(getattr(args, "matched_tol", 0.10))
    # argparse uses default None for matched_tiw_tol; interpret None as "use budget tol".
    _tol_tiw_raw = getattr(args, "matched_tiw_tol", None)
    tol_tiw = float(_tol_tiw_raw) if _tol_tiw_raw is not None else tol_budget
    tiw_cap = float(getattr(args, "tiw_budget", 0.0) or 0.0)

    def _years_col(df):
        if "false_alert_years" in df.columns:
            return "false_alert_years"
        if "eval_years" in df.columns:
            return "eval_years"
        return None

    def _ok_budget_count(df, B):
        """Return boolean mask for budget match. Uses integer episode counts when available."""
        B = float(B)
        if B <= 0:
            return df["fae_null"].notna() if "fae_null" in df.columns else pd.Series(True, index=df.index)

        ycol = _years_col(df)
        if (ycol is not None) and ("false_alert_events_count" in df.columns):
            years = df[ycol].astype(float)
            cnt = df["false_alert_events_count"].astype(float)
            tgt = (years * B).round()
            # Convert relative tolerance into an integer count band; ensure >=1 when tgt>=1.
            band = np.where(
                tgt <= 0,
                0,
                np.maximum(1.0, np.ceil(tol_budget * np.maximum(tgt, 1.0)))
            )
            return (cnt - tgt).abs() <= band

        # Fallback: rate-based band
        if "fae_null" not in df.columns:
            return pd.Series(True, index=df.index)
        return (df["fae_null"].astype(float).sub(B).abs() / B) <= tol_budget

    def _ok_tiw(df):
        if tiw_cap <= 0 or ("tiw_null" not in df.columns):
            return pd.Series(True, index=df.index)
        # One-sided: stay under cap (allow small tolerance)
        return df["tiw_null"].astype(float) <= tiw_cap * (1.0 + tol_tiw)

    # Stealth curves (matched burden): group by strength grid (paper-facing object)
    for (kind, p, d, B, strength), sub in df_trials.groupby(["kind", "panel", "detector", "budget", "attack_strength"]):
        if sub.empty:
            continue
        okB = _ok_budget_count(sub, B)
        okT = _ok_tiw(sub)
        okJ = okB & okT

        inB = sub.loc[okB]
        inJ = sub.loc[okJ]

        # We always emit a row (even if coverage=0) so the paper surface is complete.
        def _mean_or_nan(s):
            return float(s.mean()) if len(s) else float("nan")
        def _median_or_nan(s):
            return float(s.median()) if len(s) else float("nan")

        stealth_rows.append(
            {
                "kind": kind,
                "panel": p,
                "detector": d,
                "budget": float(B),
                "attack_strength": float(strength),

                "intensity_proxy_name": inB["intensity_proxy_name"].iloc[0] if ("intensity_proxy_name" in inB.columns and len(inB)) else "",
                "intensity_proxy_mean": _mean_or_nan(inB["intensity_proxy"]) if "intensity_proxy" in inB.columns else float("nan"),
                "intensity_proxy_data_mean": _mean_or_nan(inB["intensity_proxy_data"]) if "intensity_proxy_data" in inB.columns else float("nan"),

                "detect_rate_budget": _mean_or_nan(inB["detected"]),
                "detect_rate_joint": _mean_or_nan(inJ["detected"]),

                "delay_cens_mean_months_budget": _mean_or_nan(inB["delay_months_cens"]) if "delay_months_cens" in inB.columns else float("nan"),
                "delay_cens_median_months_budget": _median_or_nan(inB["delay_months_cens"]) if "delay_months_cens" in inB.columns else float("nan"),
                "delay_cens_mean_months_joint": _mean_or_nan(inJ["delay_months_cens"]) if "delay_months_cens" in inJ.columns else float("nan"),
                "delay_cens_median_months_joint": _median_or_nan(inJ["delay_months_cens"]) if "delay_months_cens" in inJ.columns else float("nan"),

                "cad_win_mean_budget": _mean_or_nan(inB["cad_win"]) if "cad_win" in inB.columns else float("nan"),
                "cad_post_mean_budget": _mean_or_nan(inB["cad_post"]) if "cad_post" in inB.columns else float("nan"),
                "dtiw_win_mean_budget": _mean_or_nan(inB["dtiw_win"]) if "dtiw_win" in inB.columns else float("nan"),
                "dtiw_post_mean_budget": _mean_or_nan(inB["dtiw_post"]) if "dtiw_post" in inB.columns else float("nan"),

                "n_trials_total": int(len(sub)),
                "n_trials_budget": int(len(inB)),
                "n_trials_joint": int(len(inJ)),
                "coverage_budget": float(okB.mean()) if len(okB) else 0.0,
                "coverage_tiw": float(okT.mean()) if len(okT) else 0.0,
                "coverage_joint": float(okJ.mean()) if len(okJ) else 0.0,

                "fae_null_mean": _mean_or_nan(inB["fae_null"]) if "fae_null" in inB.columns else float("nan"),
                "tiw_null_mean": _mean_or_nan(inB["tiw_null"]) if "tiw_null" in inB.columns else float("nan"),
                "false_alert_events_count_mean": _mean_or_nan(inB["false_alert_events_count"]) if "false_alert_events_count" in inB.columns else float("nan"),
                "false_alert_years_mean": _mean_or_nan(inB[_years_col(inB)]) if (_years_col(inB) is not None and len(inB)) else float("nan"),
            }
        )

    if stealth_rows:
        df_stealth = pd.DataFrame(stealth_rows)
        df_stealth = df_stealth.sort_values(["kind", "panel", "detector", "budget", "attack_strength"])
        # Emit both filenames for reproducibility and clearer downstream interpretation
        df_stealth.to_csv(os.path.join(out_dir, "stealth_curves_matched.csv"), index=False)
        df_stealth.to_csv(os.path.join(out_dir, "detectability_curve.csv"), index=False)

    # Explicit joint feasibility report (null burden): per (panel, detector, budget).
    feas_rows = []
    if all(c in df_trials.columns for c in ["trial", "panel", "detector", "budget"]):
        df_clean = df_trials.drop_duplicates(subset=["trial", "panel", "detector", "budget"]).copy()
        for (p, d, B), g in df_clean.groupby(["panel", "detector", "budget"]):
            okB_cnt = _ok_budget_count(g, B)
            okT = _ok_tiw(g)
            okJ = okB_cnt & okT

            # Nearest-integer feasibility (diagnostic): can B be represented within tol_budget given years?
            ycol = _years_col(g)
            if ycol is not None and "false_alert_events_count" in g.columns:
                years_med = float(g[ycol].median())
                tgt_cnt = int(round(float(B) * years_med))
                achieved_rate = (tgt_cnt / years_med) if years_med > 0 else float("nan")
                rel_err_nearest = abs(achieved_rate / float(B) - 1.0) if (years_med > 0 and float(B) > 0) else float("nan")
            else:
                years_med = float("nan")
                tgt_cnt = -1
                achieved_rate = float("nan")
                rel_err_nearest = float("nan")

            feas_rows.append(
                {
                    "panel": p,
                    "detector": d,
                    "budget": float(B),
                    "n_trials": int(len(g)),
                    "fae_null_mean": float(g["fae_null"].mean()) if "fae_null" in g.columns else float("nan"),
                    "tiw_null_mean": float(g["tiw_null"].mean()) if "tiw_null" in g.columns else float("nan"),
                    "false_alert_events_count_mean": float(g["false_alert_events_count"].mean()) if "false_alert_events_count" in g.columns else float("nan"),
                    "false_alert_events_count_min": float(g["false_alert_events_count"].min()) if "false_alert_events_count" in g.columns else float("nan"),
                    "false_alert_events_count_max": float(g["false_alert_events_count"].max()) if "false_alert_events_count" in g.columns else float("nan"),
                    "false_alert_events_count_q05": float(g["false_alert_events_count"].quantile(0.05)) if "false_alert_events_count" in g.columns else float("nan"),
                    "false_alert_events_count_q95": float(g["false_alert_events_count"].quantile(0.95)) if "false_alert_events_count" in g.columns else float("nan"),
                    "false_alert_years_median": years_med,
                    "target_count_nearest": int(tgt_cnt),
                    "achievable_rate_nearest": float(achieved_rate),
                    "rel_err_nearest": float(rel_err_nearest),

                    "frac_budget_in_band_count": float(okB_cnt.mean()) if len(okB_cnt) else 0.0,
                    "frac_tiw_safe": float(okT.mean()) if len(okT) else 0.0,
                    "frac_joint_count": float(okJ.mean()) if len(okJ) else 0.0,
                }
            )

    if feas_rows:
        df_feas = pd.DataFrame(feas_rows)
        df_feas = df_feas.sort_values(["panel", "detector", "budget"])
        df_feas.to_csv(os.path.join(out_dir, "joint_feasibility.csv"), index=False)



    # AUROC-mirage toy example snippet (static)
    try:
        from pathlib import Path
        snippet_path = Path(__file__).resolve().parent / "paper_snippets" / "auroc_mirage_toy.md"
        if snippet_path.exists():
            (out_dir / "auroc_mirage_toy.md").write_text(snippet_path.read_text(), encoding="utf-8")
            (out_dir / "auroc_mirage_toy.txt").write_text(snippet_path.read_text(), encoding="utf-8")
    except Exception:
        pass

if __name__ == "__main__":
    main()
