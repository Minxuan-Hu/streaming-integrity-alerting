from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from common import load_run_metrics, read_csv, sha256_file, write_csv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis"
MAN = ROOT / "data" / "manifests"
OUT.mkdir(parents=True, exist_ok=True)

R_REQUIRED = 4750
ALPHA = 0.05
Z_975 = 1.959963984540054


def q025_975(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return (
        float(np.quantile(values, 0.025, method="linear")),
        float(np.quantile(values, 0.975, method="linear")),
    )


def wilson_interval(successes: int, n: int, z: float = Z_975) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    z2 = z * z
    den = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / den
    half = z / den * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def stable_key(x: str) -> tuple[int, Any]:
    try:
        return 0, int(x)
    except Exception:
        return 1, x


indices_path = MAN / "paired_base_trial_resampling_indices_5000x80.npy"
indices = np.load(indices_path, allow_pickle=False).astype(np.int64)
if indices.shape != (5000, 80):
    raise RuntimeError(f"Unexpected resampling matrix shape: {indices.shape}")
R_TOTAL = indices.shape[0]

resampling_manifest = json.loads((MAN / "resampling_manifest.json").read_text(encoding="utf-8"))
if sha256_file(indices_path) != resampling_manifest["index_file_sha256"]:
    raise RuntimeError("Stored resampling matrix checksum does not match manifest")
if resampling_manifest.get("seed") != 42 or resampling_manifest.get("bit_generator") != "PCG64":
    raise RuntimeError("Unexpected stored RNG convention")

base_manifest = read_csv(MAN / "base_trial_manifest_80.csv")
base_ids = [r["base_trial"] for r in sorted(base_manifest, key=lambda r: int(r["position"]))]
if len(base_ids) != 80 or len(set(base_ids)) != 80:
    raise RuntimeError("Invalid base-trial manifest")

on, _, _ = load_run_metrics("cfpb_online_recal1_seed0")
off, _, _ = load_run_metrics("cfpb_online_recal0_seed0")


def aligned_array(m: dict[str, Any], field: str, dtype: Any = float) -> np.ndarray:
    pos = {b: i for i, b in enumerate(m["base_trials"])}
    if set(pos) != set(base_ids):
        raise RuntimeError(f"Base-trial mismatch for {m['spec'].id}")
    raw = np.asarray(m[field])
    return np.asarray([raw[pos[b]] for b in base_ids], dtype=dtype)


def aligned_numeric_from_rows(m: dict[str, Any], field: str) -> np.ndarray:
    pos = {str(r["base_trial"]): r for r in m["rows"]}
    if set(pos) != set(base_ids):
        raise RuntimeError(f"Base-trial row mismatch for {m['spec'].id}")
    vals = []
    for bid in base_ids:
        text = str(pos[bid].get(field, "")).strip()
        try:
            vals.append(float(text))
        except Exception:
            vals.append(float("nan"))
    return np.asarray(vals, dtype=float)


def bootstrap_metric_arrays(m: dict[str, Any]) -> dict[str, np.ndarray]:
    a_b = aligned_array(m, "a_b", bool)
    a_j = aligned_array(m, "a_j", bool)
    timely = aligned_array(m, "timely", bool)
    contract = aligned_array(m, "contract", bool)
    clean_tiw = aligned_numeric_from_rows(m, "tiw_null")
    corrupt_tiw = aligned_numeric_from_rows(m, "time_in_warning")

    ab_draw = a_b[indices]
    n_ab = ab_draw.sum(axis=1)
    n_aj = a_j[indices].sum(axis=1)
    n_timely_ab = (a_b & timely)[indices].sum(axis=1)
    n_contract_ab = (a_b & contract)[indices].sum(axis=1)
    n_contract_all = contract[indices].sum(axis=1)

    # Both TIW means use the full clean-defined A_B denominator.
    if np.any(a_b & ~np.isfinite(clean_tiw)):
        bad = int(np.sum(a_b & ~np.isfinite(clean_tiw)))
        raise RuntimeError(f"Nonfinite clean TIW inside A_B for {m['spec'].id}: {bad} trial(s)")
    if np.any(a_b & ~np.isfinite(corrupt_tiw)):
        bad = int(np.sum(a_b & ~np.isfinite(corrupt_tiw)))
        raise RuntimeError(f"Nonfinite corrupted TIW inside A_B for {m['spec'].id}: {bad} trial(s)")
    clean_num = np.where(a_b, clean_tiw, 0.0)[indices].sum(axis=1)
    corrupt_num = np.where(a_b, corrupt_tiw, 0.0)[indices].sum(axis=1)

    mb_timely = np.full(R_TOTAL, np.nan)
    mb_contract = np.full(R_TOTAL, np.nan)
    mean_clean = np.full(R_TOTAL, np.nan)
    mean_corrupt = np.full(R_TOTAL, np.nan)
    np.divide(n_timely_ab, n_ab, out=mb_timely, where=n_ab > 0)
    np.divide(n_contract_ab, n_ab, out=mb_contract, where=n_ab > 0)
    np.divide(clean_num, n_ab, out=mean_clean, where=n_ab > 0)
    np.divide(corrupt_num, n_ab, out=mean_corrupt, where=n_ab > 0)

    return {
        "n_A_B": n_ab.astype(int),
        "coverage_budget": n_ab / indices.shape[1],
        "coverage_joint": n_aj / indices.shape[1],
        "matched_burden_timely_at_k": mb_timely,
        "matched_burden_contract_pass_rate": mb_contract,
        "all_trial_contract_pass_rate": n_contract_all / indices.shape[1],
        "mean_tiw_clean_matched": mean_clean,
        "mean_tiw_corrupted_matched": mean_corrupt,
    }


_cache: dict[tuple[str, tuple[str, str, str]], dict[str, np.ndarray]] = {}


def boot(condition: str, key: tuple[str, str, str]) -> dict[str, np.ndarray]:
    ck = (condition, key)
    if ck not in _cache:
        source = on if condition == "ON" else off
        _cache[ck] = bootstrap_metric_arrays(source[key])
    return _cache[ck]


# ---------------------------------------------------------------------------
# Pairing checks include all attacked rows and the selected control row.
# ---------------------------------------------------------------------------
reference = on[("freeze", "score_only", "T2")]
reference_ids = reference["base_trials"]
reference_windows = {
    (r["base_trial"], r.get("incident_start"), r.get("incident_end")) for r in reference["rows"]
}
pairing_rows: list[dict[str, Any]] = []
for condition, metrics in [("ON", on), ("OFF", off)]:
    for kind in ["freeze", "corr_mix"]:
        detectors = sorted(
            [k[2] for k in metrics if k[0] == kind and k[1] == "score_only"]
        )
        for detector in detectors:
            m = metrics[(kind, "score_only", detector)]
            windows = {
                (r["base_trial"], r.get("incident_start"), r.get("incident_end")) for r in m["rows"]
            }
            pairing_rows.append({
                "condition": condition,
                "kind": kind,
                "panel": "score_only",
                "detector": detector,
                "selected_main_row": detector in {"fused_fisher", "coh_resid_t2", "cusum_factor_cov_lrt"} and kind == "freeze",
                "n_rows": m["n_total"],
                "n_unique_base_trials": len(set(m["base_trials"])),
                "base_trial_set_matches_reference": set(m["base_trials"]) == set(reference_ids),
                "base_trial_order_matches_reference": m["base_trials"] == reference_ids,
                "incident_windows_match_reference": windows == reference_windows,
            })
    # Explicit selected control row under each condition.
    m = metrics[("freeze", "all_only", "max_abs")]
    windows = {
        (r["base_trial"], r.get("incident_start"), r.get("incident_end")) for r in m["rows"]
    }
    pairing_rows.append({
        "condition": condition,
        "kind": "freeze",
        "panel": "all_only",
        "detector": "max_abs",
        "selected_main_row": True,
        "n_rows": m["n_total"],
        "n_unique_base_trials": len(set(m["base_trials"])),
        "base_trial_set_matches_reference": set(m["base_trials"]) == set(reference_ids),
        "base_trial_order_matches_reference": m["base_trials"] == reference_ids,
        "incident_windows_match_reference": windows == reference_windows,
    })

pairing_ok = all(
    r["n_rows"] == 80
    and r["n_unique_base_trials"] == 80
    and r["base_trial_set_matches_reference"]
    and r["base_trial_order_matches_reference"]
    and r["incident_windows_match_reference"]
    for r in pairing_rows
)
if not pairing_ok:
    raise RuntimeError("Expanded base-trial pairing validation failed")
write_csv(OUT / "base_trial_pairing_validation.csv", pairing_rows)
write_csv(
    OUT / "selected_main_rows_pairing_assertion.csv",
    [r for r in pairing_rows if r["selected_main_row"]],
)

# ---------------------------------------------------------------------------
# Four selected CFPB rows.
# ---------------------------------------------------------------------------
selected_specs = [
    ("Fused Fisher", "freeze", "score_only", "fused_fisher"),
    ("Coherence residual T2", "freeze", "score_only", "coh_resid_t2"),
    ("CUSUM Factor LRT", "freeze", "score_only", "cusum_factor_cov_lrt"),
    ("Max Abs control", "freeze", "all_only", "max_abs"),
]
rate_observed_counts = {
    "coverage_budget": ("n_A_B", "n_total"),
    "coverage_joint": ("n_A_J", "n_total"),
    "matched_burden_timely_at_k": ("n_timely_A_B", "n_A_B"),
    "matched_burden_contract_pass_rate": ("n_matched_contract_pass", "n_A_B"),
    "all_trial_contract_pass_rate": ("n_all_trial_contract_pass", "n_total"),
}
metric_labels = {
    "coverage_budget": "Budget coverage",
    "coverage_joint": "Joint coverage",
    "matched_burden_timely_at_k": "Matched-burden Timely@K",
    "matched_burden_contract_pass_rate": "Matched-burden contract-pass rate",
    "all_trial_contract_pass_rate": "All-trial contract-pass rate",
    "mean_tiw_clean_matched": "Mean clean-replay TIW over A_B",
    "mean_tiw_corrupted_matched": "Mean corrupted-replay TIW over A_B",
}
selected_long: list[dict[str, Any]] = []
selected_wide: list[dict[str, Any]] = []
for display, kind, panel, detector in selected_specs:
    key = (kind, panel, detector)
    m = on[key]
    b = boot("ON", key)
    wide: dict[str, Any] = {
        "display_row": display,
        "kind": kind,
        "panel": panel,
        "detector": detector,
        "n_total": m["n_total"],
        "observed_n_A_B": m["n_A_B"],
    }
    for field in metric_labels:
        values = b[field]
        valid = np.isfinite(values)
        r_valid = int(valid.sum())
        reportable = (field not in {"matched_burden_timely_at_k", "matched_burden_contract_pass_rate", "mean_tiw_clean_matched", "mean_tiw_corrupted_matched"}) or r_valid >= R_REQUIRED
        lo, hi = q025_975(values[valid]) if reportable else (float("nan"), float("nan"))
        point = float(m[field])
        successes = denominator = None
        wilson_lo = wilson_hi = float("nan")
        boundary = False
        if field in rate_observed_counts:
            num_field, den_field = rate_observed_counts[field]
            successes = int(m[num_field])
            denominator = int(m[den_field])
            boundary = successes in {0, denominator}
            if boundary:
                wilson_lo, wilson_hi = wilson_interval(successes, denominator)
        selected_long.append({
            "analysis_id": "selected_rows",
            "display_row": display,
            "kind": kind,
            "panel": panel,
            "detector": detector,
            "metric": field,
            "metric_label": metric_labels[field],
            "point_estimate": point,
            "observed_successes": successes,
            "observed_denominator": denominator,
            "R_total": R_TOTAL,
            "R_valid": r_valid,
            "valid_fraction": r_valid / R_TOTAL,
            "meets_reporting_criterion": reportable,
            "bootstrap_lower_2_5": lo,
            "bootstrap_upper_97_5": hi,
            "bootstrap_method": "paired base-trial percentile bootstrap; common resampling vectors; NumPy quantile method=linear; valid replicates only",
            "observed_boundary_proportion": boundary,
            "wilson_lower_two_sided_95": wilson_lo,
            "wilson_upper_two_sided_95": wilson_hi,
            "wilson_note": "Approximate Bernoulli boundary sensitivity conditional on sampled incident design" if boundary else "",
        })
        prefix = field
        wide[prefix] = point
        wide[prefix + "_lower_2_5"] = lo
        wide[prefix + "_upper_97_5"] = hi
        wide[prefix + "_R_valid"] = r_valid
        wide[prefix + "_wilson_upper_if_boundary"] = wilson_hi
    selected_wide.append(wide)

write_csv(OUT / "core_cfpb_selected_rows_intervals_long.csv", selected_long)
write_csv(OUT / "core_cfpb_selected_rows_intervals_wide.csv", selected_wide)

# ---------------------------------------------------------------------------
# Fixed-menu clean-workload coverage medians.
# ---------------------------------------------------------------------------
menu19_rows = read_csv(MAN / "clean_workload_menu_19.csv")
menu12_rows = read_csv(MAN / "positive_budget_menu_12.csv")
menu_defs = [
    ("full_menu", "full_fixed_19", menu19_rows, "Primary fixed detector menu; zeros retained."),
    ("positive_budget_menu", "fixed_observed_positive_budget_12", menu12_rows, "Conditional mechanism menu fixed from observed canonical positive-budget point estimates."),
]
coverage_long: list[dict[str, Any]] = []
coverage_wide: list[dict[str, Any]] = []
for analysis_id, menu_name, menu_rows, interpretation in menu_defs:
    keys = [("freeze", "score_only", r["detector"]) for r in menu_rows]
    budget_stack = np.vstack([boot("ON", k)["coverage_budget"] for k in keys])
    joint_stack = np.vstack([boot("ON", k)["coverage_joint"] for k in keys])
    budget_med = np.median(budget_stack, axis=0)
    joint_med = np.median(joint_stack, axis=0)
    drop = budget_med - joint_med
    point_budget = float(np.median([on[k]["coverage_budget"] for k in keys]))
    point_joint = float(np.median([on[k]["coverage_joint"] for k in keys]))
    arrays = {
        "median_budget_coverage": (point_budget, budget_med),
        "median_joint_coverage": (point_joint, joint_med),
        "absolute_drop_budget_minus_joint": (point_budget - point_joint, drop),
    }
    wide = {
        "analysis_id": analysis_id,
        "menu": menu_name,
        "n_configurations": len(keys),
        "interpretation": interpretation,
        "R_total": R_TOTAL,
    }
    for metric, (point, values) in arrays.items():
        lo, hi = q025_975(values)
        coverage_long.append({
            "analysis_id": analysis_id,
            "menu": menu_name,
            "n_configurations": len(keys),
            "metric": metric,
            "point_estimate": point,
            "R_total": R_TOTAL,
            "R_valid": R_TOTAL,
            "bootstrap_lower_2_5": lo,
            "bootstrap_upper_97_5": hi,
            "interpretation": interpretation,
            "bootstrap_method": "paired base-trial percentile bootstrap of fixed-menu medians; common resampling vectors",
        })
        wide[metric] = point
        wide[metric + "_lower_2_5"] = lo
        wide[metric + "_upper_97_5"] = hi
    coverage_wide.append(wide)

write_csv(OUT / "core_cfpb_coverage_menu_intervals_long.csv", coverage_long)
write_csv(OUT / "core_cfpb_coverage_menu_intervals_wide.csv", coverage_wide)

# ---------------------------------------------------------------------------
# Primary recalibration contrasts.
# ---------------------------------------------------------------------------
clean_keys = [("freeze", "score_only", r["detector"]) for r in menu19_rows]
incident_menu = read_csv(MAN / "incident_dependent_menu_38.csv")
incident_keys = [(r["kind"], "score_only", r["detector"]) for r in incident_menu]
intersection = read_csv(MAN / "recalibration_shared_menu_22.csv")
common22_keys = [(r["kind"], "score_only", r["detector"]) for r in intersection]

recal_long: list[dict[str, Any]] = []
recal_wide: list[dict[str, Any]] = []


def add_recal_metric(
    estimand: str,
    keys: list[tuple[str, str, str]],
    field: str,
    validity: np.ndarray | None = None,
    secondary: bool = False,
) -> None:
    on_stack = np.vstack([boot("ON", k)[field] for k in keys])
    off_stack = np.vstack([boot("OFF", k)[field] for k in keys])
    if validity is None:
        validity = np.all(np.isfinite(on_stack) & np.isfinite(off_stack), axis=0)
    on_med = np.median(on_stack[:, validity], axis=0)
    off_med = np.median(off_stack[:, validity], axis=0)
    diff = on_med - off_med
    paired_change = np.median(on_stack[:, validity] - off_stack[:, validity], axis=0)

    point_on = float(np.median([on[k][field] for k in keys]))
    point_off = float(np.median([off[k][field] for k in keys]))
    point_diff = point_on - point_off
    point_paired = float(np.median([on[k][field] - off[k][field] for k in keys]))
    r_valid = int(validity.sum())
    reportable = (not secondary) or r_valid >= R_REQUIRED

    metrics = [
        ("ON_fixed_menu_median", point_on, on_med),
        ("OFF_fixed_menu_median", point_off, off_med),
        ("difference_of_fixed_menu_medians_ON_minus_OFF", point_diff, diff),
    ]
    for contrast, point, values in metrics:
        lo, hi = q025_975(values) if reportable else (float("nan"), float("nan"))
        recal_long.append({
            "analysis_id": "recalibration",
            "estimand": estimand,
            "configuration_menu_size": len(keys),
            "contrast": contrast,
            "point_estimate": point,
            "R_total": R_TOTAL,
            "R_valid": r_valid,
            "valid_fraction": r_valid / R_TOTAL,
            "meets_reporting_criterion": reportable,
            "bootstrap_lower_2_5": lo,
            "bootstrap_upper_97_5": hi,
            "primary_or_secondary": "secondary matched-burden intersection" if secondary else "primary full-menu",
            "bootstrap_method": "paired base-trial percentile bootstrap; difference of fixed-menu medians is primary",
        })
    recal_wide.append({
        "estimand": estimand,
        "configuration_menu_size": len(keys),
        "point_ON_fixed_menu_median": point_on,
        "point_OFF_fixed_menu_median": point_off,
        "point_difference_of_fixed_menu_medians_ON_minus_OFF": point_diff,
        "point_median_paired_configuration_change_ON_minus_OFF": point_paired,
        "R_total": R_TOTAL,
        "R_valid": r_valid,
        "valid_fraction": r_valid / R_TOTAL,
        "meets_reporting_criterion": reportable,
        "ON_lower_2_5": q025_975(on_med)[0] if reportable else float("nan"),
        "ON_upper_97_5": q025_975(on_med)[1] if reportable else float("nan"),
        "OFF_lower_2_5": q025_975(off_med)[0] if reportable else float("nan"),
        "OFF_upper_97_5": q025_975(off_med)[1] if reportable else float("nan"),
        "difference_lower_2_5": q025_975(diff)[0] if reportable else float("nan"),
        "difference_upper_97_5": q025_975(diff)[1] if reportable else float("nan"),
        "median_paired_configuration_change_treatment": "descriptive point estimate only",
    })


add_recal_metric("budget_coverage_median_19", clean_keys, "coverage_budget")
add_recal_metric("joint_coverage_median_19", clean_keys, "coverage_joint")
add_recal_metric("all_trial_contract_pass_median_38", incident_keys, "all_trial_contract_pass_rate")

def paired_transition_counts(off_status: np.ndarray, on_status: np.ndarray) -> dict[str, np.ndarray | int]:
    off_status = np.asarray(off_status, dtype=bool)
    on_status = np.asarray(on_status, dtype=bool)
    if off_status.shape != on_status.shape:
        raise RuntimeError(f"Transition status shape mismatch: {off_status.shape} vs {on_status.shape}")
    axis = 0 if off_status.ndim > 1 else None
    return {
        "0_to_0": np.sum((~off_status) & (~on_status), axis=axis),
        "0_to_1_OFF_to_ON": np.sum((~off_status) & on_status, axis=axis),
        "1_to_0_OFF_to_ON": np.sum(off_status & (~on_status), axis=axis),
        "1_to_1": np.sum(off_status & on_status, axis=axis),
    }


# Nonzero joint status over 19 distinct clean detectors.
on_joint_counts = np.vstack([(boot("ON", k)["coverage_joint"] * 80).round().astype(int) for k in clean_keys])
off_joint_counts = np.vstack([(boot("OFF", k)["coverage_joint"] * 80).round().astype(int) for k in clean_keys])
on_pos = on_joint_counts > 0
off_pos = off_joint_counts > 0
on_frac = on_pos.mean(axis=0)
off_frac = off_pos.mean(axis=0)
frac_diff = on_frac - off_frac
transition_arrays = paired_transition_counts(off_pos, on_pos)
if not np.all(sum(np.asarray(v) for v in transition_arrays.values()) == 19):
    raise RuntimeError("Bootstrap transition counts do not sum to 19 in every replicate")
point_on_pos = sum(on[k]["coverage_joint"] > 0 for k in clean_keys)
point_off_pos = sum(off[k]["coverage_joint"] > 0 for k in clean_keys)
for contrast, point, values in [
    ("ON_nonzero_joint_fraction", point_on_pos / 19, on_frac),
    ("OFF_nonzero_joint_fraction", point_off_pos / 19, off_frac),
    ("change_in_nonzero_joint_fraction_ON_minus_OFF", (point_on_pos - point_off_pos) / 19, frac_diff),
]:
    lo, hi = q025_975(values)
    recal_long.append({
        "analysis_id": "recalibration",
        "estimand": "nonzero_joint_fraction_19",
        "configuration_menu_size": 19,
        "contrast": contrast,
        "point_estimate": point,
        "R_total": R_TOTAL,
        "R_valid": R_TOTAL,
        "valid_fraction": 1.0,
        "meets_reporting_criterion": True,
        "bootstrap_lower_2_5": lo,
        "bootstrap_upper_97_5": hi,
        "primary_or_secondary": "primary full-menu",
        "bootstrap_method": "paired base-trial bootstrap of detector-level nonzero-joint status",
    })

off_point_status = np.asarray([off[k]["coverage_joint"] > 0 for k in clean_keys], dtype=bool)
on_point_status = np.asarray([on[k]["coverage_joint"] > 0 for k in clean_keys], dtype=bool)
transition_point_raw = paired_transition_counts(off_point_status, on_point_status)
transition_point = {k: int(v) for k, v in transition_point_raw.items()}
if sum(transition_point.values()) != 19:
    raise RuntimeError(f"Observed transition counts do not sum to 19: {transition_point}")
if transition_point["0_to_1_OFF_to_ON"] + transition_point["1_to_1"] != point_on_pos:
    raise RuntimeError("Observed ON-positive count is inconsistent with transition counts")
if transition_point["1_to_0_OFF_to_ON"] + transition_point["1_to_1"] != point_off_pos:
    raise RuntimeError("Observed OFF-positive count is inconsistent with transition counts")
transition_out = []
for name, values in transition_arrays.items():
    lo, hi = q025_975(values.astype(float))
    transition_out.append({
        "transition": name,
        "point_count": int(transition_point[name]),
        "n_detectors": 19,
        "R_total": R_TOTAL,
        "bootstrap_lower_2_5_count": lo,
        "bootstrap_upper_97_5_count": hi,
        "treatment": "paired transition-count diagnostic",
    })
write_csv(OUT / "core_cfpb_recalibration_transition_intervals.csv", transition_out)

# Secondary fixed-22 matched-burden comparisons; valid only when all 22 ON/OFF defined.
on_mb_t_stack = np.vstack([boot("ON", k)["matched_burden_timely_at_k"] for k in common22_keys])
off_mb_t_stack = np.vstack([boot("OFF", k)["matched_burden_timely_at_k"] for k in common22_keys])
on_mb_c_stack = np.vstack([boot("ON", k)["matched_burden_contract_pass_rate"] for k in common22_keys])
off_mb_c_stack = np.vstack([boot("OFF", k)["matched_burden_contract_pass_rate"] for k in common22_keys])
valid22 = np.all(
    np.isfinite(on_mb_t_stack)
    & np.isfinite(off_mb_t_stack)
    & np.isfinite(on_mb_c_stack)
    & np.isfinite(off_mb_c_stack),
    axis=0,
)
if int(valid22.sum()) != 4770:
    raise RuntimeError(f"Expected 22-row support count changed: got {int(valid22.sum())}, expected 4770")
add_recal_metric(
    "matched_burden_timely_median_22",
    common22_keys,
    "matched_burden_timely_at_k",
    validity=valid22,
    secondary=True,
)
add_recal_metric(
    "matched_burden_contract_pass_median_22",
    common22_keys,
    "matched_burden_contract_pass_rate",
    validity=valid22,
    secondary=True,
)

write_csv(OUT / "core_cfpb_recalibration_intervals_long.csv", recal_long)
write_csv(OUT / "core_cfpb_recalibration_intervals_wide.csv", recal_wide)

# ---------------------------------------------------------------------------
# Calculation summary.
# ---------------------------------------------------------------------------
interval_summary = {
    "status": "CORE_CFPB_INTERVALS_COMPLETE",
    "scope": [
        "four selected CFPB operating rows",
        "fixed 19-detector clean-workload coverage menu",
        "fixed observed 12-detector positive-budget coverage menu",
        "primary recalibration full-menu contrasts",
        "secondary matched-burden recalibration contrasts over 22 configurations defined under both settings",
    ],
    "resampling_matrix": str(indices_path.relative_to(ROOT)),
    "resampling_matrix_sha256": sha256_file(indices_path),
    "R_total": R_TOTAL,
    "selected_rows_min_R_valid": min(r["R_valid"] for r in selected_long),
    "secondary_22_R_valid": int(valid22.sum()),
    "support_threshold": R_REQUIRED,
    "expanded_pairing_rows": len(pairing_rows),
    "selected_control_pairing_asserted_ON_and_OFF": True,
    "transition_counts_directly_paired_and_sum_to_19": True,
    "tiw_complete_inside_A_B_asserted": True,
}
(OUT / "core_cfpb_interval_calculation_summary.json").write_text(
    json.dumps(interval_summary, indent=2, sort_keys=True), encoding="utf-8"
)

print(json.dumps(interval_summary, indent=2, sort_keys=True))
