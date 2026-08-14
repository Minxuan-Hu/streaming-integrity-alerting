from __future__ import annotations

import csv
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import numpy as np

from common import load_run_metrics, read_csv, write_csv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis"
MAN = ROOT / "data" / "manifests"
OUT.mkdir(parents=True, exist_ok=True)

R_REQUIRED = 4750
Z_975 = 1.959963984540054


def quantiles(values: np.ndarray) -> tuple[float, float]:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if not len(valid):
        return float("nan"), float("nan")
    return (
        float(np.quantile(valid, 0.025, method="linear")),
        float(np.quantile(valid, 0.975, method="linear")),
    )


def wilson(successes: int, denominator: int) -> tuple[float, float]:
    if denominator <= 0:
        return float("nan"), float("nan")
    proportion = successes / denominator
    z2 = Z_975 * Z_975
    scale = 1.0 + z2 / denominator
    center = (proportion + z2 / (2.0 * denominator)) / scale
    half = Z_975 / scale * math.sqrt(
        proportion * (1.0 - proportion) / denominator + z2 / (4.0 * denominator**2)
    )
    return max(0.0, center - half), min(1.0, center + half)


def q3(value: float) -> str:
    rounded = Decimal(str(round(float(value), 12))).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    )
    return f"{rounded:.3f}"


def format_interval(point: float, lower: float, upper: float) -> str:
    return f"{q3(point)} [{q3(lower)}, {q3(upper)}]"


def numeric_rows(metrics: dict[str, Any], field: str, base_ids: list[str]) -> np.ndarray:
    by_trial = {str(row["base_trial"]): row for row in metrics["rows"]}
    return np.asarray([float(by_trial[base_id][field]) for base_id in base_ids], dtype=float)


indices = np.load(MAN / "paired_base_trial_resampling_indices_5000x80.npy", allow_pickle=False)
base_manifest = sorted(read_csv(MAN / "base_trial_manifest_80.csv"), key=lambda row: int(row["position"]))
base_ids = [row["base_trial"] for row in base_manifest]
if indices.shape != (5000, 80) or len(base_ids) != 80:
    raise RuntimeError("Unexpected resampling design")

selector_rows = read_csv(OUT / "cross_dataset_selector_point_estimates.csv")
selector = next(
    row for row in selector_rows if row["run"] == "cfpb_gradual_bias_seed0" and row["kind"] == "gradual_bias"
)
selections = [
    ("Naive timely row", "CUSUM Factor LRT", selector["naive_detector"]),
    ("Best matched-compliant row", "Fused Fisher", selector["best_matched_compliant_detector"]),
]
if [row[2] for row in selections] != ["cusum_factor_cov_lrt", "fused_fisher"]:
    raise RuntimeError("Unexpected CFPB GradualBias selections")

run_metrics, _, _ = load_run_metrics("cfpb_gradual_bias_seed0")
metric_labels = {
    "matched_burden_timely_at_k": "Matched-burden Timely@K",
    "matched_burden_contract_pass_rate": "Matched-burden contract-pass rate",
    "all_trial_contract_pass_rate": "All-trial contract-pass rate",
    "coverage_joint": "Joint coverage",
    "mean_tiw_clean_matched": "Mean clean-replay TIW over A_B",
    "mean_tiw_corrupted_matched": "Mean corrupted-replay TIW over A_B",
    "mean_cad_win_matched": "Mean CAD over A_B",
    "mean_false_alert_events_corrupted_matched": "False-alert events/year over A_B",
}

long_rows: list[dict[str, Any]] = []
wide_rows: list[dict[str, Any]] = []
wilson_rows: list[dict[str, Any]] = []

for role, display, detector in selections:
    metrics = run_metrics[("gradual_bias", "score_only", detector)]
    position = {base_id: index for index, base_id in enumerate(metrics["base_trials"])}
    order = np.asarray([position[base_id] for base_id in base_ids], dtype=int)

    a_b = np.asarray(metrics["a_b"], dtype=bool)[order]
    a_j = np.asarray(metrics["a_j"], dtype=bool)[order]
    timely = np.asarray(metrics["timely"], dtype=bool)[order]
    contract = np.asarray(metrics["contract"], dtype=bool)[order]
    clean_tiw = numeric_rows(metrics, "tiw_null", base_ids)
    corrupted_tiw = numeric_rows(metrics, "time_in_warning", base_ids)
    cad = numeric_rows(metrics, "cad_win", base_ids)
    false_events = numeric_rows(metrics, "false_alert_events_per_year", base_ids)

    for array_name, array in {
        "clean TIW": clean_tiw,
        "corrupted TIW": corrupted_tiw,
        "CAD": cad,
        "false-alert burden": false_events,
    }.items():
        if np.any(a_b & ~np.isfinite(array)):
            raise RuntimeError(f"Nonfinite {array_name} inside A_B for {detector}")

    a_b_draw = a_b[indices]
    n_ab = a_b_draw.sum(axis=1)
    valid = n_ab > 0

    values: dict[str, np.ndarray] = {
        "matched_burden_timely_at_k": np.divide(
            (a_b & timely)[indices].sum(axis=1), n_ab,
            out=np.full(len(indices), np.nan), where=valid,
        ),
        "matched_burden_contract_pass_rate": np.divide(
            (a_b & contract)[indices].sum(axis=1), n_ab,
            out=np.full(len(indices), np.nan), where=valid,
        ),
        "all_trial_contract_pass_rate": contract[indices].mean(axis=1),
        "coverage_joint": a_j[indices].mean(axis=1),
        "mean_tiw_clean_matched": np.divide(
            np.where(a_b, clean_tiw, 0.0)[indices].sum(axis=1), n_ab,
            out=np.full(len(indices), np.nan), where=valid,
        ),
        "mean_tiw_corrupted_matched": np.divide(
            np.where(a_b, corrupted_tiw, 0.0)[indices].sum(axis=1), n_ab,
            out=np.full(len(indices), np.nan), where=valid,
        ),
        "mean_cad_win_matched": np.divide(
            np.where(a_b, cad, 0.0)[indices].sum(axis=1), n_ab,
            out=np.full(len(indices), np.nan), where=valid,
        ),
        "mean_false_alert_events_corrupted_matched": np.divide(
            np.where(a_b, false_events, 0.0)[indices].sum(axis=1), n_ab,
            out=np.full(len(indices), np.nan), where=valid,
        ),
    }

    points = {name: float(metrics[name]) for name in values}
    counts = {
        "matched_burden_timely_at_k": (metrics["n_timely_A_B"], metrics["n_A_B"]),
        "matched_burden_contract_pass_rate": (metrics["n_matched_contract_pass"], metrics["n_A_B"]),
        "all_trial_contract_pass_rate": (metrics["n_all_trial_contract_pass"], metrics["n_total"]),
        "coverage_joint": (metrics["n_A_J"], metrics["n_total"]),
    }

    wide: dict[str, Any] = {
        "selection_role": role,
        "display_row": display,
        "detector": detector,
        "n_total": metrics["n_total"],
        "n_A_B": metrics["n_A_B"],
        "n_A_J": metrics["n_A_J"],
        "R_total": len(indices),
        "R_valid": int(valid.sum()),
    }

    for name, draws in values.items():
        valid_count = int(np.isfinite(draws).sum())
        if valid_count < R_REQUIRED:
            raise RuntimeError(f"Insufficient valid replicates for {display}: {name}")
        lower, upper = quantiles(draws)
        point = points[name]
        successes, denominator = counts.get(name, ("", ""))
        boundary = bool(name in counts and (int(successes) == 0 or int(successes) == int(denominator)))
        wilson_lower = wilson_upper = float("nan")
        if boundary:
            wilson_lower, wilson_upper = wilson(int(successes), int(denominator))
            wilson_rows.append(
                {
                    "display_row": display,
                    "metric": name,
                    "successes": successes,
                    "denominator": denominator,
                    "wilson_lower_two_sided_95": wilson_lower,
                    "wilson_upper_two_sided_95": wilson_upper,
                }
            )
        long_rows.append(
            {
                "analysis": "CFPB GradualBias",
                "selection_role": role,
                "display_row": display,
                "detector": detector,
                "metric": name,
                "metric_label": metric_labels[name],
                "point_estimate": point,
                "observed_successes": successes,
                "observed_denominator": denominator,
                "R_total": len(indices),
                "R_valid": valid_count,
                "bootstrap_lower_2_5": lower,
                "bootstrap_upper_97_5": upper,
                "formatted_point_and_interval": format_interval(point, lower, upper),
                "wilson_lower_two_sided_95": wilson_lower,
                "wilson_upper_two_sided_95": wilson_upper,
            }
        )
        wide[f"{name}_point"] = point
        wide[f"{name}_lower_2_5"] = lower
        wide[f"{name}_upper_97_5"] = upper
        wide[f"{name}_formatted"] = format_interval(point, lower, upper)
    wide_rows.append(wide)

write_csv(OUT / "cfpb_gradualbias_selected_rows_intervals_long.csv", long_rows)
write_csv(OUT / "cfpb_gradualbias_selected_rows_intervals_wide.csv", wide_rows)
write_csv(OUT / "cfpb_gradualbias_wilson_boundary_sensitivity.csv", wilson_rows)

summary = {
    "status": "PASS",
    "selected_rows": [row[1] for row in selections],
    "replicates": int(len(indices)),
    "valid_replicates_per_row": int(min(row["R_valid"] for row in wide_rows)),
    "support_threshold": R_REQUIRED,
}
(OUT / "cfpb_gradualbias_interval_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
)
print(json.dumps(summary, indent=2, sort_keys=True))
