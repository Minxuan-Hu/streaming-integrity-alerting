from __future__ import annotations

import csv
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "analysis"
RUNS = ROOT / "data" / "runs"
TABLES = ROOT / "results" / "tables"
FIGURE_DATA = ROOT / "results" / "figure_data"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURE_DATA.mkdir(parents=True, exist_ok=True)


def q3(value: float) -> str:
    if pd.isna(value):
        return "--"
    number = Decimal(str(round(float(value), 12))).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    )
    return f"{number:.3f}"


def format_interval(point: float, lower: float, upper: float, signed: bool = False) -> str:
    point_text = q3(point)
    if signed and point > 0:
        point_text = "+" + point_text
    return f"{point_text} [{q3(lower)}, {q3(upper)}]"


def save(df: pd.DataFrame, name: str, directory: Path = TABLES) -> None:
    df.to_csv(directory / name, index=False)


def detector_label(detector: str) -> str:
    labels = {
        "max_abs": "Max Abs",
        "T2": "Hotelling T²",
        "res_energy": "Residual Energy",
        "pc_t2": "Principal-Component T²",
        "factor_cov_change": "Factor-Covariance Change",
        "factor_cov_delta": "Factor-Covariance Delta",
        "factor_cov_lrt": "Factor-Cov LRT",
        "factor_cov_lrt_delta": "Factor-Cov LRT-Δ",
        "coh_resid_t2": "Coherence Residual T²",
        "coh_resid_energy": "Coherence Residual Energy",
        "fused_sp": "Fused Sum-of-p",
        "fused_fisher": "Fused Fisher",
        "fused_sp_lrt_delta": "Fused Sum-of-p + LRT-Δ",
        "fused_fisher_lrt_delta": "Fused Fisher + LRT-Δ",
        "ewma_factor_cov_lrt": "EWMA Factor LRT",
        "cusum_factor_cov_lrt": "CUSUM Factor LRT",
        "ewma_coh_resid_t2": "EWMA Coherence T²",
        "cusum_coh_resid_t2": "CUSUM Coherence T²",
        "stale": "Stale Persistence Baseline",
    }
    return labels.get(detector, detector)


# -----------------------------------------------------------------------------
# Main selected-row table and selected-row uncertainty
# -----------------------------------------------------------------------------
main = pd.read_csv(ANALYSIS / "main_operating_table_point_estimates.csv")
family = {
    "fused_fisher": "Fused",
    "coh_resid_t2": "Coherence",
    "cusum_factor_cov_lrt": "EWMA/CUSUM",
    "max_abs": "Scalar/statistical",
}
main_table = pd.DataFrame(
    {
        "Detector": main["display_row"].replace(
            {"Coherence residual T2": "Coherence residual T²", "Max Abs control": "Max Abs"}
        ),
        "Slice": main["panel"].replace({"attacked": "Attacked", "control": "Control"}),
        "Detector family": main["detector"].map(family),
        "Incident family": "Freeze",
        "Matched-burden Timely@K": main["matched_burden_timely_at_k"].map(q3),
        "Matched-burden contract pass": main["matched_burden_contract_pass_rate"].map(q3),
        "Joint coverage": main["coverage_joint"].map(q3),
        "Clean-replay TIW": main["mean_tiw_clean_matched"].map(q3),
        "Corrupted-replay TIW": main["mean_tiw_corrupted_matched"].map(q3),
        "Incident-window CAD": main["mean_cad_win_matched"].map(q3),
    }
)
save(main_table, "table_main_cfpb.csv")

core_wide = pd.read_csv(ANALYSIS / "core_cfpb_selected_rows_intervals_wide.csv")
core_uncertainty_rows = []
for _, row in core_wide.iterrows():
    core_uncertainty_rows.append(
        {
            "Detector": row["display_row"].replace("Coherence residual T2", "Coherence residual T²"),
            "Total trials": int(row["n_total"]),
            "Matched-budget trials": int(row["observed_n_A_B"]),
            "Joint-feasible trials": int(round(row["coverage_joint"] * row["n_total"])),
            "Matched-burden Timely@K [95% interval]": format_interval(
                row["matched_burden_timely_at_k"],
                row["matched_burden_timely_at_k_lower_2_5"],
                row["matched_burden_timely_at_k_upper_97_5"],
            ),
            "Matched-burden contract pass [95% interval]": format_interval(
                row["matched_burden_contract_pass_rate"],
                row["matched_burden_contract_pass_rate_lower_2_5"],
                row["matched_burden_contract_pass_rate_upper_97_5"],
            ),
            "Joint coverage [95% interval]": format_interval(
                row["coverage_joint"],
                row["coverage_joint_lower_2_5"],
                row["coverage_joint_upper_97_5"],
            ),
            "Clean-replay TIW [95% interval]": format_interval(
                row["mean_tiw_clean_matched"],
                row["mean_tiw_clean_matched_lower_2_5"],
                row["mean_tiw_clean_matched_upper_97_5"],
            ),
            "Corrupted-replay TIW [95% interval]": format_interval(
                row["mean_tiw_corrupted_matched"],
                row["mean_tiw_corrupted_matched_lower_2_5"],
                row["mean_tiw_corrupted_matched_upper_97_5"],
            ),
        }
    )
save(pd.DataFrame(core_uncertainty_rows), "table_core_selected_row_uncertainty.csv")

# -----------------------------------------------------------------------------
# Cross-dataset canonical and GradualBias ledgers
# -----------------------------------------------------------------------------
selectors = pd.read_csv(ANALYSIS / "cross_dataset_selector_point_estimates.csv")
all_points = pd.read_csv(ANALYSIS / "point_estimates_all_selected_strength_B1.csv")

coverage_by_dataset = {}
for run, dataset in [
    ("cfpb_online_recal1_seed0", "CFPB"),
    ("smap_A1_weekly_ledger_seed0", "SMAP A-1"),
    ("smap_D15_weekly_ledger_seed0", "SMAP D-15"),
]:
    clean = all_points[
        (all_points["run"] == run)
        & (all_points["kind"] == "freeze")
        & (all_points["panel"] == "attacked")
    ]
    positive = clean[clean["coverage_budget"] > 0]
    coverage_by_dataset[dataset] = (
        len(positive),
        len(clean),
        float(positive["coverage_joint"].median()),
    )

control_cad = {}
for _, row in selectors[selectors["kind"].isin(["freeze", "corr_mix"])].iterrows():
    control = all_points[
        (all_points["run"] == row["run"])
        & (all_points["kind"] == row["kind"])
        & (all_points["panel"] == "control")
        & (all_points["n_A_B"] > 0)
    ]
    control_cad[(row["run"], row["kind"])] = float(control["mean_cad_win_matched"].median())

canonical_rows = []
for _, row in selectors[selectors["kind"].isin(["freeze", "corr_mix"])].iterrows():
    n_positive, n_total, median_joint = coverage_by_dataset[row["dataset"]]
    canonical_rows.append(
        {
            "Dataset": row["dataset"],
            "Incident family": "Freeze" if row["kind"] == "freeze" else "CorrMix",
            "Naive selected detector": detector_label(row["naive_detector"]),
            "Matched-compliant selected detector": detector_label(row["best_matched_compliant_detector"]),
            "Naive matched-burden Timely@K": q3(row["naive_MB_timely"]),
            "Compliant matched-burden Timely@K": q3(row["compliant_MB_timely"]),
            "Matched-burden contract pass": q3(row["compliant_MB_contract_pass"]),
            "All-trial contract pass": q3(row["compliant_all_trial_contract_pass"]),
            "Attacked-slice incident-window CAD": q3(row["compliant_CAD"]),
            "Control-slice incident-window CAD": q3(control_cad[(row["run"], row["kind"])]),
            "Positive-budget detectors": f"{n_positive}/{n_total}",
            "Median joint coverage among positive-budget detectors": q3(median_joint),
        }
    )
canonical = pd.DataFrame(canonical_rows).sort_values(["Dataset", "Incident family"])
save(canonical, "table_cross_dataset_canonical.csv")

gradual_rows = []
for _, row in selectors[selectors["kind"] == "gradual_bias"].iterrows():
    naive_point = all_points[
        (all_points["run"] == row["run"])
        & (all_points["kind"] == "gradual_bias")
        & (all_points["panel"] == "attacked")
        & (all_points["detector"] == row["naive_detector"])
    ].iloc[0]
    n_positive, n_total, median_joint = coverage_by_dataset[row["dataset"]]
    gradual_rows.append(
        {
            "Dataset": row["dataset"],
            "Naive selected detector": detector_label(row["naive_detector"]),
            "Naive matched-burden Timely@K": q3(row["naive_MB_timely"]),
            "Naive matched-burden contract pass": q3(row["naive_MB_contract_pass"]),
            "Naive corrupted-replay TIW": q3(naive_point["mean_tiw_corrupted_matched"]),
            "Matched-compliant selected detector": detector_label(row["best_matched_compliant_detector"]),
            "Compliant matched-burden Timely@K": q3(row["compliant_MB_timely"]),
            "Matched-burden contract pass": q3(row["compliant_MB_contract_pass"]),
            "All-trial contract pass": q3(row["compliant_all_trial_contract_pass"]),
            "Joint coverage": q3(row["compliant_n_A_J"] / (80 if row["dataset"] == "CFPB" else 30)),
            "Clean-replay TIW": q3(row["compliant_mean_clean_TIW"]),
            "Corrupted-replay TIW": q3(row["compliant_mean_corrupted_TIW"]),
            "Incident-window CAD": q3(row["compliant_CAD"]),
            "Positive-budget detectors": f"{n_positive}/{n_total}",
            "Median joint coverage among positive-budget detectors": q3(median_joint),
        }
    )
save(pd.DataFrame(gradual_rows), "table_gradual_bias.csv")

# -----------------------------------------------------------------------------
# Controller and TIW-cap sensitivity
# -----------------------------------------------------------------------------
sensitivity = pd.read_csv(ANALYSIS / "recalibration_sensitivity_point_estimates.csv")
parameters = {
    "Canonical recalibration ON": ("Baseline", 60, 0.30, 1, 1.25),
    "Window 36": ("Window 36", 36, 0.30, 1, 1.25),
    "Gain 0.15": ("Gain 0.15", 60, 0.15, 1, 1.25),
    "Update every 3": ("Update every 3", 60, 0.30, 3, 1.25),
    "Max up 1.10": ("Max up 1.10", 60, 0.30, 1, 1.10),
}
controller_rows = []
for _, row in sensitivity.iterrows():
    label, window, gain, frequency, max_up = parameters[row["setting"]]
    controller_rows.append(
        {
            "Setting": label,
            "Control window": window,
            "Gain": f"{gain:.2f}",
            "Update frequency": frequency,
            "Max upward step": f"{max_up:.2f}",
            "Positive-budget detectors": f"{int(row['n_clean_positive_budget'])}/19",
            "Median budget coverage (all 19 detectors)": q3(row["median_budget_coverage_full_19"]),
            "Median joint coverage (all 19 detectors)": q3(row["median_joint_coverage_full_19"]),
            "Median budget coverage (positive-budget detectors)": q3(row["median_budget_coverage_positive_menu"]),
            "Median joint coverage (positive-budget detectors)": q3(row["median_joint_coverage_positive_menu"]),
            "Incident configurations with defined matched-budget results": f"{int(row['n_incident_rows_defined_A_B'])}/38",
            "Median matched-burden Timely@K": q3(row["median_MB_timely_defined_rows"]),
            "Median matched-burden contract pass": q3(row["median_MB_contract_pass_defined_rows"]),
            "Median all-trial contract pass": q3(row["median_all_trial_contract_pass_38"]),
            "Mean corrupted-replay TIW": q3(row["mean_corrupted_TIW_defined_rows"]),
            "Detectors with nonzero joint coverage": f"{int(row['nonzero_joint_count_19'])}/19",
        }
    )
save(pd.DataFrame(controller_rows), "table_controller_sensitivity.csv")

tiw_rows = []
for run, cap in [
    ("cfpb_online_recal1_tiw010_seed0", 0.10),
    ("cfpb_online_recal1_seed0", 0.15),
    ("cfpb_online_recal1_tiw020_seed0", 0.20),
]:
    for kind in ["freeze", "corr_mix"]:
        subset = all_points[
            (all_points["run"] == run)
            & (all_points["kind"] == kind)
            & (all_points["panel"] == "attacked")
            & (all_points["coverage_budget"] > 0)
        ]
        tiw_rows.append(
            {
                "TIW cap": f"{cap:.2f}",
                "Incident family": "Freeze" if kind == "freeze" else "CorrMix",
                "Budget-only coverage": q3(subset["coverage_budget"].median()),
                "Joint coverage": q3(subset["coverage_joint"].median()),
                "Matched-burden contract pass": q3(subset["matched_burden_contract_pass_rate"].median()),
                "Matched-burden Timely@K": q3(subset["matched_burden_timely_at_k"].median()),
                "Mean corrupted-replay TIW": q3(subset["mean_tiw_corrupted_matched"].mean()),
            }
        )
save(pd.DataFrame(tiw_rows), "table_tiw_cap_sensitivity.csv")

# -----------------------------------------------------------------------------
# Coverage and recalibration uncertainty
# -----------------------------------------------------------------------------
coverage = pd.read_csv(ANALYSIS / "core_cfpb_coverage_menu_intervals_wide.csv")
coverage_rows = []
for _, row in coverage.iterrows():
    coverage_rows.append(
        {
            "Detector set": "All 19 detectors" if row["menu"] == "full_fixed_19" else "Positive-budget 12-detector subset",
            "Detectors": int(row["n_configurations"]),
            "Median budget coverage [95% interval]": format_interval(
                row["median_budget_coverage"], row["median_budget_coverage_lower_2_5"], row["median_budget_coverage_upper_97_5"]
            ),
            "Median joint coverage [95% interval]": format_interval(
                row["median_joint_coverage"], row["median_joint_coverage_lower_2_5"], row["median_joint_coverage_upper_97_5"]
            ),
            "Budget-minus-joint drop [95% interval]": format_interval(
                row["absolute_drop_budget_minus_joint"],
                row["absolute_drop_budget_minus_joint_lower_2_5"],
                row["absolute_drop_budget_minus_joint_upper_97_5"],
            ),
        }
    )
save(pd.DataFrame(coverage_rows), "table_coverage_uncertainty.csv")

recalibration = pd.read_csv(ANALYSIS / "core_cfpb_recalibration_intervals_wide.csv")
recalibration_names = {
    "budget_coverage_median_19": "Median budget coverage",
    "joint_coverage_median_19": "Median joint coverage",
    "all_trial_contract_pass_median_38": "Median all-trial contract pass",
    "matched_burden_timely_median_22": "Median matched-burden Timely@K among configurations defined under both settings",
    "matched_burden_contract_pass_median_22": "Median matched-burden contract pass among configurations defined under both settings",
}
recalibration_rows = []
for _, row in recalibration.iterrows():
    recalibration_rows.append(
        {
            "Quantity": recalibration_names[row["estimand"]],
            "Configurations": int(row["configuration_menu_size"]),
            "Recalibration on [95% interval]": format_interval(
                row["point_ON_fixed_menu_median"], row["ON_lower_2_5"], row["ON_upper_97_5"]
            ),
            "Recalibration off [95% interval]": format_interval(
                row["point_OFF_fixed_menu_median"], row["OFF_lower_2_5"], row["OFF_upper_97_5"]
            ),
            "On minus off [95% interval]": format_interval(
                row["point_difference_of_fixed_menu_medians_ON_minus_OFF"],
                row["difference_lower_2_5"],
                row["difference_upper_97_5"],
                signed=True,
            ),
            "Total replicates": int(row["R_total"]),
            "Valid replicates": int(row["R_valid"]),
            "Valid fraction": f"{row['valid_fraction']:.3f}".rstrip("0").rstrip("."),
        }
    )
save(pd.DataFrame(recalibration_rows), "table_recalibration_uncertainty.csv")

gradual_long = pd.read_csv(ANALYSIS / "cfpb_gradualbias_selected_rows_intervals_long.csv")
gradual_pivot = gradual_long.pivot(index="display_row", columns="metric", values="formatted_point_and_interval").reset_index()
gradual_uncertainty = gradual_pivot.rename(
    columns={
        "display_row": "Detector",
        "matched_burden_timely_at_k": "Matched-burden Timely@K [95% interval]",
        "matched_burden_contract_pass_rate": "Matched-burden contract pass [95% interval]",
        "all_trial_contract_pass_rate": "All-trial contract pass [95% interval]",
        "coverage_joint": "Joint coverage [95% interval]",
        "mean_tiw_clean_matched": "Clean-replay TIW [95% interval]",
        "mean_tiw_corrupted_matched": "Corrupted-replay TIW [95% interval]",
        "mean_cad_win_matched": "Incident-window CAD [95% interval]",
        "mean_false_alert_events_corrupted_matched": "False-alert events/year [95% interval]",
    }
)
gradual_uncertainty = gradual_uncertainty[[
    "Detector",
    "Matched-burden Timely@K [95% interval]",
    "Matched-burden contract pass [95% interval]",
    "All-trial contract pass [95% interval]",
    "Joint coverage [95% interval]",
    "Clean-replay TIW [95% interval]",
    "Corrupted-replay TIW [95% interval]",
    "Incident-window CAD [95% interval]",
    "False-alert events/year [95% interval]",
]]
save(gradual_uncertainty, "table_gradual_bias_uncertainty.csv")

# -----------------------------------------------------------------------------
# Appendix diagnostic tables
# -----------------------------------------------------------------------------
canonical_on = pd.read_csv(RUNS / "cfpb_online_recal1_seed0" / "table_achieved_band1.csv")
overlap = pd.read_csv(RUNS / "cfpb_online_recal1_seed0" / "overlap_uniform_warning.csv")
control_rows = []
for panel, label in [("score_only", "Attacked slice"), ("all_only", "Control slice")]:
    freeze = canonical_on[(canonical_on["panel"] == panel) & (canonical_on["kind"] == "freeze")]
    freeze = freeze[pd.to_numeric(freeze["coverage_primary"], errors="coerce") > 0]
    uniform = overlap[(overlap["panel"] == panel) & (overlap["budget"] == 1.0)]
    row = {
        "Slice": label,
        "Rows per incident family": len(freeze),
        "Median accepted-window p0": q3(pd.to_numeric(freeze["nc2_p0"], errors="coerce").median()),
        "Median uniform overlap": q3(uniform["p_overlap_warning_uniform"].median()),
    }
    for kind, name in [("freeze", "Freeze NC-2"), ("corr_mix", "CorrMix NC-2")]:
        subset = canonical_on[
            (canonical_on["panel"] == panel)
            & (canonical_on["kind"] == kind)
            & (pd.to_numeric(canonical_on["coverage_primary"], errors="coerce") > 0)
        ]
        row[name] = f"{int(pd.to_numeric(subset['nc2_pass'], errors='coerce').sum())}/{len(subset)}"
    control_rows.append(row)
save(pd.DataFrame(control_rows), "table_validity_controls.csv")

monotonicity = pd.read_csv(RUNS / "cfpb_online_recal1_seed0" / "table_monotonicity_paper.csv")
monotonicity = monotonicity[monotonicity["budget_target"] == 1.0]
mono_rows = []
for kind in ["freeze", "corr_mix"]:
    for panel, label in [("score_only", "Attacked slice"), ("all_only", "Control slice")]:
        subset = monotonicity[(monotonicity["kind"] == kind) & (monotonicity["panel"] == panel)]
        passed = int(subset["monotone_pass"].sum())
        mono_rows.append(
            {
                "Incident family": "Freeze" if kind == "freeze" else "CorrMix",
                "Slice": label,
                "Testable curves": len(subset),
                "Monotonicity pass": f"{passed}/{len(subset)}",
            }
        )
save(pd.DataFrame(mono_rows), "table_monotonicity.csv")

families = {
    "max_abs": "Scalar/statistical", "T2": "Scalar/statistical", "res_energy": "Scalar/statistical", "pc_t2": "Scalar/statistical",
    "factor_cov_change": "Factor/covariance", "factor_cov_delta": "Factor/covariance", "factor_cov_lrt": "Factor/covariance", "factor_cov_lrt_delta": "Factor/covariance",
    "coh_resid_t2": "Coherence", "coh_resid_energy": "Coherence",
    "fused_sp": "Fused", "fused_fisher": "Fused", "fused_sp_lrt_delta": "Fused", "fused_fisher_lrt_delta": "Fused",
    "ewma_factor_cov_lrt": "EWMA/CUSUM", "cusum_factor_cov_lrt": "EWMA/CUSUM", "ewma_coh_resid_t2": "EWMA/CUSUM", "cusum_coh_resid_t2": "EWMA/CUSUM",
}
family_order = ["Scalar/statistical", "Factor/covariance", "Coherence", "Fused", "EWMA/CUSUM"]
smap_rows = []
for run, dataset in [("smap_A1_weekly_ledger_seed0", "A-1"), ("smap_D15_weekly_ledger_seed0", "D-15")]:
    for kind in ["freeze", "corr_mix"]:
        subset = all_points[
            (all_points["run"] == run)
            & (all_points["kind"] == kind)
            & (all_points["panel"] == "attacked")
        ].copy()
        subset["Detector family"] = subset["detector"].map(families)
        for detector_family in family_order:
            candidates = subset[
                (subset["Detector family"] == detector_family)
                & (subset["matched_burden_contract_pass_rate"] > 0)
            ]
            if candidates.empty:
                smap_rows.append(
                    {
                        "Dataset": dataset,
                        "Incident family": "Freeze" if kind == "freeze" else "CorrMix",
                        "Detector family": detector_family,
                        "Detector": "--",
                        "Matched-burden Timely@K": "--",
                        "Matched-burden contract pass": "--",
                        "Joint coverage": "--",
                    }
                )
            else:
                best = candidates.sort_values(
                    ["matched_burden_timely_at_k", "detector"], ascending=[False, True]
                ).iloc[0]
                smap_rows.append(
                    {
                        "Dataset": dataset,
                        "Incident family": "Freeze" if kind == "freeze" else "CorrMix",
                        "Detector family": detector_family,
                        "Detector": detector_label(best["detector"]),
                        "Matched-burden Timely@K": q3(best["matched_burden_timely_at_k"]),
                        "Matched-burden contract pass": q3(best["matched_burden_contract_pass_rate"]),
                        "Joint coverage": q3(best["coverage_joint"]),
                    }
                )
save(pd.DataFrame(smap_rows), "table_smap_family_detail.csv")

# Recalibration ablation
recal_rows = []
for metric, off_value, on_value in []:
    pass
run_tables = {
    "off": pd.read_csv(RUNS / "cfpb_online_recal0_seed0" / "table_achieved_band1.csv"),
    "on": pd.read_csv(RUNS / "cfpb_online_recal1_seed0" / "table_achieved_band1.csv"),
}
operating_metrics = [
    ("Freeze matched-burden Timely@K", "freeze", "event_timely_sla_rate"),
    ("CorrMix matched-burden Timely@K", "corr_mix", "event_timely_sla_rate"),
    ("Freeze AIW", "freeze", "event_preexisting_warning_rate"),
    ("CorrMix AIW", "corr_mix", "event_preexisting_warning_rate"),
]
for label, kind, column in operating_metrics:
    values = {}
    for state, table in run_tables.items():
        subset = table[(table["panel"] == "score_only") & (table["kind"] == kind)]
        values[state] = pd.to_numeric(subset[column], errors="coerce").median()
    recal_rows.append({"Block": "Operating-point medians", "Metric": label, "Recalibration off": q3(values["off"]), "Recalibration on": q3(values["on"])})
for label, column in [
    ("Budget-only coverage", "coverage_budget"),
    ("Joint coverage", "coverage_joint"),
    ("All-trial contract-pass rate", "contract_pass_full_rate"),
]:
    values = {}
    for state, table in run_tables.items():
        subset = table[(table["panel"] == "score_only") & (table["kind"] == "freeze")]
        values[state] = pd.to_numeric(subset[column], errors="coerce").median()
    recal_rows.append({"Block": "Operating-point medians", "Metric": label, "Recalibration off": q3(values["off"]), "Recalibration on": q3(values["on"])})

for label, value_off, value_on in [
    ("Grid-median budget-only coverage", 0.550, 0.6875),
    ("Observed nonzero-joint detectors", "0/19", "8/19"),
]:
    recal_rows.append({"Block": "Intensity-grid summaries", "Metric": label, "Recalibration off": q3(value_off) if isinstance(value_off, float) else value_off, "Recalibration on": q3(value_on) if isinstance(value_on, float) else value_on})
for kind, label in [("freeze", "Freeze monotonicity pass count"), ("corr_mix", "CorrMix monotonicity pass count")]:
    counts = {}
    for state in ["off", "on"]:
        table = pd.read_csv(RUNS / f"cfpb_online_recal{0 if state == 'off' else 1}_seed0" / "table_monotonicity_paper.csv")
        subset = table[(table["kind"] == kind) & (table["panel"] == "score_only") & (table["budget_target"] == 1.0)]
        counts[state] = f"{int(subset['monotone_pass'].sum())}/{len(subset)}"
    recal_rows.append({"Block": "Intensity-grid summaries", "Metric": label, "Recalibration off": counts["off"], "Recalibration on": counts["on"]})
for label, column in [
    ("Mean regime-pass rate", "pass_rate"),
    ("Mean TIW-violation fraction", "frac_violate_tiw"),
    ("Mean episode-budget violation fraction", "frac_violate_events"),
]:
    values = {}
    for state in ["off", "on"]:
        table = pd.read_csv(RUNS / f"cfpb_online_recal{0 if state == 'off' else 1}_seed0" / "safety_compliance_summary.csv")
        subset = table[(table["panel"] == "score_only") & (table["budget"] >= 0.75)]
        values[state] = subset[column].mean()
    recal_rows.append({"Block": "Clean-regime compliance", "Metric": label, "Recalibration off": q3(values["off"]), "Recalibration on": q3(values["on"])})
save(pd.DataFrame(recal_rows), "table_recalibration_ablation.csv")

# Drift placement
drift_rows = []
for year in [2020, 2021, 2022]:
    table = pd.read_csv(RUNS / f"cfpb_driftplace_{year}-01-01_seed0" / "drift_slice_summary_paper.csv")
    row = table[(table["slice"] == "score_only_budget_ge_0p75") & (table["drift_delta"] == 1.0)].iloc[0]
    drift_rows.append(
        {
            "Drift start": f"{year}-01-01",
            "Clean pass-rate mean": q3(row["clean_pass_rate_mean"]),
            "Drift pass-rate mean": q3(row["pass_rate_mean"]),
            "Pass-rate change": ("+" if row["pass_rate_delta_vs_clean"] >= 0 else "") + q3(row["pass_rate_delta_vs_clean"]),
            "Flip 1→0 rate": q3(row["flip_1to0_rate"]),
            "Max event-ratio increase": q3(row["max_events_ratio_delta_vs_clean"]),
            "95th-percentile TIW ratio": q3(row["tiw_ratio_p95"]),
        }
    )
save(pd.DataFrame(drift_rows), "table_drift_placement.csv")

# -----------------------------------------------------------------------------
# Figure data
# -----------------------------------------------------------------------------
figure_2a = coverage[[
    "menu", "n_configurations", "median_budget_coverage", "median_budget_coverage_lower_2_5", "median_budget_coverage_upper_97_5", "median_joint_coverage", "median_joint_coverage_lower_2_5", "median_joint_coverage_upper_97_5"
]]
save(figure_2a, "fig2a_coverage_menus.csv", FIGURE_DATA)

achieved = pd.read_csv(RUNS / "cfpb_online_recal1_seed0" / "table_achieved_band1.csv")
cad_rows = []
for panel, label in [("score_only", "Attacked"), ("all_only", "Control")]:
    subset = achieved[
        (achieved["kind"] == "freeze")
        & (achieved["panel"] == panel)
        & (pd.to_numeric(achieved["budget_target_paper"], errors="coerce") == 1.0)
        & (pd.to_numeric(achieved["coverage_primary"], errors="coerce") > 0)
    ]
    cad_rows.extend(
        [
            {"slice": label, "window": "Incident", "median_CAD": pd.to_numeric(subset["mean_cad_win"], errors="coerce").median()},
            {"slice": label, "window": "Post", "median_CAD": pd.to_numeric(subset["mean_cad_post"], errors="coerce").median()},
        ]
    )
save(pd.DataFrame(cad_rows), "fig2b_cad_localization.csv", FIGURE_DATA)

silent = pd.read_csv(ANALYSIS / "silent_suppression_summary.csv").iloc[0]
save(
    pd.DataFrame(
        [
            {"category": "No alert-stream change", "count": int(silent["n_no_alert_stream_change"]), "fraction": silent["fraction_no_alert_stream_change"]},
            {"category": "Changed + warning overlap", "count": int(silent["n_changed_with_warning_overlap"]), "fraction": silent["fraction_changed_with_warning_overlap"]},
            {"category": "Silent suppression", "count": int(silent["n_silent_suppression"]), "fraction": silent["fraction_silent_suppression"]},
        ]
    ),
    "fig2c_silent_suppression.csv",
    FIGURE_DATA,
)

scatter = main[["display_row", "panel", "matched_burden_timely_at_k", "matched_burden_contract_pass_rate", "coverage_joint"]].copy()
scatter["display_row"] = scatter["display_row"].replace({"Coherence residual T2": "Coherence T²", "Max Abs control": "Max Abs control"})
save(scatter, "fig2d_timeliness_contract_scatter.csv", FIGURE_DATA)

surface_rows = []
for state, run in [("Recalibration ON", "cfpb_online_recal1_seed0"), ("Recalibration OFF", "cfpb_online_recal0_seed0")]:
    grid = pd.read_csv(RUNS / run / "table_intensity_grid.csv")
    for column in ["budget", "attack_strength", "event_timely_sla_rate"]:
        grid[column] = pd.to_numeric(grid[column], errors="coerce")
    for kind, family_name in [("freeze", "Freeze"), ("corr_mix", "CorrMix")]:
        subset = grid[(grid["panel"] == "score_only") & (grid["kind"] == kind) & (grid["budget"] == 1.0)]
        medians = subset.groupby("attack_strength")["event_timely_sla_rate"].median()
        for intensity, value in medians.items():
            surface_rows.append({"incident_family": family_name, "recalibration": state, "intensity": intensity, "median_MB_Timely_at_K": value})
save(pd.DataFrame(surface_rows), "fig3ab_matched_burden_timeliness_surfaces.csv", FIGURE_DATA)

recalibration_index = recalibration.set_index("estimand")
nonzero = pd.read_csv(ANALYSIS / "core_cfpb_nonzero_joint_fraction_intervals.csv") if (ANALYSIS / "core_cfpb_nonzero_joint_fraction_intervals.csv").exists() else None
# The nonzero-joint rows are also present in the long recalibration output.
recal_long = pd.read_csv(ANALYSIS / "core_cfpb_recalibration_intervals_long.csv")
fig3c_rows = []
for state, key in [("Recalibration ON", "ON"), ("Recalibration OFF", "OFF")]:
    budget = recalibration_index.loc["budget_coverage_median_19"]
    contract = recalibration_index.loc["all_trial_contract_pass_median_38"]
    fig3c_rows.append({"metric": "Median budget coverage", "state": state, "point": budget[f"point_{key}_fixed_menu_median"], "lower": budget[f"{key}_lower_2_5"], "upper": budget[f"{key}_upper_97_5"]})
    fig3c_rows.append({"metric": "Median all-trial contract pass", "state": state, "point": contract[f"point_{key}_fixed_menu_median"], "lower": contract[f"{key}_lower_2_5"], "upper": contract[f"{key}_upper_97_5"]})
    contrast = f"{key}_nonzero_joint_fraction"
    row = recal_long[recal_long["contrast"] == contrast].iloc[0]
    fig3c_rows.append({"metric": "Observed nonzero-joint fraction", "state": state, "point": row["point_estimate"], "lower": row["bootstrap_lower_2_5"], "upper": row["bootstrap_upper_97_5"]})
save(pd.DataFrame(fig3c_rows), "fig3c_recalibration_summary.csv", FIGURE_DATA)

main_drift = pd.read_csv(RUNS / "cfpb_online_recal1_seed0" / "drift_slice_summary_paper.csv")
row = main_drift[(main_drift["slice"] == "score_only_budget_ge_0p75") & (main_drift["drift_delta"] == 1.0)].iloc[0]
save(
    pd.DataFrame(
        [
            {"metric": "Clean pass rate", "value": row["clean_pass_rate_mean"]},
            {"metric": "Drift pass rate", "value": row["pass_rate_mean"]},
            {"metric": "Flip 1→0 rate", "value": row["flip_1to0_rate"]},
            {"metric": "Maximum event-burden ratio", "value": row["max_events_ratio_overall"]},
            {"metric": "Maximum TIW-burden ratio", "value": row["max_tiw_ratio_overall"]},
        ]
    ),
    "fig4a_default_drift_stress.csv",
    FIGURE_DATA,
)
placement = []
for year in [2020, 2021, 2022]:
    table = pd.read_csv(RUNS / f"cfpb_driftplace_{year}-01-01_seed0" / "drift_slice_summary_paper.csv")
    row = table[(table["slice"] == "score_only_budget_ge_0p75") & (table["drift_delta"] == 1.0)].iloc[0]
    placement.append({"drift_start": year, "pass_rate": row["pass_rate_mean"], "flip_1_to_0_rate": row["flip_1to0_rate"], "max_event_ratio_increase_vs_clean": row["max_events_ratio_delta_vs_clean"]})
save(pd.DataFrame(placement), "fig4b_drift_placement.csv", FIGURE_DATA)

print(f"Wrote {len(list(TABLES.glob('*.csv')))} paper tables and {len(list(FIGURE_DATA.glob('*.csv')))} figure-data files")
