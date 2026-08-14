from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = json.loads((ROOT / "expected" / "checksums.json").read_text(encoding="utf-8"))
RESULTS = ROOT / "results"
CHECKS: list[dict[str, str | bool]] = []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(name: str, passed: bool, detail: str) -> None:
    CHECKS.append({"check": name, "passed": passed, "detail": detail})


def close(actual: float, expected: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)


for group in ["fixed_inputs", "generated_outputs"]:
    for relative_path, expected in EXPECTED[group].items():
        path = ROOT / relative_path
        exists = path.exists()
        check(f"{group}: {relative_path}", exists and sha256(path) == expected["sha256"], expected["sha256"])

for relative_path in EXPECTED["figure_files"]:
    path = ROOT / relative_path
    valid = path.exists() and path.stat().st_size > 1000
    if valid and path.suffix.lower() == ".png":
        with Image.open(path) as image:
            image.verify()
            valid = image.width > 1000 and image.height > 500
    check(f"figure: {relative_path}", valid, f"bytes={path.stat().st_size if path.exists() else 0}")

summary = json.loads((RESULTS / "analysis" / "point_estimate_build_summary.json").read_text(encoding="utf-8"))
check("19 clean detectors", summary["n_clean_detectors"] == 19, str(summary["n_clean_detectors"]))
check("12 positive-budget detectors", summary["n_positive_budget_detectors"] == 12, str(summary["n_positive_budget_detectors"]))
check("38 incident-dependent rows", summary["n_incident_rows"] == 38, str(summary["n_incident_rows"]))
check("22 rows defined under both settings", summary["n_rows_defined_under_both_settings"] == 22, str(summary["n_rows_defined_under_both_settings"]))
check("80 aligned base trials", summary["n_base_trials"] == 80, str(summary["n_base_trials"]))
check("8/19 nonzero joint ON", summary["nonzero_joint_ON"] == "8/19", summary["nonzero_joint_ON"])
check("0/19 nonzero joint OFF", summary["nonzero_joint_OFF"] == "0/19", summary["nonzero_joint_OFF"])
check("112/960 silent suppression", summary["silent_suppression"] == "112/960", summary["silent_suppression"])

coverage = pd.read_csv(RESULTS / "analysis" / "coverage_aggregation_point_estimates.csv").set_index("menu")
check("full-menu budget coverage", close(coverage.loc["full_fixed_19", "median_budget_coverage"], 0.6875), str(coverage.loc["full_fixed_19", "median_budget_coverage"]))
check("full-menu joint coverage", close(coverage.loc["full_fixed_19", "median_joint_coverage"], 0.0), str(coverage.loc["full_fixed_19", "median_joint_coverage"]))
check("positive-menu budget coverage", close(coverage.loc["fixed_observed_positive_budget_12", "median_budget_coverage"], 0.84375), str(coverage.loc["fixed_observed_positive_budget_12", "median_budget_coverage"]))
check("positive-menu joint coverage", close(coverage.loc["fixed_observed_positive_budget_12", "median_joint_coverage"], 0.025), str(coverage.loc["fixed_observed_positive_budget_12", "median_joint_coverage"]))

core = pd.read_csv(RESULTS / "analysis" / "core_cfpb_selected_rows_intervals_wide.csv").set_index("display_row")
check("Fused Fisher matched-burden Timely@K", close(core.loc["Fused Fisher", "matched_burden_timely_at_k"], 34 / 77), str(core.loc["Fused Fisher", "matched_burden_timely_at_k"]))
check("Fused Fisher matched-burden contract pass", close(core.loc["Fused Fisher", "matched_burden_contract_pass_rate"], 26 / 77), str(core.loc["Fused Fisher", "matched_burden_contract_pass_rate"]))
check("CUSUM matched-burden contract-pass boundary", close(core.loc["CUSUM Factor LRT", "matched_burden_contract_pass_rate"], 0.0), str(core.loc["CUSUM Factor LRT", "matched_burden_contract_pass_rate"]))

interval_summary = json.loads((RESULTS / "analysis" / "core_cfpb_interval_calculation_summary.json").read_text(encoding="utf-8"))
check("5,000 paired replicates", interval_summary["R_total"] == 5000, str(interval_summary["R_total"]))
check("4,770 valid replicates for shared-setting comparison", interval_summary["secondary_22_R_valid"] == 4770, str(interval_summary["secondary_22_R_valid"]))
check("4,750 reporting threshold", interval_summary["support_threshold"] == 4750, str(interval_summary["support_threshold"]))
check("78 base-trial pairing checks", interval_summary["expanded_pairing_rows"] == 78, str(interval_summary["expanded_pairing_rows"]))

validation_path = RESULTS / "validation.csv"
with validation_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["check", "passed", "detail"])
    writer.writeheader()
    writer.writerows(CHECKS)

failed = [row for row in CHECKS if not row["passed"]]
summary_out = {
    "status": "PASS" if not failed else "FAIL",
    "checks_passed": len(CHECKS) - len(failed),
    "checks_failed": len(failed),
}
(RESULTS / "reproduction_summary.json").write_text(json.dumps(summary_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary_out, indent=2, sort_keys=True))
if failed:
    for row in failed:
        print(f"FAILED: {row['check']}: {row['detail']}")
    raise SystemExit(1)
