from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "runs"

RUN_META = {
    "cfpb_online_recal1_seed0": {"dataset": "CFPB", "role": "canonical_on"},
    "cfpb_online_recal0_seed0": {"dataset": "CFPB", "role": "canonical_off"},
    "cfpb_online_recal1_tiw010_seed0": {"dataset": "CFPB", "role": "tiw_010"},
    "cfpb_online_recal1_tiw020_seed0": {"dataset": "CFPB", "role": "tiw_020"},
    "cfpb_recal_win36_seed0": {"dataset": "CFPB", "role": "sensitivity_win36"},
    "cfpb_recal_eta015_seed0": {"dataset": "CFPB", "role": "sensitivity_eta015"},
    "cfpb_recal_upd3_seed0": {"dataset": "CFPB", "role": "sensitivity_upd3"},
    "cfpb_recal_maxup110_seed0": {"dataset": "CFPB", "role": "sensitivity_maxup110"},
    "cfpb_gradual_bias_seed0": {"dataset": "CFPB", "role": "gradual_bias"},
    "smap_A1_weekly_ledger_seed0": {"dataset": "SMAP A-1", "role": "canonical"},
    "smap_D15_weekly_ledger_seed0": {"dataset": "SMAP D-15", "role": "canonical"},
    "smap_A1_gradual_bias_seed0": {"dataset": "SMAP A-1", "role": "gradual_bias"},
    "smap_D15_gradual_bias_seed0": {"dataset": "SMAP D-15", "role": "gradual_bias"},
}

PANEL_LABEL = {"score_only": "attacked", "all_only": "control", "full": "full"}


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", newline="", encoding="utf-8-sig")
    return path.open(newline="", encoding="utf-8-sig")


def read_csv(path: Path) -> list[dict[str, str]]:
    with _open_text(path) as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "" or text.upper() in {"NA", "NAN", "INFEASIBLE", "NONE"}:
        return default
    try:
        return float(text)
    except Exception:
        return default


def inum(value: Any, default: int = 0) -> int:
    value_float = fnum(value, float("nan"))
    return default if not np.isfinite(value_float) else int(round(value_float))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_manifest(run_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    with (run_dir / "run_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    tokens = shlex.split(manifest.get("cmd", ""))
    args: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--"):
            key = token[2:]
            value = "1"
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                value = tokens[index + 1]
                index += 1
            args[key] = value
        index += 1
    return manifest, args


def arg_float(args: dict[str, str], key: str, default: float) -> float:
    try:
        return float(args.get(key, default))
    except Exception:
        return float(default)


@dataclass(frozen=True)
class RowSpec:
    kind: str
    panel: str
    detector: str
    budget_nominal: float
    attack_strength: float
    attack_strength_applicable: bool
    budget_target: float = 1.0

    @property
    def key(self) -> tuple[str, str, str]:
        return self.kind, self.panel, self.detector

    @property
    def id(self) -> str:
        return f"{self.kind}|{self.panel}|{self.detector}"


def load_row_specs(run_dir: Path) -> list[RowSpec]:
    rows = read_csv(run_dir / "table_achieved_band1_raw.csv")
    specs: list[RowSpec] = []
    for row in rows:
        specs.append(
            RowSpec(
                kind=str(row["kind"]),
                panel=str(row["panel"]),
                detector=str(row["detector"]),
                budget_nominal=fnum(row.get("budget_nominal_selected"), 1.0),
                attack_strength=fnum(row.get("attack_strength_selected"), 0.0),
                attack_strength_applicable=bool(inum(row.get("attack_strength_selected_is_applicable"), 1)),
                budget_target=fnum(row.get("budget_target_paper"), 1.0),
            )
        )
    return specs


def stream_selected_trial_rows(
    run_dir: Path, specs: list[RowSpec]
) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    by_key = {spec.key: spec for spec in specs}
    selected = {spec.key: [] for spec in specs}
    trial_path = run_dir / "trial_level_metrics.csv.gz"
    if not trial_path.exists():
        trial_path = run_dir / "trial_level_metrics.csv"
    with _open_text(trial_path) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = str(row["kind"]), str(row["panel"]), str(row["detector"])
            spec = by_key.get(key)
            if spec is None:
                continue
            if not math.isclose(
                fnum(row.get("budget")), spec.budget_nominal, rel_tol=0, abs_tol=1e-10
            ):
                continue
            if spec.attack_strength_applicable and not math.isclose(
                fnum(row.get("attack_strength")), spec.attack_strength, rel_tol=0, abs_tol=1e-10
            ):
                continue
            selected[key].append(row)
    return selected


def _arr(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return np.array([fnum(row.get(field)) for row in rows], dtype=float)


def median(values: Iterable[float]) -> float:
    array = np.array(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if len(array) else float("nan")


def compute_metrics(rows: list[dict[str, str]], spec: RowSpec, args: dict[str, str]) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"No rows for {spec.id}")

    def sort_key(row: dict[str, str]) -> tuple[int, Any]:
        base_trial = str(row["base_trial"])
        try:
            return 0, int(base_trial)
        except Exception:
            return 1, base_trial

    rows = sorted(rows, key=sort_key)
    base_trials = [str(row["base_trial"]) for row in rows]
    if len(base_trials) != len(set(base_trials)):
        raise ValueError(f"Duplicate base_trial values in {spec.id}")

    matched_tol = arg_float(args, "matched_tol", 0.10)
    raw_tiw_tol = args.get("matched_tiw_tol")
    matched_tiw_tol = matched_tol if raw_tiw_tol in (None, "", "None") else float(raw_tiw_tol)
    safety_eps = arg_float(args, "safety_eps", 0.10)
    tiw_cap = arg_float(args, "tiw_budget", 0.0)

    years = _arr(rows, "false_alert_years")
    count = _arr(rows, "false_alert_events_count")
    target = np.rint(years * float(spec.budget_target))
    band = np.maximum(1.0, np.ceil(matched_tol * np.maximum(target, 1.0)))
    a_b = np.isfinite(count) & np.isfinite(target) & (np.abs(count - target) <= band)

    clean_tiw = _arr(rows, "tiw_null")
    if tiw_cap > 0:
        tiw_safe_match = np.isfinite(clean_tiw) & (clean_tiw <= (1.0 + matched_tiw_tol) * tiw_cap)
        tiw_safe_contract = np.isfinite(clean_tiw) & (clean_tiw <= (1.0 + safety_eps) * tiw_cap)
    else:
        tiw_safe_match = np.ones(len(rows), dtype=bool)
        tiw_safe_contract = np.ones(len(rows), dtype=bool)
    a_j = a_b & tiw_safe_match

    clean_fae = _arr(rows, "fae_null")
    budget_safe = np.isfinite(clean_fae) & (
        clean_fae <= (1.0 + safety_eps) * float(spec.budget_target)
    )
    timely = _arr(rows, "event_timely_sla") == 1
    contract = timely & budget_safe & tiw_safe_contract
    stored_contract = _arr(rows, "contract_pass_full") == 1
    if not np.array_equal(contract, stored_contract):
        count_mismatch = int(np.sum(contract != stored_contract))
        raise ValueError(f"Contract reconstruction mismatch for {spec.id}: {count_mismatch}")

    corrupted_tiw = _arr(rows, "time_in_warning")
    if np.any(a_b & ~np.isfinite(clean_tiw)):
        raise ValueError(f"Nonfinite clean TIW inside A_B for {spec.id}")
    if np.any(a_b & ~np.isfinite(corrupted_tiw)):
        raise ValueError(f"Nonfinite corrupted TIW inside A_B for {spec.id}")

    cad = _arr(rows, "cad_win")
    corrupted_fae = _arr(rows, "false_alert_events_per_year")
    event_success = _arr(rows, "event_success") == 1

    def mean_masked(array: np.ndarray, mask: np.ndarray) -> float:
        values = array[mask]
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if len(values) else float("nan")

    def mean_complete_matched(array: np.ndarray) -> float:
        return float(np.mean(array[a_b])) if np.any(a_b) else float("nan")

    n_total = len(rows)
    n_ab = int(a_b.sum())
    n_aj = int(a_j.sum())
    n_timely_ab = int((timely & a_b).sum())
    n_contract_ab = int((contract & a_b).sum())
    n_contract_all = int(contract.sum())

    return {
        "spec": spec,
        "rows": rows,
        "base_trials": base_trials,
        "a_b": a_b,
        "a_j": a_j,
        "budget_safe": budget_safe,
        "tiw_safe_match": tiw_safe_match,
        "tiw_safe_contract": tiw_safe_contract,
        "timely": timely,
        "contract": contract,
        "event_success": event_success,
        "n_total": n_total,
        "n_A_B": n_ab,
        "n_A_J": n_aj,
        "n_timely_A_B": n_timely_ab,
        "n_matched_contract_pass": n_contract_ab,
        "n_all_trial_contract_pass": n_contract_all,
        "coverage_budget": n_ab / n_total,
        "coverage_joint": n_aj / n_total,
        "matched_burden_timely_at_k": n_timely_ab / n_ab if n_ab else float("nan"),
        "matched_burden_contract_pass_rate": n_contract_ab / n_ab if n_ab else float("nan"),
        "all_trial_contract_pass_rate": n_contract_all / n_total,
        "mean_tiw_clean_matched": mean_complete_matched(clean_tiw),
        "mean_tiw_corrupted_matched": mean_complete_matched(corrupted_tiw),
        "mean_cad_win_matched": mean_masked(cad, a_b),
        "mean_false_alert_events_corrupted_matched": mean_masked(corrupted_fae, a_b),
        "matched_tol_budget": matched_tol,
        "matched_tol_tiw": matched_tiw_tol,
        "safety_eps": safety_eps,
        "tiw_cap": tiw_cap,
    }


def load_run_metrics(run_name: str) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, Any], dict[str, str]]:
    run_dir = DATA_ROOT / run_name
    specs = load_row_specs(run_dir)
    selected = stream_selected_trial_rows(run_dir, specs)
    manifest, args = parse_manifest(run_dir)
    metrics: dict[tuple[str, str, str], dict[str, Any]] = {}
    for spec in specs:
        rows = selected.get(spec.key, [])
        if rows:
            metrics[spec.key] = compute_metrics(rows, spec, args)
    return metrics, manifest, args


def metric_public_row(run_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    spec: RowSpec = metrics["spec"]
    meta = RUN_META.get(run_name, {"dataset": "Unknown", "role": "unknown"})
    return {
        "run": run_name,
        "dataset": meta["dataset"],
        "run_role": meta["role"],
        "kind": spec.kind,
        "panel_code": spec.panel,
        "panel": PANEL_LABEL.get(spec.panel, spec.panel),
        "detector": spec.detector,
        "budget_nominal": spec.budget_nominal,
        "budget_target": spec.budget_target,
        "attack_strength": spec.attack_strength,
        "n_total": metrics["n_total"],
        "n_A_B": metrics["n_A_B"],
        "n_A_J": metrics["n_A_J"],
        "n_timely_A_B": metrics["n_timely_A_B"],
        "n_matched_contract_pass": metrics["n_matched_contract_pass"],
        "n_all_trial_contract_pass": metrics["n_all_trial_contract_pass"],
        "coverage_budget": metrics["coverage_budget"],
        "coverage_joint": metrics["coverage_joint"],
        "matched_burden_timely_at_k": metrics["matched_burden_timely_at_k"],
        "matched_burden_contract_pass_rate": metrics["matched_burden_contract_pass_rate"],
        "all_trial_contract_pass_rate": metrics["all_trial_contract_pass_rate"],
        "mean_tiw_clean_matched": metrics["mean_tiw_clean_matched"],
        "mean_tiw_corrupted_matched": metrics["mean_tiw_corrupted_matched"],
        "mean_cad_win_matched": metrics["mean_cad_win_matched"],
        "mean_false_alert_events_corrupted_matched": metrics[
            "mean_false_alert_events_corrupted_matched"
        ],
        "matched_tol_budget": metrics["matched_tol_budget"],
        "matched_tol_tiw": metrics["matched_tol_tiw"],
        "safety_eps": metrics["safety_eps"],
        "tiw_cap": metrics["tiw_cap"],
    }
