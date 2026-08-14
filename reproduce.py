from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GENERATED_DIRS = [
    ROOT / "results" / "analysis",
    ROOT / "results" / "tables",
    ROOT / "results" / "figure_data",
    ROOT / "results" / "figures",
]
STATUS_FILES = [
    ROOT / "results" / "reproduction_summary.json",
    ROOT / "results" / "validation.csv",
]
STEPS = [
    "analysis/build_point_estimates.py",
    "analysis/core_intervals.py",
    "analysis/gradual_bias_intervals.py",
    "analysis/build_paper_outputs.py",
    "analysis/generate_figures.py",
    "analysis/validate_results.py",
]


def run_step(relative_script: str) -> None:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    analysis_path = str(ROOT / "analysis")
    env["PYTHONPATH"] = analysis_path + os.pathsep + env.get("PYTHONPATH", "")
    print(f"\n==> {relative_script}", flush=True)
    subprocess.run(
        [sys.executable, str(ROOT / relative_script)],
        cwd=ROOT,
        env=env,
        check=True,
    )


def main() -> None:
    for path in STATUS_FILES:
        path.unlink(missing_ok=True)

    for path in GENERATED_DIRS:
        shutil.rmtree(path, ignore_errors=True)
    (ROOT / "results").mkdir(exist_ok=True)

    current_step = "initialization"
    try:
        for script in STEPS:
            current_step = script
            run_step(script)
    except Exception:
        summary_path = ROOT / "results" / "reproduction_summary.json"
        if not summary_path.exists():
            summary_path.write_text(
                json.dumps({"failed_step": current_step, "status": "FAIL"}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        raise

    print("\nReproduction completed successfully.")
    print("Tables:  results/tables")
    print("Figures: results/figures")
    print("Checks:  results/validation.csv")


if __name__ == "__main__":
    main()
