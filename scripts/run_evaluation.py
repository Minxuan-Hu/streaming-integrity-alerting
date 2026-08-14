from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "runs"
PANELS = ROOT / "data" / "panels"
ENGINE = ROOT / "run_contract_eval.py"
OUTPUT = ROOT / "results" / "full_evaluations"


def available_runs() -> list[str]:
    return sorted(path.parent.name for path in RUNS.glob("*/run_manifest.json"))


def replace_option(arguments: list[str], option: str, value: str) -> None:
    try:
        position = arguments.index(option)
    except ValueError as exc:
        raise RuntimeError(f"Saved command is missing {option}") from exc
    arguments[position + 1] = value


def command_for(run_name: str) -> list[str]:
    manifest_path = RUNS / run_name / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Unknown run: {run_name}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved = shlex.split(manifest["cmd"])
    if len(saved) < 2:
        raise RuntimeError(f"Invalid command in {manifest_path}")

    panel_name = Path(manifest["panel_csv"]).name
    panel_path = PANELS / panel_name
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Processed panel not found: {panel_path}. "
            "Construct the required panel using the commands in README.md."
        )

    saved[0] = sys.executable
    saved[1] = str(ENGINE)
    replace_option(saved, "--panel_csv", str(panel_path))
    replace_option(saved, "--out_dir", str(OUTPUT / run_name))
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun one saved evaluation configuration.")
    parser.add_argument("run_name", nargs="?", help="Run directory name under data/runs")
    parser.add_argument("--list", action="store_true", help="List available configurations")
    parser.add_argument("--print-command", action="store_true", help="Print the resolved command without running it")
    args = parser.parse_args()

    runs = available_runs()
    if args.list:
        print("\n".join(runs))
        return
    if not args.run_name:
        parser.error("provide a run name or use --list")
    if args.run_name not in runs:
        parser.error(f"unknown run: {args.run_name}")

    command = command_for(args.run_name)
    print(shlex.join(command))
    if not args.print_command:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
