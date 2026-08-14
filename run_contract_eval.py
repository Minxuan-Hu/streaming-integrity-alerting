#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_contract_eval.py

Run the Streaming Integrity Monitoring **operational evaluation framework**.

This is the main entrypoint used to reproduce the paper’s results. It reads a
panel CSV (wide table with a time column) and writes framework-facing artifacts
(CSVs) to an output directory.

Usage:
  python run_contract_eval.py --panel_csv <panel.csv> --out_dir <results_dir> [flags]

All CLI flags are defined in `contract_runner.py`.
"""

from __future__ import annotations


def main() -> None:
    import contract_runner

    contract_runner.main()


if __name__ == "__main__":
    main()
