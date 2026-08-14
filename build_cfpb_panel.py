#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_cfpb_panel.py

Build the CFPB Consumer Credit Trends (CCT) monthly panel used in the paper.

This script is a small CLI entrypoint that delegates to `cfpb_panel_adapter.py`.
It outputs:
  - a wide CSV panel with a `date` column (monthly cadence)
  - a pickle copy of the same panel
  - a JSON metadata/manifest that records the signal list and the score/control split

Usage:
  python build_cfpb_panel.py \
    --raw_csv data/raw/all_data.csv \
    --out_csv data/panels/cfpb_panel.csv \
    --out_pkl data/panels/cfpb_panel.pkl \
    --out_meta data/panels/cfpb_signals_metadata.json
"""

from __future__ import annotations


def main() -> None:
    import cfpb_panel_adapter as adapter

    adapter.main()


if __name__ == "__main__":
    main()
