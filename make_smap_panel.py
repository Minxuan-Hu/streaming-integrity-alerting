#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_smap_panel.py

Convert Telemanom-format NASA SMAP/MSL arrays (train/test .npy per system ID)
into the common panel CSV interface used by the operational evaluation framework
runner (`run_contract_eval.py`).

Usage:
  python make_smap_panel.py --smap_zip SMAP.zip --series_id A-1 \
    --out_dir smap_refined --downsample_k 7 --downsample_mode mean --n_score_cols 10
"""

from __future__ import annotations


def main() -> None:
    import smap_panel_adapter

    smap_panel_adapter.main()


if __name__ == "__main__":
    main()
