"""cfpb_panel_adapter.py

Dataset adapter for CFPB Consumer Credit Trends (CCT).

Reads the CFPB "All data points" CSV and writes a wide monthly panel:
- uses Seasonally Adjusted YoY values (value_yoy)
- deterministic, fixed 54-signal column set
- explicit score-tier ("|score|") vs negative-control ("|all|all|") split
- refuses to silently aggregate duplicate (date, signal) rows
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SCORE_BUCKETS = ["Deep Subprime", "Subprime", "Near Prime", "Prime", "Superprime"]
LOAN_TYPES = ["AUT", "CRC", "MTG", "STU"]

VALUE_TYPE = "Seasonally Adjusted"
SIGNAL_FIELD = "value_yoy"


def _colname(series: str, loan_type: str, subgroup: str, subgroup_level: str) -> str:
    """Build a deterministic, regex-friendly column name."""
    series = str(series)
    loan_type = str(loan_type)
    subgroup = str(subgroup).lower()
    subgroup_level = str(subgroup_level)

    if subgroup == "all":
        # all/all negative-control surface
        return f"{series}|{loan_type}|all|all|{VALUE_TYPE}|{SIGNAL_FIELD}"

    # score-tier attacked surface
    return f"{series}|{loan_type}|{subgroup}|{subgroup_level}|{VALUE_TYPE}|{SIGNAL_FIELD}"


def build_panel(raw_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_csv)
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    # Keep only Seasonally Adjusted YoY values
    df = df[(df["value_type"] == VALUE_TYPE) & df[SIGNAL_FIELD].notna()].copy()
    df[SIGNAL_FIELD] = pd.to_numeric(df[SIGNAL_FIELD], errors="coerce")
    df = df.dropna(subset=[SIGNAL_FIELD])

    # Subgroup: all
    df_all = df[
        (df["subgroup"] == "all")
        & (df["subgroup_level"] == "all")
        & (df["series"].isin(["Inquiry Index", "Credit Tightness Index", "Originations", "Dollar Volume"]))
    ].copy()

    # Subgroup: score (Originations + Dollar Volume only)
    df_score = df[
        (df["subgroup"] == "score")
        & (df["subgroup_level"].isin(SCORE_BUCKETS))
        & (df["series"].isin(["Originations", "Dollar Volume"]))
        & (df["loan_type"].isin(LOAN_TYPES))
    ].copy()

    df_all["col"] = df_all.apply(
        lambda r: _colname(r["series"], r["loan_type"], r["subgroup"], r["subgroup_level"]), axis=1
    )
    df_score["col"] = df_score.apply(
        lambda r: _colname(r["series"], r["loan_type"], r["subgroup"], r["subgroup_level"]), axis=1
    )

    use = pd.concat([df_all, df_score], ignore_index=True)

    # Refuse to silently aggregate duplicate rows.
    dup = use.groupby(["date", "col"]).size()
    bad = dup[dup > 1]
    if len(bad) > 0:
        ex = bad.sort_values(ascending=False).head(10)
        raise RuntimeError(
            f"Duplicated (date,col) rows found (n={len(bad)}). Refusing to aggregate. "
            f"Examples:\n{ex.to_string()}"
        )

    panel = use.pivot(index="date", columns="col", values=SIGNAL_FIELD).sort_index()

    # Enforce the exact 54 expected columns in a deterministic order
    expected_all = []
    for s in ["Inquiry Index", "Credit Tightness Index"]:
        for loan in ["AUT", "CRC", "MTG"]:
            expected_all.append(_colname(s, loan, "all", "all"))

    for s in ["Originations", "Dollar Volume"]:
        for loan in ["AUT", "CRC", "MTG", "STU"]:
            expected_all.append(_colname(s, loan, "all", "all"))

    expected_score = []
    for s in ["Originations", "Dollar Volume"]:
        for loan in ["AUT", "CRC", "MTG", "STU"]:
            for sb in SCORE_BUCKETS:
                expected_score.append(_colname(s, loan, "score", sb))

    expected_cols = expected_all + expected_score

    missing = [c for c in expected_cols if c not in panel.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns (n={len(missing)}): {missing[:10]}")

    panel = panel[expected_cols]

    # Drop any dates with missing values across selected signals (full-case months)
    panel = panel.dropna(axis=0, how="any")

    return panel


def write_metadata(panel: pd.DataFrame, out_json: Path) -> None:
    meta = {
        "source": "CFPB Consumer Credit Trends (CCT) all_data.csv",
        "value_type": VALUE_TYPE,
        "signal": SIGNAL_FIELD,
        "date_min": str(panel.index.min().date()),
        "date_max": str(panel.index.max().date()),
        "n_months": int(panel.shape[0]),
        "n_signals": int(panel.shape[1]),
        "score_buckets": SCORE_BUCKETS,
        "loan_types": LOAN_TYPES,
        # Explicit panel slices used by the evaluation harness.
        "slices": {
            "all_only": [c for c in panel.columns if "|all|all|" in c],
            "score_only": [c for c in panel.columns if "|score|" in c],
            "full": list(panel.columns),
        },
        "columns": [],
    }

    for c in panel.columns:
        # c format: Series|Loan|segment1|segment2|VALUE_TYPE|SIGNAL
        parts = c.split("|")
        if len(parts) != 6:
            raise RuntimeError(f"Unexpected column format (expected 6 parts split by '|'): {c}")
        series, loan_type, seg1, seg2, value_type, signal = parts
        meta["columns"].append(
            {
                "col": c,
                "series": series,
                "loan_type": loan_type,
                "segment": f"{seg1}|{seg2}",
                "value_type": value_type,
                "signal": signal,
            }
        )

    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", type=str, default="data/raw/all_data.csv")
    ap.add_argument(
        "--out_csv",
        type=str,
        default="data/panels/cfpb_panel.csv",
    )
    ap.add_argument(
        "--out_pkl",
        type=str,
        default="data/panels/cfpb_panel.pkl",
    )
    ap.add_argument("--out_meta", type=str, default="data/panels/cfpb_signals_metadata.json")

    # Optional assertions that detect changes in the CFPB source file.
    ap.add_argument("--assert_n_signals", type=int, default=54)
    ap.add_argument("--assert_date_min", type=str, default="2008-01-01")
    ap.add_argument("--assert_date_max", type=str, default="2025-04-01")

    args = ap.parse_args()

    raw_csv = Path(args.raw_csv)
    out_csv = Path(args.out_csv)
    out_pkl = Path(args.out_pkl)
    out_meta = Path(args.out_meta)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    panel = build_panel(raw_csv)

    # Assertions
    if int(args.assert_n_signals) > 0 and panel.shape[1] != int(args.assert_n_signals):
        raise RuntimeError(f"Unexpected n_signals: got {panel.shape[1]}, expected {int(args.assert_n_signals)}")

    if str(args.assert_date_min).strip():
        exp_min = pd.to_datetime(str(args.assert_date_min))
        if panel.index.min() != exp_min:
            raise RuntimeError(f"Unexpected date_min: got {panel.index.min().date()}, expected {exp_min.date()}")

    if str(args.assert_date_max).strip():
        exp_max = pd.to_datetime(str(args.assert_date_max))
        if panel.index.max() != exp_max:
            raise RuntimeError(f"Unexpected date_max: got {panel.index.max().date()}, expected {exp_max.date()}")

    panel.to_csv(out_csv, index=True, date_format="%Y-%m-%d")
    panel.to_pickle(out_pkl)
    write_metadata(panel, out_meta)

    print(f"Wrote panel: {out_csv} ({panel.shape[0]} months, {panel.shape[1]} signals)")
    print(f"Wrote metadata: {out_meta}")


if __name__ == "__main__":
    main()
