
from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def to_month_index(dts: pd.DatetimeIndex) -> pd.PeriodIndex:
    return dts.to_period("M")

def months_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    """Return integer month difference b - a."""
    pa = a.to_period("M")
    pb = b.to_period("M")
    return (pb.year - pa.year) * 12 + (pb.month - pa.month)
