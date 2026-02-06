#!/usr/bin/env python3
"""Load fixture CSV data into DuckDB for offline demos."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.data_pipeline import cache_dataframe, DUCKDB_PATH, ensure_duckdb_schema  # noqa: E402

FIXTURES = Path(ROOT) / "tests" / "data"


def main() -> None:
    ensure_duckdb_schema()
    mapping = {
        "financials": "raw_financials",
        "daily": "raw_prices",
        "macro": "macro_series",
    }
    for name, table in mapping.items():
        path = FIXTURES / f"{name}.csv"
        df = pd.read_csv(path)
        if name == "daily":
            df = df.rename(columns={"ts_code": "ticker"})
        cache_dataframe(table, df, path=DUCKDB_PATH)
        print(f"Loaded {len(df)} rows into {table}")


if __name__ == "__main__":
    main()
