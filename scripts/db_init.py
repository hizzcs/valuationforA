#!/usr/bin/env python3
"""Initialize DuckDB schema and optionally seed sample data."""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.data_pipeline import ensure_duckdb_schema  # noqa: E402


def main() -> None:
    ensure_duckdb_schema()
    print("DuckDB schema ensured at duckdb/valuation.duckdb")


if __name__ == "__main__":
    main()
