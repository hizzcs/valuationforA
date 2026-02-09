#!/usr/bin/env python3
"""Environment validator for the valuation workspace."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

from loguru import logger

sys.path.append(str((os.path.dirname(__file__))))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

try:
    import duckdb
except Exception:  # pragma: no cover
    duckdb = None  # type: ignore

from src.data_pipeline import DUCKDB_PATH, TuShareClient, _load_env_token, ensure_duckdb_schema  # noqa: E402

LIVE_ENDPOINTS: Dict[str, Dict[str, object]] = {
    "daily": {"ts_code": "600000.SH", "limit": 50},
    "income": {"ts_code": "600000.SH", "limit": 20},
    "cashflow": {"ts_code": "600000.SH", "limit": 20},
    "balancesheet": {"ts_code": "600000.SH", "limit": 20},
    "daily_basic": {"ts_code": "600000.SH", "limit": 50},
}

FIXTURE_ENDPOINTS: Dict[str, Dict[str, object]] = {
    "daily": {"ts_code": "600000.SH"},
    "financials": {"ts_code": "600000.SH"},
    "macro": {},
    "bonds": {},
    "csi300": {},
}

REQUIRED_TABLES = {
    "valuation_runs": [
        "ticker",
        "as_of_date",
        "intrinsic_value",
        "percentile_5",
        "scenario_seed",
        "wacc_details",
        "data_quality_grade",
        "source_mode",
    ],
    "risk_profiles": [
        "ticker",
        "as_of_date",
        "beta",
        "risk_free",
        "observations",
        "trace",
        "window_start",
    ],
    "valuation_alerts": ["ticker", "alert_type", "message", "severity"],
    "valuation_backtest": ["ticker", "valuation_date"],
}


def main() -> int:
    token = os.getenv("TUSHARE_TOKEN") or _load_env_token()
    if token:
        logger.info("TUSHARE_TOKEN detected; attempting live connectivity.")
    else:
        logger.warning("TUSHARE_TOKEN missing; validation will use fixtures.")

    client = TuShareClient(token=token)
    required_endpoints = LIVE_ENDPOINTS if token else FIXTURE_ENDPOINTS
    endpoint_failures: List[str] = []
    for endpoint, params in required_endpoints.items():
        try:
            df = client.call_api(endpoint, **params)
            mode = client.last_call.get("mode", client.mode)
            if token and mode != "live":
                endpoint_failures.append(endpoint)
                logger.error("Endpoint {} expected live mode but got {}", endpoint, mode)
            if df.empty:
                endpoint_failures.append(endpoint)
                logger.error("Endpoint {} returned zero rows", endpoint)
        except Exception as exc:  # pragma: no cover - runtime guard
            endpoint_failures.append(endpoint)
            logger.exception("Endpoint {} check failed: {}", endpoint, exc)

    table_failures = verify_duckdb_tables()

    if endpoint_failures or table_failures:
        if endpoint_failures:
            logger.error("Endpoint validation failed: {}", ", ".join(endpoint_failures))
        if table_failures:
            logger.error("DuckDB schema issues: {}", "; ".join(table_failures))
        return 1

    logger.info("Environment check passed.")
    return 0


def verify_duckdb_tables() -> List[str]:
    ensure_duckdb_schema()
    if duckdb is None:
        logger.warning("duckdb package missing; skipping schema verification.")
        return []
    issues: List[str] = []
    con = duckdb.connect(Path(DUCKDB_PATH).as_posix())
    try:
        for table, expected_cols in REQUIRED_TABLES.items():
            info = con.execute(f"PRAGMA table_info('{table}')").fetchall()
            existing = {row[1] for row in info}
            missing = [col for col in expected_cols if col not in existing]
            if missing:
                issues.append(f"{table} missing columns {', '.join(missing)}")
    finally:
        con.close()
    logger.info("DuckDB schema verified at {}", DUCKDB_PATH)
    return issues


if __name__ == "__main__":
    raise SystemExit(main())
