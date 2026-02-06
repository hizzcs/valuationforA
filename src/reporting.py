"""Reporting helpers to persist valuation results."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import duckdb
except Exception:  # pragma: no cover
    duckdb = None  # type: ignore

from .scenario_engine import ScenarioSummary
from .valuation_core import ValuationResult
from .data_pipeline import DUCKDB_PATH, ensure_duckdb_schema
from .risk_params import RiskProfile


def save_valuation(
    valuation: ValuationResult,
    scenario: Optional[ScenarioSummary],
    risk: Optional[RiskProfile],
    path: Path = DUCKDB_PATH,
) -> None:
    if duckdb is None:
        raise RuntimeError("duckdb package is required to persist valuation results.")
    ensure_duckdb_schema(path)
    con = duckdb.connect(path.as_posix())
    try:
        scenario_seed = scenario.seed if scenario else None
        scenario_inputs = None
        if scenario:
            payload = {"draws": scenario.draws}
            payload.update(scenario.inputs)
            scenario_inputs = json.dumps(payload)
        wacc_payload = {"valuation": valuation.assumptions}
        if risk is not None:
            wacc_payload["risk"] = risk.trace
        con.execute(
            """
            INSERT INTO valuation_runs (
                ticker, as_of_date, method, intrinsic_value,
                percentile_5, percentile_50, percentile_95,
                scenario_seed, scenario_inputs,
                wacc_details, data_quality_grade, source_mode, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                valuation.ticker,
                valuation.as_of_date,
                valuation.method,
                valuation.intrinsic_value,
                scenario.percentile_5 if scenario else None,
                scenario.percentile_50 if scenario else None,
                scenario.percentile_95 if scenario else None,
                scenario_seed,
                scenario_inputs,
                json.dumps(wacc_payload),
                valuation.metadata.get("data_quality"),
                valuation.metadata.get("source_mode"),
                datetime.now(timezone.utc),
            ],
        )
    finally:
        con.close()


def log_risk_profile(risk: RiskProfile, path: Path = DUCKDB_PATH) -> None:
    if duckdb is None:
        raise RuntimeError("duckdb package is required to persist risk profiles.")
    ensure_duckdb_schema(path)
    con = duckdb.connect(path.as_posix())
    try:
        con.execute(
            """
            INSERT INTO risk_profiles (
                ticker, as_of_date, beta, risk_free, cost_of_equity,
                cost_of_debt, wacc, observations, std_err,
                trace, window_start, window_end, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                risk.ticker,
                risk.as_of_date,
                risk.beta,
                risk.risk_free,
                risk.cost_of_equity,
                risk.cost_of_debt,
                risk.wacc,
                risk.observations,
                risk.std_err,
                json.dumps(risk.trace),
                risk.window_start,
                risk.window_end,
                datetime.now(timezone.utc),
            ],
        )
    finally:
        con.close()


__all__ = ["save_valuation", "log_risk_profile"]
