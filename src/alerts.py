"""Valuation alert helpers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

try:
    import duckdb
except Exception:  # pragma: no cover
    duckdb = None  # type: ignore

from .data_pipeline import DUCKDB_PATH, ensure_duckdb_schema
from .scenario_engine import ScenarioSummary
from .valuation_core import ValuationResult


@dataclass
class Alert:
    ticker: str
    alert_type: str
    message: str
    severity: str


def evaluate_alerts(result: ValuationResult, scenario: Optional[ScenarioSummary]) -> List[Alert]:
    alerts: List[Alert] = []
    if result.assumptions.get("operating_margin", 0.15) < 0:
        alerts.append(Alert(result.ticker, "negative_margin", "Operating margin negative", "high"))
    if result.metadata.get("data_quality") == "C":
        alerts.append(Alert(result.ticker, "data_quality", "Data quality grade C", "medium"))
    if result.metadata.get("source_mode") != "live":
        alerts.append(Alert(result.ticker, "fixture_mode", "Using fixture data instead of live TuShare", "low"))
    if scenario:
        spread = scenario.percentile_95 / max(scenario.percentile_5, 1e-6)
        if spread >= 3:
            alerts.append(
                Alert(
                    result.ticker,
                    "scenario_spread",
                    "Monte Carlo dispersion above 3x between P95 and P5",
                    "medium",
                )
            )
    return alerts


def persist_alerts(alerts: List[Alert], path: Path = DUCKDB_PATH) -> None:
    if not alerts:
        return
    if duckdb is None:
        raise RuntimeError("duckdb package is required to persist alerts.")
    ensure_duckdb_schema(path)
    con = duckdb.connect(path.as_posix())
    try:
        for alert in alerts:
            con.execute(
                """
                INSERT INTO valuation_alerts (ticker, alert_type, message, severity, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [alert.ticker, alert.alert_type, alert.message, alert.severity, datetime.now(timezone.utc)],
            )
    finally:
        con.close()


__all__ = ["Alert", "evaluate_alerts", "persist_alerts"]
