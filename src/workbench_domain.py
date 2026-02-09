"""Domain orchestration helpers for the valuation workbench."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional

import pandas as pd

from .data_pipeline import ValidatedInputs
from .risk_params import RiskProfile
from .scenario_engine import ScenarioSummary
from .valuation_core import ValuationResult, revenue_driven_dcf, roic_driven_dcf, two_stage_dcf


METHOD_LABELS = {
    "revenue": "收入驱动 DCF",
    "roic": "ROIC 驱动 DCF",
    "two_stage": "两阶段 DCF",
}

UI_METHOD_MAPPING = {
    "收入驱动DCF": "revenue",
    "ROIC驱动DCF": "roic",
    "两阶段DCF": "two_stage",
}


@dataclass(frozen=True)
class ScenarioBand:
    """Monte Carlo valuation band."""

    p5: float
    p50: float
    p95: float


@dataclass(frozen=True)
class MethodSnapshot:
    """Read model outputs as a domain snapshot."""

    key: str
    label: str
    enterprise_value: float
    equity_value: float
    intrinsic_value_per_share: Optional[float]
    assumptions: Dict[str, float]
    cash_flows: list[float]
    discount_factors: list[float]
    terminal_value: float


@dataclass(frozen=True)
class DashboardSnapshot:
    """Aggregate dashboard context in a single domain object."""

    ticker: str
    as_of_date: date
    selected_method_key: str
    methods: Dict[str, MethodSnapshot]
    scenario_band: ScenarioBand
    market_price: Optional[float]
    margin_of_safety: Optional[float]
    risk: RiskProfile
    data_quality_grade: str
    source_mode: str
    verification: Dict[str, str]


def build_method_results(validated: ValidatedInputs, risk: RiskProfile) -> Dict[str, ValuationResult]:
    """Run baseline valuation models and keep a keyed dictionary for the UI layer."""
    return {
        "revenue": revenue_driven_dcf(validated, risk),
        "roic": roic_driven_dcf(validated, risk),
        "two_stage": two_stage_dcf(validated, risk),
    }


def to_method_snapshots(results: Dict[str, ValuationResult]) -> Dict[str, MethodSnapshot]:
    snapshots: Dict[str, MethodSnapshot] = {}
    for key, result in results.items():
        snapshots[key] = MethodSnapshot(
            key=key,
            label=METHOD_LABELS.get(key, key),
            enterprise_value=float(result.intrinsic_value),
            equity_value=float(result.equity_value),
            intrinsic_value_per_share=result.intrinsic_value_per_share,
            assumptions=result.assumptions,
            cash_flows=[float(x) for x in result.cash_flows],
            discount_factors=[float(x) for x in result.discount_factors],
            terminal_value=float(result.terminal_value),
        )
    return snapshots


def build_scenario_band(summary: ScenarioSummary) -> ScenarioBand:
    return ScenarioBand(
        p5=float(summary.percentile_5),
        p50=float(summary.percentile_50),
        p95=float(summary.percentile_95),
    )


def compute_margin_of_safety(
    intrinsic_value_per_share: Optional[float],
    market_price: Optional[float],
) -> Optional[float]:
    if intrinsic_value_per_share is None or market_price is None or market_price <= 0:
        return None
    return float(intrinsic_value_per_share / market_price - 1)


def build_dashboard_snapshot(
    *,
    ticker: str,
    as_of_date: date,
    selected_method_label: str,
    validated: ValidatedInputs,
    risk: RiskProfile,
    results: Dict[str, ValuationResult],
    scenario_summary: ScenarioSummary,
    market_price: Optional[float],
) -> DashboardSnapshot:
    selected_method_key = UI_METHOD_MAPPING.get(selected_method_label, "revenue")
    snapshots = to_method_snapshots(results)
    if selected_method_key not in snapshots:
        selected_method_key = "revenue"
    margin_of_safety = compute_margin_of_safety(
        snapshots[selected_method_key].intrinsic_value_per_share,
        market_price,
    )
    return DashboardSnapshot(
        ticker=ticker,
        as_of_date=as_of_date,
        selected_method_key=selected_method_key,
        methods=snapshots,
        scenario_band=build_scenario_band(scenario_summary),
        market_price=market_price,
        margin_of_safety=margin_of_safety,
        risk=risk,
        data_quality_grade=validated.data_quality_grade,
        source_mode=validated.metadata.get("source_mode", "fixture"),
        verification=validated.verification,
    )


def method_comparison_frame(snapshots: Dict[str, MethodSnapshot]) -> pd.DataFrame:
    rows = []
    for key, snap in snapshots.items():
        rows.append(
            {
                "method_key": key,
                "method": snap.label,
                "enterprise_value": snap.enterprise_value,
                "equity_value": snap.equity_value,
                "intrinsic_value_per_share": snap.intrinsic_value_per_share,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("enterprise_value", ascending=False).reset_index(drop=True)


def assumption_frame(snapshot: MethodSnapshot) -> pd.DataFrame:
    frame = pd.DataFrame(
        [{"参数": k, "值": v} for k, v in snapshot.assumptions.items() if v is not None]
    )
    if frame.empty:
        return frame
    return frame.sort_values("参数")


__all__ = [
    "ScenarioBand",
    "MethodSnapshot",
    "DashboardSnapshot",
    "METHOD_LABELS",
    "UI_METHOD_MAPPING",
    "build_method_results",
    "to_method_snapshots",
    "build_scenario_band",
    "compute_margin_of_safety",
    "build_dashboard_snapshot",
    "method_comparison_frame",
    "assumption_frame",
]
