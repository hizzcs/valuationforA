"""Valuation models for A股 issuers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from loguru import logger

from .data_pipeline import ValidatedInputs
from .risk_params import RiskProfile


@dataclass
class ValuationResult:
    ticker: str
    as_of_date: date
    method: str
    intrinsic_value: float
    cash_flows: List[float]
    discount_factors: List[float]
    terminal_value: float
    assumptions: Dict[str, float]
    metadata: Dict[str, str]
    equity_value: float = 0.0
    intrinsic_value_per_share: Optional[float] = None
    terminal_method: str = "perpetual"
    terminal_exit_multiple: Optional[float] = None


def _historical_growth(series: pd.Series, method: str = "cagr") -> float:
    """
    Calculate historical growth rate with multiple methods.

    Args:
        series: Historical financial data series
        method: Growth calculation method ("cagr" or "linear")

    Returns:
        float: Estimated growth rate
    """
    series = pd.to_numeric(series, errors="coerce").dropna().tail(6)
    if len(series) < 2:
        return 0.05

    if method == "cagr":
        start, end = series.iloc[0], series.iloc[-1]
        periods = len(series) - 1
        if start <= 0:
            return 0.05
        return float((end / start) ** (1 / periods) - 1)
    elif method == "linear":
        x = np.arange(len(series))
        y = series.values
        slope, _ = np.polyfit(x, np.log(y + 1e-9), 1)
        return float(np.exp(slope) - 1)
    else:
        logger.warning(f"Unknown growth method: {method}, defaulting to CAGR")
        return _historical_growth(series, "cagr")


def _operating_margin(fin: pd.DataFrame, fallback: float) -> float:
    if "revenue" not in fin or "net_profit" not in fin:
        return float(np.clip(fallback, -0.2, 0.5))
    revenue = pd.to_numeric(fin["revenue"], errors="coerce").dropna()
    profit = pd.to_numeric(fin["net_profit"], errors="coerce").dropna()
    if revenue.empty or profit.empty:
        return float(np.clip(fallback, -0.2, 0.5))
    margins = (profit / revenue).replace([np.inf, -np.inf], np.nan).dropna()
    if margins.empty:
        return float(np.clip(fallback, -0.2, 0.5))
    margin_mean = margins.tail(4).mean()
    margin_median = margins.tail(4).median()
    margin = (margin_mean + margin_median) / 2
    return float(np.clip(margin, -0.2, 0.5))


def _roic(fin: pd.DataFrame) -> float:
    if "net_profit" not in fin or "invested_capital" not in fin:
        return 0.1
    ordered = fin.sort_values("end_date") if "end_date" in fin else fin.copy()
    nopat = pd.to_numeric(ordered["net_profit"], errors="coerce")
    invested = pd.to_numeric(ordered["invested_capital"], errors="coerce")
    ratios = []
    for prof, capital in zip(nopat, invested):
        if capital and not np.isnan(capital):
            ratios.append(prof / capital)
    if not ratios:
        return 0.1
    roic_mean = np.nanmean(ratios[-4:])
    roic_median = np.nanmedian(ratios[-4:])
    roic = (roic_mean + roic_median) / 2
    return float(np.clip(roic, 0.01, 0.4))


def _reinvestment_rate(fin: pd.DataFrame, default: float = 0.3) -> float:
    ordered = fin.sort_values("end_date") if "end_date" in fin else fin.copy()
    invested = pd.to_numeric(ordered.get("invested_capital", pd.Series(dtype=float)), errors="coerce")
    nopat = pd.to_numeric(ordered.get("net_profit", pd.Series(dtype=float)), errors="coerce")
    if len(invested) >= 2 and not np.isnan(invested.iloc[-1]) and not np.isnan(invested.iloc[-2]):
        delta = invested.iloc[-1] - invested.iloc[-2]
        latest_nopat = nopat.iloc[-1] if not nopat.empty else np.nan
        if latest_nopat and not np.isnan(latest_nopat):
            ratio = max(min(delta / latest_nopat if latest_nopat else default, 0.9), 0.0)
            return float(ratio)
    if "operating_cf" in ordered and "net_profit" in ordered:
        cf = pd.to_numeric(ordered["operating_cf"], errors="coerce").dropna()
        profit = pd.to_numeric(ordered["net_profit"], errors="coerce").dropna()
        if not cf.empty and not profit.empty and profit.iloc[-1]:
            ratio = 1 - (cf.iloc[-1] / profit.iloc[-1])
            return float(np.clip(ratio, 0.0, 0.9))
    return default


def _determine_fade_years(fin: pd.DataFrame, fallback: int = 5) -> int:
    periods = max(len(fin), 1)
    derived = max(3, min(8, periods - 1))
    return derived if periods > 1 else fallback


def _terminal_value(
    last_cf: float,
    wacc: float,
    terminal_growth: float,
    terminal_method: str = "perpetual",
    exit_multiple: Optional[float] = None,
    last_nopat: Optional[float] = None,
) -> float:
    if terminal_method == "exit_multiple" and exit_multiple and last_nopat:
        return last_nopat * exit_multiple
    elif terminal_method == "perpetual":
        spread = max(wacc - terminal_growth, 0.005)
        return last_cf * (1 + terminal_growth) / spread
    else:
        logger.warning(f"Unknown terminal method: {terminal_method}, defaulting to perpetual")
        return _terminal_value(last_cf, wacc, terminal_growth)


def _valuation_metadata(validated: ValidatedInputs) -> Dict[str, str]:
    metadata = {
        "data_quality": validated.data_quality_grade,
        "source_mode": validated.metadata.get("source_mode", validated.metadata.get("source", "")),
    }
    if "verification_summary" in validated.metadata:
        metadata["verification_summary"] = validated.metadata["verification_summary"]
    return metadata


def _equity_value_per_share(enterprise_value: float, validated: ValidatedInputs) -> tuple[float, Optional[float]]:
    equity_value = enterprise_value - validated.net_debt
    shares = max(validated.shares_outstanding, 0.0)
    if shares <= 0:
        return float(equity_value), None
    return float(equity_value), float(equity_value / shares)


def _discount_factors(rate: float, periods: int) -> List[float]:
    return [1 / ((1 + rate) ** t) for t in range(1, periods + 1)]


def revenue_driven_dcf(
    validated: ValidatedInputs,
    risk: RiskProfile,
    horizon: int = 5,
    fade_years: Optional[int] = None,
    terminal_growth: float = 0.02,
    terminal_method: str = "perpetual",
    exit_multiple: Optional[float] = None,
    growth_method: str = "cagr",
) -> ValuationResult:
    fade = fade_years if fade_years is not None else _determine_fade_years(validated.statements)
    revenue_growth = _historical_growth(
        validated.statements.get("revenue", pd.Series(dtype=float)),
        method=growth_method,
    )
    fallback_margin = validated.net_profit / max(validated.revenue, 1.0)
    operating_margin = _operating_margin(validated.statements, fallback_margin)
    reinvestment_rate = _reinvestment_rate(validated.statements)
    cash_flows: List[float] = []
    revenue = validated.revenue
    last_nopat: Optional[float] = None
    for year in range(1, horizon + 1):
        fade_factor = max(0.0, 1 - (year - 1) / max(fade, 1))
        growth = terminal_growth + (revenue_growth - terminal_growth) * fade_factor
        revenue *= 1 + growth
        nopat = revenue * operating_margin
        reinvestment = nopat * reinvestment_rate
        fcff = nopat - reinvestment
        cash_flows.append(fcff)
        last_nopat = nopat
    discount_rate = max(risk.wacc, 0.01)
    discount = _discount_factors(discount_rate, horizon)
    pv_flows = sum(cf * df for cf, df in zip(cash_flows, discount))
    terminal_value = _terminal_value(
        cash_flows[-1],
        discount_rate,
        terminal_growth,
        terminal_method,
        exit_multiple,
        last_nopat,
    )
    pv_terminal = terminal_value / ((1 + discount_rate) ** horizon)
    intrinsic = pv_flows + pv_terminal
    equity_value, intrinsic_value_per_share = _equity_value_per_share(intrinsic, validated)
    assumptions = {
        "revenue_growth": revenue_growth,
        "operating_margin": operating_margin,
        "reinvestment_rate": reinvestment_rate,
        "terminal_growth": terminal_growth,
        "wacc": discount_rate,
        "fade_years": fade,
        "exit_multiple": exit_multiple,
        "growth_method": growth_method,
    }
    metadata = _valuation_metadata(validated)
    return ValuationResult(
        ticker=validated.ticker,
        as_of_date=validated.as_of_date,
        method="revenue",
        intrinsic_value=intrinsic,
        equity_value=equity_value,
        intrinsic_value_per_share=intrinsic_value_per_share,
        cash_flows=cash_flows,
        discount_factors=discount,
        terminal_value=terminal_value,
        assumptions=assumptions,
        metadata=metadata,
        terminal_method=terminal_method,
        terminal_exit_multiple=exit_multiple,
    )


def roic_driven_dcf(
    validated: ValidatedInputs,
    risk: RiskProfile,
    horizon: int = 5,
    fade_years: Optional[int] = None,
    terminal_growth: float = 0.02,
    terminal_method: str = "perpetual",
    exit_multiple: Optional[float] = None,
) -> ValuationResult:
    fade = fade_years if fade_years is not None else _determine_fade_years(validated.statements)
    roic = max(_roic(validated.statements), 0.01)
    reinvestment_rate = _reinvestment_rate(validated.statements)
    invested_capital = max(validated.invested_capital, 1.0)
    cash_flows: List[float] = []
    last_nopat: Optional[float] = None
    for year in range(1, horizon + 1):
        fade_factor = max(0.0, 1 - (year - 1) / max(fade, 1))
        roic_year = terminal_growth + (roic - terminal_growth) * fade_factor
        invested_capital *= 1 + reinvestment_rate
        nopat = invested_capital * roic_year
        reinvestment = invested_capital * reinvestment_rate
        fcff = nopat - reinvestment
        cash_flows.append(fcff)
        last_nopat = nopat
    discount_rate = max(risk.wacc, 0.01)
    discount = _discount_factors(discount_rate, horizon)
    pv_flows = sum(cf * df for cf, df in zip(cash_flows, discount))
    terminal_value = _terminal_value(
        cash_flows[-1],
        discount_rate,
        terminal_growth,
        terminal_method,
        exit_multiple,
        last_nopat,
    )
    pv_terminal = terminal_value / ((1 + discount_rate) ** horizon)
    intrinsic = pv_flows + pv_terminal
    equity_value, intrinsic_value_per_share = _equity_value_per_share(intrinsic, validated)
    assumptions = {
        "roic": roic,
        "reinvestment_rate": reinvestment_rate,
        "terminal_growth": terminal_growth,
        "wacc": discount_rate,
        "fade_years": fade,
        "exit_multiple": exit_multiple,
    }
    metadata = _valuation_metadata(validated)
    return ValuationResult(
        ticker=validated.ticker,
        as_of_date=validated.as_of_date,
        method="roic",
        intrinsic_value=intrinsic,
        equity_value=equity_value,
        intrinsic_value_per_share=intrinsic_value_per_share,
        cash_flows=cash_flows,
        discount_factors=discount,
        terminal_value=terminal_value,
        assumptions=assumptions,
        metadata=metadata,
        terminal_method=terminal_method,
        terminal_exit_multiple=exit_multiple,
    )


def two_stage_dcf(
    validated: ValidatedInputs,
    risk: RiskProfile,
    high_growth_years: int = 3,
    stable_growth_years: int = 2,
    high_growth_rate: Optional[float] = None,
    stable_growth_rate: float = 0.02,
    terminal_method: str = "perpetual",
    exit_multiple: Optional[float] = None,
    growth_method: str = "cagr",
) -> ValuationResult:
    horizon = high_growth_years + stable_growth_years
    fallback_margin = validated.net_profit / max(validated.revenue, 1.0)
    operating_margin = _operating_margin(validated.statements, fallback_margin)
    reinvestment_rate = _reinvestment_rate(validated.statements)

    if high_growth_rate is None:
        high_growth_rate = _historical_growth(
            validated.statements.get("revenue", pd.Series(dtype=float)),
            method=growth_method,
        )

    cash_flows: List[float] = []
    revenue = validated.revenue
    last_nopat: Optional[float] = None
    for year in range(1, horizon + 1):
        if year <= high_growth_years:
            growth = high_growth_rate
        else:
            fade_factor = (year - high_growth_years) / stable_growth_years
            growth = high_growth_rate + fade_factor * (stable_growth_rate - high_growth_rate)

        revenue *= 1 + growth
        nopat = revenue * operating_margin
        reinvestment = nopat * reinvestment_rate
        fcff = nopat - reinvestment
        cash_flows.append(fcff)
        last_nopat = nopat

    discount_rate = max(risk.wacc, 0.01)
    discount = _discount_factors(discount_rate, horizon)
    pv_flows = sum(cf * df for cf, df in zip(cash_flows, discount))
    terminal_value = _terminal_value(
        cash_flows[-1],
        discount_rate,
        stable_growth_rate,
        terminal_method,
        exit_multiple,
        last_nopat,
    )
    pv_terminal = terminal_value / ((1 + discount_rate) ** horizon)
    intrinsic = pv_flows + pv_terminal
    equity_value, intrinsic_value_per_share = _equity_value_per_share(intrinsic, validated)
    assumptions = {
        "high_growth_rate": high_growth_rate,
        "stable_growth_rate": stable_growth_rate,
        "operating_margin": operating_margin,
        "reinvestment_rate": reinvestment_rate,
        "wacc": discount_rate,
        "high_growth_years": high_growth_years,
        "stable_growth_years": stable_growth_years,
        "exit_multiple": exit_multiple,
        "growth_method": growth_method,
    }
    metadata = _valuation_metadata(validated)
    return ValuationResult(
        ticker=validated.ticker,
        as_of_date=validated.as_of_date,
        method="two_stage",
        intrinsic_value=intrinsic,
        equity_value=equity_value,
        intrinsic_value_per_share=intrinsic_value_per_share,
        cash_flows=cash_flows,
        discount_factors=discount,
        terminal_value=terminal_value,
        assumptions=assumptions,
        metadata=metadata,
        terminal_method=terminal_method,
        terminal_exit_multiple=exit_multiple,
    )


__all__ = ["ValuationResult", "revenue_driven_dcf", "roic_driven_dcf", "two_stage_dcf"]
