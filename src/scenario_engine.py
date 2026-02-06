"""Scenario generation and Monte Carlo simulation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .valuation_core import ValuationResult


@dataclass
class ScenarioSummary:
    seed: int
    draws: int
    percentile_5: float
    percentile_50: float
    percentile_95: float
    samples: List[float]
    inputs: Dict[str, float]


def historical_volatility(series: pd.Series) -> float:
    returns = series.pct_change().dropna()
    if returns.empty:
        return 0.2
    return float(returns.std())


def build_distribution_params(
    price_history: pd.DataFrame,
    macro: pd.DataFrame,
    base_result: ValuationResult,
) -> Dict[str, float]:
    price_vol = historical_volatility(price_history.sort_values("trade_date")["close"])
    macro_series = macro.sort_values("obs_date")["value"] if "value" in macro else pd.Series(dtype=float)
    macro_vol = historical_volatility(macro_series) if not macro_series.empty else 0.1
    base_growth = base_result.assumptions.get("revenue_growth") or base_result.assumptions.get("roic", 0.05)
    base_margin = base_result.assumptions.get("operating_margin", 0.15)
    base_wacc = base_result.assumptions.get("wacc", 0.1)
    return {
        "growth_mu": float(base_growth),
        "growth_sigma": max(price_vol, 0.03),
        "margin_mu": float(base_margin),
        "margin_sigma": max(macro_vol, 0.02),
        "wacc_mu": float(base_wacc),
        "wacc_sigma": max(price_vol / 3, 0.01),
        "price_vol": float(price_vol),
        "macro_vol": float(macro_vol),
    }


def run_monte_carlo(
    base_result: ValuationResult,
    params: Dict[str, float],
    draws: int = 5000,
    seed: int = 42,
) -> ScenarioSummary:
    rng = np.random.default_rng(seed)
    growth_shock = rng.normal(params["growth_mu"], params["growth_sigma"], draws)
    margin_shock = rng.normal(params["margin_mu"], params["margin_sigma"], draws)
    wacc_shock = rng.normal(params["wacc_mu"], params["wacc_sigma"], draws)
    samples = []
    base_cf = base_result.cash_flows[-1] if base_result.cash_flows else base_result.intrinsic_value
    base_margin = max(base_result.assumptions.get("operating_margin", params["margin_mu"]), 0.01)
    for g, m, w in zip(growth_shock, margin_shock, wacc_shock):
        cf = base_cf * (1 + g) * (m / base_margin)
        spread = max(w - g, 0.005)
        tv = cf * (1 + g) / spread
        samples.append(tv)
    percentile_5 = float(np.percentile(samples, 5))
    percentile_50 = float(np.percentile(samples, 50))
    percentile_95 = float(np.percentile(samples, 95))
    return ScenarioSummary(
        seed=seed,
        draws=draws,
        percentile_5=percentile_5,
        percentile_50=percentile_50,
        percentile_95=percentile_95,
        samples=[float(x) for x in samples[:500]],
        inputs=params,
    )


__all__ = ["ScenarioSummary", "build_distribution_params", "run_monte_carlo"]
