"""Scenario generation and Monte Carlo simulation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable

import numpy as np
import pandas as pd

from loguru import logger

from .valuation_core import ValuationResult
from .data_pipeline import ValidatedInputs
from .risk_params import RiskProfile


@dataclass
class ScenarioSummary:
    seed: int
    draws: int
    percentile_5: float
    percentile_50: float
    percentile_95: float
    samples: List[float]
    inputs: Dict[str, float]
    sensitivity_analysis: Optional[Dict[str, List[float]]] = None


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
    validation_func: Optional[Callable[[Dict[str, float]], bool]] = None,
) -> ScenarioSummary:
    rng = np.random.default_rng(seed)
    growth_shock = rng.normal(params["growth_mu"], params["growth_sigma"], draws)
    margin_shock = rng.normal(params["margin_mu"], params["margin_sigma"], draws)
    wacc_shock = rng.normal(params["wacc_mu"], params["wacc_sigma"], draws)
    samples = []
    base_cf = base_result.cash_flows[-1] if base_result.cash_flows else base_result.intrinsic_value
    base_margin = max(base_result.assumptions.get("operating_margin", params["margin_mu"]), 0.01)

    valid_samples = 0
    for i in range(draws):
        g = growth_shock[i]
        m = margin_shock[i]
        w = wacc_shock[i]

        if validation_func and not validation_func({"growth": g, "margin": m, "wacc": w}):
            continue

        g = np.clip(g, -0.5, 0.5)
        m = np.clip(m, 0.01, 0.4)
        w = np.clip(w, 0.01, 0.3)

        cf = base_cf * (1 + g) * (m / base_margin)
        spread = max(w - g, 0.005)
        tv = cf * (1 + g) / spread
        samples.append(tv)
        valid_samples += 1

    if valid_samples < draws * 0.5:
        logger.warning(f"Only {valid_samples} valid samples out of {draws}; consider adjusting distribution parameters")

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


def sensitivity_analysis(
    base_result: ValuationResult,
    validated: ValidatedInputs,
    risk: RiskProfile,
    valuation_func: Callable[[ValidatedInputs, RiskProfile, Dict[str, float]], ValuationResult],
    parameters: List[str],
    ranges: List[List[float]],
    num_points: int = 5,
) -> Dict[str, List[float]]:
    """
    Perform sensitivity analysis on valuation inputs.

    Args:
        base_result: Base valuation result
        validated: Validated inputs
        risk: Risk profile
        valuation_func: Valuation function to use
        parameters: List of parameters to analyze
        ranges: List of parameter ranges
        num_points: Number of points to evaluate per parameter

    Returns:
        Dict[str, List[float]]: Sensitivity results
    """
    sensitivity_results = {}
    base_value = base_result.intrinsic_value

    for param, param_range in zip(parameters, ranges):
        values = np.linspace(param_range[0], param_range[1], num_points)
        results = []
        for val in values:
            try:
                modified_assumptions = base_result.assumptions.copy()
                modified_assumptions[param] = float(val)
                result = valuation_func(validated, risk, **modified_assumptions)
                results.append((float(val), float(result.intrinsic_value)))
            except Exception as e:
                logger.warning(f"Error evaluating {param} = {val}: {e}")
                results.append((float(val), np.nan))

        sensitivity_results[param] = results

    return sensitivity_results


def tornado_analysis(
    base_result: ValuationResult,
    validated: ValidatedInputs,
    risk: RiskProfile,
    valuation_func: Callable,
    parameters: List[str],
    percent_change: float = 0.1,
) -> Dict[str, float]:
    """
    Perform tornado analysis by varying each parameter by ±% from base value.

    Args:
        base_result: Base valuation result
        validated: Validated inputs
        risk: Risk profile
        valuation_func: Valuation function
        parameters: List of parameters to analyze
        percent_change: Percentage change from base value

    Returns:
        Dict[str, float]: Impact of each parameter on valuation
    """
    tornado_results = {}
    base_value = base_result.intrinsic_value

    for param in parameters:
        base_param_value = base_result.assumptions.get(param, 0.05)
        lower_value = base_param_value * (1 - percent_change)
        upper_value = base_param_value * (1 + percent_change)

        try:
            modified_lower = base_result.assumptions.copy()
            modified_lower[param] = lower_value
            lower_result = valuation_func(validated, risk, **modified_lower)

            modified_upper = base_result.assumptions.copy()
            modified_upper[param] = upper_value
            upper_result = valuation_func(validated, risk, **modified_upper)

            min_val = min(lower_result.intrinsic_value, upper_result.intrinsic_value)
            max_val = max(lower_result.intrinsic_value, upper_result.intrinsic_value)
            impact = max_val - min_val
            tornado_results[param] = float(impact)
        except Exception as e:
            logger.warning(f"Error in tornado analysis for {param}: {e}")
            tornado_results[param] = float(np.nan)

    return tornado_results


__all__ = ["ScenarioSummary", "build_distribution_params", "run_monte_carlo", "sensitivity_analysis", "tornado_analysis"]
