"""Risk parameter estimation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

from loguru import logger


TracePayload = Dict[str, Union[float, str]]


@dataclass
class RiskProfile:
    ticker: str
    as_of_date: date
    beta: float
    risk_free: float
    market_risk_premium: float
    cost_of_equity: float
    cost_of_debt: float
    wacc: float
    window_start: date
    window_end: date
    observations: int
    std_err: float
    trace: TracePayload


def estimate_beta(
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    use_levered: bool = True,
    daily_returns: bool = True,
    adjust_for_outliers: bool = True,
) -> Tuple[float, TracePayload]:
    """
    Estimate beta with improved methodology including outlier adjustment and returns calculation.

    Args:
        prices: Asset price history
        benchmark: Benchmark price history
        use_levered: Whether to adjust for leverage (not implemented yet)
        daily_returns: Use daily returns instead of monthly
        adjust_for_outliers: Adjust for extreme return values

    Returns:
        Tuple[float, TracePayload]: Beta value and statistics
    """
    df = (
        prices[["trade_date", "close"]]
        .rename(columns={"close": "asset_close"})
        .merge(
            benchmark[["trade_date", "close"]].rename(columns={"close": "bench_close"}),
            on="trade_date",
            how="inner",
        )
        .sort_values("trade_date")
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["asset_ret"] = df["asset_close"].pct_change()
    df["bench_ret"] = df["bench_close"].pct_change()
    df = df.dropna()

    if adjust_for_outliers:
        asset_q1, asset_q3 = np.percentile(df["asset_ret"], [25, 75])
        asset_iqr = asset_q3 - asset_q1
        asset_lower = asset_q1 - 1.5 * asset_iqr
        asset_upper = asset_q3 + 1.5 * asset_iqr
        bench_q1, bench_q3 = np.percentile(df["bench_ret"], [25, 75])
        bench_iqr = bench_q3 - bench_q1
        bench_lower = bench_q1 - 1.5 * bench_iqr
        bench_upper = bench_q3 + 1.5 * bench_iqr
        df = df[
            (df["asset_ret"] >= asset_lower)
            & (df["asset_ret"] <= asset_upper)
            & (df["bench_ret"] >= bench_lower)
            & (df["bench_ret"] <= bench_upper)
        ]

    if len(df) < 10:
        logger.warning("Insufficient data for beta; defaulting to 1.0")
        window_start = df["trade_date"].min()
        window_end = df["trade_date"].max()
        stats: TracePayload = {
            "r2": 0.0,
            "std_err": 0.0,
            "observations": float(len(df)),
            "window_start": window_start.date().isoformat() if isinstance(window_start, pd.Timestamp) else "",
            "window_end": window_end.date().isoformat() if isinstance(window_end, pd.Timestamp) else "",
            "outlier_adjusted": adjust_for_outliers,
        }
        return 1.0, stats

    x = df["bench_ret"].to_numpy()
    y = df["asset_ret"].to_numpy()

    try:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        beta = model.coef_[0]
        residuals = y - model.predict(x.reshape(-1, 1))
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = model.score(x.reshape(-1, 1), y)
        std_err = np.sqrt(ss_res / (len(x) - 2))
    except ImportError:
        logger.warning("sklearn not available, using numpy covariance method")
        cov = np.cov(x, y)
        beta = cov[0, 1] / cov[0, 0]
        residuals = y - beta * x
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
        std_err = np.sqrt(ss_res / (len(x) - 2))

    stats: TracePayload = {
        "r2": float(r2),
        "std_err": float(std_err),
        "observations": float(len(x)),
        "window_start": df["trade_date"].iloc[0].date().isoformat(),
        "window_end": df["trade_date"].iloc[-1].date().isoformat(),
        "outlier_adjusted": adjust_for_outliers,
    }

    beta = float(np.clip(beta, 0.1, 3.0))

    return beta, stats


def derive_risk_free(bond_curve: pd.DataFrame, as_of: date, tenor: int = 10) -> Tuple[float, str]:
    if bond_curve.empty:
        return 0.03, ""
    curve = bond_curve.copy()
    curve["obs_date"] = pd.to_datetime(curve["obs_date"])  # ensure datetime
    curve = curve[curve["obs_date"] <= pd.Timestamp(as_of)]
    if curve.empty:
        return 0.03, ""
    curve = curve.sort_values("obs_date")
    latest = curve.iloc[-1]
    obs_date = latest.get("obs_date")
    return float(latest.get("value", 0.03)), str(getattr(obs_date, "date", lambda: obs_date)())


def derive_cost_of_debt(financials: pd.DataFrame, credit_spread: float = 0.02) -> float:
    if "interest_expense" in financials.columns and "total_debt" in financials.columns:
        interest = pd.to_numeric(financials["interest_expense"], errors="coerce").tail(1).iloc[0]
        debt = pd.to_numeric(financials["total_debt"], errors="coerce").tail(1).iloc[0]
        if debt and debt > 0:
            return float(interest / debt)
    logger.warning("Using default cost of debt with credit spread")
    return 0.03 + credit_spread


def build_risk_profile(
    ticker: str,
    as_of_date: date,
    price_history: pd.DataFrame,
    benchmark_history: pd.DataFrame,
    bond_curve: pd.DataFrame,
    financials: pd.DataFrame,
    tax_rate: float = 0.25,
    market_risk_premium: float = 0.05,
    target_de_ratio: float = 1.0,
    use_levered_beta: bool = False,
    industry_beta: Optional[float] = None,
) -> RiskProfile:
    prices = price_history.copy()
    benchmark = benchmark_history.copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    benchmark["trade_date"] = pd.to_datetime(benchmark["trade_date"])
    cutoff = pd.Timestamp(as_of_date)
    prices = prices[prices["trade_date"] <= cutoff]
    benchmark = benchmark[benchmark["trade_date"] <= cutoff]

    if industry_beta is not None:
        beta = industry_beta
        stats: TracePayload = {
            "r2": 0.0,
            "std_err": 0.0,
            "observations": 0.0,
            "window_start": "",
            "window_end": "",
            "outlier_adjusted": False,
        }
        logger.info(f"Using industry beta: {beta} for ticker {ticker}")
    else:
        beta, stats = estimate_beta(
            prices,
            benchmark,
            use_levered=use_levered_beta,
            adjust_for_outliers=True,
        )

    risk_free, rf_obs = derive_risk_free(bond_curve, as_of_date)
    cost_of_equity = risk_free + beta * market_risk_premium
    pretax_cost_of_debt = derive_cost_of_debt(financials)
    cost_of_debt = pretax_cost_of_debt * (1 - tax_rate)

    if use_levered_beta and "total_debt" in financials.columns and "total_equity" in financials.columns:
        debt = pd.to_numeric(financials["total_debt"], errors="coerce").tail(1).iloc[0]
        equity = pd.to_numeric(financials["total_equity"], errors="coerce").tail(1).iloc[0]
        if debt > 0 and equity > 0:
            current_de_ratio = debt / equity
            unlevered_beta = beta / (1 + (1 - tax_rate) * current_de_ratio)
            target_beta = unlevered_beta * (1 + (1 - tax_rate) * target_de_ratio)
            cost_of_equity = risk_free + target_beta * market_risk_premium
            stats["unlevered_beta"] = unlevered_beta
            stats["current_de_ratio"] = current_de_ratio

    weight_equity = 1 / (1 + target_de_ratio)
    weight_debt = 1 - weight_equity
    wacc = weight_equity * cost_of_equity + weight_debt * cost_of_debt

    window_start_str = stats.get("window_start") or as_of_date.isoformat()
    window_end_str = stats.get("window_end") or as_of_date.isoformat()
    try:
        window_start = date.fromisoformat(window_start_str)
        window_end = date.fromisoformat(window_end_str)
    except ValueError:
        window_start = window_end = as_of_date

    observations = int(stats.get("observations", 0))
    std_err = float(stats.get("std_err", 0.0))

    trace: TracePayload = {
        "beta": beta,
        "risk_free": risk_free,
        "market_risk_premium": market_risk_premium,
        "cost_of_equity": cost_of_equity,
        "cost_of_debt": cost_of_debt,
        "weight_equity": weight_equity,
        "weight_debt": weight_debt,
        "r2": stats.get("r2", 0.0),
        "std_err": std_err,
        "observations": observations,
        "beta_window_start": window_start.isoformat(),
        "beta_window_end": window_end.isoformat(),
        "risk_free_obs_date": rf_obs,
        "tax_rate": tax_rate,
        "target_de_ratio": target_de_ratio,
        "outlier_adjusted": stats.get("outlier_adjusted", False),
    }

    if "unlevered_beta" in stats:
        trace["unlevered_beta"] = stats["unlevered_beta"]
        trace["current_de_ratio"] = stats["current_de_ratio"]

    if industry_beta is not None:
        trace["industry_beta"] = industry_beta

    return RiskProfile(
        ticker=ticker,
        as_of_date=as_of_date,
        beta=beta,
        risk_free=risk_free,
        market_risk_premium=market_risk_premium,
        cost_of_equity=cost_of_equity,
        cost_of_debt=cost_of_debt,
        wacc=wacc,
        window_start=window_start,
        window_end=window_end,
        observations=observations,
        std_err=std_err,
        trace=trace,
    )


__all__ = ["RiskProfile", "estimate_beta", "derive_risk_free", "build_risk_profile"]
