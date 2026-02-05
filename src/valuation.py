from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class DcfInputs:
    base_revenue: float
    net_profit: float
    cashflow: float
    wacc: float
    terminal_growth: float
    growth_rates: tuple
    shares_outstanding: float = 1.0
    net_debt: float = 0.0


def calc_wacc(
    risk_free: float,
    beta: float,
    equity_risk_premium: float,
    cost_of_debt: float,
    tax_rate: float,
    debt_ratio: float,
) -> float:
    cost_of_equity = risk_free + beta * equity_risk_premium
    equity_ratio = 1 - debt_ratio
    wacc = equity_ratio * cost_of_equity + debt_ratio * cost_of_debt * (1 - tax_rate)
    return float(wacc)


def simple_dcf(inputs: DcfInputs) -> float:
    """Simplified DCF: project cashflow and discount (enterprise value)."""
    cashflows = []
    cf = inputs.cashflow
    for g in inputs.growth_rates:
        cf = cf * (1 + g)
        cashflows.append(cf)
    discounted = [cf / ((1 + inputs.wacc) ** (i + 1)) for i, cf in enumerate(cashflows)]
    terminal_value = cashflows[-1] * (1 + inputs.terminal_growth) / (inputs.wacc - inputs.terminal_growth)
    terminal_discounted = terminal_value / ((1 + inputs.wacc) ** len(cashflows))
    return float(np.sum(discounted) + terminal_discounted)


def equity_value_per_share(enterprise_value: float, net_debt: float, shares_outstanding: float) -> float:
    if shares_outstanding <= 0:
        return 0.0
    return float((enterprise_value - net_debt) / shares_outstanding)


def build_growth_path(base: float, delta: float, years: int = 5) -> Tuple[float, ...]:
    return tuple(max(-0.05, min(0.20, base + delta)) for _ in range(years))


def build_inputs(
    financials: Dict[str, pd.DataFrame],
    wacc: float,
    terminal_growth: float,
    growth_delta: float,
    shares_outstanding: float,
    net_debt: float,
) -> DcfInputs:
    income = financials["income"].sort_values("end_date").tail(1)
    cashflow = financials["cashflow"].sort_values("end_date").tail(1)

    base_revenue = float(income["revenue"].iloc[0]) if not income.empty else 0.0
    net_profit = float(income["net_profit"].iloc[0]) if not income.empty else 0.0
    base_cf = float(cashflow["n_cashflow_act"].iloc[0]) if not cashflow.empty else 0.0

    growth_rates = build_growth_path(0.05, growth_delta, years=5)

    return DcfInputs(
        base_revenue=base_revenue,
        net_profit=net_profit,
        cashflow=base_cf,
        wacc=wacc,
        terminal_growth=terminal_growth,
        growth_rates=growth_rates,
        shares_outstanding=shares_outstanding,
        net_debt=net_debt,
    )
