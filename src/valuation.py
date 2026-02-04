from dataclasses import dataclass
from typing import Dict

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


def simple_dcf(inputs: DcfInputs) -> float:
    """Very simplified DCF: project cashflow and discount."""
    cashflows = []
    cf = inputs.cashflow
    for g in inputs.growth_rates:
        cf = cf * (1 + g)
        cashflows.append(cf)
    discounted = [cf / ((1 + inputs.wacc) ** (i + 1)) for i, cf in enumerate(cashflows)]
    terminal_value = cashflows[-1] * (1 + inputs.terminal_growth) / (inputs.wacc - inputs.terminal_growth)
    terminal_discounted = terminal_value / ((1 + inputs.wacc) ** len(cashflows))
    return float(np.sum(discounted) + terminal_discounted)


def build_inputs(financials: Dict[str, pd.DataFrame], wacc: float, terminal_growth: float, growth_delta: float) -> DcfInputs:
    income = financials["income"].sort_values("end_date").tail(1)
    cashflow = financials["cashflow"].sort_values("end_date").tail(1)

    base_revenue = float(income["revenue"].iloc[0]) if not income.empty else 0.0
    net_profit = float(income["net_profit"].iloc[0]) if not income.empty else 0.0
    base_cf = float(cashflow["n_cashflow_act"].iloc[0]) if not cashflow.empty else 0.0

    growth_rates = tuple(max(-0.05, min(0.20, 0.05 + growth_delta)) for _ in range(5))

    return DcfInputs(
        base_revenue=base_revenue,
        net_profit=net_profit,
        cashflow=base_cf,
        wacc=wacc,
        terminal_growth=terminal_growth,
        growth_rates=growth_rates,
    )
