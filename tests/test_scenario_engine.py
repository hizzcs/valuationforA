import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data_pipeline import TuShareClient, load_inputs
from src.risk_params import build_risk_profile
from src.valuation_core import revenue_driven_dcf
from src.scenario_engine import build_distribution_params, run_monte_carlo


class ScenarioEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TuShareClient(token=None, fixtures_dir=Path("tests/data"))
        self.ticker = "600000.SH"
        self.as_of = date(2024, 12, 31)

    def test_percentiles_are_ordered(self) -> None:
        with patch("src.data_pipeline.cache_dataframe", return_value=None):
            validated = load_inputs(self.client, self.ticker, self.as_of)
        prices = self.client.call_api("daily", ts_code=self.ticker)
        prices["trade_date"] = pd.to_datetime(prices["trade_date"])
        csi300 = self.client.call_api("csi300")
        csi300["trade_date"] = pd.to_datetime(csi300["trade_date"])
        bonds = self.client.call_api("bonds")
        bonds["obs_date"] = pd.to_datetime(bonds["obs_date"])
        macro = self.client.call_api("macro")
        macro["obs_date"] = pd.to_datetime(macro["obs_date"])
        risk = build_risk_profile(self.ticker, self.as_of, prices, csi300, bonds, validated.statements)
        base_result = revenue_driven_dcf(validated, risk)
        params = build_distribution_params(prices, macro, base_result)
        summary = run_monte_carlo(base_result, params, draws=2000, seed=7)
        self.assertLess(summary.percentile_5, summary.percentile_50)
        self.assertLess(summary.percentile_50, summary.percentile_95)
        self.assertEqual(summary.draws, 2000)
        self.assertIn("wacc_mu", summary.inputs)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
