import unittest
from datetime import date
from pathlib import Path
import math
from unittest.mock import patch

import pandas as pd

from src.data_pipeline import TuShareClient, load_inputs
from src.risk_params import build_risk_profile
from src.valuation_core import revenue_driven_dcf, roic_driven_dcf, two_stage_dcf


class ValuationCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TuShareClient(token=None, fixtures_dir=Path("tests/data"))
        self.ticker = "600000.SH"
        self.as_of = date(2024, 12, 31)

    def test_revenue_dcf_intrinsic_positive(self) -> None:
        with patch("src.data_pipeline.cache_dataframe", return_value=None):
            validated = load_inputs(self.client, self.ticker, self.as_of)
        prices = self.client.call_api("daily", ts_code=self.ticker)
        prices["trade_date"] = pd.to_datetime(prices["trade_date"])
        csi300 = self.client.call_api("csi300")
        csi300["trade_date"] = pd.to_datetime(csi300["trade_date"])
        bonds = self.client.call_api("bonds")
        bonds["obs_date"] = pd.to_datetime(bonds["obs_date"])
        risk = build_risk_profile(self.ticker, self.as_of, prices, csi300, bonds, validated.statements)
        valuation = revenue_driven_dcf(validated, risk)
        self.assertGreater(valuation.intrinsic_value, 0)
        self.assertEqual(valuation.method, "revenue")
        self.assertIn("operating_margin", valuation.assumptions)
        self.assertIn("fade_years", valuation.assumptions)
        self.assertEqual(len(valuation.discount_factors), 5)
        self.assertIn(valuation.metadata.get("source_mode"), {"fixture", "live", "fallback"})

    def test_roic_dcf_consistency(self) -> None:
        with patch("src.data_pipeline.cache_dataframe", return_value=None):
            validated = load_inputs(self.client, self.ticker, self.as_of)
        prices = self.client.call_api("daily", ts_code=self.ticker)
        prices["trade_date"] = pd.to_datetime(prices["trade_date"])
        csi300 = self.client.call_api("csi300")
        csi300["trade_date"] = pd.to_datetime(csi300["trade_date"])
        bonds = self.client.call_api("bonds")
        bonds["obs_date"] = pd.to_datetime(bonds["obs_date"])
        risk = build_risk_profile(self.ticker, self.as_of, prices, csi300, bonds, validated.statements)
        valuation = roic_driven_dcf(validated, risk)
        self.assertAlmostEqual(len(valuation.cash_flows), 5)
        self.assertTrue(math.isfinite(valuation.terminal_value))
        self.assertIn("fade_years", valuation.assumptions)
        self.assertGreaterEqual(valuation.assumptions["fade_years"], 3)

    def test_two_stage_dcf_positive(self) -> None:
        with patch("src.data_pipeline.cache_dataframe", return_value=None):
            validated = load_inputs(self.client, self.ticker, self.as_of)
        prices = self.client.call_api("daily", ts_code=self.ticker)
        prices["trade_date"] = pd.to_datetime(prices["trade_date"])
        csi300 = self.client.call_api("csi300")
        csi300["trade_date"] = pd.to_datetime(csi300["trade_date"])
        bonds = self.client.call_api("bonds")
        bonds["obs_date"] = pd.to_datetime(bonds["obs_date"])
        risk = build_risk_profile(self.ticker, self.as_of, prices, csi300, bonds, validated.statements)
        valuation = two_stage_dcf(validated, risk)
        self.assertGreater(valuation.intrinsic_value, 0)
        self.assertEqual(valuation.method, "two_stage")
        self.assertTrue(math.isfinite(valuation.terminal_value))

    def test_exit_multiple_terminal_value(self) -> None:
        with patch("src.data_pipeline.cache_dataframe", return_value=None):
            validated = load_inputs(self.client, self.ticker, self.as_of)
        prices = self.client.call_api("daily", ts_code=self.ticker)
        prices["trade_date"] = pd.to_datetime(prices["trade_date"])
        csi300 = self.client.call_api("csi300")
        csi300["trade_date"] = pd.to_datetime(csi300["trade_date"])
        bonds = self.client.call_api("bonds")
        bonds["obs_date"] = pd.to_datetime(bonds["obs_date"])
        risk = build_risk_profile(self.ticker, self.as_of, prices, csi300, bonds, validated.statements)
        valuation = revenue_driven_dcf(
            validated,
            risk,
            terminal_method="exit_multiple",
            exit_multiple=10,
        )
        self.assertEqual(valuation.terminal_method, "exit_multiple")
        self.assertEqual(valuation.terminal_exit_multiple, 10)
        self.assertGreater(valuation.intrinsic_value, 0)

    def test_growth_method_variants(self) -> None:
        with patch("src.data_pipeline.cache_dataframe", return_value=None):
            validated = load_inputs(self.client, self.ticker, self.as_of)
        prices = self.client.call_api("daily", ts_code=self.ticker)
        prices["trade_date"] = pd.to_datetime(prices["trade_date"])
        csi300 = self.client.call_api("csi300")
        csi300["trade_date"] = pd.to_datetime(csi300["trade_date"])
        bonds = self.client.call_api("bonds")
        bonds["obs_date"] = pd.to_datetime(bonds["obs_date"])
        risk = build_risk_profile(self.ticker, self.as_of, prices, csi300, bonds, validated.statements)
        valuation_cagr = revenue_driven_dcf(validated, risk, growth_method="cagr")
        valuation_linear = revenue_driven_dcf(validated, risk, growth_method="linear")
        self.assertGreater(valuation_cagr.intrinsic_value, 0)
        self.assertGreater(valuation_linear.intrinsic_value, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
