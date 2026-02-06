"""Industry-specific valuation tests."""
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data_pipeline import TuShareClient, load_inputs
from src.risk_params import build_risk_profile
from src.industry_valuation import (
    estimate_industry_type,
    estimate_industry_valuation,
    IndustryType,
    IndustryValuationResult
)


class IndustryValuationTest(unittest.TestCase):
    """Industry valuation tests."""

    def setUp(self) -> None:
        self.client = TuShareClient(token=None, fixtures_dir=Path("tests/data"))
        self.ticker = "600000.SH"  # 浦发银行，属于金融行业
        self.as_of = date(2024, 12, 31)

        # 模拟商品价格数据（用于周期性行业）
        dates = pd.date_range(self.as_of - pd.DateOffset(years=3), self.as_of)
        prices = [100 + 20 * i for i in range(len(dates))]
        self.commodity_prices = pd.DataFrame({
            "date": dates,
            "price": prices
        })

        # 模拟监管数据（用于金融行业）
        regulatory_dates = pd.date_range(self.as_of - pd.DateOffset(years=2), self.as_of)
        self.regulatory_data = pd.DataFrame({
            "date": regulatory_dates,
            "capital_requirement": [0.09 for _ in range(len(regulatory_dates))]
        })

        # 模拟宏观数据（用于消费行业）
        macro_dates = pd.date_range(self.as_of - pd.DateOffset(years=2), self.as_of)
        self.macro_data = pd.DataFrame({
            "date": macro_dates,
            "gdp_growth": [0.06 for _ in range(len(macro_dates))],
            "cpi": [0.025 for _ in range(len(macro_dates))]
        })

    def test_industry_estimation(self) -> None:
        """Test industry estimation based on metadata."""
        with patch("src.data_pipeline.cache_dataframe", return_value=None):
            validated = load_inputs(self.client, self.ticker, self.as_of)

        # 测试基础行业类型
        industry_type = estimate_industry_type(self.ticker, validated)
        self.assertIsNotNone(industry_type)
        self.assertIsInstance(industry_type, IndustryType)

    def test_industry_valuation_basic(self) -> None:
        """Test basic industry valuation functionality."""
        with patch("src.data_pipeline.cache_dataframe", return_value=None):
            validated = load_inputs(self.client, self.ticker, self.as_of)

        prices = self.client.call_api("daily", ts_code=self.ticker)
        prices["trade_date"] = pd.to_datetime(prices["trade_date"])
        csi300 = self.client.call_api("csi300")
        csi300["trade_date"] = pd.to_datetime(csi300["trade_date"])
        bonds = self.client.call_api("bonds")
        bonds["obs_date"] = pd.to_datetime(bonds["obs_date"])

        risk = build_risk_profile(self.ticker, self.as_of, prices, csi300, bonds, validated.statements)

        valuation_result = estimate_industry_valuation(
            self.ticker,
            self.as_of,
            validated,
            risk,
            self.commodity_prices,
            self.regulatory_data,
            self.macro_data
        )

        self.assertIsInstance(valuation_result, IndustryValuationResult)
        self.assertGreater(valuation_result.base_value, 0)
        self.assertIn("基础情景", valuation_result.scenarios)
        self.assertGreater(len(valuation_result.scenarios), 1)

    def test_financial_industry_valuation(self) -> None:
        """Test financial industry valuation."""
        with patch("src.industry_valuation.estimate_industry_type", return_value=IndustryType.FINANCIAL):
            with patch("src.data_pipeline.cache_dataframe", return_value=None):
                validated = load_inputs(self.client, self.ticker, self.as_of)

            prices = self.client.call_api("daily", ts_code=self.ticker)
            prices["trade_date"] = pd.to_datetime(prices["trade_date"])
            csi300 = self.client.call_api("csi300")
            csi300["trade_date"] = pd.to_datetime(csi300["trade_date"])
            bonds = self.client.call_api("bonds")
            bonds["obs_date"] = pd.to_datetime(bonds["obs_date"])

            risk = build_risk_profile(self.ticker, self.as_of, prices, csi300, bonds, validated.statements)

            valuation_result = estimate_industry_valuation(
                self.ticker,
                self.as_of,
                validated,
                risk,
                self.commodity_prices,
                self.regulatory_data,
                self.macro_data
            )

            self.assertEqual(valuation_result.industry_type, "financial")
            self.assertIn("放松监管", valuation_result.scenarios)
            self.assertIn("强化监管", valuation_result.scenarios)

    def test_cyclical_industry_valuation(self) -> None:
        """Test cyclical industry valuation with simulated data."""
        with patch("src.industry_valuation.estimate_industry_type", return_value=IndustryType.CYCLICAL):
            with patch("src.data_pipeline.cache_dataframe", return_value=None):
                validated = load_inputs(self.client, self.ticker, self.as_of)

            prices = self.client.call_api("daily", ts_code=self.ticker)
            prices["trade_date"] = pd.to_datetime(prices["trade_date"])
            csi300 = self.client.call_api("csi300")
            csi300["trade_date"] = pd.to_datetime(csi300["trade_date"])
            bonds = self.client.call_api("bonds")
            bonds["obs_date"] = pd.to_datetime(bonds["obs_date"])

            risk = build_risk_profile(self.ticker, self.as_of, prices, csi300, bonds, validated.statements)

            valuation_result = estimate_industry_valuation(
                self.ticker,
                self.as_of,
                validated,
                risk,
                self.commodity_prices,
                self.regulatory_data,
                self.macro_data
            )

            self.assertEqual(valuation_result.industry_type, "cyclical")
            self.assertIn("繁荣情景", valuation_result.scenarios)
            self.assertIn("衰退情景", valuation_result.scenarios)

    def test_tech_growth_industry_valuation(self) -> None:
        """Test tech growth industry valuation with simulated data."""
        with patch("src.industry_valuation.estimate_industry_type", return_value=IndustryType.TECH_GROWTH):
            with patch("src.data_pipeline.cache_dataframe", return_value=None):
                validated = load_inputs(self.client, self.ticker, self.as_of)

            prices = self.client.call_api("daily", ts_code=self.ticker)
            prices["trade_date"] = pd.to_datetime(prices["trade_date"])
            csi300 = self.client.call_api("csi300")
            csi300["trade_date"] = pd.to_datetime(csi300["trade_date"])
            bonds = self.client.call_api("bonds")
            bonds["obs_date"] = pd.to_datetime(bonds["obs_date"])

            risk = build_risk_profile(self.ticker, self.as_of, prices, csi300, bonds, validated.statements)

            valuation_result = estimate_industry_valuation(
                self.ticker,
                self.as_of,
                validated,
                risk,
                self.commodity_prices,
                self.regulatory_data,
                self.macro_data
            )

            self.assertEqual(valuation_result.industry_type, "tech_growth")
            self.assertIn("成功情景", valuation_result.scenarios)
            self.assertIn("失败情景", valuation_result.scenarios)

    def test_consumer_industry_valuation(self) -> None:
        """Test consumer industry valuation with simulated data."""
        with patch("src.industry_valuation.estimate_industry_type", return_value=IndustryType.CONSUMER):
            with patch("src.data_pipeline.cache_dataframe", return_value=None):
                validated = load_inputs(self.client, self.ticker, self.as_of)

            prices = self.client.call_api("daily", ts_code=self.ticker)
            prices["trade_date"] = pd.to_datetime(prices["trade_date"])
            csi300 = self.client.call_api("csi300")
            csi300["trade_date"] = pd.to_datetime(csi300["trade_date"])
            bonds = self.client.call_api("bonds")
            bonds["obs_date"] = pd.to_datetime(bonds["obs_date"])

            risk = build_risk_profile(self.ticker, self.as_of, prices, csi300, bonds, validated.statements)

            valuation_result = estimate_industry_valuation(
                self.ticker,
                self.as_of,
                validated,
                risk,
                self.commodity_prices,
                self.regulatory_data,
                self.macro_data
            )

            self.assertEqual(valuation_result.industry_type, "consumer")
            self.assertIn("经济复苏", valuation_result.scenarios)
            self.assertIn("经济放缓", valuation_result.scenarios)

    def test_sensitivity_analysis(self) -> None:
        """Test sensitivity analysis."""
        with patch("src.industry_valuation.estimate_industry_type", return_value=IndustryType.CONSUMER):
            with patch("src.data_pipeline.cache_dataframe", return_value=None):
                validated = load_inputs(self.client, self.ticker, self.as_of)

            prices = self.client.call_api("daily", ts_code=self.ticker)
            prices["trade_date"] = pd.to_datetime(prices["trade_date"])
            csi300 = self.client.call_api("csi300")
            csi300["trade_date"] = pd.to_datetime(csi300["trade_date"])
            bonds = self.client.call_api("bonds")
            bonds["obs_date"] = pd.to_datetime(bonds["obs_date"])

            risk = build_risk_profile(self.ticker, self.as_of, prices, csi300, bonds, validated.statements)

            valuation_result = estimate_industry_valuation(
                self.ticker,
                self.as_of,
                validated,
                risk,
                self.commodity_prices,
                self.regulatory_data,
                self.macro_data
            )

            self.assertIsNotNone(valuation_result.sensitivity)
            self.assertIsNotNone(valuation_result.tornado)

    def test_fallback_mechanism(self) -> None:
        """Test fallback mechanism when industry-specific valuation fails."""
        with patch("src.industry_valuation.cyclical_industry_valuation") as mock:
            mock.side_effect = Exception("Simulated valuation failure")

            with patch("src.industry_valuation.estimate_industry_type", return_value=IndustryType.CYCLICAL):
                with patch("src.data_pipeline.cache_dataframe", return_value=None):
                    validated = load_inputs(self.client, self.ticker, self.as_of)

                prices = self.client.call_api("daily", ts_code=self.ticker)
                prices["trade_date"] = pd.to_datetime(prices["trade_date"])
                csi300 = self.client.call_api("csi300")
                csi300["trade_date"] = pd.to_datetime(csi300["trade_date"])
                bonds = self.client.call_api("bonds")
                bonds["obs_date"] = pd.to_datetime(bonds["obs_date"])

                risk = build_risk_profile(self.ticker, self.as_of, prices, csi300, bonds, validated.statements)

                valuation_result = estimate_industry_valuation(
                    self.ticker,
                    self.as_of,
                    validated,
                    risk,
                    self.commodity_prices,
                    self.regulatory_data,
                    self.macro_data
                )

                self.assertIsInstance(valuation_result, IndustryValuationResult)
                self.assertGreater(valuation_result.base_value, 0)
                self.assertEqual(valuation_result.industry_type, "cyclical")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
