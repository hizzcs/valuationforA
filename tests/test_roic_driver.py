import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data_pipeline import TuShareClient, load_inputs
from src.risk_params import build_risk_profile
from src.valuation_core import roic_driven_dcf


class RoicDriverTest(unittest.TestCase):
    def test_roic_and_reinvestment_populate_assumptions(self) -> None:
        client = TuShareClient(token="", fixtures_dir=Path("tests/data"))
        ticker = "600000.SH"
        as_of = date(2024, 12, 31)
        with patch("src.data_pipeline.cache_dataframe", return_value=None):
            validated = load_inputs(client, ticker, as_of)
        prices = client.call_api("daily", ts_code=ticker)
        prices["trade_date"] = pd.to_datetime(prices["trade_date"])
        csi300 = client.call_api("csi300")
        csi300["trade_date"] = pd.to_datetime(csi300["trade_date"])
        bonds = client.call_api("bonds")
        bonds["obs_date"] = pd.to_datetime(bonds["obs_date"])
        risk = build_risk_profile(ticker, as_of, prices, csi300, bonds, validated.statements)
        valuation = roic_driven_dcf(validated, risk)
        self.assertIn("roic", valuation.assumptions)
        self.assertIn("reinvestment_rate", valuation.assumptions)
        self.assertGreaterEqual(valuation.assumptions["roic"], 0)
        self.assertGreaterEqual(valuation.assumptions["reinvestment_rate"], 0)
        self.assertIn("fade_years", valuation.assumptions)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
