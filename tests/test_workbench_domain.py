import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data_pipeline import TuShareClient, load_inputs
from src.risk_params import build_risk_profile
from src.scenario_engine import ScenarioSummary
from src.workbench_domain import (
    UI_METHOD_MAPPING,
    build_dashboard_snapshot,
    build_method_results,
    compute_margin_of_safety,
    method_comparison_frame,
    to_method_snapshots,
)


class WorkbenchDomainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TuShareClient(token="", fixtures_dir=Path("tests/data"))
        self.ticker = "600000.SH"
        self.as_of = date(2024, 12, 31)

        with patch("src.data_pipeline.cache_dataframe", return_value=None):
            self.validated = load_inputs(self.client, self.ticker, self.as_of)

        prices = self.client.call_api("daily", ts_code=self.ticker)
        prices["trade_date"] = pd.to_datetime(prices["trade_date"])
        csi300 = self.client.call_api("csi300")
        csi300["trade_date"] = pd.to_datetime(csi300["trade_date"])
        bonds = self.client.call_api("bonds")
        bonds["obs_date"] = pd.to_datetime(bonds["obs_date"])
        self.risk = build_risk_profile(self.ticker, self.as_of, prices, csi300, bonds, self.validated.statements)

    def test_method_results_and_frame(self) -> None:
        results = build_method_results(self.validated, self.risk)
        self.assertEqual(set(results.keys()), {"revenue", "roic", "two_stage"})
        frame = method_comparison_frame(to_method_snapshots(results))
        self.assertEqual(len(frame), 3)
        self.assertTrue(frame["enterprise_value"].is_monotonic_decreasing)

    def test_margin_of_safety(self) -> None:
        self.assertAlmostEqual(compute_margin_of_safety(12.0, 10.0), 0.2)
        self.assertIsNone(compute_margin_of_safety(None, 10.0))
        self.assertIsNone(compute_margin_of_safety(12.0, 0))

    def test_dashboard_snapshot(self) -> None:
        results = build_method_results(self.validated, self.risk)
        summary = ScenarioSummary(
            seed=42,
            draws=1000,
            percentile_5=90.0,
            percentile_50=120.0,
            percentile_95=170.0,
            samples=[100.0, 110.0, 130.0],
            inputs={"growth_mu": 0.1},
        )
        snapshot = build_dashboard_snapshot(
            ticker=self.ticker,
            as_of_date=self.as_of,
            selected_method_label="收入驱动DCF",
            validated=self.validated,
            risk=self.risk,
            results=results,
            scenario_summary=summary,
            market_price=10.0,
        )
        self.assertEqual(snapshot.selected_method_key, UI_METHOD_MAPPING["收入驱动DCF"])
        self.assertIn("revenue", snapshot.methods)
        self.assertEqual(snapshot.scenario_band.p95, 170.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
