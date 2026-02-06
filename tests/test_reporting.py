import tempfile
import unittest
from datetime import date
from pathlib import Path

import duckdb

from src.reporting import log_risk_profile, save_valuation
from src.valuation_core import ValuationResult
from src.scenario_engine import ScenarioSummary
from src.risk_params import RiskProfile


class ReportingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "valuation.duckdb"

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_save_and_log(self) -> None:
        valuation = ValuationResult(
            ticker="600000.SH",
            as_of_date=date(2024, 12, 31),
            method="revenue",
            intrinsic_value=120.0,
            cash_flows=[10.0] * 5,
            discount_factors=[0.9] * 5,
            terminal_value=80.0,
            assumptions={"wacc": 0.08, "operating_margin": 0.2},
            metadata={"data_quality": "A", "source_mode": "fixture"},
        )
        scenario = ScenarioSummary(
            seed=123,
            draws=500,
            percentile_5=90.0,
            percentile_50=120.0,
            percentile_95=150.0,
            samples=[95.0, 110.0],
            inputs={"growth_mu": 0.05, "wacc_mu": 0.08},
        )
        risk = RiskProfile(
            ticker="600000.SH",
            as_of_date=date(2024, 12, 31),
            beta=1.1,
            risk_free=0.03,
            market_risk_premium=0.05,
            cost_of_equity=0.085,
            cost_of_debt=0.04,
            wacc=0.07,
            window_start=date(2023, 1, 1),
            window_end=date(2024, 12, 31),
            observations=240,
            std_err=0.12,
            trace={"beta": 1.1},
        )

        save_valuation(valuation, scenario, risk, path=self.db_path)
        log_risk_profile(risk, path=self.db_path)

        con = duckdb.connect(self.db_path.as_posix())
        try:
            valuation_row = con.execute(
                "SELECT percentile_50, data_quality_grade, source_mode FROM valuation_runs LIMIT 1"
            ).fetchone()
            self.assertAlmostEqual(valuation_row[0], scenario.percentile_50)
            self.assertEqual(valuation_row[1], "A")
            self.assertEqual(valuation_row[2], "fixture")

            risk_row = con.execute(
                "SELECT beta, observations, std_err FROM risk_profiles WHERE ticker='600000.SH' LIMIT 1"
            ).fetchone()
            self.assertAlmostEqual(risk_row[0], risk.beta)
            self.assertEqual(risk_row[1], risk.observations)
            self.assertAlmostEqual(risk_row[2], risk.std_err)
        finally:
            con.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
