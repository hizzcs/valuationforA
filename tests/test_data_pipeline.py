import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import duckdb
import pandas as pd

from src.data_pipeline import TuShareClient, cache_dataframe, ensure_duckdb_schema, load_financials, load_inputs


class DataPipelineTest(unittest.TestCase):
    def test_ensure_duckdb_schema_recovers_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "valuation.duckdb"
            db_path.touch()
            ensure_duckdb_schema(db_path)
            con = duckdb.connect(db_path.as_posix())
            try:
                table = con.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_name='valuation_runs'"
                ).fetchone()
                self.assertEqual(table[0], "valuation_runs")
            finally:
                con.close()

    def test_cache_dataframe_maps_ts_code_into_ticker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "valuation.duckdb"
            ensure_duckdb_schema(db_path)
            financials = pd.DataFrame(
                [
                    {
                        "ts_code": "600000.SH",
                        "end_date": "2024-12-31",
                        "revenue": 121000000,
                        "net_profit": 15000000,
                        "net_debt": 22000000,
                        "operating_cf": 17000000,
                        "invested_capital": 85000000,
                        "interest_expense": 2200000,
                        "total_debt": 54000000,
                    }
                ]
            )
            cache_dataframe("raw_financials", financials, path=db_path)
            con = duckdb.connect(db_path.as_posix())
            try:
                row = con.execute(
                    "SELECT ticker, revenue, net_profit FROM raw_financials WHERE ticker='600000.SH' LIMIT 1"
                ).fetchone()
                self.assertEqual(row[0], "600000.SH")
                self.assertEqual(row[1], 121000000)
                self.assertEqual(row[2], 15000000)
            finally:
                con.close()

    def test_load_inputs_respects_as_of_cutoff(self) -> None:
        client = TuShareClient(token="", fixtures_dir=Path("tests/data"))
        with patch("src.data_pipeline.cache_dataframe", return_value=None):
            inputs = load_inputs(client, "600000.SH", date(2023, 12, 31))
        self.assertEqual(inputs.revenue, 110000000)
        self.assertEqual(inputs.metadata["statement_end_date"], "2023-12-31")
        self.assertEqual(len(inputs.statements), 2)

    def test_load_financials_falls_back_to_standard_tushare_endpoints(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls: list[str] = []
                self.mode = "live"

            def call_api(self, endpoint: str, **params):
                self.calls.append(endpoint)
                if endpoint == "financials":
                    return pd.DataFrame()
                if endpoint == "income":
                    return pd.DataFrame(
                        [
                            {
                                "ts_code": "600570.SH",
                                "end_date": "20241231",
                                "revenue": 1000.0,
                                "n_income_attr_p": 200.0,
                                "fin_exp": 10.0,
                            }
                        ]
                    )
                if endpoint == "cashflow":
                    return pd.DataFrame(
                        [
                            {
                                "ts_code": "600570.SH",
                                "end_date": "20241231",
                                "n_cashflow_act": 300.0,
                            }
                        ]
                    )
                if endpoint == "balancesheet":
                    return pd.DataFrame(
                        [
                            {
                                "ts_code": "600570.SH",
                                "end_date": "20241231",
                                "total_liab": 600.0,
                                "money_cap": 100.0,
                                "total_assets": 1500.0,
                                "total_share": 1000000.0,
                            }
                        ]
                    )
                raise AssertionError(f"unexpected endpoint: {endpoint}")

        fake = FakeClient()
        with patch("src.data_pipeline.cache_dataframe", return_value=None):
            financials = load_financials(fake, "600570.SH", as_of=date(2024, 12, 31))
        self.assertFalse(financials.empty)
        row = financials.iloc[-1]
        self.assertEqual(row["ts_code"], "600570.SH")
        self.assertAlmostEqual(float(row["revenue"]), 1000.0)
        self.assertAlmostEqual(float(row["net_profit"]), 200.0)
        self.assertAlmostEqual(float(row["operating_cf"]), 300.0)
        self.assertAlmostEqual(float(row["net_debt"]), 500.0)
        self.assertIn("income", fake.calls)
        self.assertIn("cashflow", fake.calls)
        self.assertIn("balancesheet", fake.calls)

    def test_load_financials_switches_exchange_suffix_when_needed(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.mode = "live"

            def call_api(self, endpoint: str, **params):
                code = params.get("ts_code")
                if endpoint not in {"income", "cashflow", "balancesheet"}:
                    return pd.DataFrame()
                if code == "000933.SH":
                    return pd.DataFrame()
                if code == "000933.SZ" and endpoint == "income":
                    return pd.DataFrame(
                        [{"ts_code": "000933.SZ", "end_date": "20241231", "revenue": 100.0, "n_income_attr_p": 10.0}]
                    )
                if code == "000933.SZ" and endpoint == "cashflow":
                    return pd.DataFrame(
                        [{"ts_code": "000933.SZ", "end_date": "20241231", "n_cashflow_act": 12.0}]
                    )
                if code == "000933.SZ" and endpoint == "balancesheet":
                    return pd.DataFrame(
                        [{"ts_code": "000933.SZ", "end_date": "20241231", "total_liab": 20.0, "money_cap": 3.0, "total_assets": 50.0}]
                    )
                return pd.DataFrame()

        with patch("src.data_pipeline.cache_dataframe", return_value=None):
            df = load_financials(FakeClient(), "000933.SH", as_of=date(2025, 1, 1))
        self.assertFalse(df.empty)
        self.assertEqual(df.iloc[-1]["ts_code"], "000933.SZ")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
