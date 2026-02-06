import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import duckdb
import pandas as pd

from src.data_pipeline import TuShareClient, cache_dataframe, ensure_duckdb_schema, load_inputs


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
        client = TuShareClient(token=None, fixtures_dir=Path("tests/data"))
        with patch("src.data_pipeline.cache_dataframe", return_value=None):
            inputs = load_inputs(client, "600000.SH", date(2023, 12, 31))
        self.assertEqual(inputs.revenue, 110000000)
        self.assertEqual(inputs.metadata["statement_end_date"], "2023-12-31")
        self.assertEqual(len(inputs.statements), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
