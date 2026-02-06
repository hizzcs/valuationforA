import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from src.data_pipeline import TuShareClient, load_inputs


class DataVerificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TuShareClient(token=None, fixtures_dir=Path("tests/data"))

    def test_load_inputs_outputs_grade(self) -> None:
        with patch("src.data_pipeline.cache_dataframe", return_value=None):
            inputs = load_inputs(self.client, "600000.SH", date(2024, 12, 31))
        self.assertEqual(inputs.ticker, "600000.SH")
        self.assertIn(inputs.data_quality_grade, {"A", "B", "C"})
        self.assertTrue(inputs.revenue > 0)
        self.assertIn("cash_conversion", inputs.verification)
        self.assertIn("growth_alignment", inputs.verification)
        self.assertIn(inputs.metadata.get("source_mode"), {"fixture", "live", "fallback"})
        self.assertIn("verification_summary", inputs.metadata)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
