from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1] / "open_analysis_code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.privacy import (  # noqa: E402
    drop_margin_rows,
    find_exact_identifiers,
    suppress_rows_with_small_counts,
    validate_dof_bounds,
)


class PrivacyTests(unittest.TestCase):
    def test_exact_identifier_detection(self) -> None:
        self.assertEqual(find_exact_identifiers(["region", "HYBAS_ID"]), ["HYBAS_ID"])

    def test_complete_row_suppression(self) -> None:
        frame = pd.DataFrame(
            {
                "region": ["A", "B", "Total"],
                "Increase": [3, 20, 23],
                "Decrease": [10, 30, 40],
                "Total": [13, 50, 63],
                "Increase rate (%)": [23.1, 40.0, 36.5],
            }
        )
        protected, mask = suppress_rows_with_small_counts(frame)
        self.assertTrue(mask.iloc[0])
        self.assertTrue(np.isnan(protected.loc[0, "Increase"]))
        self.assertTrue(np.isnan(protected.loc[0, "Decrease"]))
        self.assertTrue(np.isnan(protected.loc[0, "Total"]))
        self.assertTrue(np.isnan(protected.loc[0, "Increase rate (%)"]))
        without_margins = drop_margin_rows(protected)
        self.assertNotIn("Total", without_margins["region"].tolist())

    def test_dof_bounds(self) -> None:
        valid = pd.DataFrame(
            {"dof_mean": [20.0], "dof_ci_lo": [10.0], "dof_ci_hi": [30.0]}
        )
        validate_dof_bounds(valid)
        invalid = pd.DataFrame(
            {"dof_mean": [20.0], "dof_ci_lo": [25.0], "dof_ci_hi": [30.0]}
        )
        with self.assertRaises(ValueError):
            validate_dof_bounds(invalid)


if __name__ == "__main__":
    unittest.main()
