from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "open_analysis_code" / "03_prepare_anonymized_segment_dof.py"
EXPECTED_COLUMNS = [
    "segment_public_id",
    "year",
    "dof_mean",
    "dof_ci_lo",
    "dof_ci_hi",
    "river_class",
    "river_order_group",
    "network_hierarchy_group",
    "length_group",
    "continent",
]


class AnonymizationTests(unittest.TestCase):
    def test_random_ids_and_public_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_csv = root / "controlled_input.csv"
            output_csv = root / "public_output.csv"
            key_map = root / "controlled_key_map.csv"

            rows = []
            for segment in range(20):
                for year in (2010, 2015, 2020):
                    rows.append(
                        {
                            "source_key": f"controlled-{segment}",
                            "year": year,
                            "dof_mean": 20 + segment,
                            "dof_ci_lo": 10 + segment,
                            "dof_ci_hi": 30 + segment,
                            "river_class": "broad",
                            "river_order_group": "middle",
                            "network_hierarchy_group": "tributary",
                            "length_group": "medium",
                            "continent": "Example",
                            "GOID": 1000 + segment,
                            "geometry": "not released",
                        }
                    )
            pd.DataFrame(rows).to_csv(input_csv, index=False)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(input_csv),
                    str(output_csv),
                    "--stable-key-column",
                    "source_key",
                    "--controlled-key-map",
                    str(key_map),
                    "--minimum-k",
                    "20",
                    "--acknowledge-linkage-risk",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            public = pd.read_csv(output_csv)
            self.assertEqual(list(public.columns), EXPECTED_COLUMNS)
            self.assertEqual(public["segment_public_id"].nunique(), 20)
            self.assertFalse(public["segment_public_id"].str.startswith("controlled-").any())
            self.assertNotIn("GOID", public.columns)
            self.assertNotIn("geometry", public.columns)


if __name__ == "__main__":
    unittest.main()
