#!/usr/bin/env python3
"""Audit an unpacked public-data package and reproduce aggregate figures."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = CODE_DIR.parent
REQUIRED_INPUTS = [
    Path("aggregated_dam_statistics/dam_counts_by_continent.csv"),
    Path("aggregated_dam_statistics/dam_counts_by_type.csv"),
    Path("aggregated_dam_statistics/dam_size_summary_2020.csv"),
    Path("figure_source_data/model/global_total_effects.csv"),
    Path("figure_source_data/model/global_indirect_shares.csv"),
    Path("ffr_aggregate_tables/grouped_all_years.csv"),
]


def configure_console() -> None:
    """Use UTF-8 for paths and messages on Windows terminals."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root of the unpacked Figshare public-data package.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "figures",
        help="Directory for regenerated aggregate figures.",
    )
    return parser.parse_args()


def main() -> int:
    configure_console()
    args = parse_args()
    data_root = args.data_root.resolve()
    missing = [path for path in REQUIRED_INPUTS if not (data_root / path).is_file()]
    if missing:
        print("The public-data package is incomplete or --data-root is incorrect.", file=sys.stderr)
        for path in missing:
            print(f"- missing: {path.as_posix()}", file=sys.stderr)
        return 2

    steps = [
        ["00_audit_public_release.py", "--root", str(data_root)],
        [
            "05_plot_dam_aggregates.py",
            "--data-dir",
            str(data_root / "aggregated_dam_statistics"),
            "--output",
            str(args.output_dir / "dam_aggregate_summary.png"),
        ],
        [
            "06_plot_model_summaries.py",
            "--source-dir",
            str(data_root / "figure_source_data" / "model"),
            "--output",
            str(args.output_dir / "model_effect_summary.png"),
        ],
        [
            "07_plot_ffr_aggregates.py",
            "--data",
            str(data_root / "ffr_aggregate_tables" / "grouped_all_years.csv"),
            "--output",
            str(args.output_dir / "ffr_3x3_summary.png"),
        ],
    ]

    for step in steps:
        command = [sys.executable, str(CODE_DIR / step[0]), *step[1:]]
        print(" ".join(command))
        subprocess.run(command, check=True, cwd=REPOSITORY_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
