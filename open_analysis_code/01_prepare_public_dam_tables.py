#!/usr/bin/env python3
"""Apply disclosure control to a country, continent, or basin summary table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common.privacy import (
    drop_margin_rows,
    find_exact_identifiers,
    infer_count_columns,
    infer_derived_columns,
    suppress_rows_with_small_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--count-columns",
        nargs="+",
        help="Explicit count columns; otherwise infer them from column names.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=4,
        help="Suppress integer counts from one through this value.",
    )
    return parser.parse_args()


def prepare_table(
    input_csv: Path,
    output_csv: Path,
    count_columns: list[str] | None,
    threshold: int,
) -> dict[str, object]:
    frame = pd.read_csv(input_csv)
    identifiers = find_exact_identifiers(frame.columns)
    if identifiers:
        raise ValueError(f"Exact identifiers are not allowed: {identifiers}")

    counts = count_columns or infer_count_columns(frame)
    derived = infer_derived_columns(frame)
    protected, suppressed_mask = suppress_rows_with_small_counts(
        frame,
        count_columns=counts,
        derived_columns=derived,
        threshold=threshold,
    )
    if suppressed_mask.any():
        protected = drop_margin_rows(protected)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    protected.to_csv(output_csv, index=False)
    metadata = {
        "source_filename": input_csv.name,
        "output_filename": output_csv.name,
        "rows_read": int(len(frame)),
        "rows_written": int(len(protected)),
        "rows_suppressed": int(suppressed_mask.sum()),
        "small_count_threshold": threshold,
        "count_columns": counts,
        "derived_columns": derived,
        "margin_rows_removed": int(len(frame) - len(protected)),
    }
    metadata_path = output_csv.with_suffix(output_csv.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    args = parse_args()
    metadata = prepare_table(
        args.input_csv,
        args.output_csv,
        args.count_columns,
        args.threshold,
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
