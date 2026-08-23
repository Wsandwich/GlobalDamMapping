#!/usr/bin/env python3
"""Create a de-identified level-3 or major-basin aggregate release."""

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


DISALLOWED_PUBLIC_TOKENS = {
    "new",
    "removed",
    "matched",
    "delta",
    "longitude",
    "latitude",
    "centroid",
    "confidence",
    "precision",
    "certain_mask",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--aggregation-level", required=True, choices=["lev03", "mrb"])
    parser.add_argument("--label-column", required=True)
    parser.add_argument(
        "--keep-columns",
        nargs="+",
        required=True,
        help="Explicit aggregate fields approved for public release.",
    )
    parser.add_argument("--count-columns", nargs="+")
    parser.add_argument(
        "--rename",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Rename an approved output column; may be repeated.",
    )
    parser.add_argument("--threshold", type=int, default=4)
    return parser.parse_args()


def read_input(path: Path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path), None
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("Install requirements-geo.txt to process vector data") from exc
    frame = gpd.read_file(path)
    if frame.geometry.geom_type.isin(["Point", "MultiPoint"]).any():
        raise ValueError("Point geometry is prohibited in the public basin release")
    return frame, frame.geometry


def write_output(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        pd.DataFrame(frame.drop(columns="geometry", errors="ignore")).to_csv(path, index=False)
        return
    if path.suffix.lower() not in {".geojson", ".json"}:
        raise ValueError("Public vector outputs must use GeoJSON")
    frame.to_file(path, driver="GeoJSON")


def main() -> int:
    args = parse_args()
    frame, geometry = read_input(args.input_file)
    approved = [args.label_column, *args.keep_columns]
    missing = [column for column in approved if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing approved columns: {missing}")

    normalized_approved = {column.lower() for column in approved}
    risky = sorted(
        column
        for column in normalized_approved
        if any(token in column for token in DISALLOWED_PUBLIC_TOKENS)
    )
    if risky:
        raise ValueError(f"Disallowed fine-scale or differencing fields: {risky}")

    rename_map = {args.label_column: "public_basin_label"}
    for specification in args.rename:
        if "=" not in specification:
            raise ValueError(f"Invalid rename specification: {specification}")
        old, new = specification.split("=", 1)
        if old not in approved:
            raise ValueError(f"Cannot rename an unapproved column: {old}")
        rename_map[old] = new

    selected = frame[approved].copy()
    selected = selected.rename(columns=rename_map)
    if geometry is not None:
        selected = frame[approved + [frame.geometry.name]].copy()
        selected = selected.rename(columns=rename_map)

    identifier_candidates = [
        column for column in selected.columns if column != "geometry"
    ]
    identifiers = find_exact_identifiers(identifier_candidates)
    if identifiers:
        raise ValueError(f"Exact identifiers are not allowed: {identifiers}")

    counts = (
        [rename_map.get(column, column) for column in args.count_columns]
        if args.count_columns
        else infer_count_columns(selected)
    )
    derived = infer_derived_columns(selected)
    protected, suppressed = suppress_rows_with_small_counts(
        selected,
        count_columns=counts,
        derived_columns=derived,
        threshold=args.threshold,
    )
    if suppressed.any():
        protected = drop_margin_rows(protected)

    write_output(protected, args.output_file)
    metadata = {
        "aggregation_level": args.aggregation_level,
        "rows_written": int(len(protected)),
        "suppressed_rows": int(suppressed.sum()),
        "small_count_threshold": args.threshold,
        "released_columns": [column for column in protected.columns if column != "geometry"],
        "geometry_included": "geometry" in protected.columns,
    }
    metadata_path = args.output_file.with_suffix(args.output_file.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
