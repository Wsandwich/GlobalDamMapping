"""Disclosure-control helpers used by public-data preparation scripts."""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd


PRIMARY_SUPPRESSION_MAX = 4
SUPPRESSION_VALUE = np.nan

EXACT_IDENTIFIER_COLUMNS = {
    "goid",
    "noid",
    "ndoid",
    "nuoid",
    "hybas_id",
    "bas_id",
    "reach_id",
    "geometry",
    "longitude",
    "latitude",
    "lon",
    "lat",
    "source_row",
    "source_row_id",
    "row_number",
    "upstream_id",
    "downstream_id",
    "nearest_river_goid",
    "distance_to_river",
}

SEGMENT_PUBLIC_COLUMNS = [
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

SEGMENT_QUASI_IDENTIFIERS = [
    "river_class",
    "river_order_group",
    "network_hierarchy_group",
    "length_group",
    "continent",
]


def normalize_column(name: str) -> str:
    """Return a lowercase snake-case representation of a column name."""
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(name)).strip("_").lower()
    return value


def find_exact_identifiers(columns: Iterable[str]) -> list[str]:
    """Return columns that expose an exact spatial or network identifier."""
    return [
        column
        for column in columns
        if normalize_column(column) in EXACT_IDENTIFIER_COLUMNS
    ]


def is_primary_small_count(value: object, threshold: int = PRIMARY_SUPPRESSION_MAX) -> bool:
    """Return True when a value is an integer count between one and threshold."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return numeric.is_integer() and 1 <= abs(int(numeric)) <= threshold


def infer_count_columns(frame: pd.DataFrame) -> list[str]:
    """Infer columns that contain counts rather than rates or continuous values."""
    patterns = ("count", "number", "segments", "increase", "decrease", "uncertain", "total")
    columns: list[str] = []
    for column in frame.columns:
        normalized = normalize_column(column)
        if normalized in {"2010", "2015", "2020"} or any(
            token in normalized for token in patterns
        ):
            if not any(token in normalized for token in ("rate", "pct", "percent", "mean")):
                columns.append(column)
    return columns


def infer_derived_columns(frame: pd.DataFrame) -> list[str]:
    """Infer rates, confidence intervals, and changes derived from counts."""
    patterns = ("rate", "pct", "percent", "ci", "change", "share")
    return [
        column
        for column in frame.columns
        if any(token in normalize_column(column) for token in patterns)
    ]


def suppress_rows_with_small_counts(
    frame: pd.DataFrame,
    count_columns: list[str] | None = None,
    derived_columns: list[str] | None = None,
    threshold: int = PRIMARY_SUPPRESSION_MAX,
) -> tuple[pd.DataFrame, pd.Series]:
    """Suppress all count and derived cells in rows containing a small count.

    Suppressing the complete row prevents direct recovery from a row total or
    percentage. Column margins must also be removed when any row is suppressed.
    """
    result = frame.copy()
    count_columns = count_columns or infer_count_columns(result)
    derived_columns = derived_columns or infer_derived_columns(result)
    missing = [column for column in count_columns if column not in result.columns]
    if missing:
        raise KeyError(f"Missing count columns: {missing}")

    if count_columns:
        primary_mask = result[count_columns].apply(
            lambda row: any(is_primary_small_count(value, threshold) for value in row),
            axis=1,
        )
    else:
        primary_mask = pd.Series(False, index=result.index)

    protected_columns = list(dict.fromkeys(count_columns + derived_columns))
    result.loc[primary_mask, protected_columns] = SUPPRESSION_VALUE
    return result, primary_mask


def drop_margin_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop aggregate margin rows that can undo complementary suppression."""
    labels = {"total", "sum", "all", "global total", "grand total"}
    object_columns = list(frame.select_dtypes(include=["object", "string"]).columns)
    if not object_columns:
        return frame.copy()
    margin_mask = pd.Series(False, index=frame.index)
    for column in object_columns:
        normalized = frame[column].astype(str).str.strip().str.lower()
        margin_mask |= normalized.isin(labels)
    return frame.loc[~margin_mask].copy()


def validate_dof_bounds(frame: pd.DataFrame) -> None:
    """Validate public DoF means and confidence intervals."""
    required = ["dof_mean", "dof_ci_lo", "dof_ci_hi"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing DoF columns: {missing}")
    values = frame[required].apply(pd.to_numeric, errors="raise")
    if ((values < 0) | (values > 100)).any().any():
        raise ValueError("DoF values must be within [0, 100]")
    if (values["dof_ci_lo"] > values["dof_mean"]).any():
        raise ValueError("DoF lower confidence bounds exceed means")
    if (values["dof_mean"] > values["dof_ci_hi"]).any():
        raise ValueError("DoF means exceed upper confidence bounds")
