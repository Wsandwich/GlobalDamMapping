#!/usr/bin/env python3
"""Create an optional anonymous segment-DoF table with random public IDs."""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

import pandas as pd

from common.privacy import (
    SEGMENT_PUBLIC_COLUMNS,
    SEGMENT_QUASI_IDENTIFIERS,
    validate_dof_bounds,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--stable-key-column",
        required=True,
        help="Controlled stable key used only to preserve IDs across years.",
    )
    parser.add_argument(
        "--controlled-key-map",
        type=Path,
        required=True,
        help="Sensitive key map. This path must be outside the public repository.",
    )
    parser.add_argument("--minimum-k", type=int, default=20)
    parser.add_argument(
        "--acknowledge-linkage-risk",
        action="store_true",
        help="Confirm that an external linkage-risk review has been completed.",
    )
    return parser.parse_args()


def require_outside_repository(path: Path) -> None:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return
    raise ValueError("The controlled key map must be stored outside the public repository")


def load_or_create_key_map(path: Path, source_keys: pd.Series) -> pd.DataFrame:
    normalized_keys = source_keys.astype(str)
    if path.exists():
        mapping = pd.read_csv(path, dtype=str)
        required = {"controlled_segment_key", "segment_public_id"}
        if not required.issubset(mapping.columns):
            raise KeyError(f"Controlled key map must contain {sorted(required)}")
    else:
        unique_keys = sorted(normalized_keys.unique())
        mapping = pd.DataFrame(
            {
                "controlled_segment_key": unique_keys,
                "segment_public_id": [secrets.token_urlsafe(18) for _ in unique_keys],
            }
        )

    if mapping["controlled_segment_key"].duplicated().any():
        raise ValueError("Controlled segment keys are duplicated in the key map")
    if mapping["segment_public_id"].duplicated().any():
        raise ValueError("Random public segment IDs are duplicated")

    missing = sorted(set(normalized_keys).difference(mapping["controlled_segment_key"]))
    if missing:
        additions = pd.DataFrame(
            {
                "controlled_segment_key": missing,
                "segment_public_id": [secrets.token_urlsafe(18) for _ in missing],
            }
        )
        mapping = pd.concat([mapping, additions], ignore_index=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(path, index=False)
    return mapping


def minimum_equivalence_class_size(frame: pd.DataFrame) -> int:
    unique_profiles = frame[["segment_public_id", *SEGMENT_QUASI_IDENTIFIERS]].drop_duplicates()
    sizes = unique_profiles.groupby(SEGMENT_QUASI_IDENTIFIERS, dropna=False).size()
    return int(sizes.min()) if len(sizes) else 0


def main() -> int:
    args = parse_args()
    if not args.acknowledge_linkage_risk:
        raise ValueError(
            "Anonymous segment release is optional and requires --acknowledge-linkage-risk"
        )
    require_outside_repository(args.controlled_key_map)

    source = pd.read_csv(args.input_csv)
    required_source = {
        args.stable_key_column,
        "year",
        "dof_mean",
        "dof_ci_lo",
        "dof_ci_hi",
        *SEGMENT_QUASI_IDENTIFIERS,
    }
    missing = sorted(required_source.difference(source.columns))
    if missing:
        raise KeyError(f"Missing segment fields: {missing}")

    mapping = load_or_create_key_map(args.controlled_key_map, source[args.stable_key_column])
    source_key = source[args.stable_key_column].astype(str).rename("controlled_segment_key")
    working = source.assign(controlled_segment_key=source_key)
    public = working.merge(mapping, on="controlled_segment_key", how="left", validate="many_to_one")
    public = public[SEGMENT_PUBLIC_COLUMNS].copy()
    validate_dof_bounds(public)

    minimum_k = minimum_equivalence_class_size(public)
    if minimum_k < args.minimum_k:
        raise ValueError(
            f"Minimum quasi-identifier equivalence class is {minimum_k}; "
            f"required minimum is {args.minimum_k}. Publish aggregate FFR tables instead."
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    public.to_csv(args.output_csv, index=False)
    metadata = {
        "rows": int(len(public)),
        "unique_segments": int(public["segment_public_id"].nunique()),
        "years": sorted(int(value) for value in public["year"].unique()),
        "minimum_equivalence_class_size": minimum_k,
        "public_columns": SEGMENT_PUBLIC_COLUMNS,
        "id_method": "cryptographically random token; not derived from spatial identifiers",
        "linkage_review_required": True,
    }
    metadata_path = args.output_csv.with_suffix(args.output_csv.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
