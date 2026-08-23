#!/usr/bin/env python3
"""Select approved columns from spatial-model summary outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common.privacy import find_exact_identifiers


APPROVED_COLUMNS = {
    "vif": ["region", "variable", "var_col", "VIF"],
    "moran": [
        "region",
        "var_col",
        "variable",
        "Morans_I_mean",
        "Morans_I_ci_lo",
        "Morans_I_ci_hi",
        "E_I_mean",
        "z_score_mean",
        "p_value_mean",
        "n_mean",
        "p_fdr",
        "sig_fdr",
    ],
    "effects": [
        "region",
        "dam_type",
        "variable",
        "direct_effect_mean",
        "direct_effect_ci_lo",
        "direct_effect_ci_hi",
        "indirect_effect_mean",
        "indirect_effect_ci_lo",
        "indirect_effect_ci_hi",
        "total_effect_mean",
        "total_effect_ci_lo",
        "total_effect_ci_hi",
        "p_fdr",
        "sig_fdr",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table_type", choices=sorted(APPROVED_COLUMNS))
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.input_csv)
    approved = APPROVED_COLUMNS[args.table_type]
    missing = [column for column in approved if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing model-summary columns: {missing}")
    selected = frame[approved].copy()
    identifiers = find_exact_identifiers(selected.columns)
    if identifiers:
        raise ValueError(f"Exact identifiers are not allowed: {identifiers}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_csv, index=False)
    metadata = {
        "table_type": args.table_type,
        "rows": int(len(selected)),
        "columns": approved,
        "contains_model_summaries_only": True,
    }
    metadata_path = args.output_csv.with_suffix(args.output_csv.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
