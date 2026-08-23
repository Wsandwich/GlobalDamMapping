#!/usr/bin/env python3
"""Plot public 3 x 3 free-flowing-river aggregate summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPOSITORY_ROOT / "ffr_aggregate_tables" / "grouped_all_years.csv"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs" / "figures" / "ffr_3x3_summary.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def matrix(frame: pd.DataFrame, year: int, value: str) -> tuple[np.ndarray, list[str], list[str]]:
    selected = frame[frame["year"] == year]
    pivot = selected.pivot(index="network_hierarchy", columns="riv_order_group", values=value)
    return pivot.to_numpy(), list(pivot.index), list(pivot.columns)


def heatmap(axis, values, rows, columns, title, colorbar_label):
    image = axis.imshow(values, cmap="YlOrRd", aspect="auto")
    axis.set_xticks(np.arange(len(columns)))
    axis.set_xticklabels(columns, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(rows)))
    axis.set_yticklabels(rows)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=6)
    axis.set_title(title, fontsize=7)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label, fontsize=7)


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.data)
    values_2020, rows, columns = matrix(frame, 2020, "confirmed_fragmentation_rate_pct")
    values_2010, rows_2010, columns_2010 = matrix(frame, 2010, "confirmed_fragmentation_rate_pct")
    if rows != rows_2010 or columns != columns_2010:
        raise ValueError("The 2010 and 2020 3 x 3 group layouts differ")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    heatmap(axes[0], values_2020, rows, columns, "a  Confirmed fragmentation in 2020", "Rate (%)")
    heatmap(axes[1], values_2020 - values_2010, rows, columns, "b  Change from 2010 to 2020", "Percentage points")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, metadata={"Software": "matplotlib"})
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
