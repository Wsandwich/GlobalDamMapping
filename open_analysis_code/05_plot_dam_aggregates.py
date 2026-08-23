#!/usr/bin/env python3
"""Plot public global and continental dam summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "aggregated_dam_statistics"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs" / "figures" / "dam_aggregate_summary.png"
YEARS = ["2010", "2015", "2020"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def main() -> int:
    args = parse_args()
    continent = pd.read_csv(args.data_dir / "dam_counts_by_continent.csv")
    dam_type = pd.read_csv(args.data_dir / "dam_counts_by_type.csv")
    size = pd.read_csv(args.data_dir / "dam_size_summary_2020.csv")

    continent_label = continent.columns[0]
    continent = continent[~continent[continent_label].isin(["Total", "Unknown", "Antarctica"])]
    type_label = dam_type.columns[0]
    dam_type = dam_type[dam_type[type_label] != "Total"]

    style()
    colors = ["#4C78A8", "#F2CF5B", "#E45756", "#72B7B2"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))

    x = np.arange(len(continent))
    width = 0.24
    for index, year in enumerate(YEARS):
        axes[0].bar(x + (index - 1) * width, continent[year] / 1e6, width, label=year)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(continent[continent_label], rotation=45, ha="right")
    axes[0].set_ylabel("Detected dams (millions)")
    axes[0].set_title("a", loc="left", fontsize=8, fontweight="bold")
    axes[0].legend(frameon=False)

    for index, row in dam_type.reset_index(drop=True).iterrows():
        label = str(row[type_label]).replace("_", " ")
        axes[1].plot(
            YEARS,
            row[YEARS].astype(float),
            marker="o",
            color=colors[index % len(colors)],
            label=label,
        )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Detected dams (log scale)")
    axes[1].set_title("b", loc="left", fontsize=8, fontweight="bold")
    axes[1].legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=6)

    length = size[size["metric"] == "length_m"].copy()
    length["display_type"] = length["dam_type"].astype(str).str.replace("_", " ")
    axes[2].barh(length["display_type"], length["median"], color=colors[: len(length)])
    axes[2].set_xlabel("Median detected length (m)")
    axes[2].set_title("c", loc="left", fontsize=8, fontweight="bold")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, metadata={"Software": "matplotlib"})
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
