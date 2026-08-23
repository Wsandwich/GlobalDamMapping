#!/usr/bin/env python3
"""Plot public spatial-model effect summaries without model input matrices."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPOSITORY_ROOT / "figure_source_data" / "model"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs" / "figures" / "model_effect_summary.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    effects = pd.read_csv(args.source_dir / "global_total_effects.csv")
    shares = pd.read_csv(args.source_dir / "global_indirect_shares.csv")

    effects = effects.sort_values(["dam_order", "variable_order"])
    labels = effects["dam_label"].astype(str) + ": " + effects["variable_label"].astype(str)
    y = np.arange(len(effects))
    estimate = effects["total_effect_mean"].astype(float).to_numpy()
    low = effects["total_effect_pooled_ci_lo"].astype(float).to_numpy()
    high = effects["total_effect_pooled_ci_hi"].astype(float).to_numpy()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 7.2), gridspec_kw={"width_ratios": [2.2, 1]})
    axes[0].errorbar(
        estimate,
        y,
        xerr=np.vstack([estimate - low, high - estimate]),
        fmt="o",
        color="#2F6B9A",
        ecolor="#8AA9C2",
        capsize=2,
        markersize=3,
    )
    axes[0].axvline(0, color="#777777", linewidth=0.7)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel("Standardized total effect")
    axes[0].set_title("a", loc="left", fontsize=8, fontweight="bold")

    share_summary = (
        shares.groupby(["dam_order", "dam_label"], as_index=False)
        .agg(indirect_share=("dam_mean_indirect_share_plotted", "first"))
        .sort_values("dam_order")
    )
    axes[1].barh(
        share_summary["dam_label"],
        share_summary["indirect_share"] * 100,
        color="#E59D3A",
    )
    axes[1].axvline(50, color="#777777", linewidth=0.7, linestyle="--")
    axes[1].set_xlabel("Indirect share (%)")
    axes[1].set_title("b", loc="left", fontsize=8, fontweight="bold")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, metadata={"Software": "matplotlib"})
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
