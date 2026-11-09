#!/usr/bin/env python3
"""Plot extracted-feature distributions, split by make/miss.

This is the "check the extracted features" visualization step — a sanity
check on whether the features look reasonable and whether any of them
show an obvious univariate relationship with outcome, not a claim of a
validated effect (see SUMMARY.md for that discussion; with 5 players and
only ~190 misses, any single-feature pattern here could easily be a
player effect, not a release-mechanics effect).

Outputs: outputs/figures/feature_checks/*.png

Run:
    python scripts/04_feature_distributions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spl_ft import config
from spl_ft.features import FEATURE_COLUMNS


def main():
    if not config.FEATURES_CSV.exists():
        print(f"ERROR: {config.FEATURES_CSV} not found.")
        print("BLOCKED: run `python scripts/03_extract_features.py` first.")
        sys.exit(1)

    config.FEATURE_CHECK_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.FEATURES_CSV)
    usable = df[~df["excluded"]].copy()
    usable["result"] = usable["label"].map({1: "made", 0: "missed"})

    # 1. Per-feature histogram, made vs missed overlaid.
    n = len(FEATURE_COLUMNS)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3 * nrows))
    axes = axes.flatten()
    for ax, col in zip(axes, FEATURE_COLUMNS):
        for result, color in [("made", "tab:green"), ("missed", "tab:red")]:
            vals = usable.loc[usable["result"] == result, col].dropna()
            ax.hist(vals, bins=20, alpha=0.5, label=result, color=color, density=True)
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Pre-release feature distributions by outcome (all players pooled)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out1 = config.FEATURE_CHECK_FIGURES_DIR / "feature_histograms_by_outcome.png"
    fig.savefig(out1, dpi=110)
    plt.close(fig)
    print(f"Wrote {out1}")

    # 2. Per-player boxplots for a couple of key features, to visualize how
    #    much between-player variation there is (relevant to the
    #    "model may just learn player identity" limitation).
    key_feats = ["knee_angle_min_deg", "elbow_angle_change_deg", "wrist_vel_z_mean_norm"]
    fig, axes = plt.subplots(1, len(key_feats), figsize=(4 * len(key_feats), 4))
    for ax, col in zip(axes, key_feats):
        data = [usable.loc[usable["participant_id"] == pid, col].dropna()
                for pid in sorted(usable["participant_id"].unique())]
        ax.boxplot(data, tick_labels=sorted(usable["participant_id"].unique()))
        ax.set_title(col, fontsize=9)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Per-player spread of selected features\n(illustrates between-player variation; see SUMMARY.md limitations)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out2 = config.FEATURE_CHECK_FIGURES_DIR / "feature_by_player.png"
    fig.savefig(out2, dpi=110)
    plt.close(fig)
    print(f"Wrote {out2}")

    # 3. Scatter of two interpretable features colored by outcome and
    #    shaped by player.
    fig, ax = plt.subplots(figsize=(6, 5))
    markers = ["o", "s", "^", "D", "v"]
    for marker, pid in zip(markers, sorted(usable["participant_id"].unique())):
        sub = usable[usable["participant_id"] == pid]
        for result, color in [("made", "tab:green"), ("missed", "tab:red")]:
            s = sub[sub["result"] == result]
            ax.scatter(s["knee_angle_min_deg"], s["elbow_angle_change_deg"],
                       c=color, marker=marker, alpha=0.6, s=25,
                       label=f"{pid} {result}")
    ax.set_xlabel("minimum knee angle in window (deg)")
    ax.set_ylabel("elbow angle change over window (deg)")
    ax.set_title("Knee flexion vs elbow extension, by player & outcome")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    out3 = config.FEATURE_CHECK_FIGURES_DIR / "knee_vs_elbow_scatter.png"
    fig.savefig(out3, dpi=110)
    plt.close(fig)
    print(f"Wrote {out3}")


if __name__ == "__main__":
    main()
