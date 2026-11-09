#!/usr/bin/env python3
"""Train and evaluate the majority baseline, logistic regression, and
random forest models under two protocols:

  A. Known players, later shots: per player, chronological 80/20
     train/test split (verified chronological — see
     spl_ft.data_io.trial_sort_key docstring and
     outputs/dataset_inspection/summary.md). One model is trained on the
     pooled training shots from all players and evaluated on the pooled
     held-out later shots from all players.

  B. Unseen players: leave-one-player-out cross-validation. All
     preprocessing (imputation/scaling) and the majority baseline are
     refit from scratch within each fold's training data only.

Outputs under outputs/eval_known_players/ and outputs/eval_unseen_players/:
  - results_overall.csv
  - results_by_player.csv (B only; A also reports a by-player breakdown of
    the pooled test set for context)
  - confusion_matrices.png

Run:
    python scripts/05_train_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spl_ft import config
from spl_ft.evaluation import compute_metrics, plot_confusion_matrix
from spl_ft.modeling import build_models, get_X_y, load_usable_features


def chronological_split(df: pd.DataFrame, train_fraction: float):
    """Per-player chronological split. Assumes df rows are already in
    chronological order within each participant_id (true for
    outputs/features.csv — see data_io.trial_sort_key).
    """
    train_idx = []
    test_idx = []
    for pid, sub in df.groupby("participant_id", sort=True):
        n = len(sub)
        n_train = max(1, round(train_fraction * n))
        n_train = min(n_train, n - 1) if n > 1 else n_train  # keep at least 1 test row if possible
        idx = sub.index.to_numpy()
        train_idx.extend(idx[:n_train])
        test_idx.extend(idx[n_train:])
    return np.array(train_idx), np.array(test_idx)


def run_eval_known_players(df: pd.DataFrame):
    out_dir = config.EVAL_KNOWN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    train_idx, test_idx = chronological_split(df, config.TRAIN_FRACTION)
    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    X_train, y_train = get_X_y(train_df)
    X_test, y_test = get_X_y(test_df)

    print(f"[Eval A] train n={len(train_df)} (make rate {y_train.mean():.3f}), "
          f"test n={len(test_df)} (make rate {y_test.mean():.3f})")

    models = build_models()
    overall_rows = []
    by_player_rows = []
    cms = {}

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        m = compute_metrics(y_test, y_pred)
        m["model"] = name
        overall_rows.append(m)
        cms[name] = np.array(m["confusion_matrix_tn_fp_fn_tp"]).reshape(2, 2)

        test_df_copy = test_df.copy()
        test_df_copy["y_pred"] = y_pred
        for pid, sub in test_df_copy.groupby("participant_id"):
            pm = compute_metrics(sub["label"].to_numpy(), sub["y_pred"].to_numpy())
            pm["model"] = name
            pm["participant_id"] = pid
            by_player_rows.append(pm)

    overall_df = pd.DataFrame(overall_rows)[
        ["model", "n_test", "make_rate", "accuracy", "balanced_accuracy", "confusion_matrix_tn_fp_fn_tp"]
    ]
    overall_df.to_csv(out_dir / "results_overall.csv", index=False)
    print(overall_df.to_string(index=False))

    by_player_df = pd.DataFrame(by_player_rows)[
        ["model", "participant_id", "n_test", "make_rate", "accuracy", "balanced_accuracy", "confusion_matrix_tn_fp_fn_tp"]
    ]
    by_player_df.to_csv(out_dir / "results_by_player.csv", index=False)

    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4))
    for ax, name in zip(axes, models):
        plot_confusion_matrix(ax, cms[name], f"{name}\nacc={overall_df.set_index('model').loc[name, 'accuracy']:.2f}")
    fig.suptitle("Evaluation A: known players, later shots (pooled test set)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_dir / "confusion_matrices.png", dpi=110)
    plt.close(fig)

    return overall_df, by_player_df


def run_eval_unseen_players(df: pd.DataFrame):
    out_dir = config.EVAL_UNSEEN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    players = sorted(df["participant_id"].unique())
    models = build_models()

    # pooled out-of-fold predictions, per model
    pooled_true = {name: [] for name in models}
    pooled_pred = {name: [] for name in models}
    by_player_rows = []

    for held_out in players:
        train_df = df[df["participant_id"] != held_out].reset_index(drop=True)
        test_df = df[df["participant_id"] == held_out].reset_index(drop=True)
        X_train, y_train = get_X_y(train_df)
        X_test, y_test = get_X_y(test_df)

        fold_models = build_models()  # fresh, unfit pipelines per fold
        for name, pipe in fold_models.items():
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            pooled_true[name].extend(y_test.tolist())
            pooled_pred[name].extend(y_pred.tolist())

            pm = compute_metrics(y_test, y_pred)
            pm["model"] = name
            pm["held_out_player"] = held_out
            by_player_rows.append(pm)

        print(f"[Eval B] held out {held_out}: train n={len(train_df)}, test n={len(test_df)} "
              f"(make rate {y_test.mean():.3f})")

    overall_rows = []
    cms = {}
    for name in models:
        yt = np.array(pooled_true[name])
        yp = np.array(pooled_pred[name])
        m = compute_metrics(yt, yp)
        m["model"] = name
        overall_rows.append(m)
        cms[name] = np.array(m["confusion_matrix_tn_fp_fn_tp"]).reshape(2, 2)

    overall_df = pd.DataFrame(overall_rows)[
        ["model", "n_test", "make_rate", "accuracy", "balanced_accuracy", "confusion_matrix_tn_fp_fn_tp"]
    ]
    overall_df.to_csv(out_dir / "results_overall.csv", index=False)
    print("\n[Eval B] pooled across all held-out players:")
    print(overall_df.to_string(index=False))

    by_player_df = pd.DataFrame(by_player_rows)[
        ["model", "held_out_player", "n_test", "make_rate", "accuracy", "balanced_accuracy", "confusion_matrix_tn_fp_fn_tp"]
    ]
    by_player_df.to_csv(out_dir / "results_by_player.csv", index=False)

    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4))
    for ax, name in zip(axes, models):
        plot_confusion_matrix(ax, cms[name], f"{name}\nacc={overall_df.set_index('model').loc[name, 'accuracy']:.2f}")
    fig.suptitle("Evaluation B: unseen players, pooled leave-one-player-out")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_dir / "confusion_matrices.png", dpi=110)
    plt.close(fig)

    return overall_df, by_player_df


def plot_model_comparison(eval_a_overall, eval_b_overall):
    fig, ax = plt.subplots(figsize=(7, 4))
    models = eval_a_overall["model"].tolist()
    x = np.arange(len(models))
    width = 0.2
    ax.bar(x - 1.5 * width, eval_a_overall["accuracy"], width, label="A: accuracy")
    ax.bar(x - 0.5 * width, eval_a_overall["balanced_accuracy"], width, label="A: balanced acc.")
    ax.bar(x + 0.5 * width, eval_b_overall["accuracy"], width, label="B: accuracy")
    ax.bar(x + 1.5 * width, eval_b_overall["balanced_accuracy"], width, label="B: balanced acc.")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle=":", lw=1)
    ax.set_ylabel("score")
    ax.set_title("Model comparison: known-players (A) vs unseen-players (B)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = config.FIGURES_DIR / "model_comparison.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    if not config.FEATURES_CSV.exists():
        print(f"ERROR: {config.FEATURES_CSV} not found.")
        print("BLOCKED: run `python scripts/03_extract_features.py` first.")
        sys.exit(1)

    df = load_usable_features()
    print(f"Loaded {len(df)} usable shots across {df['participant_id'].nunique()} players.")

    eval_a_overall, eval_a_by_player = run_eval_known_players(df)
    print()
    eval_b_overall, eval_b_by_player = run_eval_unseen_players(df)

    plot_model_comparison(eval_a_overall, eval_b_overall)

    print(f"\nWrote results to {config.EVAL_KNOWN_DIR} and {config.EVAL_UNSEEN_DIR}")


if __name__ == "__main__":
    main()
