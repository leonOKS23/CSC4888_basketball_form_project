#!/usr/bin/env python3
"""Extract pre-release biomechanics features for every trial.

Writes:
    outputs/features.csv     one row per shot (included or excluded)
    outputs/exclusions.csv   just the excluded rows, with reasons
    outputs/dataset_inspection/feature_extraction_summary.md

Run:
    python scripts/03_extract_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spl_ft import config
from spl_ft.data_io import list_trials_grouped_by_participant, load_trial
from spl_ft.features import extract_features_for_trial
from spl_ft.handedness import build_player_shooting_side


def main():
    if not config.FREETHROW_DATA_DIR.exists():
        print(f"ERROR: dataset not found at {config.FREETHROW_DATA_DIR}")
        print("BLOCKED: run `python scripts/00_download_data.py` first.")
        sys.exit(1)

    grouped = list_trials_grouped_by_participant()
    all_paths = [p for paths in grouped.values() for p in paths]
    shooting_side_info = build_player_shooting_side(all_paths)

    rows = []
    for pid, paths in grouped.items():
        side = shooting_side_info.get(pid, {"side": "RIGHT"})["side"]
        for p in paths:
            trial = load_trial(p)
            row = extract_features_for_trial(trial, side)
            rows.append(row.to_dict())

    df = pd.DataFrame(rows)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.FEATURES_CSV, index=False)

    excluded = df[df["excluded"]]
    excluded.to_csv(config.EXCLUSIONS_CSV, index=False)

    print(f"Wrote {config.FEATURES_CSV} ({len(df)} rows)")
    print(f"Wrote {config.EXCLUSIONS_CSV} ({len(excluded)} rows)")

    # --- Summary ---
    print("\n--- Exclusion reasons ---")
    reason_counts = excluded["exclusion_reason"].value_counts()
    print(reason_counts.to_string())

    print("\n--- Usable (included) shots per player ---")
    usable = df[~df["excluded"]]
    per_player = usable.groupby("participant_id")["label"].agg(
        n_usable="count", n_made="sum"
    )
    per_player["n_missed"] = per_player["n_usable"] - per_player["n_made"]
    per_player["make_rate"] = per_player["n_made"] / per_player["n_usable"]
    print(per_player.to_string())

    config.DATASET_INSPECTION_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = config.DATASET_INSPECTION_DIR / "feature_extraction_summary.md"
    with open(summary_path, "w") as f:
        f.write("# Feature extraction summary\n\n")
        f.write(f"Total trials processed: {len(df)}\n\n")
        f.write(f"Usable (included) shots: {len(usable)}\n")
        f.write(f"Excluded shots: {len(excluded)}\n\n")
        f.write("## Exclusion reasons\n\n```\n")
        f.write(str(reason_counts))
        f.write("\n```\n\n## Usable makes/misses per player\n\n```\n")
        f.write(per_player.to_string())
        f.write("\n```\n")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
