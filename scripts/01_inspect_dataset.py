#!/usr/bin/env python3
"""Inspect the SPL Open Data free-throw dataset and write a summary.

This script re-derives (rather than assumes) the facts about the dataset
that the rest of the pipeline depends on: file layout, schema, label
encoding, keypoint set, frame rate, vertical axis, and the lack of an
explicit release event / handedness field. It also runs the release
detector and handedness vote over every trial and reports, per player,
how many usable makes/misses result and why any shots are excluded.

Outputs:
    outputs/dataset_inspection/summary.md
    outputs/dataset_inspection/per_player_counts.csv
    outputs/dataset_inspection/handedness.csv
    outputs/dataset_inspection/release_status_counts.csv

Run:
    python scripts/01_inspect_dataset.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spl_ft import config
from spl_ft.data_io import list_trials_grouped_by_participant, load_trial, trial_sort_key
from spl_ft.handedness import build_player_shooting_side
from spl_ft.release import estimate_release


def main():
    if not config.FREETHROW_DATA_DIR.exists():
        print(f"ERROR: dataset not found at {config.FREETHROW_DATA_DIR}")
        print("BLOCKED: run `python scripts/00_download_data.py` first.")
        sys.exit(1)

    config.DATASET_INSPECTION_DIR.mkdir(parents=True, exist_ok=True)

    grouped = list_trials_grouped_by_participant()
    all_paths = [p for paths in grouped.values() for p in paths]
    print(f"Found {len(all_paths)} trial files across {len(grouped)} participants.")

    # --- Schema / top-level key check -------------------------------------
    key_sets = collections.Counter()
    results = collections.Counter()
    sampling_rates = collections.Counter()
    session_dates = collections.Counter()
    keypoint_name_sets = collections.Counter()
    landing_fields_present = collections.Counter()

    for p in all_paths:
        with open(p) as f:
            raw = json.load(f)
        key_sets[tuple(sorted(raw.keys()))] += 1
        results[raw["result"]] += 1
        sampling_rates[raw["sampling_rate"]] += 1
        session_dates[raw["trial_date"]] += 1
        kp_names = tuple(sorted(raw["tracking"][0]["data"]["player"].keys()))
        keypoint_name_sets[kp_names] += 1
        has_landing = all(
            raw.get(k) is not None for k in ("landing_x", "landing_y", "entry_angle")
        )
        landing_fields_present[has_landing] += 1

    n_keypoint_schemas = len(keypoint_name_sets)
    n_keypoints_each = {len(k): v for k, v in keypoint_name_sets.items()}

    # --- Vertical axis sanity check (toe markers near 0, head markers higher) ---
    sample = load_trial(all_paths[0])
    toe = sample.keypoint("RIGHT_BIG_TOE")
    nose = sample.keypoint("NOSE")
    toe_z = np.nanmedian(toe[:, 2])
    nose_z = np.nanmedian(nose[:, 2])

    # --- Release + handedness over the whole dataset -----------------------
    release_status_counts = collections.Counter()
    hand_votes = collections.defaultdict(collections.Counter)
    per_player_rows = []

    for pid, paths in grouped.items():
        n_made = n_missed = 0
        release_ok = 0
        release_fail_reasons = collections.Counter()
        for p in paths:
            trial = load_trial(p)
            if trial.result == config.RESULT_MADE:
                n_made += 1
            else:
                n_missed += 1
            est = estimate_release(trial)
            release_status_counts[est.status] += 1
            if est.status == "ok":
                release_ok += 1
                hand_votes[pid][est.shooting_hand] += 1
            else:
                release_fail_reasons[est.status] += 1

        per_player_rows.append({
            "participant_id": pid,
            "n_trials": len(paths),
            "n_made": n_made,
            "n_missed": n_missed,
            "make_rate": n_made / len(paths) if paths else float("nan"),
            "n_release_detected": release_ok,
            "n_release_failed": len(paths) - release_ok,
            "release_fail_reasons": dict(release_fail_reasons),
        })

    per_player_df = pd.DataFrame(per_player_rows)
    per_player_df.to_csv(config.DATASET_INSPECTION_DIR / "per_player_counts.csv", index=False)

    # --- Handedness table ---------------------------------------------------
    shooting_side_info = build_player_shooting_side(all_paths)
    hand_rows = []
    for pid, info in shooting_side_info.items():
        votes = info["votes"]
        total = info["n_votes"]
        hand_rows.append({
            "participant_id": pid,
            "assigned_shooting_side": info["side"],
            "n_votes_right": votes.get("RIGHT", 0),
            "n_votes_left": votes.get("LEFT", 0),
            "n_votes_total": total,
            "fraction_majority": info["fraction_majority"],
            "overridden": info["overridden"],
        })
    hand_df = pd.DataFrame(hand_rows).sort_values("participant_id")
    hand_df.to_csv(config.DATASET_INSPECTION_DIR / "handedness.csv", index=False)

    release_status_df = pd.DataFrame(
        sorted(release_status_counts.items()), columns=["release_status", "n_trials"]
    )
    release_status_df.to_csv(config.DATASET_INSPECTION_DIR / "release_status_counts.csv", index=False)

    # --- Write human-readable summary --------------------------------------
    lines = []
    lines.append("# SPL Open Data free-throw dataset — inspection summary\n")
    lines.append("Generated by `scripts/01_inspect_dataset.py`. All facts below were derived\n"
                 "directly from the trial JSON files and the dataset README/notebook, not assumed.\n")

    lines.append("\n## File layout\n")
    lines.append(f"- {len(all_paths)} trial files, {len(grouped)} participants: "
                 f"{', '.join(sorted(grouped.keys()))}\n")
    lines.append("- Path convention: `data/<session_date>/<participant_id>/BB_FT_<participant_id>_<trial_id>.json`\n")
    lines.append(f"- Session dates found: {dict(session_dates)}\n")
    lines.append("- IMPORTANT: trial IDs (T0001, T0002, ...) reset at the start of each capture\n"
                 "  session. Participant P0001 has trials from two sessions "
                 "(2024-08-28 and 2025-12-18); within-participant chronological order is\n"
                 "  therefore sorted by (session_date, trial_number), not trial_number alone.\n")

    lines.append("\n## Per-trial JSON schema\n")
    lines.append(f"- Top-level key sets observed (should be exactly 1 unique set): {n_keypoint_schemas if False else len(key_sets)}\n")
    for ks, count in key_sets.items():
        lines.append(f"  - {count} files with keys: {list(ks)}\n")
    lines.append("- `result`: " + str(dict(results)) + "\n")
    lines.append("- `sampling_rate` (fps): " + str(dict(sampling_rates)) + "\n")
    lines.append(f"- `landing_x`/`landing_y`/`entry_angle` present (non-null) in all trials: "
                 f"{dict(landing_fields_present)} (these describe shot *outcome* — ball position "
                 f"relative to the rim — and are intentionally excluded from model features)\n")

    lines.append("\n## Keypoints\n")
    lines.append(f"- Distinct keypoint-name sets across trials: {n_keypoint_schemas}\n")
    lines.append(f"- Keypoint-set size -> number of trials with that many keypoints: {n_keypoints_each}\n")
    lines.append(f"- Example keypoint names (first trial): {sorted(sample.player_xyz.keys())[:10]} ... "
                 f"({len(sample.player_xyz)} total)\n")
    lines.append("- Coordinates are in feet, court-centered. `z` is the vertical axis — "
                 f"confirmed empirically: median RIGHT_BIG_TOE z = {toe_z:.2f} ft, "
                 f"median NOSE z = {nose_z:.2f} ft (toe near the floor, nose near head height).\n")

    lines.append("\n## Release event\n")
    lines.append("- The dataset does **not** provide an explicit release frame/event field.\n")
    lines.append("- We estimate it from ball-to-wrist distance (see `spl_ft/release.py` and\n"
                 "  `PROJECT_LOG.md` for the full method and validation). Status counts across\n"
                 "  all trials:\n")
    for _, row in release_status_df.iterrows():
        lines.append(f"  - {row['release_status']}: {row['n_trials']}\n")

    lines.append("\n## Shooting handedness\n")
    lines.append("- The dataset does **not** provide a handedness field.\n")
    lines.append("- We estimate a per-player shooting side as the majority vote (over trials) of\n"
                 "  which wrist loses contact with the ball last. See `handedness.csv`:\n")
    for _, row in hand_df.iterrows():
        lines.append(
            f"  - {row['participant_id']}: assigned **{row['assigned_shooting_side']}** "
            f"({row['n_votes_right']} RIGHT / {row['n_votes_left']} LEFT votes, "
            f"{row['fraction_majority']*100:.1f}% majority)\n"
        )

    lines.append("\n## Per-player make/miss counts\n")
    lines.append("| Participant | Trials | Made | Missed | Make rate | Release detected | Release failed |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for _, row in per_player_df.sort_values("participant_id").iterrows():
        lines.append(
            f"| {row['participant_id']} | {row['n_trials']} | {row['n_made']} | {row['n_missed']} | "
            f"{row['make_rate']:.3f} | {row['n_release_detected']} | {row['n_release_failed']} |\n"
        )
    lines.append(
        "\nNote: this table counts trials for which a release frame could be estimated at all. "
        "Further exclusions (insufficient pre-release window length, missing keypoint data at the "
        "window boundaries) happen during feature extraction — see "
        "`outputs/exclusions.csv` and `outputs/dataset_inspection/` after running "
        "`scripts/03_extract_features.py`.\n"
    )

    summary_path = config.DATASET_INSPECTION_DIR / "summary.md"
    with open(summary_path, "w") as f:
        f.writelines(lines)

    print(f"Wrote {summary_path}")
    print(f"Wrote {config.DATASET_INSPECTION_DIR / 'per_player_counts.csv'}")
    print(f"Wrote {config.DATASET_INSPECTION_DIR / 'handedness.csv'}")
    print(f"Wrote {config.DATASET_INSPECTION_DIR / 'release_status_counts.csv'}")
    print("\n--- per-player counts ---")
    print(per_player_df.sort_values("participant_id").to_string(index=False))
    print("\n--- handedness ---")
    print(hand_df.to_string(index=False))


if __name__ == "__main__":
    main()
