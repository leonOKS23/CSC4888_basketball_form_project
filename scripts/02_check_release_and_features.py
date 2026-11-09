#!/usr/bin/env python3
"""Visualize release detection and the pre-release feature window.

For a handful of representative trials (different players, makes and
misses), plot:
  - ball-to-shooting-wrist distance over time, with the detected release
    frame and the feature window (after the safety buffer) shaded.
  - shooting-side elbow and knee angle over time, with the same window
    shaded, and the ball height.
  - shooting-side wrist/hip vertical height over time.

This is the "inspect and visualize representative examples to check
whether [release estimation] works" step called for by the assignment.
It is a sanity check for a human reader, not an automated test.

Outputs: outputs/figures/release_checks/<shot_id>.png

Run:
    python scripts/02_check_release_and_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spl_ft import config
from spl_ft.data_io import list_trials_grouped_by_participant, load_trial
from spl_ft.handedness import build_player_shooting_side
from spl_ft.kinematics import joint_angle, vertical
from spl_ft.release import estimate_release


def plot_trial(trial, shooting_side, outpath: Path):
    est = estimate_release(trial)

    t = trial.time_ms / 1000.0  # seconds
    ball_z = vertical(trial.ball_xyz)
    wrist = trial.keypoint(f"{shooting_side}_WRIST")
    shoulder = trial.keypoint(f"{shooting_side}_SHOULDER")
    elbow = trial.keypoint(f"{shooting_side}_ELBOW")
    hip = trial.keypoint(f"{shooting_side}_HIP")
    knee = trial.keypoint(f"{shooting_side}_KNEE")
    ankle = trial.keypoint(f"{shooting_side}_ANKLE")
    mid_hip = trial.keypoint("MID_HIP")

    dist_wrist_ball = np.linalg.norm(trial.ball_xyz - wrist, axis=1)
    elbow_angle = joint_angle(shoulder, elbow, wrist)
    knee_angle = joint_angle(hip, knee, ankle)

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)

    title = (f"{trial.shot_id}  result={trial.result}  fps={trial.sampling_rate:.0f}  "
             f"shooting_side={shooting_side}  release_status={est.status}")
    fig.suptitle(title, fontsize=10)

    release_t = None
    window_start_t = None
    if est.status == "ok":
        release_t = t[est.release_frame]
        buffer_frames = max(1, round(config.RELEASE_BUFFER_SEC * trial.sampling_rate))
        cutoff_frame = est.release_frame - buffer_frames
        window_frames = round(config.WINDOW_SEC * trial.sampling_rate)
        window_start_frame = max(0, cutoff_frame - window_frames)
        if 0 <= cutoff_frame < len(t):
            cutoff_t = t[cutoff_frame]
            window_start_t = t[window_start_frame]

    for ax in axes:
        if release_t is not None:
            ax.axvline(release_t, color="red", linestyle="--", lw=1, label="detected release")
        if window_start_t is not None:
            ax.axvspan(window_start_t, cutoff_t, color="tab:blue", alpha=0.15, label="feature window (post-buffer)")

    axes[0].plot(t, dist_wrist_ball, color="tab:purple")
    axes[0].axhline(config.CONTACT_DISTANCE_THRESHOLD_FT, color="gray", lw=0.8, linestyle=":",
                     label=f"contact threshold ({config.CONTACT_DISTANCE_THRESHOLD_FT} ft)")
    axes[0].set_ylabel("ball-to-wrist\ndistance (ft)")
    axes[0].legend(fontsize=7, loc="upper left")

    axes[1].plot(t, ball_z, color="tab:orange", label="ball height (z, ft)")
    axes[1].plot(t, vertical(wrist), color="tab:blue", label="wrist height (z, ft)")
    axes[1].set_ylabel("height (ft)")
    axes[1].legend(fontsize=7, loc="upper left")

    axes[2].plot(t, elbow_angle, color="tab:green")
    axes[2].set_ylabel("elbow angle (deg)")

    axes[3].plot(t, knee_angle, color="tab:brown")
    axes[3].set_ylabel("knee angle (deg)")
    axes[3].set_xlabel("time (s)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=110)
    plt.close(fig)


def main():
    config.RELEASE_CHECK_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    grouped = list_trials_grouped_by_participant()
    all_paths = [p for paths in grouped.values() for p in paths]
    shooting_side_info = build_player_shooting_side(all_paths)

    n_per_player_per_result = 2  # makes + misses examples per player
    n_written = 0
    for pid, paths in grouped.items():
        side = shooting_side_info.get(pid, {"side": "RIGHT"})["side"]
        made_paths = []
        missed_paths = []
        for p in paths:
            trial = load_trial(p)
            if trial.result == config.RESULT_MADE and len(made_paths) < n_per_player_per_result:
                made_paths.append((p, trial))
            elif trial.result == config.RESULT_MISSED and len(missed_paths) < n_per_player_per_result:
                missed_paths.append((p, trial))
            if len(made_paths) >= n_per_player_per_result and len(missed_paths) >= n_per_player_per_result:
                break
        for p, trial in made_paths + missed_paths:
            outpath = config.RELEASE_CHECK_FIGURES_DIR / f"{trial.shot_id}.png"
            plot_trial(trial, side, outpath)
            n_written += 1
            print(f"Wrote {outpath}")

    print(f"\nWrote {n_written} release-check figures to {config.RELEASE_CHECK_FIGURES_DIR}")


if __name__ == "__main__":
    main()
