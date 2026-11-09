"""Per-shot feature extraction from the pre-release window.

One row per trial is produced by :func:`extract_features_for_trial`. Rows
that cannot be computed reliably are still returned, but with
``excluded=True`` and a human-readable ``exclusion_reason`` — they are
never silently dropped or filled with fabricated values.

Features (see assignment spec / README for the full rationale):
  - elbow_angle_end_deg       shooting-side elbow angle at window end
  - knee_angle_end_deg        shooting-side knee angle at window end
  - knee_angle_min_deg        min shooting-side knee angle in window
  - elbow_angle_change_deg    elbow angle(end) - elbow angle(start)
  - knee_angle_change_deg     knee angle(end) - knee angle(start)
  - wrist_height_rel_hip_norm (wrist_z - MID_HIP_z at window end) / body_size
  - wrist_vel_z_mean_norm     mean(d wrist_z/dt) over window / body_size
  - hip_vel_z_mean_norm       mean(d MID_HIP_z/dt) over window / body_size

All features use only keypoints at or before the (buffered) release
frame — ball position is used solely to *locate* the window, never as a
feature; landing position / entry angle / result are never used as
features either.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from . import config
from .data_io import Trial
from .kinematics import body_size_ft, central_diff_velocity, joint_angle, vertical
from .release import ReleaseEstimate, estimate_release


@dataclass
class FeatureRow:
    participant_id: str
    trial_id: str
    trial_date: str
    sampling_rate: float
    shot_id: str
    result: str
    label: int
    shooting_side: str
    excluded: bool
    exclusion_reason: str = ""
    release_status: str = ""
    release_frame: Optional[int] = None
    window_start_frame: Optional[int] = None
    window_end_frame: Optional[int] = None
    window_duration_sec: Optional[float] = None
    target_window_sec: float = config.WINDOW_SEC
    body_size_ft: Optional[float] = None
    n_missing_frames_in_window: int = 0
    # features (None if excluded)
    elbow_angle_end_deg: Optional[float] = None
    knee_angle_end_deg: Optional[float] = None
    knee_angle_min_deg: Optional[float] = None
    elbow_angle_change_deg: Optional[float] = None
    knee_angle_change_deg: Optional[float] = None
    wrist_height_rel_hip_norm: Optional[float] = None
    wrist_vel_z_mean_norm: Optional[float] = None
    hip_vel_z_mean_norm: Optional[float] = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


FEATURE_COLUMNS = [
    "elbow_angle_end_deg",
    "knee_angle_end_deg",
    "knee_angle_min_deg",
    "elbow_angle_change_deg",
    "knee_angle_change_deg",
    "wrist_height_rel_hip_norm",
    "wrist_vel_z_mean_norm",
    "hip_vel_z_mean_norm",
]


def _excluded(trial: Trial, shooting_side: str, reason: str, release_est: ReleaseEstimate = None) -> FeatureRow:
    return FeatureRow(
        participant_id=trial.participant_id,
        trial_id=trial.trial_id,
        trial_date=trial.trial_date,
        sampling_rate=trial.sampling_rate,
        shot_id=trial.shot_id,
        result=trial.result,
        label=trial.label,
        shooting_side=shooting_side,
        excluded=True,
        exclusion_reason=reason,
        release_status=release_est.status if release_est else "",
        release_frame=release_est.release_frame if release_est else None,
    )


def extract_features_for_trial(trial: Trial, shooting_side: str) -> FeatureRow:
    """shooting_side must be 'RIGHT' or 'LEFT' (see spl_ft.handedness)."""
    release_est = estimate_release(trial)
    if release_est.status != "ok":
        return _excluded(trial, shooting_side, f"release_detection_failed:{release_est.status}", release_est)

    release_frame = release_est.release_frame
    buffer_frames = max(1, round(config.RELEASE_BUFFER_SEC * trial.sampling_rate))
    cutoff_frame = release_frame - buffer_frames

    if cutoff_frame < 0:
        return _excluded(trial, shooting_side, "release_too_early_for_buffer", release_est)

    window_frames = round(config.WINDOW_SEC * trial.sampling_rate)
    window_start = cutoff_frame - window_frames
    actual_start = max(0, window_start)
    actual_duration_frames = cutoff_frame - actual_start
    actual_duration_sec = actual_duration_frames * trial.dt

    if actual_duration_sec < config.MIN_WINDOW_FRACTION * config.WINDOW_SEC:
        row = _excluded(trial, shooting_side, "insufficient_pre_release_window", release_est)
        row.window_start_frame = actual_start
        row.window_end_frame = cutoff_frame
        row.window_duration_sec = actual_duration_sec
        return row

    sl = slice(actual_start, cutoff_frame + 1)  # inclusive of cutoff frame

    shoulder = trial.keypoint(f"{shooting_side}_SHOULDER")
    elbow = trial.keypoint(f"{shooting_side}_ELBOW")
    wrist = trial.keypoint(f"{shooting_side}_WRIST")
    hip_side = trial.keypoint(f"{shooting_side}_HIP")
    knee = trial.keypoint(f"{shooting_side}_KNEE")
    ankle = trial.keypoint(f"{shooting_side}_ANKLE")
    # "Hip center" for the generic (non shooting-side-specific) wrist-height
    # and hip-velocity features. We average LEFT_HIP/RIGHT_HIP rather than
    # using the dataset's own MID_HIP keypoint because MID_HIP is entirely
    # absent (all-NaN) in the 2024-08-28 capture session — a difference in
    # pose model across sessions that the dataset README warns about
    # ("Not all sessions were run using the same pose model"). LEFT_HIP and
    # RIGHT_HIP are present in every session, so this keeps that session's
    # trials usable instead of excluding them outright.
    mid_hip = (trial.keypoint("LEFT_HIP") + trial.keypoint("RIGHT_HIP")) / 2.0

    required_names = [
        f"{shooting_side}_SHOULDER", f"{shooting_side}_ELBOW", f"{shooting_side}_WRIST",
        f"{shooting_side}_HIP", f"{shooting_side}_KNEE", f"{shooting_side}_ANKLE",
        "LEFT_HIP", "RIGHT_HIP",
    ]
    required_arrays = [shoulder, elbow, wrist, hip_side, knee, ankle,
                        trial.keypoint("LEFT_HIP"), trial.keypoint("RIGHT_HIP")]
    any_missing_per_frame = np.zeros(trial.n_frames, dtype=bool)
    for arr in required_arrays:
        any_missing_per_frame |= np.isnan(arr[:, 0])
    n_missing_in_window = int(any_missing_per_frame[sl].sum())

    elbow_angle_series = joint_angle(shoulder[sl], elbow[sl], wrist[sl])
    knee_angle_series = joint_angle(hip_side[sl], knee[sl], ankle[sl])
    wrist_z_series = vertical(wrist[sl])
    mid_hip_z_series = vertical(mid_hip[sl])

    def first_last(series: np.ndarray):
        if len(series) == 0:
            return np.nan, np.nan
        return series[0], series[-1]

    elbow_start, elbow_end = first_last(elbow_angle_series)
    knee_start, knee_end = first_last(knee_angle_series)

    missing_fields = []
    if not np.isfinite(elbow_end):
        missing_fields.append("elbow_angle_at_window_end")
    if not np.isfinite(elbow_start):
        missing_fields.append("elbow_angle_at_window_start")
    if not np.isfinite(knee_end):
        missing_fields.append("knee_angle_at_window_end")
    if not np.isfinite(knee_start):
        missing_fields.append("knee_angle_at_window_start")
    if not np.isfinite(wrist_z_series[-1]) or not np.isfinite(mid_hip_z_series[-1]):
        missing_fields.append("wrist_or_hip_height_at_window_end")

    body_size = body_size_ft(shoulder, hip_side, sl)
    if not np.isfinite(body_size) or body_size <= 1e-3:
        missing_fields.append("body_size_reference")

    if missing_fields:
        row = _excluded(trial, shooting_side, "missing_required_keypoint_data: " + ",".join(missing_fields), release_est)
        row.window_start_frame = actual_start
        row.window_end_frame = cutoff_frame
        row.window_duration_sec = actual_duration_sec
        row.n_missing_frames_in_window = n_missing_in_window
        return row

    knee_min = float(np.nanmin(knee_angle_series))

    wrist_vel_series = central_diff_velocity(wrist_z_series, trial.dt)
    hip_vel_series = central_diff_velocity(mid_hip_z_series, trial.dt)
    wrist_vel_mean = float(np.nanmean(wrist_vel_series))
    hip_vel_mean = float(np.nanmean(hip_vel_series))

    wrist_height_rel_hip = float(wrist_z_series[-1] - mid_hip_z_series[-1]) / body_size

    return FeatureRow(
        participant_id=trial.participant_id,
        trial_id=trial.trial_id,
        trial_date=trial.trial_date,
        sampling_rate=trial.sampling_rate,
        shot_id=trial.shot_id,
        result=trial.result,
        label=trial.label,
        shooting_side=shooting_side,
        excluded=False,
        exclusion_reason="",
        release_status=release_est.status,
        release_frame=release_frame,
        window_start_frame=actual_start,
        window_end_frame=cutoff_frame,
        window_duration_sec=actual_duration_sec,
        body_size_ft=body_size,
        n_missing_frames_in_window=n_missing_in_window,
        elbow_angle_end_deg=float(elbow_end),
        knee_angle_end_deg=float(knee_end),
        knee_angle_min_deg=knee_min,
        elbow_angle_change_deg=float(elbow_end - elbow_start),
        knee_angle_change_deg=float(knee_end - knee_start),
        wrist_height_rel_hip_norm=wrist_height_rel_hip,
        wrist_vel_z_mean_norm=wrist_vel_mean / body_size,
        hip_vel_z_mean_norm=hip_vel_mean / body_size,
    )
