"""Shared, source-independent feature code.

Both SPL_to_features.py and raw_to_features.py convert their data into the same
COMMON SKELETON, then call the functions here. Nothing in this file knows where
the data came from, which is what keeps the feature set identical across sources.

Common skeleton
---------------
    skel : float array, shape (n_frames, 12, 3)
        Joint order = JOINTS below. Axes: x/y horizontal, z UP. Any length unit
        (SPL is feet, MediaPipe world coords are meters) because every distance
        feature is divided by torso length. Missing values are NaN.
    fps  : frames per second of that clip.
    side : 'LEFT' or 'RIGHT', the shooting hand.

All features are translation-invariant (they only use differences between
joints, never absolute position), so MediaPipe's hip-centered world coordinates
give the same features as SPL's room coordinates.
"""
import numpy as np

JOINTS = [
    'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP',
    'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
]
J = {name: i for i, name in enumerate(JOINTS)}

WINDOW_SECONDS = 0.5   # length of the pre-release window
BUFFER_SECONDS = 0.035  # gap before release so release motion is excluded (~2 frames @ 60 FPS)

FEATURES = [
    'elbow_angle_end', 'knee_angle_end', 'knee_angle_min',
    'elbow_angle_change', 'knee_angle_change',
    'wrist_above_hip_end', 'wrist_above_hip_vz_mean', 'hip_above_ankle_vz_mean',
]


def joint(skel, side, name):
    return skel[:, J[f'{side}_{name}']]


def angle(a, b, c):
    """Angle at b (degrees) for each frame, from points a-b-c of shape (n, 3)."""
    u, v = a - b, c - b
    cos = np.sum(u * v, axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def torso_length(skel):
    """Per-frame distance from mid-shoulder to mid-hip; the body-size unit."""
    mid_sh = (skel[:, J['LEFT_SHOULDER']] + skel[:, J['RIGHT_SHOULDER']]) / 2
    mid_hip = (skel[:, J['LEFT_HIP']] + skel[:, J['RIGHT_HIP']]) / 2
    return np.linalg.norm(mid_sh - mid_hip, axis=1)


def shooting_side_from_body(skel, fps):
    """Ball-free handedness guess for one clip: the arm whose elbow straightens most.

    Checked on SPL: matches the ball-based hand for 100% of P0001-P0004 shots.
    (Wrist height does NOT work: the guide-hand wrist is often higher at release.)
    Take a majority over a player's clips.
    """
    best = {}
    for side in ('LEFT', 'RIGHT'):
        el = angle(joint(skel, side, 'SHOULDER'), joint(skel, side, 'ELBOW'), joint(skel, side, 'WRIST'))
        best[side] = np.nanmax(el) if np.isfinite(el).any() else -np.inf
    return max(best, key=best.get)


def release_from_body(skel, fps, side):
    """Ball-free release estimate: frame of peak upward speed of the shooting wrist.

    Used for raw video (no ball). SPL_to_features.py reports how far this lands
    from the ball-based SPL release so you can see how good it is.
    """
    z = joint(skel, side, 'WRIST')[:, 2]
    vz = np.gradient(z) * fps
    if not np.isfinite(vz).any():
        return None
    return int(np.nanargmax(vz))


def window(n_frames, release, fps):
    """(start, end) frame indices, end inclusive, or (None, reason) if it doesn't fit."""
    if release is None:
        return None, 'no_release'
    end = release - max(1, round(BUFFER_SECONDS * fps))
    start = end - round(WINDOW_SECONDS * fps)
    if start < 0:
        return None, 'window_before_clip_start'
    if end >= n_frames:
        return None, 'release_past_clip_end'
    return (start, end), None


def extract_features(skel, fps, side, release):
    """One shot -> (dict of FEATURES, None) or (None, exclusion_reason)."""
    win, reason = window(len(skel), release, fps)
    if win is None:
        return None, reason
    start, end = win
    w = skel[start:end + 1]
    if not np.isfinite(w).all():
        return None, 'missing_keypoints_in_window'

    elbow = angle(joint(w, side, 'SHOULDER'), joint(w, side, 'ELBOW'), joint(w, side, 'WRIST'))
    knee = angle(joint(w, side, 'HIP'), joint(w, side, 'KNEE'), joint(w, side, 'ANKLE'))
    size = torso_length(w).mean()
    if not size > 0:
        return None, 'bad_torso_length'

    wrist_rel_hip = (joint(w, side, 'WRIST')[:, 2] - joint(w, side, 'HIP')[:, 2]) / size
    mid_hip_z = (w[:, J['LEFT_HIP'], 2] + w[:, J['RIGHT_HIP'], 2]) / 2
    mid_ankle_z = (w[:, J['LEFT_ANKLE'], 2] + w[:, J['RIGHT_ANKLE'], 2]) / 2
    hip_rel_ankle = (mid_hip_z - mid_ankle_z) / size
    duration = (len(w) - 1) / fps

    return dict(
        elbow_angle_end=elbow[-1],
        knee_angle_end=knee[-1],
        knee_angle_min=knee.min(),
        elbow_angle_change=elbow[-1] - elbow[0],
        knee_angle_change=knee[-1] - knee[0],
        wrist_above_hip_end=wrist_rel_hip[-1],
        # Mean vertical velocities = net change / time, in torso lengths per second.
        # Measured relative to hip / ankles (not absolute) so hip-centered MediaPipe
        # coordinates give the same numbers.
        wrist_above_hip_vz_mean=(wrist_rel_hip[-1] - wrist_rel_hip[0]) / duration,
        hip_above_ankle_vz_mean=(hip_rel_ankle[-1] - hip_rel_ankle[0]) / duration,
    ), None
