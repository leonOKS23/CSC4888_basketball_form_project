"""Generic 3D biomechanics helpers: joint angles, velocities, body-size scale.

Nothing here is specific to "release" detection or feature selection —
see :mod:`spl_ft.release` and :mod:`spl_ft.features` for that.
"""
from __future__ import annotations

import numpy as np

from .config import VERTICAL_AXIS


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at vertex ``b`` formed by points a-b-c, in degrees.

    Works elementwise on arrays of shape (..., 3). Returns NaN where any
    input is NaN or where either segment has (numerically) zero length.
    """
    ba = a - b
    bc = c - b
    ba_norm = np.linalg.norm(ba, axis=-1)
    bc_norm = np.linalg.norm(bc, axis=-1)
    denom = ba_norm * bc_norm
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_angle = np.sum(ba * bc, axis=-1) / denom
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    bad = (denom == 0) | ~np.isfinite(denom)
    angle = np.where(bad, np.nan, angle)
    return angle


def vertical(xyz: np.ndarray) -> np.ndarray:
    """Extract the vertical (z) coordinate from an (..., 3) array."""
    return xyz[..., VERTICAL_AXIS]


def central_diff_velocity(position: np.ndarray, dt: float) -> np.ndarray:
    """Velocity via ``numpy.gradient`` (central differences, same length as
    input), matching the approach used in the dataset's own example
    notebook (``usage.ipynb``) so our numbers are comparable to what the
    dataset authors demonstrate.
    """
    return np.gradient(position, dt, axis=0)


def body_size_ft(
    shoulder_shoot: np.ndarray,
    hip_shoot: np.ndarray,
    frame_idx: int | slice,
) -> float:
    """A simple, always-available body-size scale: 3D shoulder-to-hip
    ("torso") distance on the shooting side, averaged over the given
    frame(s). Used to normalize velocity/height features so they are
    roughly comparable across players of different sizes.

    We deliberately avoid a full standing-height estimate because it
    requires ankle/head markers to both be clean at the same instant,
    which is a stronger requirement than we need for a simple scale
    factor.
    """
    diff = shoulder_shoot[frame_idx] - hip_shoot[frame_idx]
    dist = np.linalg.norm(diff, axis=-1)
    return float(np.nanmean(dist))
