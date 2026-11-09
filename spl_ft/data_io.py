"""Loading SPL Open Data free-throw trial JSON files.

A trial file looks like (confirmed by directly reading real files, not
assumed from docs):

    {
      "sampling_rate": 30,                 # or 60
      "trial_date": "2024-08-28",
      "participant_id": "P0001",
      "trial_id": "T0001",
      "result": "made" | "missed",
      "landing_x": <float, inches>,         # outcome-adjacent; NOT a feature
      "landing_y": <float, inches>,         # outcome-adjacent; NOT a feature
      "entry_angle": <float, degrees>,      # outcome-adjacent; NOT a feature
      "tracking": [
        {
          "frame": 0,
          "time": 0,                        # milliseconds
          "data": {
            "ball": [x, y, z] | [NaN, NaN, NaN],
            "player": {"NOSE": [x,y,z], "LEFT_SHOULDER": [x,y,z], ...}
          }
        },
        ...
      ]
    }

All positions are in feet. ``z`` is vertical (confirmed empirically: toe
markers sit near z=0, head markers near z=5-6 ft).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import config


@dataclass
class Trial:
    """A single free-throw trial, with keypoint tracks as numpy arrays."""

    path: Path
    participant_id: str
    trial_id: str
    trial_date: str
    sampling_rate: float
    result: str  # "made" or "missed"
    landing_x: float
    landing_y: float
    entry_angle: float
    n_frames: int
    time_ms: np.ndarray  # (n_frames,)
    ball_xyz: np.ndarray  # (n_frames, 3), NaN where not tracked
    player_xyz: Dict[str, np.ndarray] = field(default_factory=dict)  # name -> (n_frames, 3)

    @property
    def shot_id(self) -> str:
        return f"{self.participant_id}_{self.trial_id}"

    @property
    def label(self) -> int:
        """1 = made, 0 = missed."""
        return 1 if self.result == config.RESULT_MADE else 0

    @property
    def dt(self) -> float:
        return 1.0 / float(self.sampling_rate)

    def keypoint(self, name: str) -> np.ndarray:
        """Return the (n_frames, 3) array for a keypoint, or all-NaN if absent."""
        arr = self.player_xyz.get(name)
        if arr is None:
            return np.full((self.n_frames, 3), np.nan)
        return arr


def _xyz_to_float(value: Optional[List]) -> List[float]:
    if value is None:
        return [np.nan, np.nan, np.nan]
    return [np.nan if v is None else float(v) for v in value]


def load_trial(path: Path) -> Trial:
    """Load one trial JSON file into a :class:`Trial`."""
    with open(path, "r") as f:
        raw = json.load(f)

    tracking = raw["tracking"]
    n_frames = len(tracking)
    time_ms = np.array([fr["time"] for fr in tracking], dtype=float)
    ball_xyz = np.array([_xyz_to_float(fr["data"]["ball"]) for fr in tracking], dtype=float)

    # Collect every keypoint name that appears (not all trials necessarily
    # have identical keypoint sets across capture sessions, per the dataset
    # README's note on differing pose models).
    all_names = set()
    for fr in tracking:
        all_names.update(fr["data"]["player"].keys())

    player_xyz: Dict[str, np.ndarray] = {}
    for name in all_names:
        col = np.full((n_frames, 3), np.nan)
        for i, fr in enumerate(tracking):
            v = fr["data"]["player"].get(name)
            if v is not None:
                col[i] = _xyz_to_float(v)
        player_xyz[name] = col

    return Trial(
        path=path,
        participant_id=raw["participant_id"],
        trial_id=raw["trial_id"],
        trial_date=raw["trial_date"],
        sampling_rate=float(raw["sampling_rate"]),
        result=raw["result"],
        landing_x=raw.get("landing_x", np.nan),
        landing_y=raw.get("landing_y", np.nan),
        entry_angle=raw.get("entry_angle", np.nan),
        n_frames=n_frames,
        time_ms=time_ms,
        ball_xyz=ball_xyz,
        player_xyz=player_xyz,
    )


def list_trial_paths(data_dir: Path = config.FREETHROW_DATA_DIR) -> List[Path]:
    """All trial JSON files, sorted by (participant_id, trial_id) using the
    filename convention ``BB_FT_<participant>_<trial>.json`` so that trial
    order reflects chronological order within a participant (per the
    dataset README: "trial IDs are simply increasing integers ... in
    chronological order").
    """
    paths = sorted(data_dir.glob("**/*.json"))
    return paths


def participant_id_from_path(path: Path) -> str:
    # Filenames look like BB_FT_P0001_T0001.json
    stem = path.stem
    parts = stem.split("_")
    return parts[2]


def trial_sort_key(path: Path):
    """Sort key giving chronological order *within* a participant.

    IMPORTANT: trial IDs (``T0001``, ``T0002``, ...) reset at the start of
    each capture session (confirmed empirically: participant P0001 has a
    ``T0001`` in both the 2024-08-28 session and the 2025-12-18 session).
    The dataset README only guarantees trial IDs are increasing *within a
    single session*. To get a session-spanning chronological order for a
    participant we therefore sort first by session date (the parent
    folder name, e.g. ``2024-08-28``), then by trial number within that
    session.
    """
    stem = path.stem  # BB_FT_P0001_T0001
    parts = stem.split("_")
    participant_id = parts[2]
    trial_num = int(parts[3].lstrip("T"))
    session_date = path.parent.parent.name  # e.g. "2024-08-28"
    return (participant_id, session_date, trial_num)


def list_trials_grouped_by_participant(
    data_dir: Path = config.FREETHROW_DATA_DIR,
) -> Dict[str, List[Path]]:
    paths = list_trial_paths(data_dir)
    paths.sort(key=trial_sort_key)
    grouped: Dict[str, List[Path]] = {}
    for p in paths:
        pid = trial_sort_key(p)[0]
        grouped.setdefault(pid, []).append(p)
    return grouped
