"""Per-player shooting-side configuration.

The dataset has no handedness field, so we derive a per-player shooting
side as the majority vote of the per-trial shooting-hand estimate from
:mod:`spl_ft.release` (computed across all of that player's usable
trials), and apply that single side consistently to every trial from
that player for feature extraction (elbow/knee angle "shooting side").

Why majority vote per player rather than per-trial hand assignment for
features: in several trials the guide hand and shooting hand separate
from the ball within 1-2 frames of each other (near-simultaneous
release), which makes the single-trial hand call noisy. The per-player
vote is overwhelmingly one-sided for every player in this dataset (see
``outputs/dataset_inspection/handedness.csv``), which is itself evidence
that it reflects a real, stable trait (true handedness) rather than
frame-level noise.

HOW TO VERIFY / OVERRIDE
------------------------
Run ``python scripts/01_inspect_dataset.py`` and check
``outputs/dataset_inspection/handedness.csv`` — it reports, per player,
the fraction of trials voting RIGHT vs LEFT. If a player's vote is not
heavily lopsided (e.g. close to 50/50), treat the automatic call with
suspicion, look at a few of that player's trials in
``outputs/figures/release_checks/``, and set an explicit override below
in ``MANUAL_SHOOTING_SIDE_OVERRIDE``.
"""
from __future__ import annotations

import collections
from pathlib import Path
from typing import Dict, List

from .data_io import Trial, load_trial
from .release import estimate_release

# If you have independently verified a player's shooting hand (e.g. by
# watching the animated trial) and it disagrees with the automatic vote,
# add an override here, e.g. {"P0005": "LEFT"}.
MANUAL_SHOOTING_SIDE_OVERRIDE: Dict[str, str] = {}


def per_trial_hand_votes(trial_paths: List[Path]) -> Dict[str, collections.Counter]:
    """Returns {participant_id: Counter({"RIGHT": n, "LEFT": m})} using only
    trials where release estimation succeeded.
    """
    votes: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for p in trial_paths:
        trial = load_trial(p)
        est = estimate_release(trial)
        if est.status == "ok" and est.shooting_hand is not None:
            votes[trial.participant_id][est.shooting_hand] += 1
    return votes


def majority_shooting_side(votes: collections.Counter) -> str:
    if not votes:
        return "RIGHT"  # arbitrary fallback; player will have no usable trials anyway
    return votes.most_common(1)[0][0]


def build_player_shooting_side(trial_paths: List[Path]) -> Dict[str, dict]:
    """Returns {participant_id: {"side": "RIGHT"/"LEFT", "votes": {...},
    "fraction_majority": float, "overridden": bool}}.
    """
    votes_by_player = per_trial_hand_votes(trial_paths)
    result = {}
    for pid, votes in votes_by_player.items():
        total = sum(votes.values())
        side = majority_shooting_side(votes)
        frac = votes[side] / total if total else float("nan")
        overridden = pid in MANUAL_SHOOTING_SIDE_OVERRIDE
        if overridden:
            side = MANUAL_SHOOTING_SIDE_OVERRIDE[pid]
        result[pid] = {
            "side": side,
            "votes": dict(votes),
            "n_votes": total,
            "fraction_majority": frac,
            "overridden": overridden,
        }
    return result
