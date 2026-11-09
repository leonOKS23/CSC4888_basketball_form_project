"""Release-frame and shooting-hand estimation.

The SPL Open Data free-throw dataset does **not** provide a release event
or a shooting-handedness field (confirmed by inspecting every trial's
top-level keys — see ``outputs/dataset_inspection/``). Both are estimated
here from ball and wrist positions, and both estimates are validated
empirically (see ``PROJECT_LOG.md`` and
``outputs/figures/release_checks/``) rather than assumed to be correct.

Algorithm (see PROJECT_LOG.md "Release detection" for the full narrative
of how this was developed and validated):

1. The ball position is only tracked while visible to the capture
   volume, so ``ball_xyz`` has NaN gaps. We find contiguous blocks of
   valid ball frames.
2. We walk blocks from *last* to *first* and pick the last block that
   contains at least one frame where the ball is within
   ``CONTACT_DISTANCE_THRESHOLD_FT`` of a wrist. This matters because in
   some trials the very last visible-ball block is the ball reappearing
   near the hoop (far from the shooter) after an occlusion gap right
   around the shot — using that block naively would miss the release
   entirely.
3. Within the chosen block, for each wrist we find the *last* frame where
   the ball is within the contact threshold ("last contact"). The
   shooting hand is whichever wrist loses contact **last** — this
   distinguishes it from the non-shooting/guide hand, which typically
   releases the ball slightly earlier.
4. The release frame is one frame after the shooting hand's last-contact
   frame.

This heuristic was validated by: (a) checking that the per-trial
shooting-hand vote is unanimous or near-unanimous *within* each player
(a real, consistent biomechanical signal should look like this; noise
would not), and (b) visually inspecting ball/wrist distance-vs-time plots
with the detected release frame marked, across players and across
made/missed trials.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from . import config
from .data_io import Trial


@dataclass
class ReleaseEstimate:
    status: str  # "ok", "never_in_contact", "no_valid_ball_data"
    shooting_hand: Optional[str] = None  # "RIGHT" or "LEFT"
    block_start: Optional[int] = None
    block_end: Optional[int] = None
    last_contact_frame: Optional[int] = None
    release_frame: Optional[int] = None
    ball_z_at_release: Optional[float] = None
    note: str = ""


def _contiguous_valid_blocks(mask: np.ndarray) -> List[Tuple[int, int]]:
    blocks = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            blocks.append((i, j - 1))
            i = j
        else:
            i += 1
    return blocks


def estimate_release(trial: Trial) -> ReleaseEstimate:
    ball = trial.ball_xyz
    valid = ~np.isnan(ball[:, 0])
    if valid.sum() == 0:
        return ReleaseEstimate(status="no_valid_ball_data", note="ball never tracked in this trial")

    blocks = _contiguous_valid_blocks(valid)

    right_wrist = trial.keypoint("RIGHT_WRIST")
    left_wrist = trial.keypoint("LEFT_WRIST")
    dist_r = np.linalg.norm(ball - right_wrist, axis=1)
    dist_l = np.linalg.norm(ball - left_wrist, axis=1)

    T = config.CONTACT_DISTANCE_THRESHOLD_FT

    chosen = None
    for (bs, be) in reversed(blocks):
        seg_min = np.minimum(dist_r[bs : be + 1], dist_l[bs : be + 1])
        if np.nanmin(seg_min) <= T:
            chosen = (bs, be)
            break

    if chosen is None:
        return ReleaseEstimate(
            status="never_in_contact",
            block_start=blocks[-1][0],
            block_end=blocks[-1][1],
            note="ball never came within contact threshold of either wrist in any tracked block",
        )

    bs, be = chosen

    def last_contact(dist: np.ndarray) -> Optional[int]:
        idx = np.where(dist[bs : be + 1] <= T)[0]
        if len(idx) == 0:
            return None
        return bs + int(idx[-1])

    lc_r = last_contact(dist_r)
    lc_l = last_contact(dist_l)

    if lc_r is None and lc_l is None:
        return ReleaseEstimate(
            status="never_in_contact", block_start=bs, block_end=be,
            note="no wrist ever within contact threshold in chosen block (unexpected)",
        )
    if lc_r is None:
        hand, lc = "LEFT", lc_l
    elif lc_l is None:
        hand, lc = "RIGHT", lc_r
    else:
        hand, lc = ("RIGHT", lc_r) if lc_r >= lc_l else ("LEFT", lc_l)

    release_frame = lc + 1 if lc + 1 <= be else lc

    return ReleaseEstimate(
        status="ok",
        shooting_hand=hand,
        block_start=bs,
        block_end=be,
        last_contact_frame=lc,
        release_frame=release_frame,
        ball_z_at_release=float(ball[release_frame, 2]),
    )
