"""SPL motion-capture trials -> one feature row per shot.

IN:   data/raw/<session>/<player>/*.json   (from scripts/download_data.py)
      Each trial: per-frame 3D positions (feet, z up) of 69 body landmarks + ball,
      plus 'result' ('made'/'missed') and 'sampling_rate'.

OUT:  data/processed/spl_features.csv
          session, participant_id, trial_id, trial_num, label (1=made, 0=missed),
          shooting_side, release_frame, body_release_offset_frames, + FEATURES
      data/processed/spl_excluded.csv
          session, participant_id, trial_id, reason   (shots dropped, and why)

Steps per trial:
  1. Keep only the 12 common-skeleton joints (pose_features.JOINTS).
  2. Release = last ball-to-wrist contact BEFORE the ball's highest point. This
     fixes the notebook detector (last contact anywhere), which picked up the
     ball coming back to the hand after the shot (2025 P0001 T0036).
  3. Shooting hand = per player/session majority of which wrist is closer to the
     ball just before release.
  4. Features from pose_features.extract_features (same code raw video uses).

The ball is only used to find release and hand, never as a feature.
Run:  .venv/bin/python scripts/SPL_to_features.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from pose_features import FEATURES, JOINTS, extract_features, release_from_body

ROOT = Path(__file__).resolve().parents[1]
BALL_CONTACT_FT = 0.79  # one ball diameter
HAND_VOTE_SECONDS = 0.2


def spl_to_skeleton(trial):
    """SPL trial dict -> (skel (n, 12, 3) z-up feet, ball (n, 3), fps)."""
    frames = trial['tracking']
    skel = np.array([[f['data']['player'].get(j) for j in JOINTS] for f in frames], dtype=float)
    ball = np.array([f['data']['ball'] for f in frames], dtype=float)
    return skel, ball, trial['sampling_rate']


def ball_release(skel, ball):
    """Last frame the ball is within one ball-width of a wrist, before the ball apex."""
    dist = np.fmin(
        np.linalg.norm(ball - skel[:, JOINTS.index('LEFT_WRIST')], axis=1),
        np.linalg.norm(ball - skel[:, JOINTS.index('RIGHT_WRIST')], axis=1),
    )
    if not np.isfinite(ball[:, 2]).any():
        return None
    apex = int(np.nanargmax(ball[:, 2]))
    contact = np.flatnonzero(np.isfinite(dist[:apex]) & (dist[:apex] <= BALL_CONTACT_FT))
    return int(contact[-1]) if len(contact) else None


def hand_vote(skel, ball, release, fps):
    """'LEFT'/'RIGHT': which wrist is closer to the ball just before release."""
    lo = max(0, release - round(HAND_VOTE_SECONDS * fps))
    seg = slice(lo, release + 1)
    left = np.nanmean(np.linalg.norm(ball[seg] - skel[seg, JOINTS.index('LEFT_WRIST')], axis=1))
    right = np.nanmean(np.linalg.norm(ball[seg] - skel[seg, JOINTS.index('RIGHT_WRIST')], axis=1))
    if not (np.isfinite(left) and np.isfinite(right)):
        return None
    return 'LEFT' if left < right else 'RIGHT'


def main():
    paths = sorted((ROOT / 'data/raw').glob('*/*/*.json'))
    if not paths:
        raise FileNotFoundError('Run scripts/download_data.py first')

    # Pass 1: release + hand vote per trial (keep skeletons, they're small).
    trials = []
    for path in paths:
        t = json.loads(path.read_text())
        skel, ball, fps = spl_to_skeleton(t)
        release = ball_release(skel, ball)
        trials.append(dict(
            session=path.parts[-3], participant_id=t['participant_id'], trial_id=t['trial_id'],
            label=int(t['result'] == 'made'), skel=skel, fps=fps, release=release,
            vote=hand_vote(skel, ball, release, fps) if release is not None else None,
        ))

    # Pass 2: one shooting side per player/session by majority vote.
    votes = {}
    for t in trials:
        if t['vote']:
            votes.setdefault((t['session'], t['participant_id']), Counter())[t['vote']] += 1
    side_of = {}
    print('Shooting hand (majority vote):')
    for key, c in sorted(votes.items()):
        side, n = c.most_common(1)[0]
        side_of[key] = side
        print(f'  {key[0]} {key[1]}: {side}  ({n}/{sum(c.values())} = {n / sum(c.values()):.0%})')

    # Pass 3: features.
    rows, excluded = [], []
    for t in trials:
        ids = dict(session=t['session'], participant_id=t['participant_id'], trial_id=t['trial_id'])
        side = side_of.get((t['session'], t['participant_id']))
        feats, reason = extract_features(t['skel'], t['fps'], side, t['release'])
        if feats is None:
            excluded.append(dict(**ids, reason=reason))
            continue
        body = release_from_body(t['skel'], t['fps'], side)
        rows.append(dict(
            **ids, trial_num=int(t['trial_id'].lstrip('T')), label=t['label'],
            shooting_side=side, release_frame=t['release'],
            body_release_offset_frames=None if body is None else body - t['release'],
            **feats,
        ))

    out = ROOT / 'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'spl_features.csv', index=False)
    ex = pd.DataFrame(excluded, columns=['session', 'participant_id', 'trial_id', 'reason'])
    ex.to_csv(out / 'spl_excluded.csv', index=False)

    print(f'\nKept {len(df)} / {len(trials)} trials -> {out / "spl_features.csv"}')
    if len(ex):
        print('Excluded:', ex.groupby(['session', 'reason']).size().to_dict())
    off = df.body_release_offset_frames.dropna()
    print(f'Body-only release vs ball release (frames): median {off.median():+.0f}, '
          f'|offset| <= 3 frames for {(off.abs() <= 3).mean():.0%} of shots')
    print(df.groupby('label')[FEATURES].median().T.round(2).to_string())


if __name__ == '__main__':
    main()
