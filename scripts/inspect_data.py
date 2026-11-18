"""Audit real trials without holding the full dataset in memory."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def release_frame(trial):
    """Original proximity heuristic, with explicit failure for absent contact."""
    frames = trial['tracking']
    ball = np.asarray([f['data']['ball'] for f in frames], dtype=float)
    distances = [np.linalg.norm(ball - np.asarray([
        f['data']['player'][side + '_WRIST'] for f in frames
    ], dtype=float), axis=1) for side in ('LEFT', 'RIGHT')]
    gap = np.fmin(*distances)
    contact = np.flatnonzero(np.isfinite(gap) & (gap <= 0.79))
    return int(contact[-1]) if len(contact) else None


def audit():
    rows, schemas, keypoints = [], Counter(), Counter()
    for path in sorted((ROOT / 'data/raw').glob('*/*/*.json')):
        trial = json.loads(path.read_text())
        frames = trial['tracking']
        schemas[tuple(sorted(trial))] += 1
        for frame in frames:
            keypoints.update(frame['data']['player'].keys())
        ball = np.asarray([f['data']['ball'] for f in frames], float)
        nose = np.asarray([f['data']['player']['NOSE'] for f in frames], float)
        toe = np.asarray([f['data']['player']['LEFT_BIG_TOE'] for f in frames], float)
        release = release_frame(trial)
        times = np.asarray([f['time'] for f in frames], float)
        fps = trial['sampling_rate']
        after = ball[release + 1:release + 1 + max(1, round(fps * .1))] if release is not None else []
        rows.append(dict(
            path=str(path.relative_to(ROOT)), session=path.parts[-3],
            participant_id=trial['participant_id'], trial_id=trial['trial_id'],
            result=trial['result'], sampling_rate=fps, frames=len(frames),
            time_step_median=float(np.median(np.diff(times))),
            time_monotonic=bool(np.all(np.diff(times) > 0)),
            frames_contiguous=[f['frame'] for f in frames] == list(range(len(frames))),
            ball_missing_frames=int((~np.isfinite(ball).all(axis=1)).sum()),
            nose_above_toe_z_fraction=float(np.mean(nose[:, 2] > toe[:, 2])),
            release_frame=release,
            release_seconds=release / fps if release is not None else np.nan,
            release_at_clip_end=release == len(frames) - 1,
            post_release_unverified=bool(len(after) < max(1, round(fps * .1)) or not np.isfinite(after).all()),
        ))
    df = pd.DataFrame(rows)
    expected = len(json.loads((ROOT / 'scripts/dataset_manifest.json').read_text())['files'])
    if len(df) != expected:
        raise ValueError(f'Download incomplete: {len(df)} / {expected}')
    median = df.groupby(['session', 'participant_id']).release_seconds.transform('median')
    df['release_timing_outlier'] = (df.release_seconds - median).abs() > .5
    out = ROOT / 'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'trial_audit.csv', index=False)
    counts = df.groupby(['session', 'participant_id', 'result']).size().unstack(fill_value=0)
    counts.to_csv(out / 'player_counts.csv')
    summary = dict(
        trials=len(df), schemas=[{'fields': list(k), 'trials': v} for k, v in schemas.items()],
        keypoint_names=sorted(keypoints),
        frame_rates=df.groupby('session').sampling_rate.unique().apply(list).to_dict(),
        no_release=int(df.release_frame.isna().sum()),
        release_timing_outliers=int(df.release_timing_outlier.sum()),
        trials_with_ball_gaps=int((df.ball_missing_frames > 0).sum()),
        post_release_unverified=int(df.post_release_unverified.sum()),
        all_times_monotonic=bool(df.time_monotonic.all()),
        all_frames_contiguous=bool(df.frames_contiguous.all()),
        min_nose_above_toe_z_fraction=float(df.nose_above_toe_z_fraction.min()),
    )
    (out / 'schema_audit.json').write_text(json.dumps(summary, indent=2, default=int) + '\n')
    print(counts.to_string())
    print(json.dumps({k: v for k, v in summary.items() if k != 'keypoint_names'}, indent=2, default=int))
    return df


if __name__ == '__main__':
    audit()
