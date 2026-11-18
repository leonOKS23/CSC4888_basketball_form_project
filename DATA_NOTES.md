# Dataset inspection

Audit of all 583 downloaded JSON trials at upstream commit
`d510cc472f80efa867def667f6b3d806085cc94d`.
Reproduce with `python scripts/inspect_data.py`. Detailed results are in
`data/processed/trial_audit.csv`, `player_counts.csv`, and `schema_audit.json`.

## Sessions and labels

| Session | Player | Made | Missed | FPS |
|---|---|---:|---:|---:|
| 2024-08-28 | P0001 | 88 | 37 | 30 |
| 2025-12-18 | P0001 | 61 | 27 | 60 |
| 2025-12-18 | P0002 | 55 | 38 | 60 |
| 2025-12-18 | P0003 | 81 | 9 | 60 |
| 2025-12-18 | P0004 | 61 | 28 | 60 |
| 2025-12-18 | P0005 | 43 | 55 | 60 |

Use the 2025 session (458 trials, five players) for the initial experiment. Keep
the 2024 session for later sensitivity work: it adds shots from only P0001, uses
a different frame rate, and has different landmark availability. P0001 is the
same athlete across sessions according to the pinned upstream documentation.

## Actual schema

Every trial has `sampling_rate`, `trial_date`, `participant_id`, `trial_id`,
`result`, `landing_x`, `landing_y`, `entry_angle`, and `tracking`.
The only observed result strings are `made` and `missed`; map them to 1 and 0
when creating model labels. Landing coordinates and entry angle must not become
features.

`tracking` is a list of frames. Each contains `frame`, `time`, and `data`;
`data.ball` is an xyz vector and `data.player` maps landmark names to xyz vectors.
There are 69 observed landmark names, listed in `schema_audit.json`. Names include
`LEFT_WRIST`, `RIGHT_WRIST`, elbows, shoulders, hips, knees, ankles, toes, face
markers, and finger markers. A key existing does not mean its values are usable:
the inspected older-session trial has null finger markers, `MID_HIP`, and `NECK`.
Missing ball coordinates also appear as nonstandard JSON `NaN`, accepted by
Python's JSON loader; NumPy conversion handles both null and NaN as missing.

Z is vertical: nose z is above left big-toe z in at least 89.6% of frames in every
trial. The exceptions deserve quality checks rather than assuming all tracking
is physically valid. The upstream documentation specifies pose/ball xyz in feet;
landing x/y use inches and entry angle uses degrees.

All frame indices are contiguous from zero and all within-trial times increase.
Median time increments are 33 (30 FPS) and 17 (60 FPS), consistent with rounded
milliseconds. Use `sampling_rate` to convert frame offsets to seconds, not the
rounded median increment or a hardcoded 60 FPS.

There is no release-event or handedness field. Trial IDs restart per player and
session. The pinned upstream README explicitly says they are chronological and
participant IDs are consistent across sessions. Actual ID ranges are contiguous
T0001 through each player's trial count. There is no absolute within-session
acquisition timestamp to independently establish ordering. Sort by numeric trial
ID within player/session for the planned chronological evaluation.

## Release detector findings

The original heuristic takes the final frame where the ball is within 0.79 feet
of either wrist. The shared implementation now returns `None` if no valid contact
exists; the notebook's outlier check uses each trial's sampling rate.

- All 583 trials have at least one proximity match; that does not establish a
  correct release estimate.
- Ball coordinates have missing frames in all 125 older-session trials and 65
  newer-session trials (190 total).
- There are 19 timing outliers more than 0.5 seconds from their player/session
  median: 18 in the older session and one in the newer session. Timing alone
  cannot distinguish detector failures from different shot routines.
- **2025 P0001 T0036 is a demonstrated failure.** Its estimate is frame 239, the
  last frame. The distance/height plot shows the ball leaving around frames
  90–100, then returning near the right wrist around frame 200. Selecting the
  last contact captures this later proximity. See
  `data/processed/release_outlier.png`. The precise release remains unverified.

Do not use T0036's current release estimate for pre-release features. Next, revise
the detector to distinguish launch from later proximity, visually check several
makes and misses per player, and retain explicit failure reasons. Shooting hand,
feature windows, feature extraction, model fitting, and evaluation are not yet
implemented. No model performance is claimed.

Source for units, chronology, and cross-session athlete identity:
[pinned SPL documentation](https://github.com/Sport-Performance-Lab/SPL-Open-Data/blob/d510cc472f80efa867def667f6b3d806085cc94d/basketball/freethrow/README.md).
