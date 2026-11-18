# Check-in 1: first SPL make/miss baseline

**Result:** the first 8 features carry only a weak signal. For players the model has seen,
balanced accuracy is 0.63; for unseen players it is about chance (0.50–0.55). Accuracy never
beats always guessing "made".

## Setup

- Data: SPL 2025-12-18 session. 458 free throws from 5 players, 66% made, 60 FPS.
  No shots were excluded.
- Features: 8 values per shot, from a 0.5 s window that ends just before release:
  - elbow angle at window end, change over window
  - knee angle at window end, minimum, change over window
  - wrist height above hip at window end, and its mean vertical velocity
  - mean vertical velocity of hip height above the ankles

  Heights and velocities are divided by torso length. Only the shooting arm and leg are used.
- Models: always-"made" baseline, logistic regression, random forest. Settings were chosen
  by grid search on training data only, with folds grouped by player.
- Balanced accuracy is the average of the correct rates for makes and for misses, so it
  isn't inflated by the 66% make rate. 0.50 means chance.

## Results

**Eval A: known players, later shots.** Each player's first 80% of trials are used for
training and the last 20% for testing (89 test shots, 66% made).

| Model | Accuracy | Balanced acc | F1 (made) | Confusion [[TN FP] [FN TP]] |
|---|---:|---:|---:|---|
| Always "made" | 0.663 | 0.500 | 0.797 | [[0 30] [0 59]] |
| Logistic regression (C=0.1) | 0.629 | **0.630** | 0.692 | [[19 11] [22 37]] |
| Random forest (depth 4) | 0.607 | 0.621 | 0.660 | [[20 10] [25 34]] |

**Eval B: unseen players.** Each player is held out in turn, and results are pooled over all
458 shots.

| Model | Accuracy | Balanced acc | F1 (made) | Confusion [[TN FP] [FN TP]] |
|---|---:|---:|---:|---|
| Always "made" | 0.657 | 0.500 | 0.793 | [[0 157] [0 301]] |
| Logistic regression | 0.537 | 0.549 | 0.592 | [[92 65] [147 154]] |
| Random forest | 0.596 | 0.499 | 0.724 | [[30 127] [58 243]] |

Balanced accuracy for each held-out player:

| Player | Shots | Make rate | Logistic regression | Random forest |
|---|---:|---:|---:|---:|
| P0001 | 88 | 0.69 | 0.56 | 0.50 |
| P0002 | 93 | 0.59 | 0.50 | 0.50 |
| P0003 | 90 | 0.90 | 0.64 | 0.50 |
| P0004 | 89 | 0.69 | 0.49 | 0.50 |
| P0005 | 98 | 0.44 | 0.53 | 0.49 |

Eval A has only 89 test shots, so its numbers could be off by roughly ±0.1.

## Files

All new code is in `scripts/`. Run the steps in this order:

```sh
.venv/bin/python scripts/SPL_to_features.py   # about 10 s
.venv/bin/python scripts/train_model.py       # about 1 min
```

| File | In | Out | What it does |
|---|---|---|---|
| `pose_features.py` | 12-joint skeleton `(frames, 12, 3)`, z up, any units, plus FPS and shooting side | Feature dict, or an exclusion reason | Shared feature code that both data sources use. It defines the common skeleton, the pre-release window, the 8 features and ball-free rules for release and handedness. |
| `SPL_to_features.py` | `data/raw/*/*/*.json` | `data/processed/spl_features.csv`, `spl_excluded.csv` | Converts SPL trials to the common skeleton. It finds release from the ball, picks the shooting hand by majority vote per player and computes the features. |
| `raw_to_features.py` | Video files, a MediaPipe `.task` model file and an optional labels CSV | `data/processed/raw_features.csv`, `raw_excluded.csv` | Runs MediaPipe to get estimated 3D landmarks, converts them to the common skeleton and runs the same feature code. Not yet run on real video. |
| `train_model.py` | `spl_features.csv` | Printed metrics | Trains and evaluates the 3 models for Eval A and Eval B. |
| `download_data.py`, `inspect_data.py`, `plot_release_outlier.py` | — | — | Earlier work: downloading the data, auditing the schema and diagnosing the release-detection failure. See `DATA_NOTES.md`. |

## Design choices

- **One feature set for both sources.** Each source has its own adapter that converts it to
  the common skeleton. Everything after that is shared code. All features are angles or are
  measured relative to the body and divided by torso length, so they don't depend on units
  (feet vs meters) or on MediaPipe's hip-centered coordinates. As a check, SPL data rewritten
  into MediaPipe's format produced identical features (difference about 1e-14).
- **Release (SPL):** the last frame the ball is within one ball width of a wrist, before the
  ball's highest point. This fixes the original detector's failure on P0001 T0036, where
  release moves from frame 239 to frame 94.
- **Release (video, no ball):** the moment the shooting wrist is moving up fastest. On SPL
  this lands a median of 4 frames (about 0.07 s) before the ball-based release, and within
  3 frames only 46% of the time.
- **Shooting hand:** majority of which wrist is closest to the ball just before release.
  P0001–P0004 are right-handed (84–100% agreement, confirmed by elbow extension).

## Caveats and next steps

- **P0005's shooting hand is unresolved.** Ball proximity says right in 57% of trials, while
  elbow extension says left in 64%. I used right. If that's wrong, P0005's features describe
  the guide arm.
- Release and handedness are estimates. They haven't been checked against ground truth.
- With only 5 players, Eval B tests generalization to new people very weakly.
- The 2024 session is extracted but not used.
- Next steps:
  1. Resolve P0005's shooting hand.
  2. Plot each feature for makes vs misses and by player (ticket 7).
  3. Try richer features, such as the full time series of the shooting arm instead of only
     the window's start and end.
  4. Install `mediapipe` and run `raw_to_features.py` on a test video.
