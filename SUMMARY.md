# Summary

Everything below actually ran, on the actual downloaded SPL Open Data dataset, in this repo, on my machine. Commands are in `README.md`, raw outputs are under `outputs/`.

## The dataset vs. what the docs say

The published README is mostly right but not 100% current for what's actually in the repo:

- It mentions a `participant_information.json` file that doesn't exist. So no demographic info, no handedness lookup, nothing like that.
- `animate.py` and `usage.ipynb` use an older keypoint naming (`R_HIP` style) that doesn't match the actual trial files (`RIGHT_HIP`, `LEFT_HIP`, etc). I went and read the real files directly instead of trusting either doc.
- One thing the README does correctly warn about, and that actually bit me: not every session used the same pose model. `MID_HIP` exists in the 2025-12-18 session but is completely missing (all NaN) in 2024-08-28. I worked around it by using `(LEFT_HIP + RIGHT_HIP) / 2` instead.

Things I checked myself rather than assumed: 5 participants, 583 trials, 389 makes / 194 misses, feet as the unit, `z` is vertical, and trial IDs reset at the start of each session (so for one participant's trials to be in actual chronological order across sessions, I had to sort by `(session_date, trial_number)`, not just trial number). Full detail in `outputs/dataset_inspection/summary.md`.

## Usable makes/misses per player

| Player | Usable shots | Made | Missed | Make rate |
|---|---|---|---|---|
| P0001 | 213 | 149 | 64 | 0.700 |
| P0002 | 93  | 55  | 38 | 0.591 |
| P0003 | 90  | 81  | 9  | 0.900 |
| P0004 | 89  | 61  | 28 | 0.685 |
| P0005 | 98  | 43  | 55 | 0.439 |

All 583 trials ended up usable, 0 excluded. (Before I fixed the `MID_HIP` issue above, that bug alone was excluding all 125 trials from the 2024-08-28 session — see `outputs/dataset_inspection/feature_extraction_summary.md` for the current, empty, exclusion list.) Zero excluded doesn't mean zero uncertainty though — see limitations below.

## Release detection: what I tried and how I checked it

No release-event field exists, so I estimate it from 3D ball-to-wrist distance (`spl_ft/release.py`):

1. Take the last contiguous block of valid ball frames that actually gets close to a hand at some point. This mattered — in roughly 40 trials the very last visible-ball block was the ball reappearing near the hoop, feet away from the shooter, after a tracking gap during the shot. Using "the last block" blindly would've put the release point in the wrong place.
2. Shooting hand = whichever wrist holds contact *longest*, not just whichever is closest overall — this is what separates the shooting hand from the guide hand, since the guide hand usually lets go first.
3. Release = one frame after that hand's last contact frame.

How I checked it:
- The per-player handedness vote is unanimous for 4 of 5 players and 81.6% for the fifth (P0005) — I looked at the disagreeing P0005 trials and the two hands were separating within 1-2 frames of each other, basically a tie, not a sign the detector is broken.
- I visually checked distance/height/angle plots for 20 trials (4 per player, mix of makes and misses), in `outputs/figures/release_checks/`. In every one I looked at, the detected release lines up with the ball-to-wrist distance shooting up right as the ball leaves the hand, and the pre-release window sits over the loading/extension phase like it should, not over the follow-through.

Still a heuristic, not ground truth — there's nothing in the dataset to score it against directly.

## Features

One row per shot, 8 pre-release features (`spl_ft/features.py`): shooting-side elbow angle at window end, shooting-side knee angle at window end, min knee angle in the window, elbow angle change over the window, knee angle change over the window, wrist height relative to hip (normalized by shoulder-to-hip length), and average normalized vertical velocity of the wrist and hip over the window. Ball position/landing/entry angle/result are never features, only used to find the window and label. Sanity-check plots are in `outputs/figures/feature_checks/`.

The per-player boxplot (`outputs/figures/feature_checks/feature_by_player.png`) makes it obvious how different the players are from each other — P0002 barely bends their knees (~155-160° minimum), P0005 bends a lot more (~110-125°). That's basically the clearest evidence for the "model might just learn who's shooting" problem I talk about below.

## Models and results

All three models (`spl_ft/modeling.py`: majority baseline, logistic regression, random forest, seed fixed, imputation/scaling inside sklearn pipelines) got trained and evaluated under both protocols. Numbers are straight from `outputs/eval_known_players/results_overall.csv` and `outputs/eval_unseen_players/results_overall.csv`.

### Known players, later shots (chronological 80/20 split per player, pooled)

Train n=465 (66.0% makes), test n=118 (69.5% makes):

| Model | Accuracy | Balanced accuracy |
|---|---|---|
| Majority baseline | 0.695 | 0.500 |
| Logistic regression | 0.712 | 0.536 |
| Random forest | **0.737** | **0.671** |

Random forest actually beats the baseline here by a decent margin, including balanced accuracy, so it's not just predicting "made" every time. See `outputs/eval_known_players/confusion_matrices.png`.

### Unseen players (leave-one-out, pooled across all 5 folds)

Pooled n=583 (66.7% makes):

| Model | Accuracy | Balanced accuracy |
|---|---|---|
| Majority baseline | 0.667 | 0.500 |
| Logistic regression | 0.607 | 0.495 |
| Random forest | 0.530 | 0.498 |

Neither model beats the baseline here, and balanced accuracy for both is basically chance. Checking `outputs/eval_unseen_players/results_by_player.csv`, it's not just one bad fold either — random forest's balanced accuracy ranges from about 0.44 to 0.56 across the 5 held-out players, bouncing around chance in both directions.

### What I think this means

Eval A shows the models can fit *something* when they've already seen the player. Eval B shows that doesn't carry over to a new player. Given how different the players look in `outputs/figures/feature_checks/feature_by_player.png`, my best guess is the random forest in eval A is partly just learning who's shooting (or that player's general make rate), not something about mechanics that holds up across people. With only 5 players that's exactly the risk I was worried about, and it's what actually showed up, not something I'm just speculating about.

## What worked, what's untested, what's unresolved

**Actually ran, with real output files:**
- Downloading/inspecting the dataset and documenting the real schema/units/axes.
- Release + handedness estimation, checked two different ways.
- Feature extraction for all 583 trials, 0 excluded.
- All three models, both evaluation protocols, saved tables and figures.

**Tested with a real photo but not a real video:**
- The MediaPipe demo (`mediapipe_demo/`). I confirmed pose detection, skeleton drawing, and the 3D angle math all work correctly using a real photo of a person looped into a short test video. I also checked the "no person in frame" path doesn't crash. I don't have an actual video, let alone a free-throw video, to test it on for real yet — see `mediapipe_demo/README.md`.

**Still open / not glossed over:**
- Release frame and handedness are estimates, nothing to validate them against.
- The unseen-player eval is only 5 folds, so don't read too much into any single one.
- Haven't run MediaPipe on a real free throw, and I haven't shown the SPL-trained classifier's features transfer to MediaPipe landmarks at all — different skeleton, different 3D estimation, different units.

## What's blocked

Nothing's blocked for the core pipeline — dataset downloaded fine, every script ran, results are real and saved under `outputs/`. The one thing I actually can't do right now is run the MediaPipe demo on a real free-throw video, because I don't have one. Once I get a video: `python mediapipe_demo/run_demo.py --video <path> --side <LEFT|RIGHT>`.
