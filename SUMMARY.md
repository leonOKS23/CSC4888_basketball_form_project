# Summary

Everything described below was actually run, on the actual downloaded
SPL Open Data free-throw dataset, in this repository, on this machine.
Exact commands are in `README.md`; raw outputs are under `outputs/`.

## What the dataset actually contains (vs. what the README says)

The published README was mostly, but not entirely, accurate/current for
the files actually in the repo at the time of download:

- It references a `participant_information.json` file that **does not
  exist** in the current repo — there is no participant demographic
  file, so we could not look up handedness, height, etc. from metadata.
- `basketball/freethrow/animate.py` and `usage.ipynb` use an older,
  different keypoint-name convention (e.g. `R_HIP`) than the one
  actually present in the trial JSON files (e.g. `RIGHT_HIP`,
  `LEFT_HIP`, ...). We read the real files directly rather than trusting
  either document.
- One thing the README *did* correctly warn about and that we ran into:
  "Not all sessions were run using the same pose model." The `MID_HIP`
  keypoint is present in the 2025-12-18 session but entirely absent
  (all-NaN) for the 2024-08-28 session. We worked around this by
  computing hip center as `(LEFT_HIP + RIGHT_HIP) / 2` instead.

Confirmed empirically (not assumed): 5 participants, 583 trials, 389
makes / 194 misses, units in feet, `z` is vertical, trial IDs reset at
the start of each capture session (so cross-session chronological order
for a participant needed `(session_date, trial_number)`, not just
`trial_number`). Full detail: `outputs/dataset_inspection/summary.md`.

## Per-player usable makes/misses (after full feature-extraction pipeline)

| Player | Usable shots | Made | Missed | Make rate |
|---|---|---|---|---|
| P0001 | 213 | 149 | 64 | 0.700 |
| P0002 | 93  | 55  | 38 | 0.591 |
| P0003 | 90  | 81  | 9  | 0.900 |
| P0004 | 89  | 61  | 28 | 0.685 |
| P0005 | 98  | 43  | 55 | 0.439 |

**All 583 trials ended up usable** — 0 excluded. (Earlier in development,
using the literal `MID_HIP` field instead of `(LEFT_HIP+RIGHT_HIP)/2`
excluded all 125 trials from the 2024-08-28 session; see
`outputs/dataset_inspection/feature_extraction_summary.md` for the
current, final exclusion breakdown, which is empty.) This is a genuinely
clean result, not a claim glossing over problems — but see "Limitations"
below for why "0 excluded" doesn't mean "0 uncertainty."

## Release detection: what we tried and how we validated it

The dataset has no release-event field. We estimate release from
ball-to-wrist 3D distance (`spl_ft/release.py`):

1. Select the last contiguous ball-visible block that actually contains
   a close ball/wrist approach (this matters — in ~40 trials the *very
   last* visible-ball block was the ball reappearing near the hoop,
   several feet from the shooter, after an occlusion gap during the
   shot itself; naively using "the last block" would have put the
   release point at the wrong place or found none).
2. The shooting hand is whichever wrist keeps the ball closest *latest*
   (not simply closest overall) — this correctly separates the shooting
   hand from the guide hand, which tends to separate first.
3. Release = one frame after that hand's last contact frame.

**Validation performed (both done, both passed):**
- Per-player handedness vote is unanimous for 4/5 players (100% of
  trials voted the same hand) and 81.6% for the fifth (P0005) — and
  inspecting the "disagreeing" P0005 trials showed the two hands
  separated within 1-2 frames of each other (a near-simultaneous
  release), i.e. measurement noise at a tie, not evidence of a
  different/ambiguous shooting hand. See
  `outputs/dataset_inspection/handedness.csv`.
- Visual inspection of distance-vs-time, height-vs-time, and
  elbow/knee-angle-vs-time plots for 20 representative trials (4 per
  player, mix of makes/misses) in `outputs/figures/release_checks/`.
  In every inspected example, the detected release frame lines up with
  the ball-wrist distance curve rising sharply and monotonically away
  from the hand, and the shaded pre-release window sits over the visible
  loading/extension phase (knee flexion minimum, then elbow/knee
  extension), not over the follow-through.

This is a validated **heuristic**, not ground truth — there is no
labeled release frame in the dataset to score it against numerically.

## Features

One row per shot, 8 pre-release features (`spl_ft/features.py`):
shooting-side elbow angle at window end, shooting-side knee angle at
window end, minimum knee angle in window, elbow angle change over
window, knee angle change over window, wrist height relative to hip
(normalized by shoulder-to-hip "torso" length), and mean normalized
vertical velocity of the wrist and of the hip over the window. Ball
position, landing position, entry angle, and shot result are never used
as features — only used to locate the window and as the label.
Feature-sanity plots: `outputs/figures/feature_checks/`.

The per-player boxplot
(`outputs/figures/feature_checks/feature_by_player.png`) makes the
between-player variation obvious — e.g. P0002's minimum knee angle
during the shot (~155-160°, barely bending the knees) is nowhere near
P0005's (~110-125°). This single plot is the clearest evidence for the
"a model might just learn who's shooting" limitation discussed below.

## Models and evaluation — actual results

All three models (`spl_ft/modeling.py`: majority-class baseline,
logistic regression, random forest; fixed seed 42; imputation/scaling
inside sklearn `Pipeline`s) were trained and evaluated under both
protocols. Numbers below are read directly from
`outputs/eval_known_players/results_overall.csv` and
`outputs/eval_unseen_players/results_overall.csv`.

### A. Known players, later shots (chronological 80/20 split per player, pooled)

Train n=465 (66.0% makes), test n=118 (69.5% makes):

| Model | Accuracy | Balanced accuracy |
|---|---|---|
| Majority baseline | 0.695 | 0.500 |
| Logistic regression | 0.712 | 0.536 |
| Random forest | **0.737** | **0.671** |

The random forest beats the majority baseline by a real margin here,
including on balanced accuracy (0.67 vs 0.50), meaning it's catching
some misses, not just always predicting "made." Confusion matrices:
`outputs/eval_known_players/confusion_matrices.png`.

### B. Unseen players (leave-one-player-out, pooled across all 5 folds)

Pooled n=583 (66.7% makes):

| Model | Accuracy | Balanced accuracy |
|---|---|---|
| Majority baseline | 0.667 | 0.500 |
| Logistic regression | 0.607 | 0.495 |
| Random forest | 0.530 | 0.498 |

**Neither trained model beats the majority baseline here, and balanced
accuracy for both is essentially 0.50** (chance). Per-player breakdown
(`outputs/eval_unseen_players/results_by_player.csv`) shows this isn't
one bad fold dragging down an otherwise-good result — it's inconsistent
fold to fold (e.g. random forest balanced accuracy ranges from ~0.44 to
~0.56 across the 5 held-out players, straddling chance in both
directions).

### Interpretation

Evaluation A shows the models can fit *something* useful in-distribution
(same players, later shots). Evaluation B shows that signal does not
transfer to a held-out player. The most likely explanation, given the
per-player feature spread shown in
`outputs/figures/feature_checks/feature_by_player.png`, is that at least
part of what Evaluation A's random forest is learning is player
identity (who is shooting) and/or player-level make-rate, not a
release-mechanics signal that generalizes across people. With 5 players
this is exactly the risk the assignment asked us to watch for, and it is
what we observed, not something we're speculating might happen.

## What worked / what was tested / what's unresolved

**Worked and tested, end-to-end, with real output files:**
- Dataset download, inspection, and documentation of real schema/units/axes.
- Release-frame + handedness estimation, with two independent validation checks.
- Feature extraction for all 583 trials, 0 excluded.
- All three models trained and evaluated under both protocols, with
  saved tables and figures.

**Tested with a real image, but not with an actual video or an actual
basketball shot:**
- The MediaPipe demo (`mediapipe_demo/`). Pose detection, skeleton
  overlay, and 3D joint-angle computation were verified on a short
  synthetic video built from one real photo of a person (so detection,
  drawing, and angle math are all confirmed correct), and the "no person
  detected" path was verified with a synthetic video containing no
  person. **No actual video with real motion, and no basketball free
  throw video, was available at the time of writing** — see
  `mediapipe_demo/README.md` for exactly what remains untested.

**Unresolved / explicitly flagged, not glossed over:**
- Release frame and handedness are estimated, not verified against
  ground truth (none exists in this dataset).
- The unseen-player evaluation is only 5 folds; any single fold's
  result should not be over-read.
- We have not run the MediaPipe demo on a real free throw, and the
  SPL-trained classifier's features have not been shown to transfer to
  MediaPipe-derived landmarks at all (different skeleton, different 3D
  estimation method, different units).

## Blocked items

Nothing is currently blocked for the core pipeline — the dataset
downloaded successfully, all scripts ran successfully, and results are
real and saved under `outputs/`. The one blocked item is: **no sample
free-throw video was provided**, so the MediaPipe demo could not be run
against real basketball footage. Next step for that: provide a video
and run `python mediapipe_demo/run_demo.py --video <path> --side <LEFT|RIGHT>`.
