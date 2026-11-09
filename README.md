# Basketball Free Throw Form — Pre-Release Biomechanics Classifier

**Course project (CSC4888, Computer Vision).** Not a production system,
not an attempt at state-of-the-art performance — a small, reproducible
pipeline for a focused research question.

## Research question

> Can we predict whether a free throw is made or missed using a player's
> body mechanics **before** the ball is released?

See `PROJECT_LOG.md` for a running log of decisions and findings, and
`SUMMARY.md` for the final write-up of what worked, what was tested, and
the actual results.

## Dataset

[SPL Open Data](https://github.com/Sport-Performance-Lab/SPL-Open-Data)
(basketball free-throw subset), by MLSE's Sport Performance Lab: 3D
markerless motion-capture data + shot outcome for 583 free-throw trials
across 5 players. License: CC BY-NC-SA 4.0 (non-commercial — respected
here; this is a non-commercial class project).

## Project layout

```
spl_ft/                     Shared pipeline code (importable package)
  config.py                 Paths + all tunable constants, documented
  data_io.py                 Trial loading, chronological sort key
  kinematics.py               Joint angles, velocities, body-size scale
  release.py                  Release-frame + shooting-hand estimation
  handedness.py               Per-player shooting-side aggregation
  features.py                  Pre-release feature extraction
  modeling.py                  sklearn Pipelines (baseline/logreg/RF)
  evaluation.py                Metrics + confusion-matrix plotting
scripts/
  00_download_data.py          Download the dataset
  01_inspect_dataset.py        Dataset inspection -> outputs/dataset_inspection/
  02_check_release_and_features.py   Release/window visualizations
  03_extract_features.py       Build outputs/features.csv
  04_feature_distributions.py  Feature-check plots
  05_train_eval.py              Train + run both evaluations
mediapipe_demo/               Optional, standalone MediaPipe demo (see its own README)
outputs/                       All generated CSVs/figures (tracked in git as deliverables)
data/raw/                      Downloaded dataset (NOT tracked in git — see .gitignore)
```

## Setup

Requires Python 3.10+ and `git`.

```bash
cd CSC4888_basketball_form_project
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If `mediapipe`/`opencv-python` fail to install for your platform
(they're only needed for the optional demo), comment out those two
lines in `requirements.txt`, reinstall, and skip straight to the core
pipeline below.

## Run (core pipeline)

Run in order from the repo root, with the venv activated:

```bash
# 1. Download the dataset (≈1.2 GB; sparse checkout of just the
#    basketball/freethrow folder of SPL-Open-Data)
python scripts/00_download_data.py

# 2. Inspect the dataset: schema, per-player make/miss counts,
#    release-detection + handedness validation
python scripts/01_inspect_dataset.py

# 3. Visualize release detection + the pre-release window on example trials
python scripts/02_check_release_and_features.py

# 4. Extract pre-release biomechanics features -> outputs/features.csv
python scripts/03_extract_features.py

# 5. Plot feature distributions (sanity check)
python scripts/04_feature_distributions.py

# 6. Train + evaluate the 3 models under both protocols
python scripts/05_train_eval.py
```

Each script prints what it's doing and where it wrote its outputs. All
of them are safe to re-run (they overwrite their own outputs).

## Run (optional MediaPipe demo)

Separate from the core pipeline above — see
[`mediapipe_demo/README.md`](mediapipe_demo/README.md) for setup
(including the required model-file download) and usage. In short:

```bash
python mediapipe_demo/run_demo.py --video /path/to/free_throw.mp4 --side RIGHT
```

**This demo is independent of, and has not been shown to be compatible
with, the SPL-trained classifier above** — see that README for why.

## Key design decisions (see `PROJECT_LOG.md` for the full narrative)

- **No release-event field exists in the dataset.** Release is estimated
  from ball-to-wrist distance (`spl_ft/release.py`), validated by (a)
  near-unanimous per-player handedness votes and (b) visual inspection
  (`outputs/figures/release_checks/`).
- **No handedness field exists.** Shooting side is the per-player
  majority vote of which wrist releases the ball last; see
  `outputs/dataset_inspection/handedness.csv` and
  `spl_ft/handedness.py` for how to verify/override it.
- **A 0.05 s safety buffer** is subtracted from the detected release
  frame before the ~0.5 s pre-release feature window, to reduce (not
  guarantee-eliminate) the risk of leaking post-release/in-release
  information, since release detection is itself a heuristic.
- **Ball position is used only to locate the window**, never as a
  feature. `landing_x`, `landing_y`, `entry_angle`, and `result` are
  never used as features.
- Per the dataset README, not all capture sessions used the same pose
  model: the `MID_HIP` keypoint is entirely absent in the 2024-08-28
  session. We use `(LEFT_HIP + RIGHT_HIP) / 2` instead of `MID_HIP` for
  the hip-center-based features, which keeps that session's trials
  usable instead of excluding them.

## Known limitations

- Only **5 players**, 583 trials total. The unseen-player evaluation
  (B) is a leave-one-of-5-out evaluation; its per-fold results have very
  high variance and should not be treated as a general claim about
  "unseen shooters."
- Release frame and shooting handedness are **estimated**, not ground
  truth — there is no label to check them against.
- A model can learn to distinguish *players* (who have very different
  mechanics and make rates — see `outputs/figures/feature_checks/feature_by_player.png`)
  rather than learning a release-mechanics signal that generalizes. This
  is consistent with what we observed: results are clearly better in
  Evaluation A (known players) than Evaluation B (unseen players) — see
  `SUMMARY.md`.

## Git / attribution policy

This repository is managed locally. Commits are not pushed to any
remote by the assistant that helped build it, and no "made with
Cursor"/AI-attribution tags are added to commits or files — see
`.cursor/rules/git-push-policy.mdc`. Only the project owner pushes.
