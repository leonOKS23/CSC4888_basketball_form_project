# Basketball Free Throw Form — Pre-Release Biomechanics Classifier

CSC4888 (Computer Vision) course project. This is not meant to be a production system or SOTA model, just a small, reproducible project for a specific question I wanted to answer.

## The question

Can I predict whether a free throw is made or missed just from a player's body mechanics *before* they release the ball?

`PROJECT_LOG.md` has my running notes on decisions/findings as I went. `SUMMARY.md` has the actual write-up of what worked and what the results were.

## Dataset

I'm using the free-throw subset of [SPL Open Data](https://github.com/Sport-Performance-Lab/SPL-Open-Data), released by MLSE's Sport Performance Lab. It's 3D markerless motion-capture data plus shot outcomes for 583 free throws across 5 players. License is CC BY-NC-SA 4.0 (non-commercial), which is fine since this is just a class project.

## Layout

```
spl_ft/                     shared code, importable
  config.py                 paths + constants
  data_io.py                 loading trials, chronological sort key
  kinematics.py               joint angles, velocities, body-size scale
  release.py                  release-frame + shooting-hand estimation
  handedness.py               per-player shooting side
  features.py                  the actual feature extraction
  modeling.py                  sklearn pipelines (baseline/logreg/RF)
  evaluation.py                metrics + confusion matrix plots
scripts/
  00_download_data.py          downloads the dataset
  01_inspect_dataset.py        schema/counts -> outputs/dataset_inspection/
  02_check_release_and_features.py   plots to sanity-check the release/window
  03_extract_features.py       builds outputs/features.csv
  04_feature_distributions.py  more sanity-check plots
  05_train_eval.py              trains + runs both evaluations
mediapipe_demo/                separate, optional demo (own README)
outputs/                       everything the scripts generate
data/raw/                      downloaded dataset, not tracked in git
```

## Setup

Needs Python 3.10+ and git.

```bash
cd CSC4888_basketball_form_project
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If `mediapipe`/`opencv-python` don't install cleanly for you, comment those two lines out of `requirements.txt` and reinstall — they're only needed for the optional demo, not the main pipeline.

## Running it

In order, from the repo root, venv active:

```bash
# 1. download the dataset (~1.2GB, sparse checkout of basketball/freethrow)
python scripts/00_download_data.py

# 2. inspect the dataset — schema, per-player counts, release + handedness checks
python scripts/01_inspect_dataset.py

# 3. plot release detection on some example trials
python scripts/02_check_release_and_features.py

# 4. extract the actual features -> outputs/features.csv
python scripts/03_extract_features.py

# 5. plot feature distributions as a sanity check
python scripts/04_feature_distributions.py

# 6. train + evaluate all 3 models, both protocols
python scripts/05_train_eval.py
```

They all print what they're doing and where they wrote stuff. Safe to re-run anytime, they just overwrite their own outputs.

## Optional MediaPipe demo

Separate from everything above, see [`mediapipe_demo/README.md`](mediapipe_demo/README.md) for setup (you need to grab a model file) and usage:

```bash
python mediapipe_demo/run_demo.py --video /path/to/free_throw.mp4 --side RIGHT
```

Important: this demo has nothing to do with the classifier above, see that README for why.

## Decisions I made along the way (more detail in PROJECT_LOG.md)

- There's no release-event field in the dataset, so I estimate it from ball-to-wrist distance (`spl_ft/release.py`). I checked this two ways — per-player handedness votes come out basically unanimous, and I eyeballed the release point on a bunch of example plots (`outputs/figures/release_checks/`).
- No handedness field either, so shooting side is just whichever wrist lets go of the ball last, majority-voted per player. `outputs/dataset_inspection/handedness.csv` has the numbers if you want to check/override it (`spl_ft/handedness.py`).
- I subtract a 0.05s buffer from the detected release before taking the ~0.5s pre-release window, just to be safe since release detection isn't perfect.
- Ball position is only used to find the window, never as a feature. Same for landing position, entry angle, and the result itself — none of that goes into the model.
- Turns out not every session used the same pose model — `MID_HIP` is completely missing in the 2024-08-28 session. I just compute it myself as `(LEFT_HIP + RIGHT_HIP) / 2` instead of excluding that whole session.

## Limitations

- Only 5 players, 583 trials. The "unseen player" evaluation is leave-one-of-5-out, so take any single fold's result with a grain of salt.
- Release frame and handedness are both estimates — there's no ground truth label to check them against.
- A model can pick up on which *player* is shooting rather than anything about mechanics, since the 5 players are pretty different from each other and have different make rates (see `outputs/figures/feature_checks/feature_by_player.png`). This actually matches what I saw: things looked a lot better when the model had already seen the player (eval A) than when it hadn't (eval B). More in `SUMMARY.md`.

## Note on pushing

I'm not pushing any of this to a remote through an assistant, and no AI-attribution tags get added anywhere — see `.cursor/rules/git-push-policy.mdc`. I push it myself when I'm ready.
