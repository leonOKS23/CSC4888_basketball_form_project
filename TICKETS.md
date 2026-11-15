# Project plan / tickets

Goal: predict whether a free throw is made or missed, using body mechanics
from *before* the ball is released, using the SPL Open Data free-throw
mocap dataset.

Work through these roughly in order. Each one is small enough to do (and
understand) in one sitting. `basketball_shot_analyzer.ipynb` already has
a rough first pass at some of this (data download, a release-frame
finder, some outlier checking, a plotting helper) - use it as a starting
point, don't throw it away.

---

### 0. Project setup

- Make a venv, install `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
- Write down those deps in a `requirements.txt`.
- Confirm `basketball_shot_analyzer.ipynb` still runs top to bottom.

**Done when:** the notebook runs, you have a venv + requirements.txt.

---

### 1. Get the data

- Download the dataset. The notebook does this with `wget` + `unzip` of
  the whole SPL-Open-Data repo (cell 2). That works but it's a full
  clone (~1.5GB) - fine for now, can optimize later if it's annoying.
- Decide where it lives on disk (e.g. `data/raw/`) and make sure that's
  gitignored - we don't want to commit a gigabyte of JSON.
- Confirm you can load trials into a dataframe (like cell 3 of the
  notebook) - one row per trial.

**Done when:** you can glob the trial JSON files and load them into a
dataframe/list without errors.

---

### 2. Actually understand the data (don't trust the README blindly)

Read a few real trial JSON files yourself and answer:

- What fields exist at the top level of each trial? (`result`,
  `sampling_rate`, `participant_id`, `trial_id`, etc.)
- How is make/miss encoded? (check the actual string values)
- What's in `tracking`? What are the keypoint names, and do they match
  what the README says? (the notebook comment already hints the dataset
  structure doesn't perfectly match the docs)
- Which axis is vertical? (hint: check a toe keypoint vs. a head keypoint)
- Is the frame rate the same for every trial? (the notebook only loads
  the `2025-12-18` session - check if there's another session folder
  with a different fps, and decide whether to include it)
- Is there a release-event field, or a handedness field anywhere? (spoiler: no, you'll need to figure both out - see tickets 3 and 4)

Write your findings down somewhere (a short notes file, or just keep
them in this repo) along with **per-player make/miss counts**.

**Done when:** you have a short write-up of the schema + a table of
makes/misses per player.

---

### 3. Find the release frame

The notebook's `release_frame()` (cell 4) is a starting point: find the
last frame where the ball is within one ball-width of either wrist.

Things to check / possibly fix:
- Does this work the same way across *every* trial, or are there trials
  where the ball-tracking data has gaps and the "last frame near a
  wrist" isn't actually near the real release? Worth checking a handful
  of outliers from `find_release_frames_outliers()` to see if they're
  genuinely bad shots or a bug in the detector.
- Does it reliably pick the *shooting* hand, or could it sometimes grab
  the guide hand instead?
- Use `plot_shot()` (cell 4) on a handful of trials per player (mix of
  makes and misses) and eyeball whether the red release line actually
  lines up with the ball leaving the hand.

**Done when:** you've visually checked release detection on several
trials per player and you're confident (or you've written down exactly
why you're not, for which trials).

---

### 4. Figure out shooting hand (no handedness field exists)

- Idea: for each trial, check which wrist is the one still holding the
  ball right up until release. Do that for all of one player's trials
  and take the majority.
- Sanity check: if a player's votes are a clean majority (like 90%+ one
  hand), that's a good sign it's picking up something real and not
  noise. If it's close to 50/50, look at a few of those trials directly.

**Done when:** you have one shooting side (LEFT/RIGHT) assigned per
player, and some way to double check it if it looks wrong.

---

### 5. Define the pre-release window

- Pick a buffer before the detected release (small, e.g. a couple of
  frames) so you're not accidentally grabbing release/post-release
  motion.
- Pick a window length before that buffer (~0.5 seconds, using the
  trial's actual frame rate, not a hardcoded frame count).
- Some trials might not have a full 0.5s available before release
  (release happens too early in the clip) - decide what to do with
  those (exclude them, and say how many).

**Done when:** for every trial you know the exact start/end frame of
its feature window, and you know how many trials got excluded and why.

---

### 6. Feature extraction

For each trial, using only data inside the pre-release window, compute:

- shooting-side elbow angle at the end of the window
- shooting-side knee angle at the end of the window
- minimum knee angle during the window
- change in elbow angle over the window
- change in knee angle over the window
- wrist height relative to hip, normalized by some body-size measure
- average vertical wrist velocity over the window, normalized
- average vertical hip velocity over the window, normalized

Write one row per shot to a CSV: participant_id, trial_id, label,
whatever exclusion/quality info you need, and the features above.

Rules:
- Ball position can be used to *find* the window, never as a feature.
- Don't use landing position / entry angle / result as features.
- If something's missing (NaN keypoint, etc.), log it and exclude that
  row - don't fill in a made-up number.

**Done when:** you have a `features.csv` with one row per shot, and a
list of anything that got excluded + why.

---

### 7. Sanity-check the features

- Plot each feature's distribution split by make/miss.
- Plot features split by player - this tells you how different the
  players are from each other, which matters later (see ticket 10).

**Done when:** you have a couple of plots and nothing in them looks
obviously broken (e.g. all-zero, all-identical, wildly-out-of-range
values).

---

### 8. Build the model pipeline

Three models, scikit-learn, simple defaults, fixed random seed:

1. Majority-class baseline (`DummyClassifier`)
2. Logistic regression
3. Random forest

Put imputation/scaling inside an sklearn `Pipeline` so it only ever gets
fit on training data, never on test data.

**Done when:** you can fit all three on some training rows and predict
on held-out rows.

---

### 9. Evaluation A: known players, later shots

- For each player, split chronologically - first ~80% of their shots
  train, last ~20% test. (Double check trial IDs are actually in time
  order before trusting this - see ticket 2.)
- Pool all players' training rows into one training set, train one
  model, test on the pooled held-out rows.
- Report: accuracy, balanced accuracy, confusion matrix, test set size,
  make rate.

**Done when:** you have those numbers for all 3 models.

---

### 10. Evaluation B: unseen players

- Leave-one-player-out: hold out one player entirely, train on the
  rest, test on them. Repeat for every player.
- Refit the pipeline (imputer/scaler/model) from scratch each fold -
  nothing should leak from other folds.
- Report overall numbers (pooled across folds) *and* per-held-out-player
  numbers.

**Done when:** you have both the pooled and per-player numbers for all
3 models.

Expect eval B to look worse than eval A. If it doesn't, that's worth
double-checking, not celebrating - with only 5 players it's easy for a
model to learn "who is shooting" instead of anything about mechanics.

---

### 11. Write up results

- Compare A vs B, and talk about why they might differ (player
  identity, make-rate differences between players - check the
  per-player feature plots from ticket 7).
- Be upfront about: release frame + handedness being estimated and not
  verified against any ground truth, and only having 5 players.

**Done when:** there's a short doc with the actual numbers and these
caveats, not just vibes.

---

### 12. (Optional / stretch) MediaPipe demo

A separate script, not connected to the trained model above:

- Takes a video file.
- Runs MediaPipe Pose Landmarker on it, draws the skeleton, saves the
  output video.
- Plots elbow/knee angle over time.
- Needs a `.task` model file downloaded separately (see MediaPipe's
  docs) - document exactly where to get it.

Worth noting clearly: MediaPipe's pose landmarks are a different
skeleton/estimation method than the SPL mocap data, so this demo isn't
proof the trained model works on regular video - it's just a
visualization tool.

**Done when:** it runs on at least one test video and produces a
sensible-looking overlay + angle plot.
