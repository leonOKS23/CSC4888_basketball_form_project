# Project Log

Running log of what this project is, what has been done, and what is left.
Updated as work progresses. See `SUMMARY.md` for the final write-up and
`README.md` for setup/run instructions.

## Research question

Can we predict whether a basketball free throw is made or missed using a
player's body mechanics **before** the ball is released, using the SPL
Open Data free-throw motion-capture dataset?

## Status: core pipeline complete and verified end-to-end; MediaPipe demo
code-complete but untested on real video (none provided yet)

## Git policy (hard rule)

- This agent does **not** push to any remote and does **not** add
  Cursor/AI attribution to commits, code, or docs. See
  `.cursor/rules/git-push-policy.mdc`. Only the user pushes.

## Timeline / decisions

- **Dataset inspection** (done): Cloned
  `Sport-Performance-Lab/SPL-Open-Data` and inspected the real README,
  `usage.ipynb`, `animate.py`, and raw trial JSON files directly (did not
  trust README text alone, since it was out of date in places — e.g. it
  mentions a `participant_information.json` that does not exist in the
  current repo, and `animate.py` uses an older short keypoint-name
  convention that does not match the actual trial JSON files).
  - Each trial is one JSON file: `data/<session_date>/<participant_id>/BB_FT_<participant_id>_<trial_id>.json`.
  - Top-level fields per trial: `sampling_rate`, `trial_date`,
    `participant_id`, `trial_id`, `result` (`"made"`/`"missed"`),
    `landing_x`, `landing_y`, `entry_angle`, `tracking` (list of per-frame
    dicts with `frame`, `time` (ms), `data.ball` (xyz, feet, NaN when not
    visible), `data.player` (dict of 69 named keypoints, each xyz in
    feet)).
  - 5 participants (P0001–P0005), 583 trials total, 389 made / 194 missed.
  - Units: feet. Court-centered coordinate system. `z` is vertical
    (confirmed empirically: toe/ankle z ≈ 0, head z ≈ 5–6).
  - Frame rate: 30 fps for the 2024-08-28 session, 60 fps for the
    2025-12-18 session (both stored per-trial in `sampling_rate`, and
    confirmed against the `time` field deltas).
  - No explicit release-event field, no explicit shooting-handedness
    field. Both were derived (see below) and documented with validation.
  - `landing_x`, `landing_y`, `entry_angle` are provided for every trial
    (including misses) — these are **outcome-adjacent** fields and are
    never used as model features, per the assignment.
  - Full inspection output: `outputs/dataset_inspection/`.

- **Release detection** (done, validated by visualization): No ground
  truth release frame exists in the data. Implemented and validated a
  ball-to-wrist-distance heuristic (see `spl_ft/release.py`):
  1. Ball tracking has gaps; we select the last contiguous ball-visible
     block that actually contains a close hand/ball approach (distance
     under a threshold), to avoid mistaking a later "ball re-acquired
     near the hoop" block for the holding/release block.
  2. Within that block, the shooting hand is the wrist that keeps the
     ball in contact *longest* (closest to the end of the block), not
     simply the closest at the moment of minimum distance — this
     distinguishes the shooting hand from the guide hand, which lets go
     first.
  3. Release frame = first frame after last contact where the ball stays
     clearly separated from that wrist.
  - Validated two ways: (a) per-player majority-vote shooting hand is
    unanimous or near-unanimous per player (see
    `outputs/dataset_inspection/handedness.csv`), which is what we'd
    expect if the detector is picking up a real, consistent signal
    rather than noise; (b) visual inspection of distance curves +
    detected release frame for representative made/missed trials across
    players, saved under `outputs/figures/release_checks/`.
  - This is a heuristic, not ground truth. Documented explicitly as a
    limitation in `SUMMARY.md`.

- **Features** (done): see `spl_ft/features.py` and
  `outputs/features.csv`. Initially used the dataset's own `MID_HIP`
  keypoint for hip-center features, which turned out to be entirely
  absent in the 2024-08-28 session (confirming the README's warning
  that pose models differ across sessions) and excluded all 125 of that
  session's trials. Fixed by computing hip center as
  `(LEFT_HIP + RIGHT_HIP) / 2` instead — all 583 trials are now usable
  (0 excluded).

- **Models / evaluation** (done, ran successfully): see
  `scripts/05_train_eval.py` and `SUMMARY.md` for actual numbers.
  Evaluation A (known players): random forest accuracy 0.737 / balanced
  accuracy 0.671 vs. majority baseline 0.695 / 0.500. Evaluation B
  (unseen players, leave-one-out): no model beat the majority baseline;
  balanced accuracy ~0.50 (chance) for both trained models, pooled
  across all 5 held-out-player folds. This gap between A and B is
  exactly the "may just learn player identity" risk flagged by the
  assignment, and the per-player feature boxplots
  (`outputs/figures/feature_checks/feature_by_player.png`) support that
  reading.

- **MediaPipe demo** (done, code-complete; **not runnable end-to-end
  without a sample video** — none was provided at the time of writing).
  See `mediapipe_demo/README.md` for exactly what was and wasn't tested.

## Known limitations (see SUMMARY.md for full discussion)

- Only 5 participants total — any "unseen player" result is based on a
  leave-one-of-5-out evaluation and should not be over-interpreted.
- Release frame and shooting handedness are both inferred, not ground
  truth.
- MediaPipe demo pose landmarks are not the same skeleton/model as the
  SPL motion-capture data; the SPL-trained classifier has not been
  validated on MediaPipe-extracted features and should not be assumed to
  transfer.
