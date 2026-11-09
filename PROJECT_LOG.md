# Project Log

Just keeping a running log here of what I'm doing and why, as I go. `SUMMARY.md` has the real write-up and `README.md` has setup/run instructions.

## The question

Can I predict whether a free throw is made or missed from a player's body mechanics *before* release, using the SPL Open Data free-throw mocap dataset?

## Status

Core pipeline is done and I ran it end to end. MediaPipe demo is written and works mechanically but I haven't tested it on an actual video yet (don't have one).

## Pushing

Not pushing anything through an assistant, no AI-attribution tags anywhere. See `.cursor/rules/git-push-policy.mdc`. I'll push myself.

## Notes as I went

**Dataset inspection** — cloned `Sport-Performance-Lab/SPL-Open-Data` and actually read through the README, `usage.ipynb`, `animate.py`, and some raw trial JSONs instead of just trusting the docs. Good thing I did, because the docs are a bit stale — the README talks about a `participant_information.json` that doesn't actually exist in the repo, and `animate.py`/`usage.ipynb` use an older short keypoint naming (like `R_HIP`) that doesn't match what's actually in the trial files (`RIGHT_HIP` etc).

What I found:
- Each trial is its own JSON: `data/<session_date>/<participant_id>/BB_FT_<participant_id>_<trial_id>.json`
- Top level fields: `sampling_rate`, `trial_date`, `participant_id`, `trial_id`, `result` (made/missed), `landing_x`, `landing_y`, `entry_angle`, and `tracking` (per-frame: `frame`, `time` in ms, `data.ball` xyz in feet, `data.player` dict of 69 named keypoints).
- 5 participants (P0001–P0005), 583 trials, 389 made / 194 missed.
- Everything's in feet, court-centered. `z` is vertical — checked this myself (toe markers sit near z=0, head markers around z=5-6ft).
- Frame rate: 30fps for the 2024-08-28 session, 60fps for 2025-12-18 (both sessions are in the `sampling_rate` field and line up with the `time` deltas).
- No release-event field, no handedness field anywhere. Had to derive both myself (see below).
- `landing_x`/`landing_y`/`entry_angle` exist for every single trial, makes and misses. These describe the outcome of the shot so I never use them as model inputs.
- Full dump of this is in `outputs/dataset_inspection/`.

**Release detection** — no ground truth for this exists in the data, so I built a heuristic based on ball-to-wrist distance (`spl_ft/release.py`):
1. Ball tracking has gaps in it. I pick the last contiguous block of valid ball frames that actually has the ball getting close to a hand at some point — otherwise I'd sometimes grab a later block where the ball reappears near the hoop, way after release.
2. Shooting hand = whichever wrist keeps the ball closest for *longest* (not just whichever is closest at the single closest moment) — this matters because the guide hand tends to let go slightly before the shooting hand.
3. Release frame = the frame right after that hand's last contact.

I checked this two ways: (a) the per-player shooting-hand vote comes out unanimous or close to it for every player (`outputs/dataset_inspection/handedness.csv`) — if this were just noise I wouldn't expect that; (b) I looked at the actual distance-vs-time plots with the detected release marked for a bunch of trials across all players, makes and misses (`outputs/figures/release_checks/`), and the release point lines up with where the ball visibly leaves the hand every time I checked.

Still a heuristic though, not ground truth. I say so in `SUMMARY.md`.

**Features** — see `spl_ft/features.py` / `outputs/features.csv`. First pass I used the dataset's own `MID_HIP` keypoint for the hip-center features, and that ended up excluding all 125 trials from the 2024-08-28 session because `MID_HIP` is just completely absent for that session (different pose model, which the README does warn about). Switched to computing it myself as `(LEFT_HIP + RIGHT_HIP) / 2` and now all 583 trials are usable.

**Models/eval** — ran all three models (majority baseline, logreg, random forest) under both protocols. Random forest does meaningfully better than baseline when evaluated on known players (0.737 vs 0.695 accuracy, 0.671 vs 0.500 balanced accuracy), but when I leave a whole player out, neither trained model beats the baseline — balanced accuracy is basically coin-flip (~0.50) pooled across all 5 held-out folds. Kind of expected given only 5 players, but good to actually see it rather than assume it. Numbers and more discussion in `SUMMARY.md`.

**MediaPipe demo** — code works, I tested the actual mechanics (pose detection, skeleton drawing, 3D angle math) on a real photo looped into a short test video, and that all checks out. But I don't have an actual free-throw video to run it on yet, so that part is untested. Details in `mediapipe_demo/README.md`.

## Limitations worth remembering

- Only 5 players total, so the "unseen player" numbers are from a tiny leave-one-out setup and shouldn't be over-interpreted.
- Release frame and handedness are inferred, not ground truth.
- MediaPipe's pose landmarks are a totally different skeleton/model from the SPL mocap data — I have not shown the classifier transfers to MediaPipe features at all, and I'm not claiming it does.
