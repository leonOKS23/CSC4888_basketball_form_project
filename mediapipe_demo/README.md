# MediaPipe pose demo (optional, standalone)

This is a separate, optional script that takes a local video and overlays
a MediaPipe pose skeleton on it, plus plots shooting-side elbow/knee
angle over time.

**This demo is not connected to the SPL-trained classifier.** It uses a
completely different pose-estimation system (MediaPipe Pose, a monocular
RGB model with 33 landmarks) than the SPL motion-capture dataset the
classifier in `scripts/05_train_eval.py` was trained on (a markerless
multi-camera system with 69 named keypoints, in real-world feet). The
two have not been shown to produce comparable angle values, and the
classifier has **not** been validated on MediaPipe-extracted features.
Treat this demo purely as a visualization tool.

## Setup

1. Install the extra dependencies (already in `requirements.txt`, but
   listed here for clarity since they're optional / heavier than the
   core pipeline):

   ```bash
   pip install mediapipe opencv-python
   ```

   Tested working with `mediapipe==1.2.2` / `opencv-python==5.0.0` on
   Python 3.12 (macOS). If `pip install mediapipe` fails on your
   platform/Python version, see
   [MediaPipe's install docs](https://ai.google.dev/edge/mediapipe/solutions/setup_python)
   — it has narrower platform support than the rest of this project.

2. Download a Pose Landmarker model file (`.task`). This is a binary
   model file, not included in this repo (see `.gitignore`). Pick one
   of the following (lite is fastest, heavy is most accurate) and save
   it to `mediapipe_demo/models/pose_landmarker_lite.task`:

   ```bash
   mkdir -p mediapipe_demo/models
   curl -L -o mediapipe_demo/models/pose_landmarker_lite.task \
     https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
   ```

   Other variants (swap `_lite` for `_full` or `_heavy` in both the URL
   and `--model` path):
   - `pose_landmarker_full`
   - `pose_landmarker_heavy`

   These URLs are documented on
   [Google's MediaPipe Pose Landmarker task page](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker).

## Run

```bash
python mediapipe_demo/run_demo.py --video /path/to/free_throw.mp4 --side RIGHT
```

- `--side` is the shooter's shooting hand (`LEFT` or `RIGHT`). It is
  **not** auto-detected — set it based on which hand releases the ball
  in your video.
- `--model` lets you point at a different `.task` file if you didn't
  use the default path.
- `--out-dir` controls where outputs go (default: `mediapipe_demo/outputs/`).

Outputs:
- `<video_stem>_pose_overlay.mp4` — input video with the pose skeleton drawn on it.
- `<video_stem>_angles.png` — elbow and knee angle vs. time.

## What has and hasn't been tested

At the time this was written, no sample basketball free-throw video had
been provided, so the following is what could and couldn't be verified:

**Tested and working:**
- The full pipeline (video read → MediaPipe `PoseLandmarker` in `VIDEO`
  running mode → skeleton overlay drawn with OpenCV → overlay video
  written → 3D world-landmark joint angles computed → angle plot saved)
  runs without errors and produces a correctly overlaid skeleton and
  plausible joint angles, verified using a single real photo of a person
  (public MediaPipe sample asset) repeated as a short synthetic video.
  Elbow/knee angles computed from it were visually consistent with the
  pose in the photo (e.g., ~97-101° knee angle for a visibly bent front
  knee in a lunge).
- Graceful handling of "no person detected" (confirmed with a synthetic
  video containing no person: the script completes, writes the overlay
  video, and reports zero detections instead of crashing or fabricating
  angles).

**Not tested (no sample video available):**
- Behavior on an actual video with real temporal human motion (as
  opposed to a repeated static frame).
- Behavior on an actual basketball free-throw motion specifically.
- Robustness to occlusion, fast motion blur, or camera angles typical of
  a phone-recorded free throw.

If/when a sample video is provided, re-run this script against it and
update this section with what was actually observed.
