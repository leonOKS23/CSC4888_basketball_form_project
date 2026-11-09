# MediaPipe pose demo (optional, separate)

This is a separate script that takes a local video and overlays a MediaPipe pose skeleton on it, plus plots shooting-side elbow/knee angle over time.

**This has nothing to do with the SPL-trained classifier.** It's a completely different pose system — MediaPipe Pose (monocular RGB, 33 landmarks) vs the SPL motion-capture data the classifier was trained on (markerless multi-camera, 69 named keypoints, real-world feet). I haven't shown the two produce comparable angles at all, and I haven't run the classifier on anything MediaPipe produces. Treat this as just a visualization tool, nothing more.

## Setup

1. Install the extra deps (already in `requirements.txt`, listed here since they're heavier/more optional than the rest):

   ```bash
   pip install mediapipe opencv-python
   ```

   I tested this with `mediapipe==1.2.2` / `opencv-python==5.0.0` on Python 3.12, macOS. If `pip install mediapipe` fails for you, check [MediaPipe's install docs](https://ai.google.dev/edge/mediapipe/solutions/setup_python) — it's pickier about platform/Python version than the rest of this project.

2. Grab a Pose Landmarker model file (`.task`). Not included in this repo (it's gitignored), you need to download it yourself. Save it to `mediapipe_demo/models/pose_landmarker_lite.task`:

   ```bash
   mkdir -p mediapipe_demo/models
   curl -L -o mediapipe_demo/models/pose_landmarker_lite.task \
     https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
   ```

   There's also `_full` and `_heavy` versions if you want more accuracy (slower) — just swap the name in both the URL and the `--model` flag. URLs are from [Google's Pose Landmarker page](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker).

## Running it

```bash
python mediapipe_demo/run_demo.py --video /path/to/free_throw.mp4 --side RIGHT
```

- `--side` is the shooting hand (`LEFT` or `RIGHT`) — I'm not auto-detecting this, just set it based on the video.
- `--model` if your `.task` file is somewhere else.
- `--out-dir` for where to put the outputs (defaults to `mediapipe_demo/outputs/`).

You get:
- `<video>_pose_overlay.mp4` — the input video with the skeleton drawn on top.
- `<video>_angles.png` — elbow and knee angle over time.

## What I actually tested vs. what I haven't

I don't have a sample free-throw video yet, so here's exactly what I could and couldn't check:

**Works, confirmed:**
- The whole pipeline (read video → MediaPipe PoseLandmarker in video mode → draw skeleton with OpenCV → write overlay video → compute 3D joint angles from world landmarks → plot) runs without errors and gives a correctly drawn skeleton with angles that make sense. I checked this using a real photo of a person (public MediaPipe sample image) looped into a short fake "video" — detection, drawing, and the angle math all check out. For example it read ~97-101° for the knee angle on a visibly bent front knee in a lunge pose, which is about right.
- It handles "no person in the frame" gracefully too — tried a synthetic video with nothing in it and it just finishes and reports zero detections instead of crashing or making up numbers.

**Haven't tested:**
- An actual video with real motion in it (as opposed to one photo repeated).
- An actual basketball free throw.
- Occlusion, motion blur, weird camera angles — the stuff you'd actually run into with a phone recording of a free throw.

Once I have a real video, I'll run this against it and come back and update this section with what actually happened.
