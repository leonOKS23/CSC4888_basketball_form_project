#!/usr/bin/env python3
"""Optional MediaPipe Pose Landmarker demo for a local free-throw video.

THIS SCRIPT IS INDEPENDENT OF THE SPL-TRAINED CLASSIFIER.
==========================================================
It does not run the model trained in scripts/05_train_eval.py, and it
should NOT be treated as evidence that that classifier works on
real-world video. MediaPipe's pose landmarks use a different landmark
set, a different (monocular, image-plane-ish) 3D estimation method, and
different units/coordinate conventions than the SPL motion-capture data
the classifier was trained on. The two have not been shown to be
comparable. This demo exists only to extract and visualize body-mechanics
angles from an arbitrary video, as a separate deliverable.

What this script does:
  1. Reads a local video file.
  2. Runs MediaPipe's PoseLandmarker (Tasks API) on every frame.
  3. Writes an output video with the pose skeleton overlaid.
  4. Computes shooting-side elbow and knee angle per frame (using
     MediaPipe's 3D world landmarks) and plots them over time.

Setup
-----
1. Install dependencies (see requirements.txt):
       pip install mediapipe opencv-python
2. Download a MediaPipe Pose Landmarker model file (not included in this
   repo — see mediapipe_demo/README.md for exact download commands/URLs).
   Default expected path: mediapipe_demo/models/pose_landmarker_lite.task
3. Run:
       python mediapipe_demo/run_demo.py --video /path/to/video.mp4

See mediapipe_demo/README.md for full details, troubleshooting, and
exactly what has/hasn't been tested without a sample video.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError as e:  # pragma: no cover
    print("ERROR: mediapipe is not installed. Run `pip install mediapipe opencv-python`.")
    raise

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = THIS_DIR / "models" / "pose_landmarker_lite.task"

# Standard MediaPipe Pose landmark indices (33-point model). These are a
# *different* landmark set from the SPL dataset's 69 named keypoints —
# there is no 1:1 correspondence, and no claim is made that angles
# computed here are numerically comparable to the SPL-derived features.
LANDMARK_IDX = {
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
}

# Connections drawn for the skeleton overlay (image-plane landmarks).
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]


def joint_angle_3d(a, b, c) -> float:
    """Angle at vertex b formed by points a-b-c, in degrees. a/b/c are
    length-3 arrays (x, y, z)."""
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    ba = a - b
    bc = c - b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na == 0 or nc == 0:
        return float("nan")
    cos_angle = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def draw_skeleton(frame, landmarks_px):
    for (i, j) in POSE_CONNECTIONS:
        if i in landmarks_px and j in landmarks_px:
            cv2.line(frame, landmarks_px[i], landmarks_px[j], (0, 255, 0), 2)
    for idx, (x, y) in landmarks_px.items():
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="Path to a local input video file")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH),
                         help=f"Path to a MediaPipe pose_landmarker .task model file "
                              f"(default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--side", choices=["LEFT", "RIGHT"], default="RIGHT",
                         help="Shooting side to compute elbow/knee angles for (default: RIGHT). "
                              "This is NOT auto-detected; set it based on the video.")
    parser.add_argument("--out-dir", default=str(THIS_DIR / "outputs"),
                         help="Directory to write the overlay video and angle plot")
    args = parser.parse_args()

    video_path = Path(args.video)
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"BLOCKED: video file not found: {video_path}")
        sys.exit(1)
    if not model_path.exists():
        print(f"BLOCKED: MediaPipe model file not found: {model_path}")
        print("See mediapipe_demo/README.md for download instructions.")
        sys.exit(1)

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"BLOCKED: could not open video file: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = out_dir / f"{video_path.stem}_pose_overlay.mp4"
    writer = cv2.VideoWriter(str(out_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    elbow_name_l, elbow_name_r = "LEFT_ELBOW", "RIGHT_ELBOW"
    shoulder_idx = LANDMARK_IDX[f"{args.side}_SHOULDER"]
    elbow_idx = LANDMARK_IDX[f"{args.side}_ELBOW"]
    wrist_idx = LANDMARK_IDX[f"{args.side}_WRIST"]
    hip_idx = LANDMARK_IDX[f"{args.side}_HIP"]
    knee_idx = LANDMARK_IDX[f"{args.side}_KNEE"]
    ankle_idx = LANDMARK_IDX[f"{args.side}_ANKLE"]

    times_s, elbow_angles, knee_angles = [], [], []
    frame_idx = 0
    n_detected = 0

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            timestamp_ms = int((frame_idx / fps) * 1000)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                n_detected += 1
                lm_2d = result.pose_landmarks[0]
                landmarks_px = {
                    i: (int(lm_2d[i].x * width), int(lm_2d[i].y * height))
                    for i in range(len(lm_2d))
                    if i in set(LANDMARK_IDX.values())
                }
                frame_bgr = draw_skeleton(frame_bgr, landmarks_px)

                if result.pose_world_landmarks:
                    lm_3d = result.pose_world_landmarks[0]

                    def pt(i):
                        return (lm_3d[i].x, lm_3d[i].y, lm_3d[i].z)

                    elbow_angle = joint_angle_3d(pt(shoulder_idx), pt(elbow_idx), pt(wrist_idx))
                    knee_angle = joint_angle_3d(pt(hip_idx), pt(knee_idx), pt(ankle_idx))
                    times_s.append(frame_idx / fps)
                    elbow_angles.append(elbow_angle)
                    knee_angles.append(knee_angle)

            writer.write(frame_bgr)
            frame_idx += 1

    cap.release()
    writer.release()

    print(f"Processed {frame_idx} frames; pose detected in {n_detected} frames.")
    print(f"Wrote overlay video: {out_video_path}")

    if times_s:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(times_s, elbow_angles, color="tab:green")
        axes[0].set_ylabel(f"{args.side} elbow angle (deg)")
        axes[1].plot(times_s, knee_angles, color="tab:brown")
        axes[1].set_ylabel(f"{args.side} knee angle (deg)")
        axes[1].set_xlabel("time (s)")
        fig.suptitle(f"MediaPipe pose angles: {video_path.name}")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_plot_path = out_dir / f"{video_path.stem}_angles.png"
        fig.savefig(out_plot_path, dpi=110)
        plt.close(fig)
        print(f"Wrote angle plot: {out_plot_path}")
    else:
        print("No pose was detected in any frame — no angle plot was produced. "
              "Check that the video actually shows a person, and that --side is set correctly.")


if __name__ == "__main__":
    main()
