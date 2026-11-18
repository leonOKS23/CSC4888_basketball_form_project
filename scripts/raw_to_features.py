"""Raw video (your own recordings) -> one feature row per shot, same features as SPL.

IN:   one or more video files, one free throw per clip, e.g. data/own/*.mp4
      --model path/to/pose_landmarker_*.task  (MediaPipe Pose Landmarker model,
        download from https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
      --labels optional CSV with columns: video, label  (1=made, 0=missed)
      --side LEFT|RIGHT optional; otherwise guessed per clip from elbow extension

OUT:  data/processed/raw_features.csv
          video, label, shooting_side, release_frame, fps, + pose_features.FEATURES
      data/processed/raw_excluded.csv
          video, reason

Steps per clip:
  1. MediaPipe Pose Landmarker -> 33 *world* landmarks per frame (estimated 3D,
     meters, origin at the hip center, y pointing DOWN).
  2. mediapipe_to_skeleton: keep the 12 common joints and rotate so z is UP.
     Units (meters vs SPL feet) don't matter: features are divided by torso length.
  3. Release = peak upward speed of the shooting wrist (no ball available).
     On SPL this lands a median ~4 frames (~0.07 s @ 60 FPS) BEFORE the
     ball-based release; see SPL_to_features.py output.
  4. Features from pose_features.extract_features (identical code to SPL).

Needs: pip install mediapipe opencv-python   (not in requirements.txt yet)
Run:   .venv/bin/python scripts/raw_to_features.py data/own/*.mp4 --model models/pose_landmarker_heavy.task
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pose_features import JOINTS, extract_features, release_from_body, shooting_side_from_body

ROOT = Path(__file__).resolve().parents[1]

# MediaPipe Pose landmark indices for the 12 common joints.
MEDIAPIPE_INDEX = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12, 'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15, 'RIGHT_WRIST': 16, 'LEFT_HIP': 23, 'RIGHT_HIP': 24,
    'LEFT_KNEE': 25, 'RIGHT_KNEE': 26, 'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
}


def mediapipe_to_skeleton(world):
    """(n, 33, 3) MediaPipe world landmarks (x, y-down, z) -> (n, 12, 3) z-up skeleton.

    Rotation (x, y, z) -> (x, z, -y): proper rotation, keeps left/right handedness.
    """
    sub = world[:, [MEDIAPIPE_INDEX[j] for j in JOINTS]]
    return np.stack([sub[..., 0], sub[..., 2], -sub[..., 1]], axis=-1)


def video_to_world_landmarks(video_path, model_path):
    """Video -> ((n, 33, 3) world landmarks with NaN where no person found, fps)."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO, num_poses=1,
    )
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    with PoseLandmarker.create_from_options(options) as landmarker:
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(image, int(i * 1000 / fps))
            if result.pose_world_landmarks:
                frames.append([[p.x, p.y, p.z] for p in result.pose_world_landmarks[0]])
            else:
                frames.append([[np.nan] * 3] * 33)
            i += 1
    cap.release()
    return np.asarray(frames, dtype=float), fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('videos', nargs='+', type=Path)
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--labels', type=Path)
    ap.add_argument('--side', choices=['LEFT', 'RIGHT'])
    ap.add_argument('--out', type=Path, default=ROOT / 'data/processed/raw_features.csv')
    args = ap.parse_args()

    labels = {}
    if args.labels:
        labels = dict(pd.read_csv(args.labels).astype({'video': str}).values)

    rows, excluded = [], []
    for video in args.videos:
        world, fps = video_to_world_landmarks(video, args.model)
        skel = mediapipe_to_skeleton(world)
        side = args.side or shooting_side_from_body(skel, fps)
        release = release_from_body(skel, fps, side)
        feats, reason = extract_features(skel, fps, side, release)
        if feats is None:
            excluded.append(dict(video=str(video), reason=reason))
            continue
        rows.append(dict(video=str(video), label=labels.get(video.name, labels.get(str(video))),
                         shooting_side=side, release_frame=release, fps=fps, **feats))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    pd.DataFrame(excluded, columns=['video', 'reason']).to_csv(
        args.out.with_name('raw_excluded.csv'), index=False)
    print(f'Kept {len(rows)} / {len(args.videos)} clips -> {args.out}')


if __name__ == '__main__':
    main()
