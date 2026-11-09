"""Central configuration/constants for the SPL free-throw pipeline.

Keeping these in one place makes every assumption we made about the
dataset (confirmed empirically — see PROJECT_LOG.md) explicit and easy
to audit/change.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
# Where the SPL-Open-Data repo gets cloned by scripts/00_download_data.py
SPL_REPO_DIR = DATA_RAW_DIR / "SPL-Open-Data"
# The free-throw trial JSON files live under this subfolder of the SPL repo.
FREETHROW_DATA_DIR = SPL_REPO_DIR / "basketball" / "freethrow" / "data"

OUTPUTS_DIR = REPO_ROOT / "outputs"
DATASET_INSPECTION_DIR = OUTPUTS_DIR / "dataset_inspection"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RELEASE_CHECK_FIGURES_DIR = FIGURES_DIR / "release_checks"
FEATURE_CHECK_FIGURES_DIR = FIGURES_DIR / "feature_checks"
EVAL_KNOWN_DIR = OUTPUTS_DIR / "eval_known_players"
EVAL_UNSEEN_DIR = OUTPUTS_DIR / "eval_unseen_players"
FEATURES_CSV = OUTPUTS_DIR / "features.csv"
EXCLUSIONS_CSV = OUTPUTS_DIR / "exclusions.csv"

# ---------------------------------------------------------------------------
# Dataset conventions (confirmed by inspecting the real files — see
# PROJECT_LOG.md / outputs/dataset_inspection/)
# ---------------------------------------------------------------------------
# Vertical axis index into each [x, y, z] keypoint/ball position.
VERTICAL_AXIS = 2  # 'z'

# Keypoint names as they literally appear in the trial JSON files.
KEYPOINTS_NEEDED = [
    "RIGHT_SHOULDER", "LEFT_SHOULDER",
    "RIGHT_ELBOW", "LEFT_ELBOW",
    "RIGHT_WRIST", "LEFT_WRIST",
    "RIGHT_HIP", "LEFT_HIP",
    "RIGHT_KNEE", "LEFT_KNEE",
    "RIGHT_ANKLE", "LEFT_ANKLE",
]

RESULT_MADE = "made"
RESULT_MISSED = "missed"

# ---------------------------------------------------------------------------
# Release detection (ball-to-wrist distance heuristic; see spl_ft/release.py)
# ---------------------------------------------------------------------------
# A ball is considered "in contact" with a wrist when the 3D distance
# between the ball center and that wrist marker is below this threshold
# (feet). A regulation basketball has a ~0.39 ft radius; we allow some
# slack for marker placement / hand size.
CONTACT_DISTANCE_THRESHOLD_FT = 0.8

# After the detected "last contact" frame, how many subsequent frames must
# stay separated (used as a sanity check that the ball is actually flying
# away rather than a single noisy frame).
RELEASE_CONFIRM_FRAMES = 3

# ---------------------------------------------------------------------------
# Pre-release feature window
# ---------------------------------------------------------------------------
# Buffer subtracted from the detected release frame before the analysis
# window ends, to reduce the risk of leaking post-release / in-release
# information (hand-ball separation is a heuristic estimate, not ground
# truth, so we deliberately back off a little further).
RELEASE_BUFFER_SEC = 0.05

# Length of the pre-release analysis window.
WINDOW_SEC = 0.5

# Minimum fraction of WINDOW_SEC that must actually be available (i.e. not
# clipped by the start of the trial) for a shot to be usable.
MIN_WINDOW_FRACTION = 0.8

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TRAIN_FRACTION = 0.8
