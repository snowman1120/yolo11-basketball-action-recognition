"""
Detection Constants

Constants used for pose and ball detection matching the React implementation.
"""

# -----------------------------
# Runtime / pipeline settings
# -----------------------------
# These replace normalize/config.json. Edit this file to change defaults.

# Model paths
POSE_MODEL_PATH = "models/yolo11x-pose.pt"
BALL_MODEL_PATH = "models/yolo11n-trained.onnx"

# Inference device
DEVICE = "cpu"  # "cpu" or "cuda"

# Enable/disable detectors by default
ENABLE_BALL = True
ENABLE_POSE = True

# Video sampling
# - Set to 0 (or any falsy value) to process all frames.
TARGET_FPS = 30

# Output defaults
DEFAULT_OUTPUT_FILE = "output.json"

# Label stored in output JSON metadata + per-frame entries
DEFAULT_LABEL = "other"

# Use ONNX for pose detection (matches React implementation more closely)
USE_ONNX_POSE = False

# Model input size
INPUT_SIZE = 640

# Pose detection constants
NUM_KEYPOINTS = 17  # COCO format
KEYPOINT_DIMS = 3  # x, y, confidence
POSE_ATTRS_PER_DET = 5 + (NUM_KEYPOINTS * KEYPOINT_DIMS)  # 5 (bbox + obj_conf) + 51 (keypoints)

# Ball/Hoop detection constants
NUM_CLASSES = 2  # ball, hoop
CLASS_NAMES = ["ball", "hoop"]
ATTRS_PER_DET = 4 + NUM_CLASSES  # 4 (bbox) + 2 (class scores) = 6

# Thresholds / tuning parameters
BALL_IOU_THRESH = 0.55
POSE_IOU_THRESH = 0.5
BALL_CONF_THRESH = 0.2
POSE_CONF_THRESH = 0.15
#
# Note: this threshold is used for "full body" coverage scoring when selecting
# the best pose (not for filtering keypoints out of the output JSON).
KEYPOINT_CONF_THRESH = 0.2

