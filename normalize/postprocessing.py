"""
Post-processing Utilities

Handles Non-Maximum Suppression (NMS) and filtering operations
for detection results.
Matches the React implementation in react/postprocessing.ts
"""

from typing import List
from utils import nms, BoundingBox
from constants import BALL_IOU_THRESH, POSE_IOU_THRESH


class BallDetection(BoundingBox):
    """Ball/Hoop detection with class and score."""
    def __init__(self, x1: float, y1: float, x2: float, y2: float, 
                 score: float, class_id: int, class_name: str):
        super().__init__(x1, y1, x2, y2)
        self.score = score
        self.classId = class_id
        self.class_name = class_name


class PoseDetection(BoundingBox):
    """Pose detection with keypoints and score."""
    def __init__(self, x1: float, y1: float, x2: float, y2: float, 
                 score: float, keypoints: List[dict]):
        super().__init__(x1, y1, x2, y2)
        self.score = score
        self.keypoints = keypoints


def apply_ball_hoop_nms(detections: List[BallDetection], iou_thresh: float = None) -> List[BallDetection]:
    """
    Apply class-wise Non-Maximum Suppression to ball/hoop detections.
    
    Groups detections by class, sorts by confidence, and removes
    overlapping detections above the IoU threshold.
    
    Args:
        detections: Array of ball/hoop detections
        iou_thresh: IoU threshold for NMS (defaults to BALL_IOU_THRESH from constants)
    
    Returns:
        Filtered detections after NMS
    """
    if iou_thresh is None:
        iou_thresh = BALL_IOU_THRESH
    return nms(detections, iou_thresh, lambda det: det.classId)


def apply_pose_nms(detections: List[PoseDetection], iou_thresh: float = None) -> List[PoseDetection]:
    """
    Apply Non-Maximum Suppression to pose detections.
    
    Sorts detections by confidence and removes overlapping
    detections above the IoU threshold.
    
    Args:
        detections: Array of pose detections
        iou_thresh: IoU threshold for NMS (defaults to POSE_IOU_THRESH from constants)
    
    Returns:
        Filtered detections after NMS
    """
    if iou_thresh is None:
        iou_thresh = POSE_IOU_THRESH
    return nms(detections, iou_thresh)

