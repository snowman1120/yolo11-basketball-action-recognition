"""
Pose Detection Parser

Parses YOLO pose model output for human pose detections.
Matches the React implementation in react/parsers/pose-detection.ts
"""

import numpy as np
from typing import List, Callable, Dict, Any
from postprocessing import PoseDetection
from utils import to_corner_coordinates
from constants import INPUT_SIZE, KEYPOINT_DIMS, NUM_KEYPOINTS, POSE_ATTRS_PER_DET

# COCO keypoint names (17 keypoints)
KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"
]


def parse_pose_layout(
    detection_count: int,
    get_value: Callable[[int, int], float],
    confidence_threshold: float,
    original_width: int,
    original_height: int
) -> List[PoseDetection]:
    """
    Parse pose detections from a specific tensor layout.
    
    Args:
        detection_count: Number of detections
        get_value: Function to get value at (detection_index, attribute_offset)
        confidence_threshold: Minimum confidence threshold
        original_width: Original image width
        original_height: Original image height
        
    Returns:
        List of parsed pose detections
    """
    detections: List[PoseDetection] = []
    
    for i in range(detection_count):
        # Extract bounding box (center-size format)
        center_x = get_value(i, 0)
        center_y = get_value(i, 1)
        width = get_value(i, 2)
        height = get_value(i, 3)
        object_confidence = get_value(i, 4)
        score = object_confidence
        
        # Filter by confidence threshold
        if score < confidence_threshold:
            continue
        
        # Convert to corner coordinates
        bbox = to_corner_coordinates(
            center_x,
            center_y,
            width,
            height,
            original_width,
            original_height
        )
        
        # Extract keypoints
        keypoints: List[Dict[str, Any]] = []
        keypoint_start_offset = 5  # After bbox (4) + object conf (1)
        
        for k in range(NUM_KEYPOINTS):
            base_offset = keypoint_start_offset + k * KEYPOINT_DIMS
            keypoint_x = get_value(i, base_offset)
            keypoint_y = get_value(i, base_offset + 1)
            keypoint_confidence = get_value(i, base_offset + 2)
            
            keypoints.append({
                "name": KEYPOINT_NAMES[k] if k < len(KEYPOINT_NAMES) else f"keypoint_{k}",
                "x": (keypoint_x / INPUT_SIZE) * original_width,
                "y": (keypoint_y / INPUT_SIZE) * original_height,
                "confidence": keypoint_confidence
            })
        
        detections.append(PoseDetection(
            bbox.x1, bbox.y1, bbox.x2, bbox.y2,
            score, keypoints
        ))
    
    return detections


def parse_pose_detections(
    output: np.ndarray,
    original_width: int,
    original_height: int,
    confidence_threshold: float = 0.15
) -> List[PoseDetection]:
    """
    Parse pose detections from YOLO pose model output.
    
    Handles multiple tensor layouts:
    - [1, POSE_ATTRS_PER_DET, N] (channel-first)
    - [1, N, POSE_ATTRS_PER_DET] (detection-first)
    - [N, POSE_ATTRS_PER_DET] (no batch dimension)
    - Flat array (fallback)
    
    Args:
        output: Model output tensor as numpy array
        original_width: Original image/video width
        original_height: Original image/video height
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        Array of parsed pose detections
    """
    # Flatten to 1D for easier handling
    data = output.flatten()
    dims = output.shape
    
    # Handle different tensor layouts
    if len(dims) == 3 and dims[0] == 1 and dims[1] == POSE_ATTRS_PER_DET:
        # Layout: [1, POSE_ATTRS_PER_DET, N] - channel-first
        detection_count = dims[2]
        stride = detection_count
        
        def get_value(cell: int, offset: int) -> float:
            return float(data[offset * stride + cell])
        
        return parse_pose_layout(
            detection_count,
            get_value,
            confidence_threshold,
            original_width,
            original_height
        )
    
    if len(dims) == 3 and dims[0] == 1 and dims[2] == POSE_ATTRS_PER_DET:
        # Layout: [1, N, POSE_ATTRS_PER_DET] - detection-first
        detection_count = dims[1]
        stride = dims[2]
        
        def get_value(index: int, offset: int) -> float:
            return float(data[index * stride + offset])
        
        return parse_pose_layout(
            detection_count,
            get_value,
            confidence_threshold,
            original_width,
            original_height
        )
    
    if len(dims) == 2 and dims[1] == POSE_ATTRS_PER_DET:
        # Layout: [N, POSE_ATTRS_PER_DET] - no batch dimension
        detection_count = dims[0]
        stride = dims[1]
        
        def get_value(index: int, offset: int) -> float:
            return float(data[index * stride + offset])
        
        return parse_pose_layout(
            detection_count,
            get_value,
            confidence_threshold,
            original_width,
            original_height
        )
    
    # Fallback: treat as flat array
    flat_count = len(data) // POSE_ATTRS_PER_DET
    
    def get_value(index: int, offset: int) -> float:
        return float(data[index * POSE_ATTRS_PER_DET + offset])
    
    return parse_pose_layout(
        flat_count,
        get_value,
        confidence_threshold,
        original_width,
        original_height
    )

