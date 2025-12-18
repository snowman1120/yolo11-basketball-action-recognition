"""
Ball Detection Parser

Parses YOLO model output for ball and hoop detections.
Matches the React implementation in react/parsers/ball-detection.ts
"""

import numpy as np
from typing import List, Set, Literal, Callable
from postprocessing import BallDetection
from utils import to_corner_coordinates
from constants import ATTRS_PER_DET, NUM_CLASSES, CLASS_NAMES

ClassFilter = Literal["ball", "hoop", "both"]


def get_enabled_class_indices(class_filter: ClassFilter) -> Set[int]:
    """
    Get enabled class indices based on filter.
    
    Args:
        class_filter: Filter for which classes to include
        
    Returns:
        Set of enabled class indices
    """
    if class_filter == "ball":
        return {0}
    elif class_filter == "hoop":
        return {1}
    else:  # "both"
        return {0, 1}


def parse_ball_hoop_layout(
    detection_count: int,
    get_value: Callable[[int, int], float],
    enabled_classes: Set[int],
    confidence_threshold: float,
    original_width: int,
    original_height: int
) -> List[BallDetection]:
    """
    Parse detections from a specific tensor layout.
    
    Args:
        detection_count: Number of detections
        get_value: Function to get value at (detection_index, attribute_offset)
        enabled_classes: Set of enabled class indices
        confidence_threshold: Minimum confidence threshold
        original_width: Original image width
        original_height: Original image height
        
    Returns:
        List of parsed ball/hoop detections
    """
    detections: List[BallDetection] = []
    
    for i in range(detection_count):
        # Extract bounding box (center-size format)
        center_x = get_value(i, 0)
        center_y = get_value(i, 1)
        width = get_value(i, 2)
        height = get_value(i, 3)
        
        # Find best class and confidence
        best_class = -1
        best_score = float('-inf')
        
        for class_index in range(NUM_CLASSES):
            score = get_value(i, 4 + class_index)
            if score > best_score:
                best_score = score
                best_class = class_index
        
        # Filter by confidence threshold and enabled classes
        if best_class == -1 or best_score < confidence_threshold:
            continue
        if best_class not in enabled_classes:
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
        
        class_name = CLASS_NAMES[best_class] if best_class < len(CLASS_NAMES) else f"class_{best_class}"
        
        detections.append(BallDetection(
            bbox.x1, bbox.y1, bbox.x2, bbox.y2,
            best_score, best_class, class_name
        ))
    
    return detections


def parse_ball_detections(
    output: np.ndarray,
    original_width: int,
    original_height: int,
    class_filter: ClassFilter = "both",
    confidence_threshold: float = 0.4
) -> List[BallDetection]:
    """
    Parse ball/hoop detections from YOLO model output.
    
    Handles multiple tensor layouts:
    - [1, ATTRS_PER_DET, N] (channel-first)
    - [1, N, ATTRS_PER_DET] (detection-first)
    - [N, ATTRS_PER_DET] (no batch dimension)
    - Flat array (fallback)
    
    Args:
        output: Model output tensor as numpy array
        original_width: Original image/video width
        original_height: Original image/video height
        class_filter: Filter for which classes to include
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        Array of parsed detections
    """
    enabled_classes = get_enabled_class_indices(class_filter)
    
    # Flatten to 1D for easier handling
    data = output.flatten()
    dims = output.shape
    
    # Handle different tensor layouts
    if len(dims) == 3 and dims[0] == 1 and dims[1] == ATTRS_PER_DET:
        # Layout: [1, ATTRS_PER_DET, N] - channel-first
        detection_count = dims[2]
        stride = detection_count
        
        def get_value(index: int, offset: int) -> float:
            return float(data[offset * stride + index])
        
        return parse_ball_hoop_layout(
            detection_count,
            get_value,
            enabled_classes,
            confidence_threshold,
            original_width,
            original_height
        )
    
    if len(dims) == 3 and dims[0] == 1 and dims[2] == ATTRS_PER_DET:
        # Layout: [1, N, ATTRS_PER_DET] - detection-first
        detection_count = dims[1]
        stride = dims[2]
        
        def get_value(index: int, offset: int) -> float:
            return float(data[index * stride + offset])
        
        return parse_ball_hoop_layout(
            detection_count,
            get_value,
            enabled_classes,
            confidence_threshold,
            original_width,
            original_height
        )
    
    if len(dims) == 2 and dims[1] == ATTRS_PER_DET:
        # Layout: [N, ATTRS_PER_DET] - no batch dimension
        detection_count = dims[0]
        stride = dims[1]
        
        def get_value(index: int, offset: int) -> float:
            return float(data[index * stride + offset])
        
        return parse_ball_hoop_layout(
            detection_count,
            get_value,
            enabled_classes,
            confidence_threshold,
            original_width,
            original_height
        )
    
    # Fallback: treat as flat array
    flat_length = len(data)
    detection_count = flat_length // ATTRS_PER_DET
    
    def get_value(index: int, offset: int) -> float:
        return float(data[index * ATTRS_PER_DET + offset])
    
    return parse_ball_hoop_layout(
        detection_count,
        get_value,
        enabled_classes,
        confidence_threshold,
        original_width,
        original_height
    )


def get_class_name(class_id: int) -> str:
    """
    Get human-readable class name from class ID.
    
    Args:
        class_id: Class index
        
    Returns:
        Class name string
    """
    return CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"

