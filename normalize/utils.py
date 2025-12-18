"""
Core Detection Utilities

Fundamental utility functions for detection operations.
"""

import numpy as np
from typing import Dict, List, Callable, Optional, TypeVar
from dataclasses import dataclass

INPUT_SIZE = 640  # Standard YOLO input size


@dataclass
class BoundingBox:
    """Bounding box in corner coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float


def compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    IoU measures the overlap between two bounding boxes, used for
    evaluating detection quality and Non-Maximum Suppression.
    
    Args:
        box_a: First bounding box
        box_b: Second bounding box
    
    Returns:
        IoU value between 0 (no overlap) and 1 (perfect overlap)
    """
    # Calculate intersection rectangle
    intersection_x1 = max(box_a.x1, box_b.x1)
    intersection_y1 = max(box_a.y1, box_b.y1)
    intersection_x2 = min(box_a.x2, box_b.x2)
    intersection_y2 = min(box_a.y2, box_b.y2)
    
    # Calculate intersection area
    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height
    
    # Calculate individual box areas
    area_a = (box_a.x2 - box_a.x1) * (box_a.y2 - box_a.y1)
    area_b = (box_b.x2 - box_b.x1) * (box_b.y2 - box_b.y1)
    
    # Calculate union area
    union_area = max(area_a + area_b - intersection_area, 1e-6)
    
    return intersection_area / union_area


T = TypeVar('T', bound=BoundingBox)


def nms(
    detections: List[T],
    iou_thresh: float,
    group_by: Optional[Callable[[T], str | int]] = None
) -> List[T]:
    """
    Generic Non-Maximum Suppression for any bounding-box-like detections.
    Optionally group detections (e.g., by class) before suppression.
    
    Args:
        detections: List of detections with bounding boxes and scores
        iou_thresh: IoU threshold for suppression
        group_by: Optional function to group detections (e.g., by class)
    
    Returns:
        Filtered detections after NMS
    """
    if len(detections) == 0:
        return []
    
    groups: Dict[str, List[T]] = {}
    
    if group_by:
        for det in detections:
            key = str(group_by(det))
            if key not in groups:
                groups[key] = []
            groups[key].append(det)
    else:
        groups['__all__'] = detections
    
    def suppress_group(group: List[T]) -> List[T]:
        sorted_group = sorted(group, key=lambda x: x.score, reverse=True)
        kept: List[T] = []
        
        while len(sorted_group) > 0:
            current = sorted_group.pop(0)
            kept.append(current)
            i = len(sorted_group) - 1
            while i >= 0:
                if compute_iou(current, sorted_group[i]) >= iou_thresh:
                    sorted_group.pop(i)
                i -= 1
        
        return kept
    
    result: List[T] = []
    for group in groups.values():
        result.extend(suppress_group(group))
    
    return result


def to_corner_coordinates(
    cx: float,
    cy: float,
    width: float,
    height: float,
    original_width: int,
    original_height: int,
    default_width: int = INPUT_SIZE,
    default_height: int = INPUT_SIZE
) -> BoundingBox:
    """
    Convert center-size format to corner coordinates.
    
    Args:
        cx: Center x coordinate (in model input space)
        cy: Center y coordinate (in model input space)
        width: Width (in model input space)
        height: Height (in model input space)
        original_width: Original image width
        original_height: Original image height
        default_width: Model input width (default: INPUT_SIZE)
        default_height: Model input height (default: INPUT_SIZE)
    
    Returns:
        BoundingBox in original image coordinates
    """
    cx_scaled = (cx / default_width) * original_width
    cy_scaled = (cy / default_height) * original_height
    width_scaled = (width / default_width) * original_width
    height_scaled = (height / default_height) * original_height
    
    return BoundingBox(
        x1=cx_scaled - width_scaled / 2,
        y1=cy_scaled - height_scaled / 2,
        x2=cx_scaled + width_scaled / 2,
        y2=cy_scaled + height_scaled / 2
    )

