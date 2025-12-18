"""
Image/Video Preprocessing Utilities

Handles conversion of images/video frames to tensors
suitable for YOLO model inference.
Matches the React implementation in react/preprocessing.ts
"""

import numpy as np
import cv2
from typing import Tuple
from constants import INPUT_SIZE


def preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Preprocess an image to a tensor for YOLO model input.
    
    Resizes the input to INPUT_SIZE x INPUT_SIZE and converts to normalized
    RGB float32 tensor in CHW format (channels, height, width).
    
    Args:
        image: Input image as numpy array (H, W, C) in BGR format (OpenCV default)
    
    Returns:
        Tuple of (preprocessed_tensor, original_width, original_height)
        - preprocessed_tensor: numpy array of shape (1, 3, INPUT_SIZE, INPUT_SIZE) in RGB format
        - original_width: Original image width
        - original_height: Original image height
    """
    original_height, original_width = image.shape[:2]
    
    # Resize image to INPUT_SIZE x INPUT_SIZE
    resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] and convert to float32
    normalized = rgb_image.astype(np.float32) / 255.0
    
    # Convert HWC to CHW format (channels, height, width)
    # Shape: (INPUT_SIZE, INPUT_SIZE, 3) -> (3, INPUT_SIZE, INPUT_SIZE)
    chw_image = np.transpose(normalized, (2, 0, 1))
    
    # Add batch dimension: (1, 3, INPUT_SIZE, INPUT_SIZE)
    batched = np.expand_dims(chw_image, axis=0)
    
    return batched, original_width, original_height

