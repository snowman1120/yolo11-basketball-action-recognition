"""
Main Detection Script for Basketball Player Pose and Ball Detection

Uses yolo11x-pose.pt for pose detection and yolo11n-trained.onnx for ball/hoop detection.
Processes video files and outputs detection results in JSON format matching data.json.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import onnxruntime as ort
from ultralytics import YOLO

from preprocessing import preprocess_image
from postprocessing import BallDetection, PoseDetection, apply_ball_hoop_nms, apply_pose_nms
from constants import (
    INPUT_SIZE,
    BALL_IOU_THRESH,
    POSE_IOU_THRESH,
    BALL_CONF_THRESH,
    POSE_CONF_THRESH,
    KEYPOINT_CONF_THRESH,
    POSE_MODEL_PATH,
    BALL_MODEL_PATH,
    DEVICE,
    ENABLE_BALL,
    ENABLE_POSE,
    TARGET_FPS,
    DEFAULT_OUTPUT_FILE,
    DEFAULT_LABEL,
    USE_ONNX_POSE,
)
from parsers import parse_ball_detections, parse_pose_detections, ClassFilter, KEYPOINT_NAMES
from ball_tracking_pipeline import compute_ball_centers_from_json, BallCenters


class BasketballDetector:
    """Main detector class for pose and ball detection."""
    
    def __init__(
        self,
        pose_model_path: str = "models/yolo11x-pose.pt",
        ball_model_path: str = "models/yolo11n-trained.pt",
        device: str = "cuda",
        ball_conf_thresh: float = BALL_CONF_THRESH,
        pose_conf_thresh: float = POSE_CONF_THRESH,
        keypoint_conf_thresh: float = KEYPOINT_CONF_THRESH,
        ball_iou_thresh: float = BALL_IOU_THRESH,
        pose_iou_thresh: float = POSE_IOU_THRESH,
        use_onnx_pose: bool = False,
        label="other"
    ):
        """
        Initialize the detector with pose and ball models.
        
        Args:
            pose_model_path: Path to pose model (.pt for Ultralytics YOLO or .onnx for ONNX)
            ball_model_path: Path to ball model (.pt for Ultralytics YOLO or .onnx for ONNX)
            device: Device to run inference on ('cpu' or 'cuda')
            ball_conf_thresh: Confidence threshold for ball detection
            pose_conf_thresh: Confidence threshold for pose detection
            keypoint_conf_thresh: Confidence threshold for keypoints
            ball_iou_thresh: IoU threshold for ball detection NMS
            pose_iou_thresh: IoU threshold for pose detection NMS
            use_onnx_pose: If True, use ONNX for pose detection (matches React implementation)
        """
        self.ball_conf_thresh = ball_conf_thresh
        self.pose_conf_thresh = pose_conf_thresh
        self.keypoint_conf_thresh = keypoint_conf_thresh
        self.ball_iou_thresh = ball_iou_thresh
        self.pose_iou_thresh = pose_iou_thresh
        self.use_onnx_pose = use_onnx_pose
        self.pose_model_path = pose_model_path
        self.ball_model_path = ball_model_path
        self.label = label
        
        # Load pose model
        print(f"Loading pose model from {pose_model_path}...")
        if use_onnx_pose or pose_model_path.endswith('.onnx'):
            # Use ONNX for pose detection (matches React implementation)
            self.pose_session = ort.InferenceSession(
                pose_model_path,
                providers=['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.pose_input_name = self.pose_session.get_inputs()[0].name
            self.pose_output_names = [output.name for output in self.pose_session.get_outputs()]
            self.pose_model = None
        else:
            # Use Ultralytics YOLO (easier to use, but different from React)
            self.pose_model = YOLO(pose_model_path)
            self.pose_model.to(device)
            self.pose_session = None
        
        # Load ball model
        print(f"Loading ball model from {ball_model_path}...")
        if ball_model_path.endswith('.onnx'):
            # Use ONNX for ball detection (matches React implementation)
            self.ball_session = ort.InferenceSession(
                ball_model_path,
                providers=['CPUExecutionProvider'] if device == 'cpu' else ['AzureExecutionProvider', 'CPUExecutionProvider']
            )
            self.ball_input_name = self.ball_session.get_inputs()[0].name
            self.ball_output_names = [output.name for output in self.ball_session.get_outputs()]
            self.ball_model = None
        else:
            # Use Ultralytics YOLO for .pt files
            self.ball_model = YOLO(ball_model_path)
            self.ball_model.to(device)
            self.ball_session = None
            self.ball_input_name = None
            self.ball_output_names = None
        
        print("Models loaded successfully!")
    
    def detect_ball(self, image: np.ndarray, class_filter: ClassFilter = "both") -> List[BallDetection]:
        """
        Detect ball and hoop in the image using ball model.
        Supports both Ultralytics YOLO and ONNX models.
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR format
            class_filter: Filter for which classes to include ("ball", "hoop", or "both")
        
        Returns:
            List of BallDetection objects
        """
        if self.ball_session is not None:
            # Use ONNX for ball detection (matches React implementation)
            return self._detect_ball_onnx(image, class_filter)
        else:
            # Use Ultralytics YOLO
            return self._detect_ball_yolo(image, class_filter)
    
    def _detect_ball_onnx(self, image: np.ndarray, class_filter: ClassFilter = "both") -> List[BallDetection]:
        """
        Detect ball using ONNX model (matches React implementation).
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR format
            class_filter: Filter for which classes to include ("ball", "hoop", or "both")
        
        Returns:
            List of BallDetection objects
        """
        # Preprocess image
        tensor, orig_width, orig_height = preprocess_image(image)
        
        # Run inference
        outputs = self.ball_session.run(self.ball_output_names, {self.ball_input_name: tensor})
        
        # Get output tensor (handle different output formats)
        output = outputs[0]  # Shape can vary: [1, N, ATTRS_PER_DET] or [1, ATTRS_PER_DET, N] etc.
        
        # Use the parser matching React implementation
        detections = parse_ball_detections(
            output,
            orig_width,
            orig_height,
            class_filter=class_filter,
            confidence_threshold=self.ball_conf_thresh
        )
        
        # Apply NMS (already done in parser, but keeping for consistency)
        return apply_ball_hoop_nms(detections, self.ball_iou_thresh)
    
    def _detect_ball_yolo(self, image: np.ndarray, class_filter: ClassFilter = "both") -> List[BallDetection]:
        """
        Detect ball using Ultralytics YOLO.
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR format
            class_filter: Filter for which classes to include ("ball", "hoop", or "both")
        
        Returns:
            List of BallDetection objects
        """
        # NOTE: Ultralytics handles BGR->RGB conversion internally for OpenCV/numpy images.
        # Passing an already-RGB numpy array can result in a double channel swap and missed detections.
        results = self.ball_model(image, conf=self.ball_conf_thresh, verbose=False)
        
        detections: List[BallDetection] = []
        orig_height, orig_width = image.shape[:2]
        
        # Map class filter to enabled class indices (assuming 0=ball, 1=hoop based on parsers)
        enabled_classes = set()
        if class_filter == "ball":
            enabled_classes.add(0)
        elif class_filter == "hoop":
            enabled_classes.add(1)
        else:  # "both"
            enabled_classes.add(0)
            enabled_classes.add(1)
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                # Get bounding box
                conf = float(box.conf[0])
                if conf < self.ball_conf_thresh:
                    continue
                
                # Get class
                cls_id = int(box.cls[0])
                if cls_id not in enabled_classes:
                    continue
                
                # YOLO returns boxes in xyxy format
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get class name
                class_name = result.names[cls_id] if cls_id in result.names else f"class_{cls_id}"
                
                detections.append(BallDetection(
                    float(x1), float(y1), float(x2), float(y2),
                    conf, cls_id, class_name
                ))
        
        # Apply NMS
        return apply_ball_hoop_nms(detections, self.ball_iou_thresh)
    
    def detect_pose(self, image: np.ndarray) -> List[PoseDetection]:
        """
        Detect pose keypoints in the image using pose model.
        Supports both Ultralytics YOLO and ONNX models.
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR format
        
        Returns:
            List of PoseDetection objects
        """
        if self.pose_session is not None:
            # Use ONNX for pose detection (matches React implementation)
            return self._detect_pose_onnx(image)
        else:
            # Use Ultralytics YOLO (easier to use, but different from React)
            return self._detect_pose_yolo(image)
    
    def _detect_pose_onnx(self, image: np.ndarray) -> List[PoseDetection]:
        """
        Detect pose using ONNX model (matches React implementation).
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR format
        
        Returns:
            List of PoseDetection objects
        """
        # Preprocess image
        tensor, orig_width, orig_height = preprocess_image(image)
        
        # Run inference
        outputs = self.pose_session.run(self.pose_output_names, {self.pose_input_name: tensor})
        
        # Get output tensor (handle different output formats)
        output = outputs[0]  # Shape can vary: [1, N, POSE_ATTRS_PER_DET] or [1, POSE_ATTRS_PER_DET, N] etc.
        
        # Use the parser matching React implementation
        detections = parse_pose_detections(
            output,
            orig_width,
            orig_height,
            confidence_threshold=self.pose_conf_thresh
        )
        
        # Apply NMS
        return apply_pose_nms(detections, self.pose_iou_thresh)
    
    def _detect_pose_yolo(self, image: np.ndarray) -> List[PoseDetection]:
        """
        Detect pose using Ultralytics YOLO (easier to use, but different from React).
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR format
        
        Returns:
            List of PoseDetection objects
        """
        # NOTE: Ultralytics handles BGR->RGB conversion internally for OpenCV/numpy images.
        results = self.pose_model(image, conf=self.pose_conf_thresh, verbose=False)
        
        detections: List[PoseDetection] = []
        orig_height, orig_width = image.shape[:2]
        
        for result in results:
            # Get boxes and keypoints
            boxes = result.boxes
            keypoints_data = result.keypoints
            
            if boxes is None or len(boxes) == 0 or keypoints_data is None:
                continue
            
            for i, box in enumerate(boxes):
                # Get bounding box
                conf = float(box.conf[0])
                if conf < self.pose_conf_thresh:
                    continue
                
                # YOLO returns boxes in xyxy format
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get keypoints for this detection
                if keypoints_data.data is not None and len(keypoints_data.data) > i:
                    kp_data = keypoints_data.data[i].cpu().numpy()  # Shape: [17, 3] (x, y, conf)
                    
                    # Process keypoints
                    keypoints = []
                    for j, (name, kp) in enumerate(zip(KEYPOINT_NAMES, kp_data)):
                        x, y, kp_conf = kp
                        
                        # Ultralytics YOLO returns keypoints in original image coordinates
                        # No scaling needed
                        x_scaled = float(x)
                        y_scaled = float(y)
                        
                        # Only include keypoints above confidence threshold
                        # if kp_conf >= self.keypoint_conf_thresh:
                        keypoints.append({
                            "name": name,
                            "x": x_scaled,
                            "y": y_scaled,
                            "confidence": float(kp_conf)
                        })
                    
                    detections.append(PoseDetection(
                        float(x1), float(y1), float(x2), float(y2),
                        conf, keypoints
                    ))
        
        # Apply NMS
        return apply_pose_nms(detections, self.pose_iou_thresh)
    
    def _select_best_ball(self, ball_detections: List[BallDetection]) -> Optional[BallDetection]:
        """
        Select the ball detection with the highest confidence.
        
        Args:
            ball_detections: List of ball detections
        
        Returns:
            Best ball detection or None if no detections
        """
        if not ball_detections:
            return None
        
        # Filter to only ball class (not hoop)
        ball_only = [det for det in ball_detections if det.class_name == "ball"]
        if not ball_only:
            return None
        
        # Return the one with highest confidence
        return max(ball_only, key=lambda det: det.score)
    
    def _fill_missing_ball_detections(
        self,
        frames_data: List[Dict[str, Any]],
        ball_centers: BallCenters
    ) -> List[Dict[str, Any]]:
        """
        Fill missing ball detections using interpolated ball centers from tracking.
        
        Args:
            frames_data: List of frame detection dictionaries
            ball_centers: BallCenters object with interpolated centers and radii
        
        Returns:
            Updated frames_data with missing ball detections filled
        """
        if len(frames_data) != len(ball_centers.centers_full):
            print(f"Warning: Frame count mismatch ({len(frames_data)} vs {len(ball_centers.centers_full)})")
            return frames_data
        
        # Track the last known score for interpolated detections
        last_known_score = 0.5  # Default score for interpolated detections
        
        for i, frame_data in enumerate(frames_data):
            # Check if frame already has ball detection
            has_ball = frame_data.get("ballDetections") and len(frame_data["ballDetections"]) > 0
            
            if not has_ball:
                # Get interpolated center and radius
                center = ball_centers.centers_full[i]
                radius = ball_centers.radii_full[i]
                
                # Check if interpolated values are valid (not NaN)
                if np.isfinite(center[0]) and np.isfinite(center[1]) and np.isfinite(radius) and radius > 0:
                    # Ensure minimum radius for valid bbox
                    min_radius = 2.0
                    radius = max(radius, min_radius)
                    
                    # Convert center and radius to bbox
                    x1 = float(center[0] - radius)
                    y1 = float(center[1] - radius)
                    x2 = float(center[0] + radius)
                    y2 = float(center[1] + radius)
                    
                    # Ensure bbox is valid (x2 > x1, y2 > y1)
                    if x2 > x1 and y2 > y1:
                        # Add interpolated ball detection
                        frame_data["ballDetections"] = [{
                            "class": "ball",
                            "bbox": {
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2
                            },
                            "score": last_known_score,
                            "interpolated": True  # Flag to indicate this is interpolated
                        }]
            else:
                # Update last known score from actual detection
                if frame_data["ballDetections"]:
                    last_known_score = frame_data["ballDetections"][0].get("score", last_known_score)
        
        return frames_data
    
    def _select_best_player(self, pose_detections: List[PoseDetection]) -> Optional[PoseDetection]:
        """
        Select the player with the best full body detection.
        Considers both bounding box confidence and keypoint confidences.
        
        Args:
            pose_detections: List of pose detections
        
        Returns:
            Best pose detection or None if no detections
        """
        if not pose_detections:
            return None
        
        # Calculate a combined score for each detection
        # This considers both bbox confidence and keypoint quality
        best_detection = None
        best_score = -1.0
        
        for det in pose_detections:
            # Calculate average keypoint confidence
            keypoint_confs = [kp["confidence"] for kp in det.keypoints]
            avg_keypoint_conf = sum(keypoint_confs) / len(keypoint_confs) if keypoint_confs else 0.0
            
            # Count keypoints above threshold (for full body detection)
            high_conf_keypoints = sum(1 for conf in keypoint_confs if conf >= self.keypoint_conf_thresh)
            keypoint_coverage = high_conf_keypoints / len(keypoint_confs) if keypoint_confs else 0.0
            area = (det.x2 - det.x1) * (det.y2 - det.y1)
            
            # Combined score: weighted combination of bbox confidence, avg keypoint conf, coverage, and area
            # Weight: 20% bbox confidence, 40% avg keypoint confidence, 20% keypoint coverage, 20% area
            combined_score = (
                0.2 * det.score +
                0.4 * avg_keypoint_conf +
                0.2 * keypoint_coverage +
                0.2 * area
            )
            
            if combined_score > best_score:
                best_score = combined_score
                best_detection = det
        
        return best_detection
    
    def process_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> Dict[str, Any]:
        """
        Process a single frame and return detections.
        Returns only the best ball and best player (full body).
        
        Args:
            frame: Video frame as numpy array
            frame_number: Frame number in video
            timestamp: Timestamp in seconds
        
        Returns:
            Dictionary with frame detections matching data.json format
        """
        # Detect ball and hoop
        ball_detections = self.detect_ball(frame)
        
        # Detect poses
        pose_detections = self.detect_pose(frame)
        
        # Select best ball (highest confidence)
        best_ball = self._select_best_ball(ball_detections)
        
        # Select best player (considering keypoint confidence for full body)
        best_player = self._select_best_player(pose_detections)
        
        # Format ball detection
        ball_detections_json = []
        if best_ball:
            ball_detections_json.append({
                "class": best_ball.class_name,
                "bbox": {
                    "x1": best_ball.x1,
                    "y1": best_ball.y1,
                    "x2": best_ball.x2,
                    "y2": best_ball.y2
                },
                "score": best_ball.score
            })
        
        # Format pose detection
        pose_detections_json = []
        if best_player:
            pose_detections_json.append({
                "bbox": {
                    "x1": best_player.x1,
                    "y1": best_player.y1,
                    "x2": best_player.x2,
                    "y2": best_player.y2
                },
                "score": best_player.score,
                "keypoints": best_player.keypoints
            })
        
        return {
            "frame": frame_number,
            "timestamp": timestamp,
            "label": self.label,
            "ballDetections": ball_detections_json,
            "poseDetections": pose_detections_json
        }
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        target_fps: Optional[float] = None,
        enable_ball: bool = True,
        enable_pose: bool = True
    ) -> Dict[str, Any]:
        """
        Process a video file and extract detections.
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output JSON (optional)
            target_fps: Target FPS for frame extraction (None = process all frames)
            enable_ball: Enable ball detection
            enable_pose: Enable pose detection
        
        Returns:
            Dictionary with all detections matching data.json format
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_duration = total_frames / video_fps if video_fps > 0 else 0
        
        print(f"Video properties:")
        print(f"  Resolution: {video_width}x{video_height}")
        print(f"  FPS: {video_fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {video_duration:.2f}s")
        
        # Calculate frame skip if target_fps is specified
        frame_skip = 1
        if target_fps and video_fps > 0:
            frame_skip = max(1, int(video_fps / target_fps))
            print(f"  Target FPS: {target_fps}, processing every {frame_skip} frame(s)")
        
        frames_data = []
        frame_number = 0
        processed_frames = 0
        
        print("\nProcessing video...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if target_fps is specified
            if frame_number % frame_skip != 0:
                frame_number += 1
                continue
            
            timestamp = frame_number / video_fps if video_fps > 0 else frame_number * 0.033
            
            # Process frame
            if enable_ball or enable_pose:
                frame_data = self.process_frame(frame, processed_frames, timestamp)
                
                # Only include enabled detections
                if not enable_ball:
                    frame_data["ballDetections"] = []
                if not enable_pose:
                    frame_data["poseDetections"] = []
                
                frames_data.append(frame_data)
                processed_frames += 1
                
                if processed_frames % 10 == 0:
                    print(f"  Processed {processed_frames} frames...")
            
            frame_number += 1
        
        cap.release()
        
        # Create output structure matching data.json
        output = {
            "metadata": {
                "videoDuration": video_duration,
                "totalFrames": total_frames,
                "processedFrames": processed_frames,
                "videoFPS": video_fps,
                "targetFPS": target_fps if target_fps else video_fps,
                "videoWidth": video_width,
                "videoHeight": video_height,
                "extractionDate": datetime.now().isoformat(),
                "label": self.label,
                "detectionSettings": {
                    "enableBallDetection": enable_ball,
                    "enablePoseDetection": enable_pose,
                    "classFilter": "both",
                    "ballConfThresh": self.ball_conf_thresh,
                    "ballIoUThresh": self.ball_iou_thresh,
                    "poseConfThreshold": self.pose_conf_thresh,
                    "poseIouThreshold": self.pose_iou_thresh,
                    "keypointConfThresh": self.keypoint_conf_thresh
                }
            },
            "frames": frames_data,
            "modelVersions": {
                "poseDetection": Path(self.pose_model_path).name,
                "ballDetection": Path(self.ball_model_path).name
            }
        }
        
        # Apply ball tracking to compensate for missing ball detections
        # if enable_ball:
        #     print("\nApplying ball tracking to compensate for missing detections...")
        #     try:
        #         ball_centers = compute_ball_centers_from_json(output, use_tracker=True, tracker_alpha=1.0)
        #         frames_data = self._fill_missing_ball_detections(frames_data, ball_centers)
        #         output["frames"] = frames_data
        #         print("Ball tracking applied successfully!")
        #     except Exception as e:
        #         print(f"Warning: Ball tracking failed: {e}")
        #         print("Continuing with original detections...")
        
        # Save to file if output_path is specified
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"\nResults saved to {output_path}")
        
        print(f"\nProcessing complete! Processed {processed_frames} frames.")
        
        return output


def main():
    """Main function to run detection on a video file."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Basketball Player Pose and Ball Detection")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to output JSON file (overrides constants)")
    parser.add_argument("--pose-model", type=str, default=None, help="Path to pose model (overrides constants)")
    parser.add_argument("--ball-model", type=str, default=None, help="Path to ball model (overrides constants)")
    parser.add_argument("--target-fps", type=float, default=None, help="Target FPS for frame extraction (overrides constants; set 0 to process all frames)")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Device to use (overrides constants)")
    parser.add_argument("--no-ball", action="store_true", help="Disable ball detection (overrides constants)")
    parser.add_argument("--no-pose", action="store_true", help="Disable pose detection (overrides constants)")
    parser.add_argument("--label", type=str, default=None, help="Label stored in output JSON (overrides constants)")
    parser.add_argument("--onnx-pose", action="store_true", help="Use ONNX pose inference (overrides constants)")
    
    args = parser.parse_args()
    
    # Defaults come from constants.py (no config.json)
    pose_model_path = args.pose_model or POSE_MODEL_PATH
    ball_model_path = args.ball_model or BALL_MODEL_PATH
    device = args.device or DEVICE
    output_path = args.output or DEFAULT_OUTPUT_FILE
    target_fps = TARGET_FPS if args.target_fps is None else args.target_fps
    
    # Detection settings (command-line args override constants)
    enable_ball = (not args.no_ball) and ENABLE_BALL
    enable_pose = (not args.no_pose) and ENABLE_POSE
    ball_conf_thresh = BALL_CONF_THRESH
    pose_conf_thresh = POSE_CONF_THRESH
    keypoint_conf_thresh = KEYPOINT_CONF_THRESH
    ball_iou_thresh = BALL_IOU_THRESH
    pose_iou_thresh = POSE_IOU_THRESH
    label = args.label or DEFAULT_LABEL
    use_onnx_pose = bool(args.onnx_pose) or USE_ONNX_POSE
    
    # Initialize detector
    detector = BasketballDetector(
        pose_model_path=pose_model_path,
        ball_model_path=ball_model_path,
        device=device,
        ball_conf_thresh=ball_conf_thresh,
        pose_conf_thresh=pose_conf_thresh,
        keypoint_conf_thresh=keypoint_conf_thresh,
        ball_iou_thresh=ball_iou_thresh,
        pose_iou_thresh=pose_iou_thresh,
        use_onnx_pose=use_onnx_pose,
        label=label
    )
    
    # Process video
    detector.process_video(
        video_path=args.video,
        output_path=output_path,
        target_fps=target_fps,
        enable_ball=enable_ball,
        enable_pose=enable_pose
    )


if __name__ == "__main__":
    main()

