"""
Export Frames Script

Extracts frames from video and saves them as images along with their corresponding JSON data.
Can work with existing detection output JSON or extract frames independently.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse


def draw_detections_on_frame(
    frame: np.ndarray,
    frame_data: Dict[str, Any],
    draw_ball: bool = True,
    draw_pose: bool = True
) -> np.ndarray:
    """
    Draw bounding boxes and keypoints on a frame.
    
    Args:
        frame: Input frame as numpy array (BGR format)
        frame_data: Frame detection data from JSON
        draw_ball: Whether to draw ball/hoop detections
        draw_pose: Whether to draw pose detections
    
    Returns:
        Frame with detections drawn
    """
    frame_copy = frame.copy()
    
    # COCO pose skeleton connections (keypoint pairs to draw lines between)
    SKELETON_CONNECTIONS = [
        # Head
        ("nose", "left_eye"),
        ("left_eye", "left_ear"),
        ("nose", "right_eye"),
        ("right_eye", "right_ear"),
        # Torso
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        # Left arm
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        # Right arm
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        # Left leg
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        # Right leg
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]
    
    # Draw ball detections
    if draw_ball and "ballDetections" in frame_data:
        for det in frame_data["ballDetections"]:
            bbox = det.get("bbox", {})
            x1, y1, x2, y2 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0)), \
                            int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
            score = det.get("score", 0)
            class_name = det.get("class", "unknown")
            
            # Color based on class
            if "ball" in class_name.lower():
                color = (0, 255, 0)  # Green for ball
            elif "hoop" in class_name.lower():
                color = (255, 0, 0)  # Blue for hoop
            else:
                color = (0, 255, 255)  # Yellow for other
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {score:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 5), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw pose detections
    if draw_pose and "poseDetections" in frame_data:
        for det in frame_data["poseDetections"]:
            bbox = det.get("bbox", {})
            x1, y1, x2, y2 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0)), \
                            int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
            score = det.get("score", 0)
            keypoints = det.get("keypoints", [])
            
            # Draw bounding box (cyan for pose)
            color = (255, 255, 0)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{score:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 5), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Create keypoint lookup dictionary by name
            kp_dict = {}
            for kp in keypoints:
                name = kp.get("name", "")
                kp_dict[name] = kp
            
            # Draw skeleton lines (connections between keypoints)
            line_color = (0, 255, 255)  # Yellow for skeleton lines
            line_thickness = 2
            min_confidence = 0.5  # Minimum confidence to draw connection
            
            for kp1_name, kp2_name in SKELETON_CONNECTIONS:
                if kp1_name in kp_dict and kp2_name in kp_dict:
                    kp1 = kp_dict[kp1_name]
                    kp2 = kp_dict[kp2_name]
                    
                    conf1 = kp1.get("confidence", 0)
                    conf2 = kp2.get("confidence", 0)
                    
                    # Draw line if both keypoints have sufficient confidence
                    if conf1 >= min_confidence and conf2 >= min_confidence:
                        x1_kp = int(kp1.get("x", 0))
                        y1_kp = int(kp1.get("y", 0))
                        x2_kp = int(kp2.get("x", 0))
                        y2_kp = int(kp2.get("y", 0))
                        cv2.line(frame_copy, (x1_kp, y1_kp), (x2_kp, y2_kp), line_color, line_thickness)
            
            # Draw keypoints (circles)
            for kp in keypoints:
                x = int(kp.get("x", 0))
                y = int(kp.get("y", 0))
                conf = kp.get("confidence", 0)
                
                if conf > 0.5:  # Only draw high confidence keypoints
                    cv2.circle(frame_copy, (x, y), 3, (0, 0, 255), -1)
    
    return frame_copy


def export_frames_from_video(
    video_path: str,
    output_dir: str,
    json_data: Optional[Dict[str, Any]] = None,
    draw_detections: bool = False,
    image_format: str = "jpg",
    frame_prefix: str = "frame"
) -> None:
    """
    Export frames from video as images and JSON files.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save frames and JSON files
        json_data: Optional detection JSON data (if None, only extracts frames)
        draw_detections: Whether to draw detections on images
        image_format: Image format (jpg, png, etc.)
        frame_prefix: Prefix for frame filenames
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    images_dir = output_path / "images"
    json_dir = output_path / "json"
    images_dir.mkdir(exist_ok=True)
    json_dir.mkdir(exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path}")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {video_fps}")
    print(f"  Output directory: {output_dir}")
    print(f"  Draw detections: {draw_detections}")
    print()
    
    # Create frame data lookup if JSON is provided
    frame_data_map = {}
    if json_data and "frames" in json_data:
        for frame_data in json_data["frames"]:
            frame_idx = frame_data.get("frame", -1)
            frame_data_map[frame_idx] = frame_data
    
    frame_number = 0
    exported_count = 0
    
    print("Exporting frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get frame data if available
        frame_data = frame_data_map.get(frame_number, {
            "frame": frame_number,
            "timestamp": frame_number / video_fps if video_fps > 0 else frame_number * 0.033,
            "ballDetections": [],
            "poseDetections": []
        })
        
        # Prepare frame for saving
        frame_to_save = frame.copy()
        if draw_detections and json_data:
            frame_to_save = draw_detections_on_frame(frame_to_save, frame_data)
        
        # Save image
        image_filename = f"{frame_prefix}_{frame_number:06d}.{image_format}"
        image_path = images_dir / image_filename
        cv2.imwrite(str(image_path), frame_to_save)
        
        # Save JSON
        json_filename = f"{frame_prefix}_{frame_number:06d}.json"
        json_path = json_dir / json_filename
        with open(json_path, 'w') as f:
            json.dump(frame_data, f, indent=2)
        
        exported_count += 1
        if exported_count % 10 == 0:
            print(f"  Exported {exported_count} frames...")
        
        frame_number += 1
    
    cap.release()
    
    print(f"\nExport complete!")
    print(f"  Total frames exported: {exported_count}")
    print(f"  Images saved to: {images_dir}")
    print(f"  JSON files saved to: {json_dir}")


def export_frames_from_json(
    video_path: str,
    json_path: str,
    output_dir: str,
    draw_detections: bool = False,
    image_format: str = "jpg",
    frame_prefix: str = "frame"
) -> None:
    """
    Export frames using detection data from JSON file.
    Only exports frames that are present in the JSON (processed frames).
    
    Args:
        video_path: Path to input video file
        json_path: Path to detection JSON file
        output_dir: Directory to save frames and JSON files
        draw_detections: Whether to draw detections on images
        image_format: Image format (jpg, png, etc.)
        frame_prefix: Prefix for frame filenames
    """
    # Load JSON data
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    images_dir = output_path / "images"
    json_dir = output_path / "json"
    images_dir.mkdir(exist_ok=True)
    json_dir.mkdir(exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path}")
    print(f"JSON: {json_path}")
    print(f"  Total frames in video: {total_frames}")
    print(f"  Processed frames in JSON: {len(json_data.get('frames', []))}")
    print(f"  Output directory: {output_dir}")
    print(f"  Draw detections: {draw_detections}")
    print()
    
    # Create frame data lookup
    frame_data_map = {}
    processed_frame_indices = []
    if "frames" in json_data:
        for frame_data in json_data["frames"]:
            frame_idx = frame_data.get("frame", -1)
            frame_data_map[frame_idx] = frame_data
            processed_frame_indices.append(frame_idx)
    
    # Sort processed frame indices
    processed_frame_indices.sort()
    
    # Find frame skip if needed (based on target_fps in metadata)
    frame_skip = 1
    if "metadata" in json_data:
        metadata = json_data["metadata"]
        video_fps_meta = metadata.get("videoFPS", video_fps)
        target_fps = metadata.get("targetFPS", video_fps_meta)
        if target_fps and video_fps_meta > 0:
            frame_skip = max(1, int(video_fps_meta / target_fps))
    
    frame_number = 0
    exported_count = 0
    
    print("Exporting frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if this frame was processed (exists in JSON)
        if frame_number in frame_data_map:
            frame_data = frame_data_map[frame_number]
            
            # Prepare frame for saving
            frame_to_save = frame.copy()
            if draw_detections:
                frame_to_save = draw_detections_on_frame(frame_to_save, frame_data)
            
            # Save image
            image_filename = f"{frame_prefix}_{frame_number:06d}.{image_format}"
            image_path = images_dir / image_filename
            cv2.imwrite(str(image_path), frame_to_save)
            
            # Save JSON
            json_filename = f"{frame_prefix}_{frame_number:06d}.json"
            json_path_out = json_dir / json_filename
            with open(json_path_out, 'w') as f:
                json.dump(frame_data, f, indent=2)
            
            exported_count += 1
            if exported_count % 10 == 0:
                print(f"  Exported {exported_count} frames...")
        
        frame_number += 1
    
    cap.release()
    
    print(f"\nExport complete!")
    print(f"  Total frames exported: {exported_count}")
    print(f"  Images saved to: {images_dir}")
    print(f"  JSON files saved to: {json_dir}")


def main():
    """Main function to run frame export."""
    parser = argparse.ArgumentParser(
        description="Export video frames as images and JSON files per frame"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Path to detection JSON file (optional, if provided only exports processed frames)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for frames and JSON files"
    )
    parser.add_argument(
        "--draw",
        action="store_true",
        help="Draw detections (bounding boxes and keypoints) on images"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jpg",
        choices=["jpg", "jpeg", "png"],
        help="Image format (default: jpg)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame",
        help="Prefix for frame filenames (default: frame)"
    )
    
    args = parser.parse_args()
    
    if args.json:
        # Export using JSON data (only processed frames)
        export_frames_from_json(
            video_path=args.video,
            json_path=args.json,
            output_dir=args.output,
            draw_detections=args.draw,
            image_format=args.format,
            frame_prefix=args.prefix
        )
    else:
        # Export all frames from video
        export_frames_from_video(
            video_path=args.video,
            output_dir=args.output,
            json_data=None,
            draw_detections=False,  # Can't draw without JSON data
            image_format=args.format,
            frame_prefix=args.prefix
        )


if __name__ == "__main__":
    main()

