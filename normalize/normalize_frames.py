"""
Frame Normalization Script

Takes detection JSON output from detect.py and normalizes frame data using normalize.py.
Outputs normalized features for use in downstream models.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
import argparse
from parsers import KEYPOINT_NAMES
from normalize import build_frame_features, FrameFeatures


def normalize_detection_json(
    input_json: Union[str, Dict[str, Any]],
    output_path: Optional[str] = None,
    fps: Optional[float] = None,
    tracker_alpha: float = 1.0,
    interpolate_ball: bool = True,
) -> FrameFeatures:
    """
    Normalize frame data from detection JSON file or dictionary.
    
    Args:
        input_json: Path to detection JSON file (output from detect.py) OR detection JSON dictionary
        output_path: Optional path to save normalized features as JSON
        fps: Optional FPS override (if not provided, uses metadata from JSON)
        tracker_alpha: Alpha parameter for ball tracker smoothing
    
    Returns:
        FrameFeatures object with normalized data
    """
    # Load detection JSON (from file or use provided dict)
    if isinstance(input_json, str):
        print(f"Loading detection JSON from {input_json}...")
        with open(input_json, 'r') as f:
            det_json = json.load(f)
    else:
        print("Using provided detection JSON dictionary...")
        det_json = input_json
    
    # Build normalized features
    print("Normalizing frame data...")
    features = build_frame_features(
        det_json,
        fps=fps,
        tracker_alpha=tracker_alpha,
        interpolate_ball=interpolate_ball,
    )
    
    print(f"Normalization complete! Processed {len(features.timestamps)} frames.")
    
    # Save normalized features if output path is provided
    if output_path:
        save_normalized_features(features, output_path, det_json)
    
    return features


def save_normalized_features(
    features: FrameFeatures,
    output_path: str,
    original_json: Dict[str, Any]
):
    """
    Save normalized features to JSON file.
    
    Args:
        features: FrameFeatures object with normalized data
        output_path: Path to save output JSON
        original_json: Original detection JSON (for metadata preservation)
    """
    # Convert numpy arrays to lists for JSON serialization
    output = {
        "metadata": original_json.get("metadata", {}),
        "normalized_features": {
            "timestamps": features.timestamps.tolist(),

            # masks
            "pose_present": features.pose_present.tolist(),
            "ball_present": features.ball_present.tolist(),
            "ball_measured": features.ball_measured.tolist(),

            # ball (normalized + raw)
            "ball_norm": features.ball_norm.tolist(),                # (cx, cy) in person-centric coords
            "ball_center": features.ball_center.tolist(),            # pixel coords (optional)
            "ball_diameter": features.ball_diameter.tolist(),        # pixels (optional)
            "ball_diameter_norm": features.ball_diameter_norm.tolist(),  # diameter/scale (recommended)
            "ball_vel": features.ball_vel.tolist(),
            "ball_acc": features.ball_acc.tolist(),
            "ball_ax": features.ball_ax.tolist(),
            "ball_ay": features.ball_ay.tolist(),
            "ball_acc_mag": features.ball_acc_mag.tolist(),

            # derived
            "hand_ball_dist": features.hand_ball_dist.tolist(),
            "hand_ball_dist_L": features.hand_ball_dist_L.tolist(),
            "hand_ball_dist_R": features.hand_ball_dist_R.tolist(),
            "hand_used": features.hand_used.tolist(),
            "release_score": features.release_score.tolist(),
            "wrist_head_rel_y_L": features.wrist_head_rel_y_L.tolist(),
            "wrist_head_rel_y_R": features.wrist_head_rel_y_R.tolist(),
            "elbow_angle_L": features.elbow_angle_L.tolist(),
            "elbow_angle_R": features.elbow_angle_R.tolist(),
            "ball_y_local_min": features.ball_y_local_min.tolist(),
            "ball_y_local_max": features.ball_y_local_max.tolist(),
            "ball_vy_sign_flip": features.ball_vy_sign_flip.tolist(),
            "ankle_y": features.ankle_y.tolist(),
            "ball_speed": features.ball_speed.tolist(),
            "ball_vx": features.ball_vx.tolist(),
            "ball_vy": features.ball_vy.tolist(),
            "ball_dir_sin": features.ball_dir_sin.tolist(),
            "ball_dir_cos": features.ball_dir_cos.tolist(),

            # global context (optional but useful)
            "center": features.center.tolist(),                      # pixels
            "scale": features.scale.tolist(),
            "center_y_img": features.center_y_img.tolist(),          # 0..1 if videoHeight exists else 0s

            # pose
            "keypoints_normalized": {
                name: features.kp_norm[name].tolist()
                for name in KEYPOINT_NAMES  # fixed order
            },
            "keypoints_confidence": {
                name: features.kp_conf[name].tolist()
                for name in KEYPOINT_NAMES  # fixed order
            },
        }
    }
    
    # Save to file (compact format to minimize size)
    with open(output_path, 'w') as f:
        json.dump(output, f, separators=(',', ':'))
    
    print(f"Normalized features saved to {output_path}")


def main():
    """Main function to normalize detection JSON files."""
    parser = argparse.ArgumentParser(description="Normalize frame data from detection JSON")
    parser.add_argument("--input", type=str, required=True, help="Path to input detection JSON file")
    parser.add_argument("--output", type=str, default=None, help="Path to output normalized features JSON")
    parser.add_argument("--fps", type=float, default=None, help="FPS override (if not provided, uses metadata)")
    parser.add_argument("--tracker-alpha", type=float, default=1.0, help="Ball tracker smoothing alpha (default: 1.0)")
    parser.add_argument(
        "--no-interpolate-ball",
        action="store_true",
        help="Disable ball trajectory interpolation (keep NaNs when the detector missed the ball).",
    )
    
    args = parser.parse_args()
    
    # Generate default output path if not provided
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_normalized.json")
    
    # Normalize detection JSON
    normalize_detection_json(
        input_json=args.input,
        output_path=args.output,
        fps=args.fps,
        tracker_alpha=args.tracker_alpha,
        interpolate_ball=not args.no_interpolate_ball,
    )


if __name__ == "__main__":
    main()

