"""
Main Pipeline Script

Root file that orchestrates the detection and normalization pipeline:
1. Runs detection on video using detect.py
2. Normalizes frame data using normalize_frames.py

Supports both single video file and bulk processing of videos in a folder.
"""

import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict

from detect import BasketballDetector
from normalize_frames import normalize_detection_json
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
    DEFAULT_LABEL,
    USE_ONNX_POSE,
)

def run_pipeline(
    video_path: str,
    normalized_output: Optional[str] = None,
    fps: Optional[float] = None,
    tracker_alpha: float = 1.0,
    interpolate_ball: bool = False,
    label: str = DEFAULT_LABEL,
    **detection_kwargs
):
    """
    Run the complete pipeline: detection followed by normalization.
    Only saves the normalized JSON output, not the intermediate detection JSON.
    
    Args:
        video_path: Path to input video file
        normalized_output: Path to save normalized features JSON (if None, auto-generated)
        fps: Optional FPS override for normalization
        tracker_alpha: Alpha parameter for ball tracker smoothing
        **detection_kwargs: Additional arguments to pass to BasketballDetector
    """
    print("=" * 60)
    print("BASKETBALL DETECTION AND NORMALIZATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Run detection
    print("\n[STEP 1/2] Running detection on video...")
    print("-" * 60)
    
    # Pull explicit detector overrides out of detection_kwargs to avoid
    # passing duplicate keyword arguments to BasketballDetector(...).
    pose_model_path = detection_kwargs.pop("pose_model_path", POSE_MODEL_PATH)
    ball_model_path = detection_kwargs.pop("ball_model_path", BALL_MODEL_PATH)
    device = detection_kwargs.pop("device", DEVICE)
    use_onnx_pose = detection_kwargs.pop("use_onnx_pose", USE_ONNX_POSE)

    # Initialize detector
    detector = BasketballDetector(
        pose_model_path=pose_model_path,
        ball_model_path=ball_model_path,
        device=device,
        ball_conf_thresh=BALL_CONF_THRESH,
        pose_conf_thresh=POSE_CONF_THRESH,
        keypoint_conf_thresh=KEYPOINT_CONF_THRESH,
        ball_iou_thresh=BALL_IOU_THRESH,
        pose_iou_thresh=POSE_IOU_THRESH,
        use_onnx_pose=use_onnx_pose,
        label=label,
        **detection_kwargs
    )
    
    # Get enable flags (can be overridden by kwargs)
    enable_ball = detection_kwargs.get("enable_ball", ENABLE_BALL)
    enable_pose = detection_kwargs.get("enable_pose", ENABLE_POSE)
    target_fps = detection_kwargs.get("target_fps", TARGET_FPS)
    
    # Remove these from kwargs as they're passed separately to process_video
    detection_kwargs.pop("enable_ball", None)
    detection_kwargs.pop("enable_pose", None)
    detection_kwargs.pop("target_fps", None)
    
    # Process video (don't save detection JSON to disk)
    detection_result = detector.process_video(
        video_path=video_path,
        output_path=None,  # Don't save detection JSON
        target_fps=target_fps,
        enable_ball=enable_ball,
        enable_pose=enable_pose
    )
    
    print(f"\n✓ Detection complete! Processed {len(detection_result.get('frames', []))} frames.")
    
    # Step 2: Normalize frame data
    print("\n[STEP 2/2] Normalizing frame data...")
    print("-" * 60)
    
    # Determine normalized output path
    if normalized_output is None:
        video_path_obj = Path(video_path)
        normalized_output = str(video_path_obj.parent / f"{video_path_obj.stem}.json")
    
    # Normalize detection JSON (pass dictionary directly, not file path)
    features = normalize_detection_json(
        input_json=detection_result,  # Pass dictionary directly
        output_path=normalized_output,
        fps=fps,
        tracker_alpha=tracker_alpha,
        interpolate_ball=interpolate_ball,
    )
    
    print(f"\n✓ Normalization complete! Features saved to {normalized_output}")
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Normalized output: {normalized_output}")
    print(f"Total frames processed: {len(features.timestamps)}")
    print("=" * 60)
    
    return detection_result, features


def get_video_files(folder_path: str, extensions: Optional[List[str]] = None, recursive: bool = False) -> List[Path]:
    """
    Get all video files from a folder (optionally including subfolders).
    
    Args:
        folder_path: Path to folder containing videos
        extensions: List of video file extensions (default: ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'])
        recursive: If True, search in all subfolders (folder/**)
    
    Returns:
        List of Path objects for video files
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    video_files: List[Path] = []
    globber = folder.rglob if recursive else folder.glob
    for ext in extensions:
        video_files.extend(globber(f"*{ext}"))
        video_files.extend(globber(f"*{ext.upper()}"))
    
    # Remove duplicates and sort
    video_files = sorted(set(video_files))
    return video_files


def _derive_label_from_path(root_folder: Path, video_file: Path, fallback_label: str = "other") -> str:
    """
    Derive label from the first subfolder under the bulk root:
      root/SHOOT/clip.mp4  -> label = "SHOOT"
      root/SHOOT/sub/clip.mp4 -> label = "SHOOT"
      root/clip.mp4 -> label = fallback_label
    """
    try:
        rel = video_file.relative_to(root_folder)
    except ValueError:
        # If video_file isn't under root_folder for some reason, fall back
        return fallback_label

    # rel.parts includes filename; we need at least ["label", "file.ext"] to infer a subfolder label
    if len(rel.parts) >= 2:
        return rel.parts[0]
    return fallback_label


def run_bulk_pipeline(
    folder_path: str,
    output_folder: Optional[str] = None,
    fps: Optional[float] = None,
    tracker_alpha: float = 1.0,
    interpolate_ball: bool = False,
    label: str = DEFAULT_LABEL,
    **detection_kwargs
):
    """
    Run the pipeline on all video files in a folder.
    
    Args:
        folder_path: Path to folder containing video files
        output_folder: Folder to save normalized JSON files (if None, saves in same folder as videos)
        fps: Optional FPS override for normalization
        tracker_alpha: Alpha parameter for ball tracker smoothing
        **detection_kwargs: Additional arguments to pass to BasketballDetector
    """
    print("=" * 60)
    print("BULK BASKETBALL DETECTION AND NORMALIZATION PIPELINE")
    print("=" * 60)
    
    bulk_root = Path(folder_path)
    # Get all video files (recursive: folder/**)
    video_files = get_video_files(folder_path, recursive=True)
    
    if not video_files:
        print(f"\n⚠ No video files found in folder: {folder_path}")
        return
    
    print(f"\nFound {len(video_files)} video file(s) to process:")
    # Show a compact, relative view so labels/subfolders are obvious
    for i, video_file in enumerate(video_files[:50], 1):
        try:
            rel = video_file.relative_to(bulk_root)
            print(f"  {i}. {rel.as_posix()}")
        except ValueError:
            print(f"  {i}. {video_file.name}")
    if len(video_files) > 50:
        print(f"  ... and {len(video_files) - 50} more")
    
    # Determine output folder
    if output_folder is None:
        # Default: save next to each input video (including subfolders)
        output_folder = None
    else:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    
    # Pull explicit detector overrides out of detection_kwargs to avoid
    # passing duplicate keyword arguments to BasketballDetector(...).
    pose_model_path = detection_kwargs.pop("pose_model_path", POSE_MODEL_PATH)
    ball_model_path = detection_kwargs.pop("ball_model_path", BALL_MODEL_PATH)
    device = detection_kwargs.pop("device", DEVICE)
    use_onnx_pose = detection_kwargs.pop("use_onnx_pose", USE_ONNX_POSE)

    # Initialize detector once (reused for all videos)
    detector = BasketballDetector(
        pose_model_path=pose_model_path,
        ball_model_path=ball_model_path,
        device=device,
        ball_conf_thresh=BALL_CONF_THRESH,
        pose_conf_thresh=POSE_CONF_THRESH,
        keypoint_conf_thresh=KEYPOINT_CONF_THRESH,
        ball_iou_thresh=BALL_IOU_THRESH,
        pose_iou_thresh=POSE_IOU_THRESH,
        use_onnx_pose=use_onnx_pose,
        label=label,  # will be overwritten per video when processing folder/subfolder structure
        **detection_kwargs
    )
    
    # Get enable flags
    enable_ball = detection_kwargs.get("enable_ball", ENABLE_BALL)
    enable_pose = detection_kwargs.get("enable_pose", ENABLE_POSE)
    target_fps = detection_kwargs.get("target_fps", TARGET_FPS)
    
    # Process each video
    results = []
    successful = 0
    failed = 0
    label_counts: Dict[str, int] = {}
    
    for idx, video_file in enumerate(video_files, 1):
        print("\n" + "=" * 60)
        print(f"Processing video {idx}/{len(video_files)}: {video_file.name}")
        print("=" * 60)
        
        try:
            # Determine label from folder/subfolder structure
            derived_label = _derive_label_from_path(bulk_root, video_file, fallback_label=label)
            detector.label = derived_label
            label_counts[derived_label] = label_counts.get(derived_label, 0) + 1

            # Determine output path
            # - If output_folder is None: write next to the input video (same folder)
            # - Else: mirror the input folder structure under output_folder to avoid name collisions
            if output_folder is None:
                output_path = video_file.with_suffix(".json")
            else:
                rel = video_file.relative_to(bulk_root)
                output_path = Path(output_folder) / rel.with_suffix(".json")
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Run detection
            print("\n[STEP 1/2] Running detection on video...")
            print("-" * 60)
            print(f"Label: {derived_label}")
            
            detection_result = detector.process_video(
                video_path=str(video_file),
                output_path=None,  # Don't save detection JSON
                target_fps=target_fps,
                enable_ball=enable_ball,
                enable_pose=enable_pose
            )
            
            print(f"\n✓ Detection complete! Processed {len(detection_result.get('frames', []))} frames.")
            
            # Step 2: Normalize frame data
            print("\n[STEP 2/2] Normalizing frame data...")
            print("-" * 60)
            
            features = normalize_detection_json(
                input_json=detection_result,
                output_path=str(output_path),
                fps=fps,
                tracker_alpha=tracker_alpha,
                interpolate_ball=interpolate_ball,
            )
            
            print(f"\n✓ Normalization complete! Features saved to {output_path}")
            print(f"  Total frames processed: {len(features.timestamps)}")
            
            results.append({
                "video": str(video_file),
                "output": str(output_path),
                "frames": len(features.timestamps),
                "status": "success"
            })
            successful += 1
            
        except Exception as e:
            print(f"\n✗ Error processing {video_file.name}: {str(e)}")
            results.append({
                "video": str(video_file),
                "output": None,
                "frames": 0,
                "status": "failed",
                "error": str(e)
            })
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("BULK PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total videos processed: {len(video_files)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    if output_folder is None:
        print("Output: next to each input video (same folder as video)")
    else:
        print(f"Output folder: {output_folder}")
    if label_counts:
        print("Labels (from subfolders):")
        for k in sorted(label_counts.keys()):
            print(f"  {k}: {label_counts[k]}")
    print("=" * 60)
    
    if failed > 0:
        print("\nFailed videos:")
        for result in results:
            if result["status"] == "failed":
                print(f"  - {Path(result['video']).name}: {result.get('error', 'Unknown error')}")
    
    return results


def main():
    """Main function to run the pipeline from command line."""
    parser = argparse.ArgumentParser(
        description="Basketball Detection and Normalization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single video with default config
  python pipeline.py --video videos/0000162.mp4
  
  # Process single video with custom output path
  python pipeline.py --video videos/0000162.mp4 --normalized-output features.json
  
  # Process single video with custom FPS and tracker settings
  python pipeline.py --video videos/0000162.mp4 --fps 30 --tracker-alpha 0.7
  
  # Process all videos in a folder (outputs: {video_name}.json)
  python pipeline.py --folder videos/
  
  # Process all videos with custom output folder
  python pipeline.py --folder videos/ --output-folder output/
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", type=str, help="Path to input video file")
    input_group.add_argument("--folder", type=str, help="Path to folder containing video files")
    
    parser.add_argument("--normalized-output", type=str, default=None, help="Path to save normalized features JSON (single video only)")
    parser.add_argument("--output-folder", type=str, default=None, help="Folder to save normalized JSON files (bulk processing only)")
    parser.add_argument("--fps", type=float, default=None, help="FPS override for normalization")
    parser.add_argument("--tracker-alpha", type=float, default=1.0, help="Ball tracker smoothing alpha")
    parser.add_argument(
        "--no-interpolate-ball",
        action="store_true",
        help="Disable ball trajectory interpolation during normalization (keep NaNs when missing).",
    )
    
    # Detection model overrides
    parser.add_argument("--pose-model", type=str, default=None, help="Path to pose model (overrides constants)")
    parser.add_argument("--ball-model", type=str, default=None, help="Path to ball model (overrides constants)")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Device to use (overrides constants)")
    parser.add_argument("--target-fps", type=float, default=None, help="Target FPS for frame extraction (overrides constants; set 0 to process all frames)")
    parser.add_argument("--no-ball", action="store_true", help="Disable ball detection")
    parser.add_argument("--no-pose", action="store_true", help="Disable pose detection")
    parser.add_argument("--interpolate-ball", action="store_true", help="Enable ball trajectory interpolation during normalization")
    parser.add_argument("--label", type=str, default=DEFAULT_LABEL, help="Label for the detection")
    
    args = parser.parse_args()
    
    # Prepare detection kwargs
    detection_kwargs = {}
    if args.pose_model:
        detection_kwargs["pose_model_path"] = args.pose_model
    if args.ball_model:
        detection_kwargs["ball_model_path"] = args.ball_model
    if args.device:
        detection_kwargs["device"] = args.device
    if args.target_fps:
        detection_kwargs["target_fps"] = args.target_fps
    if args.no_ball:
        detection_kwargs["enable_ball"] = False
    if args.no_pose:
        detection_kwargs["enable_pose"] = False
    
    # Run pipeline (single video or bulk)
    if args.folder:
        # Bulk processing
        run_bulk_pipeline(
            folder_path=args.folder,
            output_folder=args.output_folder,
            fps=args.fps,
            tracker_alpha=args.tracker_alpha,
            interpolate_ball=args.interpolate_ball,
            label=args.label,
            **detection_kwargs
        )
    else:
        # Single video processing
        run_pipeline(
            video_path=args.video,
            normalized_output=args.normalized_output,
            fps=args.fps,
            tracker_alpha=args.tracker_alpha,
            interpolate_ball=args.interpolate_ball,
            label=args.label,
            **detection_kwargs
        )


if __name__ == "__main__":
    main()

