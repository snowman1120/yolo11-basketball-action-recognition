"""
Ball Tracking Pipeline

Provides robust ball tracking and interpolation functionality for basketball video analysis.
Handles ball detection smoothing, filtering, and temporal interpolation across frames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ========================  Ball Tracker  ========================


@dataclass
class BallState:
    """Internal state for the ball tracker.

    Attributes
    ----------
    center : np.ndarray
        Current ball center [x, y] in pixels.
    radius : float
        Estimated ball radius in pixels.
    """
    center: np.ndarray
    radius: float


class BallTracker:
    """Simple, robust ball tracker used to clean bbox detections.

    The tracker applies exponential smoothing to ball detections and filters out
    suspicious detections (heavily occluded or incorrectly shaped).

    Design:
      - Only called on frames that have a ball bbox.
      - Keeps a smoothed/filtered center and radius.
      - Ignores detections that look heavily occluded or have crazy shape.
      - Does NOT try to predict missing frames; that can optionally be handled
        later by interpolation over the whole sequence.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        aspect_range: Tuple[float, float] = (0.5, 1.8),
        min_radius_ratio: float = 0.3,
    ) -> None:
        """Create a BallTracker.

        Parameters
        ----------
        alpha : float, default=1.0
            Smoothing factor when updating from a new detection.
            center_new = alpha * meas + (1 - alpha) * previous.
            Higher values (closer to 1.0) trust new measurements more.
        aspect_range : Tuple[float, float], default=(0.5, 1.8)
            Allowed w/h range for a roughly circular ball.
            Detections outside this range are considered invalid.
        min_radius_ratio : float, default=0.3
            If new_radius / old_radius falls below this, the detection is
            considered heavily occluded / wrong and is ignored.
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if aspect_range[0] >= aspect_range[1]:
            raise ValueError(f"aspect_range must be (min, max) with min < max")
        if not (0.0 < min_radius_ratio <= 1.0):
            raise ValueError(f"min_radius_ratio must be in (0, 1], got {min_radius_ratio}")

        self.alpha = float(alpha)
        self.aspect_range = aspect_range
        self.min_radius_ratio = float(min_radius_ratio)
        self.state: Optional[BallState] = None

    @staticmethod
    def bbox_to_measurement(bbox: Dict[str, float]) -> Tuple[np.ndarray, float, float, float]:
        """Convert bbox dict -> (center, radius, width, height).

        Parameters
        ----------
        bbox : dict
            Must have keys: x1, y1, x2, y2 (pixels).

        Returns
        -------
        center : np.ndarray
            Ball center [x, y] in pixels.
        radius : float
            Approximate ball radius in pixels.
        width : float
            Bounding box width in pixels.
        height : float
            Bounding box height in pixels.
        """
        try:
            x1 = float(bbox["x1"])
            y1 = float(bbox["y1"])
            x2 = float(bbox["x2"])
            y2 = float(bbox["y2"])
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid bbox format: {bbox}") from e

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox: x2 <= x1 or y2 <= y1")

        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        w = x2 - x1
        h = y2 - y1

        # Approximate radius as average of half-width and half-height
        r_meas = 0.25 * (w + h)

        center = np.array([cx, cy], dtype=np.float32)
        return center, float(r_meas), float(w), float(h)

    def _is_measurement_healthy(
        self,
        r_meas: float,
        w: float,
        h: float,
        r_prev: Optional[float],
    ) -> bool:
        """Heuristics to decide if a bbox measurement is trustworthy.

        Checks:
          - Aspect ratio (shape should be roughly round)
          - Radius shrink (strong occlusion / cropping detection)

        NOTE: We do NOT check how far the center jumps here, only:
          - aspect ratio (shape),
          - huge radius shrink (strong occlusion / cropping).

        Parameters
        ----------
        r_meas : float
            Measured radius.
        w : float
            Bounding box width.
        h : float
            Bounding box height.
        r_prev : Optional[float]
            Previous radius (None if first detection).

        Returns
        -------
        bool
            True if measurement is healthy, False otherwise.
        """
        # 1) Aspect ratio roughly round (but allow some slack)
        if h <= 1e-3 or w <= 1e-3:
            return False
        aspect = w / h
        if not (self.aspect_range[0] <= aspect <= self.aspect_range[1]):
            return False

        # 2) Radius: only penalize huge shrink.
        if r_prev is not None and r_prev > 0:
            ratio = r_meas / r_prev
            if ratio < self.min_radius_ratio:
                return False

        return True

    def reset(self) -> None:
        """Reset internal state (start tracking a new sequence)."""
        self.state = None

    def update_from_bbox(self, bbox: Dict[str, float]) -> Tuple[np.ndarray, float]:
        """Update tracker with a new bbox and return cleaned (center, radius).

        Parameters
        ----------
        bbox : dict
            Dictionary with keys x1, y1, x2, y2 (pixels).

        Returns
        -------
        Tuple[np.ndarray, float]
            (center [x,y] in pixels, radius in pixels).
        """
        center_meas, r_meas, w, h = self.bbox_to_measurement(bbox)

        # First observation -> initialize directly from measurement
        if self.state is None:
            self.state = BallState(center=center_meas, radius=r_meas)
            return center_meas, r_meas

        c_prev = self.state.center
        r_prev = self.state.radius

        healthy = self._is_measurement_healthy(r_meas, w, h, r_prev)

        if healthy:
            # Simple exponential smoothing
            alpha = self.alpha
            c_new = alpha * center_meas + (1.0 - alpha) * c_prev
            r_new = alpha * r_meas + (1.0 - alpha) * r_prev
        else:
            # Suspicious detection (occluded / wrong) -> ignore, keep previous
            c_new = c_prev
            r_new = r_prev

        self.state = BallState(center=c_new, radius=r_new)
        return c_new, r_new

    def update_from_detection(self, det: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """Update tracker from a detection dict that has a 'bbox' field.

        Parameters
        ----------
        det : dict
            Detection dictionary with 'bbox' key.

        Returns
        -------
        Tuple[np.ndarray, float]
            (center [x,y] in pixels, radius in pixels).
        """
        if "bbox" not in det:
            raise ValueError("Detection dict must have 'bbox' key")
        return self.update_from_bbox(det["bbox"])


# ========================  Centers + Interpolation  ========================


@dataclass
class BallCenters:
    """Container for ball center trajectories.

    Attributes
    ----------
    timestamps : np.ndarray
        Shape (N,), timestamps in seconds.
    centers_raw : np.ndarray
        Shape (N, 2), raw bbox centers (no smoothing), NaN where no detection.
    radii_raw : np.ndarray
        Shape (N,), raw radii derived from bbox sizes, NaN when missing.
    centers_tracker : np.ndarray
        Shape (N, 2), centers cleaned by BallTracker on detection frames,
        NaN where no detection.
    radii_tracker : np.ndarray
        Shape (N,), radii cleaned by BallTracker on detection frames,
        NaN where no detection.
    centers_full : np.ndarray
        Shape (N, 2), output centers for all frames.
        If interpolation is enabled, this is a fully-filled trajectory
        (no NaNs unless there were no detections at all). If interpolation
        is disabled, this equals the chosen basis and may contain NaNs.
    radii_full : np.ndarray
        Shape (N,), output radii for all frames.
        Same interpolation behavior as `centers_full`.
    """

    timestamps: np.ndarray
    centers_raw: np.ndarray
    radii_raw: np.ndarray
    centers_tracker: np.ndarray
    radii_tracker: np.ndarray
    centers_full: np.ndarray
    radii_full: np.ndarray


def load_detection_json(path: str) -> Dict[str, Any]:
    """Load a detection JSON file produced by the detection pipeline.

    Parameters
    ----------
    path : str
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed detection JSON.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Detection JSON file not found: {path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {path}: {e}", e.doc, e.pos)


def bbox_to_center_radius(bbox: Dict[str, float]) -> Tuple[np.ndarray, float]:
    """Convert bbox dict -> (center, radius).

    Parameters
    ----------
    bbox : dict
        Must have keys: x1, y1, x2, y2 (pixels).

    Returns
    -------
    center : np.ndarray
        Ball center [x, y] in pixels.
    radius : float
        Approximate ball radius in pixels.
    """
    center, radius, _, _ = BallTracker.bbox_to_measurement(bbox)
    return center, radius


def interpolate_1d_values(
    timestamps: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Fill all missing 1D values by linear interpolation in time.

    This is a general-purpose 1D interpolation function that can be used
    for ball diameters, radii, or any other 1D time series data.

    values[k] may contain NaNs; we fill them as follows:

      - If there are 0 valid values -> return values unchanged (all NaNs).
      - If there is exactly 1 valid value -> fill ALL frames with that value.
      - If there are 2+ valid values:
          * prefix before first valid frame: use first valid value,
          * suffix after last valid frame: use last valid value,
          * interior gaps: linear interpolation in time between neighbors.

    Parameters
    ----------
    timestamps : np.ndarray
        Shape (N,), timestamps in seconds.
    values : np.ndarray
        Shape (N,), array with NaNs for missing frames.

    Returns
    -------
    filled : np.ndarray
        Shape (N,), array with missing frames filled.
    """
    ts = np.asarray(timestamps, dtype=np.float32)
    V = np.asarray(values, dtype=np.float32)

    if V.shape[0] != ts.shape[0]:
        raise ValueError(f"timestamps and values must have same length, got {ts.shape[0]} and {V.shape[0]}")

    filled = V.copy()

    # Valid frames = where value is finite
    valid = np.isfinite(V)
    idx_valid = np.where(valid)[0]

    if idx_valid.size == 0:
        # No valid values at all
        return filled

    if idx_valid.size == 1:
        # Only one detection: fill everything with that value
        filled[:] = V[idx_valid[0]]
        return filled

    # 1) Fill prefix (before first detection) with first valid value
    first = idx_valid[0]
    filled[:first] = V[first]

    # 2) Fill suffix (after last detection) with last valid value
    last = idx_valid[-1]
    filled[last + 1 :] = V[last]

    # 3) Fill gaps between valid detections by linear interpolation
    for i0, i1 in zip(idx_valid, idx_valid[1:]):
        if i1 == i0 + 1:
            # Consecutive frames, no gap
            continue

        t0, t1 = ts[i0], ts[i1]
        v0, v1 = V[i0], V[i1]

        for k in range(i0 + 1, i1):
            if t1 > t0:
                alpha = (ts[k] - t0) / (t1 - t0)
            else:
                # Fallback: interpolate by index if timestamps are degenerate
                alpha = (k - i0) / (i1 - i0)
            filled[k] = (1.0 - alpha) * v0 + alpha * v1

    return filled


def interpolate_centers(
    timestamps: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """Fill all missing centers by linear interpolation in time.

    centers[k] may contain NaNs; we fill them as follows:

      - If there are 0 valid centers -> return centers unchanged (all NaNs).
      - If there is exactly 1 valid center -> fill ALL frames with that center.
      - If there are 2+ valid centers:
          * prefix before first valid frame: use first valid center,
          * suffix after last valid frame: use last valid center,
          * interior gaps: linear interpolation in time between neighbors.

    Parameters
    ----------
    timestamps : np.ndarray
        Shape (N,), timestamps in seconds.
    centers : np.ndarray
        Shape (N, 2), array with NaNs for missing frames.

    Returns
    -------
    filled : np.ndarray
        Shape (N, 2), array with missing frames filled.
    """
    ts = np.asarray(timestamps, dtype=np.float32)
    C = np.asarray(centers, dtype=np.float32)

    if C.shape[0] != ts.shape[0]:
        raise ValueError(f"timestamps and centers must have same length, got {ts.shape[0]} and {C.shape[0]}")

    filled = C.copy()

    # Valid frames = where both x and y are finite
    valid = np.isfinite(C[:, 0]) & np.isfinite(C[:, 1])
    idx_valid = np.where(valid)[0]

    if idx_valid.size == 0:
        # No valid centers at all
        return filled

    if idx_valid.size == 1:
        # Only one detection: fill everything with that center
        filled[:] = C[idx_valid[0]]
        return filled

    # 1) Fill prefix (before first detection) with first valid center
    first = idx_valid[0]
    filled[:first] = C[first]

    # 2) Fill suffix (after last detection) with last valid center
    last = idx_valid[-1]
    filled[last + 1 :] = C[last]

    # 3) Fill gaps between valid detections by linear interpolation
    for i0, i1 in zip(idx_valid, idx_valid[1:]):
        if i1 == i0 + 1:
            # Consecutive frames, no gap
            continue

        t0, t1 = ts[i0], ts[i1]
        c0, c1 = C[i0], C[i1]

        for k in range(i0 + 1, i1):
            if t1 > t0:
                alpha = (ts[k] - t0) / (t1 - t0)
            else:
                # Fallback: interpolate by index if timestamps are degenerate
                alpha = (k - i0) / (i1 - i0)
            filled[k] = (1.0 - alpha) * c0 + alpha * c1

    return filled


def compute_ball_centers_from_frames(
    frames: List[Dict[str, Any]],
    tracker: Optional[BallTracker] = None,
    dt: Optional[float] = None,
    interpolate: bool = False,
) -> BallCenters:
    """Compute ball centers with BallTracker + interpolation from a list of frames.

    Steps:
      1) For each frame:
           - If there is a ball bbox:
               * compute raw center from bbox.
               * if tracker is given, update it and get cleaned center.
           - If no ball bbox:
               * centers_raw[i] = NaN
               * centers_tracker[i] = NaN  (no prediction here).

      2) Choose the basis for interpolation:
           - If tracker is provided: use centers_tracker.
           - Else: use centers_raw.

      3) If interpolate=True: interpolate in time to fill ALL frames (pure interpolation).
         If interpolate=False: keep the chosen basis as-is (NaNs remain where missing).

    Parameters
    ----------
    frames : List[Dict[str, Any]]
        List of frame dictionaries with ballDetections.
    tracker : Optional[BallTracker]
        Optional BallTracker instance for cleaning detections.
        If None, uses raw bbox centers directly.
    dt : Optional[float]
        Time delta between frames in seconds. Used as fallback if frames
        don't have timestamps. Defaults to 0.1 (10 fps).
    interpolate : bool, default=False
        If True, fill missing frames by linear interpolation in time.
        If False, do not fill missing frames; `centers_full/radii_full` will
        equal the chosen basis (`centers_tracker/radius_tracker` if tracker
        is provided else `centers_raw/radius_raw`).

    Returns
    -------
    BallCenters
        Object containing timestamps and ball centers (raw, tracked, and interpolated).

    Example
    -------
        >>> tracker = BallTracker(alpha=1.0)
        >>> bc = compute_ball_centers_from_frames(frames, tracker=tracker)
        >>> ball_xy = bc.centers_full  # shape (N, 2)
    """
    if not frames:
        raise ValueError("frames list cannot be empty")

    N = len(frames)
    timestamps = np.zeros(N, dtype=np.float32)
    centers_raw = np.full((N, 2), np.nan, dtype=np.float32)
    radii_raw = np.full(N, np.nan, dtype=np.float32)
    centers_tracker = np.full((N, 2), np.nan, dtype=np.float32)
    radii_tracker = np.full(N, np.nan, dtype=np.float32)

    # Default dt fallback (10 fps)
    if dt is None:
        dt = 0.1

    # Reset tracker if provided
    if tracker is not None:
        tracker.reset()

    for i, fr in enumerate(frames):
        # Extract timestamp with fallback
        ts = fr.get("timestamp", i * dt)
        timestamps[i] = float(ts)

        # Find ball detections
        ball_dets = [
            d for d in fr.get("ballDetections", [])
            if d.get("class") == "ball"
        ]

        if not ball_dets:
            continue

        # Use highest-score detection
        best = max(ball_dets, key=lambda d: d.get("score", 0.0))

        # Extract bbox
        if "bbox" not in best:
            continue

        try:
            # Raw center from bbox
            center_raw, radius_raw = bbox_to_center_radius(best["bbox"])
            centers_raw[i] = center_raw
            radii_raw[i] = radius_raw

            # Cleaned center from tracker, if provided
            if tracker is not None:
                center_clean, radius_clean = tracker.update_from_detection(best)
                centers_tracker[i] = center_clean
                radii_tracker[i] = radius_clean
        except (ValueError, KeyError) as e:
            # Skip invalid detections
            print(e)
            continue

    # Basis for interpolation
    if tracker is not None:
        centers_basis = centers_tracker
        radius_basis = radii_tracker
    else:
        centers_basis = centers_raw
        radius_basis = radii_raw

    if interpolate:
        centers_full = interpolate_centers(timestamps, centers_basis)
        radii_full = interpolate_1d_values(timestamps, radius_basis)
    else:
        centers_full = centers_basis.copy()
        radii_full = radius_basis.copy()
    return BallCenters(
        timestamps=timestamps,
        centers_raw=centers_raw,
        radii_raw=radii_raw,
        centers_tracker=centers_tracker,
        radii_tracker=radii_tracker,
        centers_full=centers_full,
        radii_full=radii_full,
    )


def compute_ball_centers_from_json(
    det_json: Dict[str, Any],
    use_tracker: bool = True,
    tracker_alpha: float = 1.0,
    interpolate: bool = False,
) -> BallCenters:
    """Convenience wrapper for full detection JSON (with 'frames' key).

    Parameters
    ----------
    det_json : dict
        Detection JSON with 'frames' key.
    use_tracker : bool, default=True
        If True, use BallTracker to clean detections before interpolation.
        If False, interpolate raw bbox centers directly.
    tracker_alpha : float, default=1.0
        Smoothing factor for BallTracker.
    interpolate : bool, default=False
        If True, fill missing frames by linear interpolation in time.
        If False, keep NaNs for missing frames.

    Returns
    -------
    BallCenters
        Object containing timestamps and ball centers.

    Raises
    ------
    ValueError
        If det_json does not have 'frames' key or frames list is empty.
    """
    if "frames" not in det_json:
        raise ValueError("det_json must have 'frames' key")

    frames = det_json.get("frames", [])
    if not frames:
        raise ValueError("frames list cannot be empty")

    tracker = BallTracker(alpha=tracker_alpha) if use_tracker else None
    return compute_ball_centers_from_frames(frames, tracker=tracker, interpolate=interpolate)
