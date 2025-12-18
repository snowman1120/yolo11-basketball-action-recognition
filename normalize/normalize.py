from typing import Any, Dict, Optional
import numpy as np
from dataclasses import dataclass
from parsers import KEYPOINT_NAMES
from ball_tracking_pipeline import BallTracker, compute_ball_centers_from_frames
from constants import KEYPOINT_CONF_THRESH

@dataclass
class FrameFeatures:
    # --- time / validity ---
    timestamps: np.ndarray                  # (N,)
    pose_present: np.ndarray                # (N,) bool

    # --- ball (normalized by center/scale) ---
    ball_norm: np.ndarray                   # (N,2)
    ball_present: np.ndarray                # (N,) bool  (we have an estimate: tracked/interpolated AND pose_present)
    ball_measured: np.ndarray               # (N,) bool  (detector actually saw the ball bbox in that frame)

    ball_center: np.ndarray                 # (N,2) pixels (NaN-safe output)
    ball_diameter: np.ndarray               # (N,) pixels (NaN-safe output)
    ball_diameter_norm: np.ndarray          # (N,) ball_diameter / scale

    # --- ball kinematics (normalized space) ---
    ball_vel: np.ndarray                    # (N,2) (vx,vy)
    ball_speed: np.ndarray                  # (N,)
    ball_vx: np.ndarray                     # (N,)
    ball_vy: np.ndarray                     # (N,)

    ball_acc: np.ndarray                    # (N,2) (ax,ay)
    ball_ax: np.ndarray                     # (N,)
    ball_ay: np.ndarray                     # (N,)
    ball_acc_mag: np.ndarray                # (N,)

    ball_dir_sin: np.ndarray                # (N,) ~ vy / speed
    ball_dir_cos: np.ndarray                # (N,) ~ vx / speed

    # --- hand/pose-ball interaction ---
    hand_ball_dist: np.ndarray              # (N,) min(dist(ball,left_wrist), dist(ball,right_wrist))
    hand_ball_dist_L: np.ndarray            # (N,) dist(ball,left_wrist) (0 if invalid)
    hand_ball_dist_R: np.ndarray            # (N,) dist(ball,right_wrist) (0 if invalid)
    hand_used: np.ndarray                   # (N,) int8: 0 left, 1 right, -1 none/unknown
    release_score: np.ndarray               # (N,) Δ(min hand-ball dist) (positive spike near release)

    # --- pose geometry cues ---
    wrist_head_rel_y_L: np.ndarray          # (N,) (head_y - left_wrist_y) in normalized coords
    wrist_head_rel_y_R: np.ndarray          # (N,) (head_y - right_wrist_y)
    elbow_angle_L: np.ndarray               # (N,) radians at left_elbow (0 if invalid)
    elbow_angle_R: np.ndarray               # (N,) radians at right_elbow (0 if invalid)

    # --- simple temporal event flags (float 0/1) ---
    ball_y_local_min: np.ndarray            # (N,) 1 if local min of ball_y
    ball_y_local_max: np.ndarray            # (N,) 1 if local max of ball_y
    ball_vy_sign_flip: np.ndarray           # (N,) 1 if vy sign flips vs previous frame

    # --- other useful signals already in your pipeline ---
    ankle_y: np.ndarray                     # (N,) normalized ankle y (avg)
    center_y_img: np.ndarray                # (N,) global vertical position of center (0..1)

    # --- raw pose outputs ---
    kp_norm: Dict[str, np.ndarray]          # each (N,2)
    kp_conf: Dict[str, np.ndarray]          # each (N,)
    center: np.ndarray                      # (N,2) pixels
    scale: np.ndarray                       # (N,) pixels


def _select_main_pose(frame: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    poses = frame.get("poseDetections", [])
    if not poses:
        return None
    return max(poses, key=lambda p: p.get("score", 0.0))

def _ema_1d(x: np.ndarray, alpha: float) -> np.ndarray:
    y = x.copy()
    prev = np.nan
    for i in range(len(x)):
        xi = x[i]
        if not np.isfinite(xi):
            y[i] = prev if np.isfinite(prev) else xi
            continue
        prev = xi if not np.isfinite(prev) else (alpha * xi + (1.0 - alpha) * prev)
        y[i] = prev
    return y

def _ffill_2d(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    last = np.array([np.nan, np.nan], dtype=y.dtype)
    for i in range(len(y)):
        if np.all(np.isfinite(y[i])):
            last = y[i]
        elif np.all(np.isfinite(last)):
            y[i] = last
    return y

def _bfill_2d(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    last = np.array([np.nan, np.nan], dtype=y.dtype)
    for i in range(len(y) - 1, -1, -1):
        if np.all(np.isfinite(y[i])):
            last = y[i]
        elif np.all(np.isfinite(last)):
            y[i] = last
    return y



def _angle_at_joint(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC at point B in radians. Returns 0.0 if degenerate."""
    u = a - b
    v = c - b
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-6 or nv < 1e-6:
        return 0.0
    cosang = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.arccos(cosang))


def _infer_fps_from_timestamps(frames: list) -> Optional[float]:
    ts = [fr.get("timestamp", None) for fr in frames]
    # Use float64 to reduce precision loss when timestamps are e.g. multiples of 0.1
    # (float32 can turn 0.1 into 0.100000001..., which slightly perturbs inferred FPS).
    ts = np.array([t for t in ts if t is not None], dtype=np.float64)
    if len(ts) < 3:
        return None
    d = np.diff(ts)
    d = d[np.isfinite(d) & (d > 1e-6)]
    if len(d) == 0:
        return None
    fps = float(1.0 / np.median(d))
    # If we're extremely close to an integer FPS (common after fixed-rate sampling),
    # snap to that integer to avoid surprising values like 9.99999999046.
    fps_int = round(fps)
    if abs(fps - fps_int) <= 1e-3:
        return float(fps_int)
    return fps

def build_frame_features(
    det_json: Dict[str, Any],
    fps: Optional[float] = None,
    tracker_alpha: float = 1.0,
    deriv_smooth_alpha: Optional[float] = None,
    interpolate_ball: bool = False,
) -> FrameFeatures:
    """
    Build per-frame (pose + ball) features, normalized by a body-centric center/scale.

    Key design choices (important for model quality):
      - Missing pose/ball values are filled with 0.0 *but* you must use the masks:
            pose_present, ball_present, ball_measured
        so the model can distinguish "missing" from "real zero".
      - ball_present means you have an estimate (tracked/interpolated) AND pose_present.
      - ball_measured means the detector actually produced a ball bbox in that frame.
    """
    frames = det_json.get("frames", [])
    if not frames:
        raise ValueError("det_json has no frames.")

    meta = det_json.get("metadata", {})
    W = meta.get("videoWidth", None)
    H = meta.get("videoHeight", None)

    # ---- FPS: timestamps -> min(videoFPS,targetFPS) -> provided ----
    if fps is None:
        fps_ts = _infer_fps_from_timestamps(frames)
        if fps_ts is not None:
            fps = fps_ts
        else:
            video_fps = meta.get("videoFPS", None)
            target_fps = meta.get("targetFPS", None)
            if video_fps is None and target_fps is None:
                raise ValueError("fps not provided and cannot infer from timestamps or metadata.")
            candidates = [v for v in [video_fps, target_fps] if v is not None]
            fps = float(min(candidates))

    fps = float(fps)
    default_dt = 1.0 / fps
    N = len(frames)

    # Allocate arrays
    timestamps = np.zeros(N, dtype=np.float32)
    kp_xy = {name: np.full((N, 2), np.nan, dtype=np.float32) for name in KEYPOINT_NAMES}
    kp_conf = {name: np.zeros(N, dtype=np.float32) for name in KEYPOINT_NAMES}

    # Pose + timestamps
    for i, fr in enumerate(frames):
        timestamps[i] = float(fr.get("timestamp", i * default_dt))
        pose = _select_main_pose(fr)
        if pose is not None:
            for kp in pose.get("keypoints", []):
                name = kp.get("name")
                if name in kp_xy:
                    kp_xy[name][i, 0] = float(kp.get("x", np.nan))
                    kp_xy[name][i, 1] = float(kp.get("y", np.nan))
                    kp_conf[name][i] = float(kp.get("confidence", 0.0))

    # ball_measured: detector saw a valid ball bbox in that frame (pre-tracking/interp)
    ball_measured = np.zeros(N, dtype=bool)
    for i, fr in enumerate(frames):
        dets = [
            d for d in fr.get("ballDetections", [])
            if d.get("class") == "ball" and isinstance(d.get("bbox", None), dict)
        ]
        ball_measured[i] = (len(dets) > 0)

    # Ball tracking (+ optional interpolation) (pixel coords)
    ball_tracker = BallTracker(alpha=tracker_alpha)
    ball_centers = compute_ball_centers_from_frames(
        frames,
        tracker=ball_tracker,
        dt=default_dt,
        interpolate=interpolate_ball,
    )
    # Keep raw (may contain NaNs) for presence gating; sanitize only for outputs later.
    ball_xy = ball_centers.centers_full.astype(np.float32)
    ball_radii = ball_centers.radii_full.astype(np.float32)

    # Center: mid-hip -> mid-shoulder -> fill
    left_hip = kp_xy["left_hip"]
    right_hip = kp_xy["right_hip"]
    left_sh = kp_xy["left_shoulder"]
    right_sh = kp_xy["right_shoulder"]

    mid_hip = 0.5 * (left_hip + right_hip)
    mid_sh = 0.5 * (left_sh + right_sh)

    center = mid_hip.copy()
    bad_center = ~np.isfinite(center).all(axis=1)
    center[bad_center] = mid_sh[bad_center]

    center = _ffill_2d(center)
    center = _bfill_2d(center)

    if not np.any(np.isfinite(center).all(axis=1)):
        raise ValueError("Cannot determine a valid center from keypoints (hips/shoulders missing).")

    # Scale: shoulder -> hip -> median, then smooth
    shoulder_w = np.linalg.norm(left_sh - right_sh, axis=1).astype(np.float32)
    hip_w = np.linalg.norm(left_hip - right_hip, axis=1).astype(np.float32)

    scale = shoulder_w.copy()
    bad_scale = ~np.isfinite(scale) | (scale < 1e-3)
    scale[bad_scale] = hip_w[bad_scale]

    good = np.isfinite(scale) & (scale > 1e-3)
    if not np.any(good):
        raise ValueError("Cannot determine a valid scale from keypoints.")
    median_scale = float(np.nanmedian(scale[good]))
    scale[~good] = median_scale

    scale = _ema_1d(scale, alpha=0.2).astype(np.float32)
    scale_col = scale.reshape(-1, 1)

    pose_present = np.isfinite(center).all(axis=1) & np.isfinite(scale) & (scale > 1e-3)

    # Normalize keypoints
    kp_norm: Dict[str, np.ndarray] = {}
    for name in KEYPOINT_NAMES:
        kp_norm[name] = (kp_xy[name] - center) / scale_col

    # Normalize ball (pixel -> body-centric normalized)
    ball_norm = (ball_xy - center) / scale_col

    # ball_present: do we have an estimate (tracked/interpolated) AND pose_present
    ball_present = np.isfinite(ball_xy).all(axis=1) & pose_present

    ball_phys = ball_measured & ball_present  # measured (real) ball only

    # Fill missing kp/ball with 0 (model-safe), but keep masks to disambiguate
    for name in KEYPOINT_NAMES:
        valid_kp = np.isfinite(kp_norm[name]).all(axis=1) & pose_present
        kp_norm[name][~valid_kp] = 0.0
        kp_conf[name][~valid_kp] = 0.0

    ball_norm[~ball_present] = 0.0

    # Optional: extra smoothing for *derivatives* (vx/vy/ax/ay) when tracker_alpha is high.
    # We keep ball_norm itself (static position feature) unchanged; only kinematics use the smoothed path.
    if deriv_smooth_alpha is None:
        deriv_smooth_alpha = 0.35 if tracker_alpha >= 0.999 else 0.0

    if float(deriv_smooth_alpha) > 0.0:
        bn = ball_norm.copy()
        bn[~ball_phys] = np.nan
        bx = _ema_1d(bn[:, 0], alpha=float(deriv_smooth_alpha))
        by = _ema_1d(bn[:, 1], alpha=float(deriv_smooth_alpha))
        ball_norm_for_deriv = np.stack([bx, by], axis=1).astype(np.float32)
    else:
        ball_norm_for_deriv = ball_norm

    # Useful y feature: ankle height (already fine)
    left_ankle_norm = kp_norm["left_ankle"]
    right_ankle_norm = kp_norm["right_ankle"]
    ankle_y = (0.5 * (left_ankle_norm[:, 1] + right_ankle_norm[:, 1])).astype(np.float32)

    # Global vertical position feature (0..1) if H known
    if H is not None and float(H) > 0:
        center_y_img = (center[:, 1] / float(H)).astype(np.float32)
    else:
        center_y_img = np.zeros(N, dtype=np.float32)

    # ----------------- Ball kinematics (normalized space) -----------------
    ball_vel = np.zeros_like(ball_norm, dtype=np.float32)
    for i in range(1, N):
        if not (ball_phys[i] and ball_phys[i - 1]):
            continue
        dt_i = float(timestamps[i] - timestamps[i - 1])
        if dt_i <= 1e-6:
            dt_i = default_dt
        ball_vel[i] = (ball_norm_for_deriv[i] - ball_norm_for_deriv[i - 1]) / dt_i

    ball_speed = np.linalg.norm(ball_vel, axis=1).astype(np.float32)
    ball_vx = ball_vel[:, 0].astype(np.float32)
    ball_vy = ball_vel[:, 1].astype(np.float32)

    ball_acc = np.zeros_like(ball_vel, dtype=np.float32)
    for i in range(2, N):
        if not (ball_phys[i] and ball_phys[i - 1] and ball_phys[i - 2]):
            continue
        dt_i = float(timestamps[i] - timestamps[i - 1])
        if dt_i <= 1e-6:
            dt_i = default_dt
        ball_acc[i] = (ball_vel[i] - ball_vel[i - 1]) / dt_i

    ball_ax = ball_acc[:, 0].astype(np.float32)
    ball_ay = ball_acc[:, 1].astype(np.float32)
    ball_acc_mag = np.linalg.norm(ball_acc, axis=1).astype(np.float32)

    # Direction encoding that avoids atan2 wrap: (cos,sin) = (vx/speed, vy/speed)
    eps = 1e-6
    ball_dir_cos = np.zeros(N, dtype=np.float32)
    ball_dir_sin = np.zeros(N, dtype=np.float32)
    ok_dir = ball_speed > eps
    ball_dir_cos[ok_dir] = ball_vx[ok_dir] / (ball_speed[ok_dir] + eps)
    ball_dir_sin[ok_dir] = ball_vy[ok_dir] / (ball_speed[ok_dir] + eps)

    # ----------------- Hand-ball distances + release -----------------
    left_wrist_norm = kp_norm["left_wrist"]
    right_wrist_norm = kp_norm["right_wrist"]
    lw_ok = (kp_conf["left_wrist"] >= KEYPOINT_CONF_THRESH) & pose_present
    rw_ok = (kp_conf["right_wrist"] >= KEYPOINT_CONF_THRESH) & pose_present

    hand_ball_dist_L = np.zeros(N, dtype=np.float32)
    hand_ball_dist_R = np.zeros(N, dtype=np.float32)
    hand_ball_dist = np.zeros(N, dtype=np.float32)
    hand_used = np.full(N, -1, dtype=np.int8)
    dmin_valid = np.zeros(N, dtype=bool)

    for i in range(N):
        if not ball_present[i]:
            continue

        dL = np.inf
        dR = np.inf
        if lw_ok[i]:
            dL = float(np.linalg.norm(ball_norm[i] - left_wrist_norm[i]))
            hand_ball_dist_L[i] = dL
        if rw_ok[i]:
            dR = float(np.linalg.norm(ball_norm[i] - right_wrist_norm[i]))
            hand_ball_dist_R[i] = dR

        d = min(dL, dR)
        if np.isfinite(d):
            hand_ball_dist[i] = float(d)
            dmin_valid[i] = True
            hand_used[i] = 0 if dL <= dR else 1

    release_score = np.zeros(N, dtype=np.float32)
    for i in range(1, N):
        if dmin_valid[i] and dmin_valid[i - 1]:
            release_score[i] = hand_ball_dist[i] - hand_ball_dist[i - 1]

    # ----------------- Pose geometry cues (shoot vs pass) -----------------
    # Head proxy (COCO-style pose usually has "nose")
    head_norm = kp_norm.get("nose", None)
    head_ok = None
    if head_norm is not None and "nose" in kp_conf:
        head_ok = (kp_conf["nose"] >= KEYPOINT_CONF_THRESH) & pose_present
    else:
        head_norm = np.zeros((N, 2), dtype=np.float32)
        head_ok = np.zeros(N, dtype=bool)

    wrist_head_rel_y_L = np.zeros(N, dtype=np.float32)
    wrist_head_rel_y_R = np.zeros(N, dtype=np.float32)
    for i in range(N):
        if not head_ok[i]:
            continue
        if lw_ok[i]:
            wrist_head_rel_y_L[i] = float(head_norm[i, 1] - left_wrist_norm[i, 1])
        if rw_ok[i]:
            wrist_head_rel_y_R[i] = float(head_norm[i, 1] - right_wrist_norm[i, 1])

    # Elbow extension angles: angle(shoulder, elbow, wrist)
    elbow_angle_L = np.zeros(N, dtype=np.float32)
    elbow_angle_R = np.zeros(N, dtype=np.float32)

    l_ok = (kp_conf.get("left_shoulder", 0.0) > 0.0) & (kp_conf.get("left_elbow", 0.0) > 0.0) & (kp_conf.get("left_wrist", 0.0) > 0.0) & pose_present
    r_ok = (kp_conf.get("right_shoulder", 0.0) > 0.0) & (kp_conf.get("right_elbow", 0.0) > 0.0) & (kp_conf.get("right_wrist", 0.0) > 0.0) & pose_present

    for i in range(N):
        if l_ok[i]:
            elbow_angle_L[i] = _angle_at_joint(kp_norm["left_shoulder"][i], kp_norm["left_elbow"][i], kp_norm["left_wrist"][i])
        if r_ok[i]:
            elbow_angle_R[i] = _angle_at_joint(kp_norm["right_shoulder"][i], kp_norm["right_elbow"][i], kp_norm["right_wrist"][i])

    # ----------------- Simple temporal event flags -----------------
    ball_y_local_min = np.zeros(N, dtype=np.float32)
    ball_y_local_max = np.zeros(N, dtype=np.float32)
    for i in range(1, N - 1):
        if not (ball_phys[i - 1] and ball_phys[i] and ball_phys[i + 1]):
            continue
        y0 = float(ball_norm_for_deriv[i - 1, 1])
        y1 = float(ball_norm_for_deriv[i, 1])
        y2 = float(ball_norm_for_deriv[i + 1, 1])
        if y0 > y1 < y2:
            ball_y_local_min[i] = 1.0
        if y0 < y1 > y2:
            ball_y_local_max[i] = 1.0

    ball_vy_sign_flip = np.zeros(N, dtype=np.float32)
    for i in range(1, N):
        if not (ball_phys[i] and ball_phys[i - 1]):
            continue
        if abs(ball_vy[i]) < 1e-5 or abs(ball_vy[i - 1]) < 1e-5:
            continue
        if np.sign(ball_vy[i]) != np.sign(ball_vy[i - 1]):
            ball_vy_sign_flip[i] = 1.0

    # ----------------- Ball size outputs -----------------
    ball_diameter = (ball_radii * 2.0).astype(np.float32)

    # Model/JSON-safe outputs: replace any NaN/inf values with 0 without affecting gating logic.
    ball_xy_out = np.nan_to_num(ball_xy, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    ball_diameter_out = np.nan_to_num(ball_diameter, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # normalize ball size by scale (very helpful)
    ball_diameter_norm = np.zeros(N, dtype=np.float32)
    ok = np.isfinite(ball_diameter) & np.isfinite(scale) & (scale > 1e-6)
    ball_diameter_norm[ok] = ball_diameter[ok] / scale[ok]

    return FrameFeatures(
        timestamps=timestamps.astype(np.float32),
        pose_present=pose_present.astype(bool),

        ball_norm=ball_norm.astype(np.float32),
        ball_present=ball_present.astype(bool),
        ball_measured=ball_measured.astype(bool),

        ball_center=ball_xy_out,
        ball_diameter=ball_diameter_out,
        ball_diameter_norm=ball_diameter_norm.astype(np.float32),

        ball_vel=ball_vel.astype(np.float32),
        ball_speed=ball_speed.astype(np.float32),
        ball_vx=ball_vx.astype(np.float32),
        ball_vy=ball_vy.astype(np.float32),

        ball_acc=ball_acc.astype(np.float32),
        ball_ax=ball_ax.astype(np.float32),
        ball_ay=ball_ay.astype(np.float32),
        ball_acc_mag=ball_acc_mag.astype(np.float32),

        ball_dir_sin=ball_dir_sin.astype(np.float32),
        ball_dir_cos=ball_dir_cos.astype(np.float32),

        hand_ball_dist=hand_ball_dist.astype(np.float32),
        hand_ball_dist_L=hand_ball_dist_L.astype(np.float32),
        hand_ball_dist_R=hand_ball_dist_R.astype(np.float32),
        hand_used=hand_used.astype(np.int8),
        release_score=release_score.astype(np.float32),

        wrist_head_rel_y_L=wrist_head_rel_y_L.astype(np.float32),
        wrist_head_rel_y_R=wrist_head_rel_y_R.astype(np.float32),
        elbow_angle_L=elbow_angle_L.astype(np.float32),
        elbow_angle_R=elbow_angle_R.astype(np.float32),

        ball_y_local_min=ball_y_local_min.astype(np.float32),
        ball_y_local_max=ball_y_local_max.astype(np.float32),
        ball_vy_sign_flip=ball_vy_sign_flip.astype(np.float32),

        ankle_y=ankle_y.astype(np.float32),
        center_y_img=center_y_img.astype(np.float32),

        kp_norm=kp_norm,
        kp_conf=kp_conf,
        center=center.astype(np.float32),
        scale=scale.astype(np.float32),
    )
