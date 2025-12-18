# features_pack_and_standardize_fixed.py
# v3: stable keypoint order + ball_measured gating + strict one-hot hand_used + safe masks
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

EPS = 1e-6

# COCO-17 keypoint order (common convention)
KEYPOINT_ORDER = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

def _as_float32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)

def _as_bool(x) -> np.ndarray:
    return np.asarray(x, dtype=bool)

def _interp_1d(t_new: np.ndarray, t_old: np.ndarray, y_old: np.ndarray) -> np.ndarray:
    return np.interp(t_new, t_old, y_old).astype(np.float32)

def _nearest_bool(t_new: np.ndarray, t_old: np.ndarray, b_old: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(t_old, t_new, side="left")
    idx = np.clip(idx, 0, len(t_old) - 1)
    idx2 = np.clip(idx - 1, 0, len(t_old) - 1)
    choose_left = (np.abs(t_new - t_old[idx2]) <= np.abs(t_new - t_old[idx]))
    idx_final = np.where(choose_left, idx2, idx)
    return b_old[idx_final].astype(bool)

def resample_to_fps(
    X: np.ndarray,
    timestamps: np.ndarray,
    target_fps: float,
    bool_idx: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample (T,D) to uniform target_fps.

    Continuous dims: linear interpolation.
    Boolean dims: nearest neighbor (thresholded >0.5 before NN).
    """
    timestamps = _as_float32(timestamps)
    X = _as_float32(X)
    if len(timestamps) < 2:
        return X, timestamps
    t0 = float(timestamps[0])
    t1 = float(timestamps[-1])
    if t1 <= t0 + 1e-6:
        return X, timestamps

    dt = 1.0 / float(target_fps)
    t_new = np.arange(t0, t1 + 0.5 * dt, dt, dtype=np.float32)

    D = X.shape[1]
    X_new = np.zeros((len(t_new), D), dtype=np.float32)

    bool_idx = np.asarray(bool_idx, dtype=np.int64) if bool_idx is not None else np.array([], dtype=np.int64)
    bool_mask = np.zeros(D, dtype=bool)
    if len(bool_idx) > 0:
        bool_mask[bool_idx] = True
    cont_idx = np.where(~bool_mask)[0]

    for j in cont_idx:
        X_new[:, j] = _interp_1d(t_new, timestamps, X[:, j])

    for j in bool_idx:
        b = X[:, j] > 0.5
        X_new[:, j] = _nearest_bool(t_new, timestamps, b).astype(np.float32)

    return X_new, t_new

def _safe_dt(t: np.ndarray) -> np.ndarray:
    dt = np.diff(t, prepend=t[0])
    dt = np.maximum(dt, 1e-3)
    return dt.astype(np.float32)

def _derivative_1d(x: np.ndarray, t: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    x = _as_float32(x)
    t = _as_float32(t)
    dt = _safe_dt(t)
    dx = np.zeros_like(x, dtype=np.float32)
    dx[1:] = (x[1:] - x[:-1]) / dt[1:]
    if valid is not None:
        v = _as_bool(valid)
        ok = v & np.roll(v, 1)
        ok[0] = False
        dx = np.where(ok, dx, 0.0).astype(np.float32)
    return np.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def _derivative_2d(xy: np.ndarray, t: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    xy = _as_float32(xy)
    vx = _derivative_1d(xy[:, 0], t, valid)
    vy = _derivative_1d(xy[:, 1], t, valid)
    return np.stack([vx, vy], axis=1).astype(np.float32)

def _angle_at_joint(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    a = _as_float32(a); b = _as_float32(b); c = _as_float32(c)
    ba = a - b
    bc = c - b
    ba_n = np.linalg.norm(ba, axis=1) + EPS
    bc_n = np.linalg.norm(bc, axis=1) + EPS
    cosang = (ba[:, 0]*bc[:, 0] + ba[:, 1]*bc[:, 1]) / (ba_n * bc_n)
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.arccos(cosang).astype(np.float32)
    return np.nan_to_num(ang, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def _l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1).astype(np.float32)

def _local_extrema_flags(y: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T = len(y)
    y = _as_float32(y)
    v = _as_bool(valid)
    local_min = np.zeros(T, dtype=np.float32)
    local_max = np.zeros(T, dtype=np.float32)
    if T < 3:
        return local_min, local_max
    y0, y1, y2 = y[:-2], y[1:-1], y[2:]
    v_mid = v[1:-1] & v[:-2] & v[2:]
    local_min[1:-1] = ((y0 > y1) & (y2 > y1) & v_mid).astype(np.float32)
    local_max[1:-1] = ((y0 < y1) & (y2 < y1) & v_mid).astype(np.float32)
    return local_min, local_max

def _sign_flip_flag(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    x = _as_float32(x)
    v = _as_bool(valid)
    s = np.sign(x)
    ok = v & np.roll(v, 1)
    ok[0] = False
    nonzero = (s != 0) & (np.roll(s, 1) != 0)
    flip = ((s != np.roll(s, 1)) & ok & nonzero).astype(np.float32)
    return flip

def _get_kp_series(
    kp_norm: Dict[str, Any],
    kp_conf: Dict[str, Any],
    name: str,
    T: int
) -> Tuple[np.ndarray, np.ndarray]:
    xy = _as_float32(kp_norm.get(name, np.zeros((T, 2), np.float32)))
    cf = _as_float32(kp_conf.get(name, np.zeros((T,), np.float32)))
    return xy, cf

def pack_from_normalized_json(
    data: Dict[str, Any],
    target_fps: float = 30.0,
    kp_conf_thresh: float = 0.2,
    keypoint_order: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Pack normalized JSON into (T,D) with engineered features.

    Returns:
      X             (T,D) float32 (resampled)
      feature_names list[str]
      idx_cont      indices to standardize
      idx_bool      indices that are boolean-like (0/1)
      timestamps    (T,) seconds (resampled)
    """
    nf = data["normalized_features"]
    t = _as_float32(nf["timestamps"])
    T = len(t)

    # core signals (safe defaults)
    ball_norm = _as_float32(nf.get("ball_norm", np.zeros((T, 2), np.float32)))
    ball_speed = _as_float32(nf.get("ball_speed", np.zeros(T, np.float32)))
    ball_vy = _as_float32(nf.get("ball_vy", np.zeros(T, np.float32)))
    hand_ball_dist = _as_float32(nf.get("hand_ball_dist", np.zeros(T, np.float32)))
    ankle_y = _as_float32(nf.get("ankle_y", np.zeros(T, np.float32)))
    ball_diam_norm = _as_float32(nf.get("ball_diameter_norm", np.zeros(T, np.float32)))

    pose_present = _as_bool(nf.get("pose_present", np.zeros(T, bool)))
    ball_present = _as_bool(nf.get("ball_present", np.zeros(T, bool)))
    ball_measured = _as_bool(nf.get("ball_measured", ball_present))

    # use measured ball only for physics/events
    ball_valid = (ball_present & ball_measured)

    # keypoints
    kp_norm = nf.get("keypoints_normalized", {})
    kp_conf = nf.get("keypoints_confidence", {})
    kp_names = list(keypoint_order) if keypoint_order is not None else KEYPOINT_ORDER

    kp_xy_list = []
    kp_cf_list = []
    for name in kp_names:
        xy, cf = _get_kp_series(kp_norm, kp_conf, name, T)
        kp_xy_list.append(xy)
        kp_cf_list.append(cf.reshape(-1, 1))
    kp_xy = np.concatenate(kp_xy_list, axis=1).astype(np.float32)  # (T,2*K)
    kp_cf = np.concatenate(kp_cf_list, axis=1).astype(np.float32)  # (T,K)

    # kinematics
    if "ball_vel" in nf:
        ball_vel = _as_float32(nf["ball_vel"])
    else:
        ball_vel = _derivative_2d(ball_norm, t, valid=ball_valid)
    ball_vx = ball_vel[:, 0]
    ball_vy_derived = ball_vel[:, 1]
    if "ball_vy" not in nf:
        ball_vy = ball_vy_derived

    if "ball_acc" in nf:
        ball_acc = _as_float32(nf["ball_acc"])
    else:
        ball_acc = _derivative_2d(ball_vel, t, valid=ball_valid)
    ball_ax = ball_acc[:, 0]
    ball_ay = ball_acc[:, 1]

    speed = np.sqrt(ball_vx**2 + ball_vy**2) + EPS
    ball_dir_cos = (ball_vx / speed).astype(np.float32)
    ball_dir_sin = (ball_vy / speed).astype(np.float32)
    ball_dir_cos = np.where(ball_valid, ball_dir_cos, 0.0).astype(np.float32)
    ball_dir_sin = np.where(ball_valid, ball_dir_sin, 0.0).astype(np.float32)

    # wrists/head
    lw, lwc = _get_kp_series(kp_norm, kp_conf, "left_wrist", T)
    rw, rwc = _get_kp_series(kp_norm, kp_conf, "right_wrist", T)
    wrist_ok_L = pose_present & (lwc >= kp_conf_thresh)
    wrist_ok_R = pose_present & (rwc >= kp_conf_thresh)
    hb_ok = pose_present & ball_valid

    dL = np.where(hb_ok & wrist_ok_L, _l2(ball_norm, lw), 0.0).astype(np.float32)
    dR = np.where(hb_ok & wrist_ok_R, _l2(ball_norm, rw), 0.0).astype(np.float32)

    dMin = np.zeros(T, dtype=np.float32)
    both = hb_ok & wrist_ok_L & wrist_ok_R
    onlyL = hb_ok & wrist_ok_L & ~wrist_ok_R
    onlyR = hb_ok & wrist_ok_R & ~wrist_ok_L
    dMin = np.where(both, np.minimum(dL, dR), dMin).astype(np.float32)
    dMin = np.where(onlyL, dL, dMin).astype(np.float32)
    dMin = np.where(onlyR, dR, dMin).astype(np.float32)

    # strict one-hot hand used
    hand_left = np.zeros(T, dtype=np.float32)
    hand_right = np.zeros(T, dtype=np.float32)
    hand_unknown = np.ones(T, dtype=np.float32)

    hand_left[onlyL] = 1.0
    hand_right[onlyR] = 1.0
    hand_unknown[onlyL | onlyR] = 0.0

    pickL = both & (dL <= dR)  # tie -> left
    pickR = both & (dR < dL)
    hand_left[pickL] = 1.0
    hand_right[pickR] = 1.0
    hand_unknown[both] = 0.0

    release_score = _derivative_1d(dMin, t, valid=hb_ok)

    # pose cues
    ls, _ = _get_kp_series(kp_norm, kp_conf, "left_shoulder", T)
    le, _ = _get_kp_series(kp_norm, kp_conf, "left_elbow", T)
    rs, _ = _get_kp_series(kp_norm, kp_conf, "right_shoulder", T)
    re, _ = _get_kp_series(kp_norm, kp_conf, "right_elbow", T)

    elbow_angle_L = _angle_at_joint(ls, le, lw)
    elbow_angle_R = _angle_at_joint(rs, re, rw)
    elbow_angle_L = np.where(pose_present, elbow_angle_L, 0.0).astype(np.float32)
    elbow_angle_R = np.where(pose_present, elbow_angle_R, 0.0).astype(np.float32)

    head, headc = _get_kp_series(kp_norm, kp_conf, "nose", T)
    head_ok = pose_present & (headc >= kp_conf_thresh)
    wrist_head_rel_y_L = (head[:, 1] - lw[:, 1]).astype(np.float32)
    wrist_head_rel_y_R = (head[:, 1] - rw[:, 1]).astype(np.float32)
    wrist_head_rel_y_L = np.where(head_ok & wrist_ok_L, wrist_head_rel_y_L, 0.0).astype(np.float32)
    wrist_head_rel_y_R = np.where(head_ok & wrist_ok_R, wrist_head_rel_y_R, 0.0).astype(np.float32)

    # dribble-ish flags
    ball_y_local_min, ball_y_local_max = _local_extrema_flags(ball_norm[:, 1], ball_valid)
    ball_vy_sign_flip = _sign_flip_flag(ball_vy, ball_valid)

    # build matrix
    cols: List[np.ndarray] = []
    names: List[str] = []

    cols.append(ball_norm); names += ["ball_x", "ball_y"]
    cols.append(ball_speed.reshape(-1, 1)); names += ["ball_speed"]
    cols.append(ball_vy.reshape(-1, 1)); names += ["ball_vy"]
    cols.append(hand_ball_dist.reshape(-1, 1)); names += ["hand_ball_dist"]
    cols.append(ankle_y.reshape(-1, 1)); names += ["ankle_y"]
    cols.append(ball_diam_norm.reshape(-1, 1)); names += ["ball_diam_norm"]

    cols.append(pose_present.astype(np.float32).reshape(-1, 1)); names += ["pose_present"]
    cols.append(ball_present.astype(np.float32).reshape(-1, 1)); names += ["ball_present"]
    cols.append(ball_measured.astype(np.float32).reshape(-1, 1)); names += ["ball_measured"]

    cols.append(ball_vx.reshape(-1, 1)); names += ["ball_vx"]
    cols.append(ball_ax.reshape(-1, 1)); names += ["ball_ax"]
    cols.append(ball_ay.reshape(-1, 1)); names += ["ball_ay"]
    cols.append(ball_dir_cos.reshape(-1, 1)); names += ["ball_dir_cos"]
    cols.append(ball_dir_sin.reshape(-1, 1)); names += ["ball_dir_sin"]

    cols.append(dL.reshape(-1, 1)); names += ["hand_ball_dist_L"]
    cols.append(dR.reshape(-1, 1)); names += ["hand_ball_dist_R"]
    cols.append(dMin.reshape(-1, 1)); names += ["hand_ball_dist_min"]
    cols.append(release_score.reshape(-1, 1)); names += ["release_score"]
    cols.append(hand_left.reshape(-1, 1)); names += ["hand_used_left"]
    cols.append(hand_right.reshape(-1, 1)); names += ["hand_used_right"]
    cols.append(hand_unknown.reshape(-1, 1)); names += ["hand_used_unknown"]

    cols.append(wrist_head_rel_y_L.reshape(-1, 1)); names += ["wrist_head_rel_y_L"]
    cols.append(wrist_head_rel_y_R.reshape(-1, 1)); names += ["wrist_head_rel_y_R"]
    cols.append(elbow_angle_L.reshape(-1, 1)); names += ["elbow_angle_L"]
    cols.append(elbow_angle_R.reshape(-1, 1)); names += ["elbow_angle_R"]

    cols.append(ball_y_local_min.reshape(-1, 1)); names += ["ball_y_local_min"]
    cols.append(ball_y_local_max.reshape(-1, 1)); names += ["ball_y_local_max"]
    cols.append(ball_vy_sign_flip.reshape(-1, 1)); names += ["ball_vy_sign_flip"]

    cols.append(kp_xy)
    for name in kp_names:
        names += [f"{name}_x", f"{name}_y"]

    cols.append(kp_cf)
    for name in kp_names:
        names += [f"{name}_conf"]

    X = np.concatenate(cols, axis=1).astype(np.float32)

    bool_names = [
        "pose_present", "ball_present", "ball_measured",
        "hand_used_left", "hand_used_right", "hand_used_unknown",
        "ball_y_local_min", "ball_y_local_max", "ball_vy_sign_flip",
    ]
    idx_bool = np.array([names.index(n) for n in bool_names], dtype=np.int64)

    conf_idx = np.array([i for i, n in enumerate(names) if n.endswith("_conf")], dtype=np.int64)
    exclude = set(idx_bool.tolist()) | set(conf_idx.tolist())
    idx_cont = np.array([i for i in range(len(names)) if i not in exclude], dtype=np.int64)

    Xr, tr = resample_to_fps(X, t, target_fps=target_fps, bool_idx=idx_bool)
    return Xr, names, idx_cont, idx_bool, tr

def fit_standardizer(X_list: List[np.ndarray], idx_cont: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx_cont = np.asarray(idx_cont, dtype=np.int64)
    all_cont = np.concatenate([X[:, idx_cont] for X in X_list], axis=0).astype(np.float64)
    mean = all_cont.mean(axis=0)
    std = np.maximum(all_cont.std(axis=0), EPS)
    return mean.astype(np.float32), std.astype(np.float32)

def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray, idx_cont: np.ndarray) -> np.ndarray:
    X = X.copy().astype(np.float32)
    idx_cont = np.asarray(idx_cont, dtype=np.int64)
    X[:, idx_cont] = (X[:, idx_cont] - mean) / std
    return X

def save_standardizer(path: str, mean: np.ndarray, std: np.ndarray, feature_names: List[str], idx_cont: np.ndarray) -> None:
    np.savez(
        path,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        feature_names=np.array(feature_names),
        idx_cont=np.asarray(idx_cont, dtype=np.int64),
    )

def load_standardizer(path: str) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    z = np.load(path, allow_pickle=True)
    mean = z["mean"].astype(np.float32)
    std = z["std"].astype(np.float32)
    feature_names = [str(x) for x in z["feature_names"].tolist()]
    idx_cont = z["idx_cont"].astype(np.int64)
    return mean, std, feature_names, idx_cont
