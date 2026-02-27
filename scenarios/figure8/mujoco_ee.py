import numpy as np
import mujoco
from mujoco import mjtObj


def rotmat_log(R: np.ndarray) -> np.ndarray:
    """
    Map SO(3) rotation matrix to a rotation vector (axis * angle).
    Robust enough for tracking rewards.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = np.trace(R)
    cos_angle = (tr - 1.0) * 0.5
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if angle < 1e-6:
        # small-angle approximation: log(R) ~ vee(R - R^T)/2
        v = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ], dtype=np.float64) * 0.5
        return v

    denom = 2.0 * np.sin(angle)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float64) / (denom + 1e-12)

    return axis * angle


def body_pose_world(model: mujoco.MjModel, data: mujoco.MjData, body_name: str):
    bid = mujoco.mj_name2id(model, mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"Body not found: {body_name}")
    p = data.xpos[bid].copy()
    R = data.xmat[bid].reshape(3, 3).copy()
    return p, R


def pose_in_frame(p_w: np.ndarray, R_w: np.ndarray, p_F: np.ndarray, R_F: np.ndarray):
    """
    Convert world pose (p_w, R_w) into frame F defined by (p_F, R_F) in world:
      p_F^T (p_w - p_F),  R_F^T R_w
    """
    p_rel = R_F.T @ (p_w - p_F)
    R_rel = R_F.T @ R_w
    return p_rel, R_rel


def pick_arm_joint_actuators(model: mujoco.MjModel, n_arm: int = 7):
    """
    Menagerie Panda scene often has joint position actuators.
    We pick actuators that attach to joints whose names look like arm joints
    (excluding fingers), and return:
      act_ids, joint_ids, qpos_addrs, dof_addrs, jnt_ranges
    """
    act_ids = []
    joint_ids = []

    for a in range(model.nu):
        jid = int(model.actuator_trnid[a, 0])
        if jid < 0:
            continue
        jname = mujoco.mj_id2name(model, mjtObj.mjOBJ_JOINT, jid) or ""
        lname = jname.lower()
        if "finger" in lname or "gripper" in lname:
            continue
        # Heuristic: keep "joint" names first, then anything else hinge/slide
        if "joint" in lname or "panda_joint" in lname or lname.startswith("joint"):
            act_ids.append(a)
            joint_ids.append(jid)

    if len(act_ids) < n_arm:
        # Fallback: take first non-finger joint actuators
        act_ids = []
        joint_ids = []
        for a in range(model.nu):
            jid = int(model.actuator_trnid[a, 0])
            if jid < 0:
                continue
            jname = mujoco.mj_id2name(model, mjtObj.mjOBJ_JOINT, jid) or ""
            lname = jname.lower()
            if "finger" in lname or "gripper" in lname:
                continue
            act_ids.append(a)
            joint_ids.append(jid)
            if len(act_ids) >= n_arm:
                break

    if len(act_ids) < n_arm:
        raise RuntimeError(f"Could not find {n_arm} arm joint actuators. Found {len(act_ids)}.")

    act_ids = act_ids[:n_arm]
    joint_ids = joint_ids[:n_arm]

    qpos_addrs = [int(model.jnt_qposadr[j]) for j in joint_ids]
    dof_addrs = [int(model.jnt_dofadr[j]) for j in joint_ids]
    jnt_ranges = np.array([model.jnt_range[j].copy() for j in joint_ids], dtype=np.float64)

    return np.array(act_ids, dtype=int), np.array(joint_ids, dtype=int), np.array(qpos_addrs, dtype=int), np.array(dof_addrs, dtype=int), jnt_ranges
