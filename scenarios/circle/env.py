import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from .reference import CircleReference
from ..figure8.mujoco_ee import (
    body_pose_world,
    pose_in_frame,
    rotmat_log,
    pick_arm_joint_actuators,
)


class CircleTrackEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        xml_path: str = "mujoco_menagerie/franka_emika_panda/scene.xml",
        ee_body: str = "hand",
        control_hz: int = 40,
        episode_seconds: float = 10.0,
        action_scale: float = 0.03,
        seed: int = 0,
        radius: float = 0.20,
        f_hz: float = 0.7,
        alpha: float = 0.08,
        theta_max_deg: float = 6.0,
        terminate_pos_err_m: float = 1.20,
        termination_grace_steps: int = 25,
        reward_pos_w: float = 25.0,
        reward_ori_w: float = 1.0,
        reward_act_w: float = 0.02,
    ):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.ee_body = ee_body

        self.sim_dt = float(self.model.opt.timestep)
        self.control_hz = int(control_hz)
        self.control_dt = 1.0 / self.control_hz
        self.n_substeps = max(1, int(round(self.control_dt / self.sim_dt)))
        self.dt = self.n_substeps * self.sim_dt

        self.max_steps = int(round(episode_seconds / self.dt))
        self.step_count = 0

        (
            self.act_ids,
            self.joint_ids,
            self.qpos_addrs,
            self.dof_addrs,
            self.jnt_ranges,
        ) = pick_arm_joint_actuators(self.model, n_arm=7)

        self.action_scale = float(action_scale)
        self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        p_hand0, R_hand0 = body_pose_world(self.model, self.data, self.ee_body)

        pc = p_hand0 + np.array([0.10, 0.00, -0.10], dtype=np.float64)
        R_plane = np.eye(3, dtype=np.float64)

        R0 = R_hand0.copy()
        u_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        theta_max = np.deg2rad(theta_max_deg)

        self.ref = CircleReference(
            radius=radius,
            pc_world=pc,
            R_plane_world=R_plane,
            f_hz=f_hz,
            alpha=alpha,
            R0_world=R0,
            u_world=u_world,
            theta_max_rad=theta_max,
            seed=seed,
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        obs_dim = 7 + 7 + 3 + 3 + 3 + 3 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.q_cmd = np.zeros(7, dtype=np.float64)

        self.w_p = float(reward_pos_w)
        self.w_R = float(reward_ori_w)
        self.w_u = float(reward_act_w)
        self.terminate_pos_err_m = float(terminate_pos_err_m)
        self.termination_grace_steps = int(termination_grace_steps)

    def _get_q_qd(self):
        q = np.array([self.data.qpos[i] for i in self.qpos_addrs], dtype=np.float64)
        qd = np.array([self.data.qvel[i] for i in self.dof_addrs], dtype=np.float64)
        return q, qd

    def _set_ctrl_from_qcmd(self):
        for k, a in enumerate(self.act_ids):
            self.data.ctrl[a] = self.q_cmd[k]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        q, _ = self._get_q_qd()
        self.q_cmd[:] = q
        self._set_ctrl_from_qcmd()

        self.ref.reset(t0=0.0, random_phase=True)
        self.step_count = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        mujoco.mj_forward(self.model, self.data)

        p_w, R_w = body_pose_world(self.model, self.data, self.ee_body)
        p_d_w, R_d_w, v_d_w, w_d_w, phase = self.ref.eval()

        p_T = self.ref.pc
        R_T = self.ref.Rp

        p_T_cur, R_T_cur = pose_in_frame(p_w, R_w, p_T, R_T)
        p_T_des, R_T_des = pose_in_frame(p_d_w, R_d_w, p_T, R_T)

        v_T_des = R_T.T @ v_d_w
        w_T_des = R_T.T @ w_d_w

        dp = p_T_cur - p_T_des
        R_err = R_T_des.T @ R_T_cur
        dR = rotmat_log(R_err)

        q, qd = self._get_q_qd()

        obs = np.concatenate([q, qd, dp, dR, v_T_des, w_T_des, phase], axis=0).astype(np.float32)
        return obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(7)

        self.q_cmd += self.action_scale * np.clip(action, -1.0, 1.0)

        lo = self.jnt_ranges[:, 0]
        hi = self.jnt_ranges[:, 1]
        self.q_cmd = np.clip(self.q_cmd, lo, hi)

        self._set_ctrl_from_qcmd()
        self.ref.step(self.dt)

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        dp = obs[14:17].astype(np.float64)
        dR = obs[17:20].astype(np.float64)

        pos_err = float(np.linalg.norm(dp))
        ori_err = float(np.linalg.norm(dR))

        reward = - (
            self.w_p * (pos_err ** 2)
            + self.w_R * (ori_err ** 2)
            + self.w_u * float(np.dot(action, action))
        )

        self.step_count += 1

        terminated = False
        truncated = False

        if not np.isfinite(reward):
            terminated = True
        elif self.step_count >= self.termination_grace_steps and pos_err > self.terminate_pos_err_m:
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        info = {
            "pos_err": pos_err,
            "ori_err": ori_err,
        }
        return obs, reward, terminated, truncated, info
