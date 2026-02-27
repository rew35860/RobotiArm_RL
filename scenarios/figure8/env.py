import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from .reference import Figure8Reference
from .mujoco_ee import (
    body_pose_world,
    pose_in_frame,
    rotmat_log,
    pick_arm_joint_actuators,
)


class Figure8TrackEnv(gym.Env):
    """
    Gymnasium env:
      - Robot: Menagerie Panda in MuJoCo
      - EE frame: body 'hand'
      - Control: delta joint targets -> written into data.ctrl (position actuators)
      - Reference: figure-8 position + sinusoidal speed modulation, orientation oscillation at 2x freq

    Observation (float32):
      [q(7), qd(7), dp_T(3), dR_T_rotvec(3), v_d_T(3), w_d_T(3), sinphi, cosphi]
    Action (float32):
      delta_q_cmd(7) in radians per step (scaled)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        xml_path: str = "mujoco_menagerie/franka_emika_panda/scene.xml",
        ee_body: str = "hand",
        control_hz: int = 20,
        episode_seconds: float = 10.0,
        action_scale: float = 0.05,   # rad per step (before clipping)
        seed: int = 0,
        # Reference params (specialized tracker defaults)
        a: float = 0.08,
        b: float = 0.06,
        f_hz: float = 1.5,           # "decently fast" once it trains; reduce if unstable
        alpha: float = 0.3,          # speed modulation depth
        theta_max_deg: float = 15.0,
        terminate_pos_err_m: float = 0.50,
        termination_grace_steps: int = 25,
        reward_pos_w: float = 10.0,
        reward_ori_w: float = 2.0,
        reward_act_w: float = 0.01,
    ):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.ee_body = ee_body

        # Control timing
        self.sim_dt = float(self.model.opt.timestep)
        self.control_hz = int(control_hz)
        self.control_dt = 1.0 / self.control_hz
        self.n_substeps = max(1, int(round(self.control_dt / self.sim_dt)))
        self.dt = self.n_substeps * self.sim_dt  # actual control step duration

        # Episode length in control steps
        self.max_steps = int(round(episode_seconds / self.dt))
        self.step_count = 0

        # Actuator/joint mapping
        (
            self.act_ids,
            self.joint_ids,
            self.qpos_addrs,
            self.dof_addrs,
            self.jnt_ranges,
        ) = pick_arm_joint_actuators(self.model, n_arm=7)

        self.action_scale = float(action_scale)
        self.rng = np.random.default_rng(seed)

        # Fixed reference frame for specialized training:
        # Choose a reachable center near the default pose.
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        p_hand0, R_hand0 = body_pose_world(self.model, self.data, self.ee_body)

        pc = p_hand0 + np.array([0.25, 0.0, -0.15], dtype=np.float64)  # mild offset
        R_plane = np.eye(3, dtype=np.float64)  # figure-8 in world XY plane by default

        # Nominal orientation: use current hand orientation as baseline
        R0 = R_hand0.copy()
        u_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        theta_max = np.deg2rad(theta_max_deg)

        self.a = float(a)
        self.b = float(b)
        self.ref = Figure8Reference(
            a=a,
            b=b,
            pc_world=pc,
            R_plane_world=R_plane,
            f_hz=f_hz,
            alpha=alpha,
            R0_world=R0,
            u_world=u_world,
            theta_max_rad=theta_max,
            seed=seed,
        )

        # Spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        obs_dim = 7 + 7 + 3 + 3 + 3 + 3 + 2 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Command state (joint target per controlled actuator)
        self.q_cmd = np.zeros(7, dtype=np.float64)

        # Reward weights
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
        # Write joint position targets into actuators
        for k, a in enumerate(self.act_ids):
            self.data.ctrl[a] = self.q_cmd[k]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Home Pose
        q_home = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785], dtype=np.float64)

        # Apply the pose to the internal MuJoCo joint positions (qpos)
        for i, addr in enumerate(self.qpos_addrs):
            self.data.qpos[addr] = q_home[i]

        # Keep default posture; you can randomize later (after it works)
        mujoco.mj_forward(self.model, self.data)
        
        # RANDOMIZATION LOGIC:
        # 1. GET THE CURRENT HAND POSITION (The "Home" Pose)
        p_hand0, _ = body_pose_world(self.model, self.data, self.ee_body)

        # 2. Pick new sizes and positions
        # Width (a) and Height (b)
        self.a = self.np_random.uniform(0.10, 0.17) 
        self.b = self.np_random.uniform(0.08, 0.12)
        
        # Center position (pc): Randomized forward (x) and slightly left (y)
        # This avoids the Joint 4 "wall" at -0.07 rad
        pc_x = self.np_random.uniform(0.25, 0.40)
        pc_y = self.np_random.uniform(0.00, 0.10) 
        new_pc = p_hand0 + np.array([pc_x, pc_y, -0.15])

        # 3. Update the reference trajectory with these new values
        # We pass the new a, b, and pc into the reference generator
        self.ref.a = self.a
        self.ref.b = self.b
        self.ref.pc_world = new_pc

        # Initialize q_cmd to current joint positions
        q, _ = self._get_q_qd()
        self.q_cmd[:] = q
        self._set_ctrl_from_qcmd()

        self.ref.reset(t0=0.0, random_phase=True)
        self.step_count = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        # Ensure kinematics updated
        mujoco.mj_forward(self.model, self.data)

        # Current EE pose in world
        p_w, R_w = body_pose_world(self.model, self.data, self.ee_body)

        # Desired reference (world)
        p_d_w, R_d_w, v_d_w, w_d_w, phase = self.ref.eval()

        # Trajectory frame is (pc, R_plane) in world.
        # Here we stored them inside the ref as self.ref.pc and self.ref.Rp.
        p_T = self.ref.pc
        R_T = self.ref.Rp

        # Convert both current and desired into trajectory frame T
        p_T_cur, R_T_cur = pose_in_frame(p_w, R_w, p_T, R_T)
        p_T_des, R_T_des = pose_in_frame(p_d_w, R_d_w, p_T, R_T)

        v_T_des = R_T.T @ v_d_w
        w_T_des = R_T.T @ w_d_w

        dp = p_T_cur - p_T_des
        R_err = R_T_des.T @ R_T_cur  # desired^-1 * current
        dR = rotmat_log(R_err)

        q, qd = self._get_q_qd()
        # Create a small array for the randomized parameters
        task_params = np.array([self.a, self.b], dtype=np.float32)

        obs = np.concatenate([q, qd, dp, dR, v_T_des, w_T_des, phase, task_params], axis=0).astype(np.float32)
        return obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(7)

        # Update joint command (delta targets), clip by joint limits
        self.q_cmd += self.action_scale * np.clip(action, -1.0, 1.0)

        # Apply joint limits using jnt_ranges (min/max for each joint)
        lo = self.jnt_ranges[:, 0]
        hi = self.jnt_ranges[:, 1]
        self.q_cmd = np.clip(self.q_cmd, lo, hi)

        self._set_ctrl_from_qcmd()

        # Advance reference by control dt (not per substep)
        self.ref.step(self.dt)

        # Step simulation
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # Parse error components from obs (by construction)
        # obs = [q(7), qd(7), dp(3), dR(3), v_d(3), w_d(3), phase(2), task_params(2)]
        dp = obs[14:17].astype(np.float64)
        dR = obs[17:20].astype(np.float64)

        pos_err = float(np.linalg.norm(dp))
        ori_err = float(np.linalg.norm(dR))

        reward = - (self.w_p * (pos_err ** 2) + self.w_R * (ori_err ** 2) + self.w_u * float(np.dot(action, action)))

        self.step_count += 1

        terminated = False
        truncated = False

        # Terminate on numerical issues, or on severe divergence after a short grace period.
        # A grace window avoids collapsing episodes too early during initial exploration.
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
