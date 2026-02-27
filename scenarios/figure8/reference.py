import numpy as np


def skew(u: np.ndarray) -> np.ndarray:
    ux, uy, uz = u
    return np.array([[0.0, -uz,  uy],
                     [uz,  0.0, -ux],
                     [-uy, ux,  0.0]], dtype=np.float64)


def rot_axis_angle(u: np.ndarray, theta: float) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    u = u / (np.linalg.norm(u) + 1e-12)
    K = skew(u)
    I = np.eye(3)
    return I + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


class Figure8Reference:
    """
    Generates:
      p_d(t), R_d(t), v_d(t), w_d(t)
    where:
      - p_d traces a figure-8 with phase phi(t)
      - phi_dot(t) has sinusoidal modulation (velocity profile)
      - orientation oscillates at 2x positional frequency
    """

    def __init__(
        self,
        a: float,
        b: float,
        pc_world: np.ndarray,
        R_plane_world: np.ndarray,
        f_hz: float,
        alpha: float,
        R0_world: np.ndarray,
        u_world: np.ndarray,
        theta_max_rad: float,
        seed: int = 0,
    ):
        self.a = float(a)
        self.b = float(b)
        self.pc = np.asarray(pc_world, dtype=np.float64).reshape(3)
        self.Rp = np.asarray(R_plane_world, dtype=np.float64).reshape(3, 3)
        self.f = float(f_hz)
        self.alpha = float(alpha)
        self.R0 = np.asarray(R0_world, dtype=np.float64).reshape(3, 3)
        self.u = np.asarray(u_world, dtype=np.float64).reshape(3)
        self.u = self.u / (np.linalg.norm(self.u) + 1e-12)
        self.theta_max = float(theta_max_rad)

        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.phi = 0.0

    def reset(self, t0: float = 0.0, random_phase: bool = True):
        self.t = float(t0)
        self.phi = float(self.rng.uniform(0.0, 2.0 * np.pi)) if random_phase else 0.0

    def _phi_dot(self, t: float) -> float:
        # sinusoidal speed modulation; keep alpha < 1 to avoid negative speed
        return 2.0 * np.pi * self.f * (1.0 + self.alpha * np.sin(2.0 * np.pi * self.f * t))

    def step(self, dt: float):
        # integrate phase forward
        phi_dot = self._phi_dot(self.t)
        self.phi += phi_dot * dt
        self.t += dt
        return self.eval()

    def eval(self):
        t = self.t
        phi = self.phi
        phi_dot = self._phi_dot(t)

        # Local figure-8 (plane coordinates)
        x = self.a * np.sin(phi)
        y = self.b * np.sin(2.0 * phi)
        p_local = np.array([y, x, 0.0], dtype=np.float64)

        # Desired world position
        p_d = self.pc + self.Rp @ p_local

        # Desired world linear velocity (analytic)
        dx_dphi = self.a * np.cos(phi)
        dy_dphi = 2.0 * self.b * np.cos(2.0 * phi)
        v_local = np.array([dy_dphi, dx_dphi, 0.0], dtype=np.float64) * phi_dot
        v_d = self.Rp @ v_local

        # Orientation oscillation at 2x frequency (positional trajectory is ~f)
        theta = self.theta_max * np.sin(4.0 * np.pi * self.f * t)
        R_osc = rot_axis_angle(self.u, theta)
        R_d = self.R0 @ R_osc

        # Angular velocity in world (axis fixed in world)
        theta_dot = self.theta_max * (4.0 * np.pi * self.f) * np.cos(4.0 * np.pi * self.f * t)
        w_d = self.u * theta_dot

        # Phase encoding (useful for RL)
        phase = np.array([np.sin(phi), np.cos(phi)], dtype=np.float64)

        return p_d, R_d, v_d, w_d, phase

    def get_ref(self, t: float):
        """Returns the desired (p_d, R_d) for a given time t without stepping."""
        # We need the phase phi at time t. 
        # Since phi changes based on the speed modulation alpha, 
        # we can estimate it or just use the current internal phi if you want a simple version.
        
        # Best way: Call eval() logic without updating self.t or self.phi
        phi = self.phi # This uses the current "live" phase of the simulation
        
        # Local figure-8 (plane coordinates)
        x = self.a * np.sin(phi)
        y = self.b * np.sin(2.0 * phi)
        p_local = np.array([y, x, 0.0], dtype=np.float64)
        
        # Desired world position
        p_d = self.pc + self.Rp @ p_local
        
        # Orientation oscillation
        theta = self.theta_max * np.sin(4.0 * np.pi * self.f * t)
        R_osc = rot_axis_angle(self.u, theta)
        R_d = self.R0 @ R_osc
        
        return p_d, R_d