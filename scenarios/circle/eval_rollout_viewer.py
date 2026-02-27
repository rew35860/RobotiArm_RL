import time
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from pathlib import Path
import argparse

from .env import CircleTrackEnv

MODEL_PREFIX = "ppo_circle_panda"


def get_model_path(model_path: str | None = None, model_rank: int | None = None) -> str:
    if model_path is not None:
        return model_path
    model_dir = Path("models")
    candidates = sorted(
        model_dir.glob(f"{MODEL_PREFIX}*.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No model found matching: {model_dir / (MODEL_PREFIX + '*.zip')}")
    if model_rank is None:
        print("Select model:")
        for idx, candidate in enumerate(candidates, start=1):
            print(f"  {idx}) {candidate}")
        while True:
            raw = input("Enter number: ").strip()
            if raw.isdigit():
                pick = int(raw)
                if 1 <= pick <= len(candidates):
                    return str(candidates[pick - 1])
            print("Invalid choice. Enter a valid number from the list.")

    if model_rank < 1:
        raise ValueError("model_rank must be >= 1")
    if model_rank > len(candidates):
        raise FileNotFoundError(
            f"Requested model_rank={model_rank}, but only {len(candidates)} model(s) found for prefix '{MODEL_PREFIX}'"
        )
    return str(candidates[model_rank - 1])


def draw_ee_trail(viewer, points, radius=0.004):
    if not points:
        return

    max_geoms = len(viewer.user_scn.geoms)
    n = min(len(points), max_geoms)
    start = len(points) - n

    viewer.user_scn.ngeom = 0
    mat = np.eye(3, dtype=np.float64).ravel()

    for i in range(n):
        p = points[start + i]
        alpha = 0.15 + 0.80 * ((i + 1) / n)
        rgba = np.array([0.30, 0.75, 1.00, alpha], dtype=np.float32)
        size = np.array([radius, 0.0, 0.0], dtype=np.float64)

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=size,
            pos=p,
            mat=mat,
            rgba=rgba,
        )
        viewer.user_scn.ngeom += 1


def main(model_path: str | None = None, model_rank: int | None = None):
    env = CircleTrackEnv(
        xml_path="mujoco_menagerie/franka_emika_panda/scene.xml",
        ee_body="hand",
        control_hz=40,
        episode_seconds=10.0,
        radius=0.20,
        f_hz=0.7,
        alpha=0.08,
        theta_max_deg=6.0,
        action_scale=0.03,
        terminate_pos_err_m=1.20,
        termination_grace_steps=25,
        reward_pos_w=25.0,
        reward_ori_w=1.0,
        reward_act_w=0.02,
        seed=0,
    )

    model = PPO.load(get_model_path(model_path=model_path, model_rank=model_rank))

    obs, _ = env.reset()
    ee_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, env.ee_body)
    if ee_body_id < 0:
        raise ValueError(f"EE body not found: {env.ee_body}")

    pos_errs = []
    ori_errs = []
    rewards = []
    ee_trail = []
    trail_stride = 2
    trail_max_points = 300

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        step = 0
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            pos_errs.append(info["pos_err"])
            ori_errs.append(info["ori_err"])
            rewards.append(reward)

            if step % 10 == 0:
                print(
                    f"step={step:4d}  pos_err={info['pos_err']:.4f} m  "
                    f"ori_err={info['ori_err']:.4f} rad  reward={reward:.4f}"
                )

            if step % trail_stride == 0:
                ee_trail.append(env.data.xpos[ee_body_id].copy())
                if len(ee_trail) > trail_max_points:
                    ee_trail = ee_trail[-trail_max_points:]

            with viewer.lock():
                draw_ee_trail(viewer, ee_trail)

            viewer.sync()
            time.sleep(env.dt)

            step += 1
            if terminated or truncated:
                break

    pos_errs = np.array(pos_errs)
    ori_errs = np.array(ori_errs)
    rewards = np.array(rewards)

    def rms(x):
        return float(np.sqrt(np.mean(x * x)))

    print("\nEpisode summary (circle):")
    print(f"  steps: {len(pos_errs)}")
    print(f"  pos_err: mean={pos_errs.mean():.4f} m, rms={rms(pos_errs):.4f} m, max={pos_errs.max():.4f} m")
    print(f"  ori_err: mean={ori_errs.mean():.4f} rad, rms={rms(ori_errs):.4f} rad, max={ori_errs.max():.4f} rad")
    print(f"  reward:  mean={rewards.mean():.4f}, sum={rewards.sum():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rollout viewer for circle scenario")
    parser.add_argument("--model-path", default=None, help="Optional explicit model zip path")
    parser.add_argument("--model-rank", type=int, default=None, help="Optional model rank by recency (1=latest)")
    args = parser.parse_args()
    main(model_path=args.model_path, model_rank=args.model_rank)
