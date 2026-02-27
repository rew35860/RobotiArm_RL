import time
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from pathlib import Path
import argparse

from .env import Figure8TrackEnv

MODEL_PREFIX = "ppo_figure8_panda_large_fit_v2"


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


def draw_ee_trail(viewer, ee_points, goal_points, radius=0.004):
    viewer.user_scn.ngeom = 0
    mat = np.eye(3, dtype=np.float64).ravel()
    max_geoms = len(viewer.user_scn.geoms)

    # 1. DRAW THE GOAL TRAIL (RED)
    n_goal = min(len(goal_points), max_geoms // 2) # Use half the buffer for goals
    for i in range(n_goal):
        p = goal_points[i]
        # Red with slight transparency
        rgba = np.array([1.0, 0.0, 0.0, 0.3], dtype=np.float32) 
        
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, 0.0, 0.0], dtype=np.float64),
            pos=p,
            mat=mat,
            rgba=rgba,
        )
        viewer.user_scn.ngeom += 1

    # 2. DRAW THE ROBOT TRAIL (GREEN)
    n_ee = min(len(ee_points), max_geoms - viewer.user_scn.ngeom)
    for i in range(n_ee):
        p = ee_points[i]
        # Bright Green
        rgba = np.array([0.10, 0.95, 0.35, 0.8], dtype=np.float32)
        
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, 0.0, 0.0], dtype=np.float64),
            pos=p,
            mat=mat,
            rgba=rgba,
        )
        viewer.user_scn.ngeom += 1

def main(model_path: str | None = None, model_rank: int | None = None):
    # Use the same env params you trained with
    env = Figure8TrackEnv(
        xml_path="mujoco_menagerie/franka_emika_panda/scene.xml",
        ee_body="hand",
        control_hz=40,
        episode_seconds=20.0,
        f_hz=1.0,
        a=0.14,
        b=0.10,
        alpha=0.08,
        theta_max_deg=2.0,
        action_scale=0.08,
        terminate_pos_err_m=1.20,
        termination_grace_steps=25,
        reward_pos_w=1000.0,
        reward_ori_w=0.5,
        reward_act_w=0.01,
        seed=24,
    )

    num_episodes = 50  # Change as needed; set to 1 for a single episode, or more for batch evaluation
    successful_episodes = 0
    all_means = []
    model = PPO.load(get_model_path(model_path=model_path, model_rank=model_rank))

    for i in range(num_episodes):
        obs, info = env.reset()
        ee_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, env.ee_body)
        if ee_body_id < 0:
            raise ValueError(f"EE body not found: {env.ee_body}")

        pos_errs = []
        ori_errs = []
        rewards = []
        ee_trail = []
        goal_trail = []
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
                    target_pos, _ = env.ref.get_ref(env.data.time)
                    goal_trail.append(target_pos.copy())

                    if len(ee_trail) > trail_max_points:
                        ee_trail = ee_trail[-trail_max_points:]
                        goal_trail = goal_trail[-trail_max_points:]

                with viewer.lock():
                    
                    draw_ee_trail(viewer, ee_trail, goal_trail)

                viewer.sync()
                # match sim pace (optional; remove if you want it as fast as possible)
                time.sleep(env.dt)

                step += 1
                if terminated or truncated:
                    break

        pos_errs = np.array(pos_errs)
        ori_errs = np.array(ori_errs)
        rewards = np.array(rewards)

        def rms(x): return float(np.sqrt(np.mean(x * x)))

        print("\nEpisode summary:")
        print(f"  steps: {len(pos_errs)}")
        print(f"  pos_err: mean={pos_errs.mean():.4f} m, rms={rms(pos_errs):.4f} m, max={pos_errs.max():.4f} m")
        print(f"  ori_err: mean={ori_errs.mean():.4f} rad, rms={rms(ori_errs):.4f} rad, max={ori_errs.max():.4f} rad")
        print(f"  reward:  mean={rewards.mean():.4f}, sum={rewards.sum():.4f}")

        all_means.append((pos_errs.mean(), ori_errs.mean(), rewards.mean()))
        if pos_errs.mean() < 0.05:
            successful_episodes += 1
    
    results = np.array(all_means)
    overall_means = np.mean(results, axis=0)

    print("\nOverall summary:")
    print(f"  successful episodes (pos_err mean < 0.05 m): {successful_episodes}/{num_episodes} ({successful_episodes/num_episodes*100:.1f}%)")
    print(f"  average pos_err mean: {overall_means[0]:.4f} m")
    print(f"  average ori_err mean: {overall_means[1]:.4f} rad")
    print(f"  average reward mean: {overall_means[2]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rollout viewer for figure8 scenario")
    parser.add_argument("--model-path", default=None, help="Optional explicit model zip path")
    parser.add_argument("--model-rank", type=int, default=None, help="Optional model rank by recency (1=latest)")
    args = parser.parse_args()
    main(model_path=args.model_path, model_rank=args.model_rank)
