import numpy as np
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


def evaluate(n_episodes=20, model_path: str | None = None, model_rank: int | None = None):
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

    ep_stats = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        pos_errs = []
        ori_errs = []
        rewards = []
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            pos_errs.append(info["pos_err"])
            ori_errs.append(info["ori_err"])
            rewards.append(reward)

        pos_errs = np.array(pos_errs)
        ori_errs = np.array(ori_errs)
        rewards = np.array(rewards)

        ep_stats.append({
            "len": len(pos_errs),
            "pos_rms": float(np.sqrt(np.mean(pos_errs**2))),
            "pos_max": float(np.max(pos_errs)),
            "ori_rms": float(np.sqrt(np.mean(ori_errs**2))),
            "ori_max": float(np.max(ori_errs)),
            "rew_sum": float(np.sum(rewards)),
            "terminated": bool(terminated),
        })

    return ep_stats


def main(model_path: str | None = None, model_rank: int | None = None):
    stats = evaluate(30, model_path=model_path, model_rank=model_rank)

    lens = np.array([s["len"] for s in stats])
    pos_rms = np.array([s["pos_rms"] for s in stats])
    pos_max = np.array([s["pos_max"] for s in stats])
    ori_rms = np.array([s["ori_rms"] for s in stats])
    term_rate = np.mean([s["terminated"] for s in stats])

    def pct(x, p):
        return float(np.percentile(x, p))

    print("Batch evaluation (circle):")
    print(f"  episodes: {len(stats)}")
    print(f"  mean ep_len: {lens.mean():.1f}  (min={lens.min()}, max={lens.max()})")
    print(f"  termination rate: {term_rate*100:.1f}%")
    print(f"  pos_rms: mean={pos_rms.mean():.4f} m  p50={pct(pos_rms,50):.4f}  p95={pct(pos_rms,95):.4f}")
    print(f"  pos_max: mean={pos_max.mean():.4f} m  p95={pct(pos_max,95):.4f}")
    print(f"  ori_rms: mean={ori_rms.mean():.4f} rad  p95={pct(ori_rms,95):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch eval for circle scenario")
    parser.add_argument("--model-path", default=None, help="Optional explicit model zip path")
    parser.add_argument("--model-rank", type=int, default=None, help="Optional model rank by recency (1=latest)")
    args = parser.parse_args()
    main(model_path=args.model_path, model_rank=args.model_rank)
