from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from .env import Figure8TrackEnv
from ..common import select_model_path, make_timestamped_model_stem
import time

MODEL_PREFIX = "ppo_figure8_panda_large_fit_v2"


def make_env():
    env = Figure8TrackEnv(
        xml_path="mujoco_menagerie/franka_emika_panda/scene.xml",
        ee_body="hand",
        control_hz=40,
        episode_seconds=20.0,
        # If training is unstable, reduce f_hz first (e.g., 1.0), then ramp later.
        f_hz=1.0,
        a=0.14,
        b=0.10,
        alpha=0.08,
        theta_max_deg=2.0,
        action_scale=0.05,
        terminate_pos_err_m=1.20,
        termination_grace_steps=25,
        reward_pos_w=1000.0,
        reward_ori_w=0.5,
        reward_act_w=0.02,
        seed=0,
    )
    return Monitor(env)


def build_model(env, device: str = "cpu"):
    return PPO(
        policy="MlpPolicy",
        env=env,
        device=device,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        gamma=0.98,
        gae_lambda=0.95,
        learning_rate=2e-4,
        clip_range=0.15,
        ent_coef=0.005,
    )


def main(
    total_timesteps: int = 500_000,
    device: str = "cpu",
    mode: str = "new",
    model_path: str | None = None,
    model_rank: int | None = None,
    interactive_model_select: bool = False,
):
    env = DummyVecEnv([make_env])

    if mode == "resume":
        resume_path = select_model_path(
            prefix=MODEL_PREFIX,
            model_path=model_path,
            model_rank=model_rank,
            interactive=interactive_model_select and model_path is None and model_rank is None,
        )
        model = PPO.load(resume_path, env=env, device=device)
        model.learning_rate = 5e-4  # Reset learning rate to avoid decay issues

        reset_num_timesteps = False
        print(f"Resuming from {resume_path}")
    else:
        model = build_model(env, device=device)
        reset_num_timesteps = True

    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=reset_num_timesteps)
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time)/60:.1f} minutes")

    model_path_stem = make_timestamped_model_stem(MODEL_PREFIX)
    model.save(str(model_path_stem))
    print(f"Saved to {model_path_stem}.zip")


if __name__ == "__main__":
    main()
