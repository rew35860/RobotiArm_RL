from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from .env import CircleTrackEnv
from ..common import select_model_path, make_timestamped_model_stem


MODEL_PREFIX = "ppo_circle_panda"


def make_env():
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
    return Monitor(env)


def build_model(env, device: str = "cpu"):
    return PPO(
        policy="MlpPolicy",
        env=env,
        device=device,
        verbose=1,
        n_steps=4096,
        batch_size=512,
        gamma=0.995,
        gae_lambda=0.95,
        learning_rate=2e-4,
        clip_range=0.15,
        ent_coef=0.0,
    )


def main(
    total_timesteps: int = 800_000,
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
        reset_num_timesteps = False
        print(f"Resuming from {resume_path}")
    else:
        model = build_model(env, device=device)
        reset_num_timesteps = True

    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=reset_num_timesteps)
    model_path_stem = make_timestamped_model_stem(MODEL_PREFIX)
    model.save(str(model_path_stem))
    print(f"Saved to {model_path_stem}.zip")


if __name__ == "__main__":
    main()
