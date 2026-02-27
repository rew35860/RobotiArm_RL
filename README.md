# Figure-8 Robot Tracking — Setup Guide

This project uses **MuJoCo**, **Gymnasium**, and **Stable-Baselines3** for reinforcement learning–based robotic control.

> ⚠️ This setup was tested on **Ubuntu 22.04 running inside WSL2 (Windows Subsystem for Linux)**.  
> It should also work on native Ubuntu 22.04 / 24.04.

---

## 1. System Dependencies (Ubuntu)

Update package lists and install required system libraries:

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip
sudo apt-get install -y libglfw3 libglew-dev libgl1-mesa-glx libosmesa6
```

## 2. Create Python Virtual Environment

Create and activate a clean Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

## 3. Install Python Dependencies

Install MuJoCo, Gymnasium, and Stable-Baselines3:

```bash
pip install mujoco gymnasium "stable-baselines3[extra]"
```

## 4. Robot Assets (MuJoCo Menagerie)

The Franka Emika Panda model is sourced from the official DeepMind Menagerie:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

## 5. Automated Setup (Optional)

If a setup script is provided, run:

```bash
bash setup.sh
```

# 🏋️ Training
To train the PPO agent:
```bash
python train_ppo.py
```

# 👁️ Evaluation / Viewer
To run a trained model and visualize the rollout:
```bash
python eval_rollout_viewer.py
```

# ⚠️ Usage Reminder

Always activate the virtual environment before running training or rollout scripts:

```bash
source .venv/bin/activate
```
