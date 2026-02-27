#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip
sudo apt-get install -y libglfw3 libgl1-mesa-glx libosmesa6

# GLEW package name differs by Ubuntu release.
if apt-cache show libglew2.2 >/dev/null 2>&1; then
  sudo apt-get install -y libglew2.2
elif apt-cache show libglew2.1 >/dev/null 2>&1; then
  sudo apt-get install -y libglew2.1
elif apt-cache show libglew2.0 >/dev/null 2>&1; then
  sudo apt-get install -y libglew2.0
else
  sudo apt-get install -y libglew-dev
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

pip install mujoco gymnasium "stable-baselines3[extra]"

if [ ! -d "mujoco_menagerie" ]; then
  git clone https://github.com/google-deepmind/mujoco_menagerie.git
else
  echo "mujoco_menagerie already exists, skipping clone"
fi

echo "Setup complete. Activate env with: source .venv/bin/activate"
