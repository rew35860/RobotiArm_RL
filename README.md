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

## 🏋️ Training
To train the PPO agent:
```bash
python train_ppo.py
```

## 👁️ Evaluation / Viewer
To run a trained model and visualize the rollout:
```bash
python eval_rollout_viewer.py
```

## ⚠️ Usage Reminder

Always activate the virtual environment before running training or rollout scripts:

```bash
source .venv/bin/activate
```

## 📊 Performance Validation

To evaluate the policy, we conducted inference tests across different spatial scales and temporal frequencies.

> [!IMPORTANT]
> **Visualization Key:**
> * **Red Line:** Target trajectory.
> * **Green Line:** Actual end-effector path.

---

<details>
  <summary>▶ Click to view Inference Testing: Small Out-of-Distribution (1.0 Hz vs 1.3 Hz)</summary>

  <table style="width: 100%; border-collapse: collapse;">
    <tr>
      <th style="text-align: center;">1.0 Hz Frequency</th>
      <th style="text-align: center;">1.3 Hz Frequency</th>
    </tr>
    <tr>
      <td style="width: 50%; padding: 5px;">
        <img src="docs/videos/Figure8_10_06.gif" alt="1.0Hz Small OOD" style="width: 100%;">
      </td>
      <td style="width: 50%; padding: 5px;">
        <img src="docs/videos/Figure8_10_06_13Hz.gif" alt="1.3Hz Small OOD" style="width: 100%;">
      </td>
    </tr>
    <tr>
      <td style="text-align: center; font-size: 0.9em;">
        <b>Mean Position Error:</b> 2.17 cm
      </td>
      <td style="text-align: center; font-size: 0.9em;">
        <b>Mean Position Error:</b> 2.11 cm
      </td>
    </tr>
  </table>

  <p align="center">
    <i><b>Note on Methodology:</b> Results demonstrate <b>zero-shot generalization</b> to a compressed figure-8 scale ($0.10 \times 0.06$ m) entirely unseen during the training phase. To ensure statistical significance, reported metrics are the <b>grand mean</b> derived from 10 randomized episodes. For each 20-second test rollout, a temporal average of the tracking error is computed, followed by an ensemble average across all 10 episodes.</i>
  </p>
</details>

<details>
  <summary>▶ Click to view Inference Testing: Large Out-of-Distribution (1.0 Hz vs 1.3 Hz)</summary>

  <table style="width: 100%; border-collapse: collapse;">
    <tr>
      <th style="text-align: center;">1.0 Hz Frequency</th>
      <th style="text-align: center;">1.3 Hz Frequency</th>
    </tr>
    <tr>
      <td style="width: 50%; padding: 5px;">
        <img src="docs/videos/Figure8_22_15.gif" alt="1.0Hz Large OOD" style="width: 100%;">
      </td>
      <td style="width: 50%; padding: 5px;">
        <img src="docs/videos/Figure8_22_15_13Hz.gif" alt="1.3Hz Large OOD" style="width: 100%;">
      </td>
    </tr>
    <tr>
      <td style="text-align: center; font-size: 0.9em;">
        <b>Mean Position Error:</b> 4.99 cm
      </td>
      <td style="text-align: center; font-size: 0.9em;">
        <b>Mean Position Error:</b> 6.80 cm
      </td>
    </tr>
  </table>

  <p align="center">
    <i><b>Note on Methodology:</b> Results demonstrate <b>zero-shot generalization</b> to an expanded figure-8 scale ($0.22 \times 0.15$ m). Reported metrics are the <b>grand mean</b> of 10 randomized 20-second episodes. 
    <br><br>
    <b>Kinematic Limit Analysis:</b> Our testing identifies a critical physical boundary for the Franka Panda. For any trajectory where the scale parameter $a > 0.20$ m, the robot arm reaches the edge of its operational workspace. At these scales, the joints (specifically Joints 4 and 5) encounter <b>hardware saturation</b>, preventing the end-effector from completing the full path. The increased error and 0% success rate at 1.3 Hz reflect this mechanical constraint rather than policy divergence.</i>
  </p>
</details>

<details>
  <summary>▶ Click to view Inference Testing: In-Distribution Baseline (1.0 Hz vs 1.3 Hz)</summary>

  <table style="width: 100%; border-collapse: collapse;">
    <tr>
      <th style="text-align: center;">1.0 Hz Frequency (Baseline)</th>
      <th style="text-align: center;">1.3 Hz Frequency (Stress Test)</th>
    </tr>
    <tr>
      <td style="width: 50%; padding: 5px;">
        <img src="docs/videos/Figure8_14_10.gif" alt="1.0Hz In-Distribution" style="width: 100%;">
      </td>
      <td style="width: 50%; padding: 5px;">
        <img src="docs/videos/Figure8_14_10_13Hz.gif" alt="1.3Hz In-Distribution" style="width: 100%;">
      </td>
    </tr>
    <tr>
      <td style="text-align: center; font-size: 0.9em;">
        <b>Mean Position Error:</b> 2.62 cm
      </td>
      <td style="text-align: center; font-size: 0.9em;">
        <b>Mean Position Error:</b> 2.86 cm
      </td>
    </tr>
  </table>

  <p align="center">
    <i><b>Note on Methodology:</b> These results represent the <b>In-Distribution</b> performance on the training scale ($0.14 \times 0.10$ m). Reported metrics are the <b>grand mean</b> derived from 10 randomized episodes. For each 20-second test rollout, a temporal average of the tracking error is computed, followed by an ensemble average across all 10 episodes to ensure statistical robustness.</i>
  </p>
</details>

<details>
  <summary>▶ Click to view Inference Testing: Specialized Policy (0.75 Hz Baseline)</summary>

  <table style="width: 100%; border-collapse: collapse;">
    <tr>
      <th style="text-align: center;">In-Distribution Testing (0.75 Hz)</th>
    </tr>
    <tr>
      <td style="text-align: center; padding: 5px;">
        <img src="docs/videos/Figure8_14_10_specalized.gif" alt="1.0Hz Specialized Baseline" style="width: 70%;">
      </td>
    </tr>
    <tr>
      <td style="text-align: center; font-size: 0.9em;">
        <b>Mean Position Error:</b> 1.84 cm
      </td>
    </tr>
  </table>

  <p align="center">
    <i><b>Note on Methodology:</b> These results represent the <b>Specialized Policy</b> performance on the standard training scale ($0.14 \times 0.10$ m) at the target frequency of 0.75 Hz. To ensure statistical significance, reported metrics are the <b>grand mean</b> derived from 50 randomized episodes. For each 20-second test rollout, a temporal average of the tracking error is computed, followed by an ensemble average across all 50 episodes.</i>
  </p>
</details>

