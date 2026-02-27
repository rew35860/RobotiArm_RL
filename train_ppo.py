import argparse
from importlib import import_module


SCENARIO_MODULES = {
    "circle": "scenarios.circle.train_ppo",
    "figure8": "scenarios.figure8.train_ppo",
    # "star5": "scenarios.star5.train_ppo",
    # "triangle": "scenarios.triangle.train_ppo",
}


def select_mode_interactive() -> str:
    print("Select training mode:")
    print("  1) new    (start from scratch)")
    print("  2) resume (continue from existing model)")

    while True:
        raw = input("Enter number (default: 1): ").strip()
        if raw in ("", "1", "new"):
            return "new"
        if raw in ("2", "resume"):
            return "resume"
        print("Invalid choice. Enter 1/new or 2/resume.")


def select_scenario_interactive() -> str:
    names = sorted(SCENARIO_MODULES.keys())
    print("Select scenario:")
    for idx, name in enumerate(names, start=1):
        print(f"  {idx}) {name}")

    default_name = "figure8" if "figure8" in names else names[0]
    prompt = f"Enter number (default: {default_name}): "

    while True:
        raw = input(prompt).strip()
        if raw == "":
            return default_name
        if raw.isdigit():
            pick = int(raw)
            if 1 <= pick <= len(names):
                return names[pick - 1]
        if raw in SCENARIO_MODULES:
            return raw
        print("Invalid choice. Enter a number from the list or a scenario name.")


def main():
    parser = argparse.ArgumentParser(description="Train PPO on a selected scenario")
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_MODULES.keys()),
        default=None,
        help="Scenario name to train (if omitted, an interactive selector is shown)",
    )
    parser.add_argument(
        "--mode",
        choices=["new", "resume"],
        default=None,
        help="Training mode (if omitted, interactive selector is shown)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional explicit model path for resume mode",
    )
    parser.add_argument(
        "--model-rank",
        type=int,
        default=None,
        help="Optional model rank by recency for resume mode (1=latest)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Optional override for total training timesteps",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for SB3 (default: cpu)",
    )
    args = parser.parse_args()

    scenario = args.scenario or select_scenario_interactive()
    mode = args.mode or select_mode_interactive()

    module = import_module(SCENARIO_MODULES[scenario])

    kwargs = {
        "device": args.device,
        "mode": mode,
        "model_path": args.model_path,
        "model_rank": args.model_rank,
        "interactive_model_select": mode == "resume" and args.model_path is None and args.model_rank is None,
    }
    if args.timesteps is not None:
        kwargs["total_timesteps"] = args.timesteps

    module.main(**kwargs)


if __name__ == "__main__":
    main()
