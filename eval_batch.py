import argparse
from importlib import import_module


SCENARIO_MODULES = {
    "circle": "scenarios.circle.eval_batch",
    "figure8": "scenarios.figure8.eval_batch",
    # "star5": "scenarios.star5.eval_batch",
    # "triangle": "scenarios.triangle.eval_batch",
}


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
    parser = argparse.ArgumentParser(description="Batch-evaluate PPO on a selected scenario")
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_MODULES.keys()),
        default=None,
        help="Scenario name to evaluate (if omitted, an interactive selector is shown)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional explicit model zip path to evaluate",
    )
    parser.add_argument(
        "--model-rank",
        type=int,
        default=None,
        help="Optional model rank by recency (1=latest, 2=previous, ...). If omitted, prompts exact selection.",
    )
    args = parser.parse_args()

    scenario = args.scenario or select_scenario_interactive()
    module = import_module(SCENARIO_MODULES[scenario])
    module.main(model_path=args.model_path, model_rank=args.model_rank)


if __name__ == "__main__":
    main()
