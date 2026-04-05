from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from behavior_poisoning.config import load_config
from behavior_poisoning.evaluate import evaluate_saved_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved clean cooperative baseline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "clean_mappo.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to a saved PPO zip file or MAPPO checkpoint directory.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Optional override for the number of evaluation episodes.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy evaluation instead of deterministic actions.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open a window and render the agent trajectories during evaluation.",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.08,
        help="Seconds to wait between rendered steps. Ignored unless --render is set.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    summary = evaluate_saved_model(
        model_path=args.model_path,
        config=config,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        render_mode="human" if args.render else None,
        step_delay=args.step_delay if args.render else 0.0,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
