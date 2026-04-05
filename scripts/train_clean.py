from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from behavior_poisoning.train_clean import train_clean_baseline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a clean cooperative baseline on Simple Spread."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "clean_mappo.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Optional override for total training timesteps.",
    )
    args = parser.parse_args()

    result = train_clean_baseline(
        config_path=args.config,
        total_timesteps=args.total_timesteps,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
