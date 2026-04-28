from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from behavior_poisoning.config import load_config
from behavior_poisoning.evaluate import evaluate_saved_model


BASELINES = {
    "shared_rmappo": {
        "config": REPO_ROOT / "configs" / "clean_mappo.yaml",
        "model_path": REPO_ROOT / "results" / "checkpoints" / "clean_rmappo_seed7",
    },
    "separated_rmappo": {
        "config": REPO_ROOT / "configs" / "clean_mappo_separated.yaml",
        "model_path": REPO_ROOT / "results" / "checkpoints" / "clean_rmappo_separated_seed7",
    },
    "ppo_fallback": {
        "config": REPO_ROOT / "configs" / "clean_ppo.yaml",
        "model_path": REPO_ROOT / "results" / "checkpoints" / "clean_ppo_seed7.zip",
    },
}


def _smoke_check_one(
    name: str,
    *,
    config_path: Path,
    model_path: Path,
    episodes: int,
) -> dict[str, Any]:
    if not config_path.exists():
        return {
            "baseline": name,
            "status": "missing_config",
            "config_path": str(config_path),
        }
    if not model_path.exists():
        return {
            "baseline": name,
            "status": "missing_model",
            "config_path": str(config_path),
            "model_path": str(model_path),
        }

    config = load_config(config_path)
    summary = evaluate_saved_model(
        model_path=model_path,
        config=config,
        episodes=episodes,
    )
    return {
        "baseline": name,
        "status": "ok",
        "config_path": str(config_path),
        "model_path": str(model_path),
        "algorithm": config.training.algorithm,
        "team_reward_mean": summary["team_reward_mean"],
        "episode_length_mean": summary["episode_length_mean"],
        "episodes": summary["episodes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run short evaluation smoke checks against saved local baseline checkpoints."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="Number of evaluation episodes to run for each available baseline.",
    )
    parser.add_argument(
        "--baseline",
        choices=sorted(BASELINES),
        nargs="*",
        default=list(BASELINES),
        help="Optional subset of baselines to smoke check.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 if any requested baseline is missing or fails.",
    )
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    failures = 0
    for name in args.baseline:
        spec = BASELINES[name]
        try:
            result = _smoke_check_one(
                name,
                config_path=spec["config"],
                model_path=spec["model_path"],
                episodes=args.episodes,
            )
        except Exception as exc:
            failures += 1
            result = {
                "baseline": name,
                "status": "error",
                "config_path": str(spec["config"]),
                "model_path": str(spec["model_path"]),
                "error": f"{type(exc).__name__}: {exc}",
            }
        else:
            if result["status"] != "ok":
                failures += 1
        results.append(result)

    print(json.dumps({"results": results}, indent=2))
    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
