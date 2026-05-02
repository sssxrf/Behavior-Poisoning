from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from behavior_poisoning.analysis import DEFAULT_EXPERIMENTS
from behavior_poisoning.config import load_config
from behavior_poisoning.train_clean import train_clean_baseline


def _write_config_variant(
    *,
    source_config: Path,
    seed: int,
    output_dir: Path,
    total_timesteps: int | None,
    evaluation_episodes: int,
    attack_probability: float | None,
    config_dir: Path,
) -> Path:
    config = load_config(source_config)
    config.seed = seed
    config.output_dir = str(output_dir)
    config.training.evaluation_episodes = evaluation_episodes
    if total_timesteps is not None:
        config.training.total_timesteps = total_timesteps
    if attack_probability is not None and config.attack.enabled:
        config.attack.probability = attack_probability

    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{source_config.stem}_seed{seed}.yaml"
    path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
    return path


def _record_from_training(name: str, label: str, seed: int, result: dict[str, Any]) -> dict[str, Any]:
    summary = result["evaluation_summary"]
    attack_summary = result.get("attack_summary") or summary.get("attack_summary") or {}
    return {
        "name": name,
        "label": label,
        "seed": seed,
        "episodes": summary["episodes"],
        "team_reward_mean": summary["team_reward_mean"],
        "team_reward_std": summary["team_reward_std"],
        "final_min_distance_sum_mean": summary["final_min_distance_sum_mean"],
        "final_unique_landmarks_mean": summary["final_unique_landmarks_mean"],
        "max_unique_landmarks_mean": summary["max_unique_landmarks_mean"],
        "collision_pair_events_mean": summary.get("collision_pair_events_mean"),
        "collision_step_rate_mean": summary.get("collision_step_rate_mean"),
        "final_collision_pairs_mean": summary.get("final_collision_pairs_mean"),
        "max_collision_pairs_mean": summary.get("max_collision_pairs_mean"),
        "poisoned_action_count": attack_summary.get("poisoned_action_count"),
        "mean_effective_probability": attack_summary.get("mean_effective_probability"),
        "checkpoint_root": result["checkpoint_root"],
        "evaluation_path": result["evaluation_path"],
    }


def _add_deltas(records: list[dict[str, Any]]) -> None:
    clean_by_seed = {
        record["seed"]: record["team_reward_mean"]
        for record in records
        if record["name"] == "clean_rmappo"
    }
    for record in records:
        clean_reward = clean_by_seed.get(record["seed"])
        if clean_reward is None:
            record["reward_delta_vs_clean"] = None
            record["reward_degradation_vs_clean"] = None
            continue
        record["reward_delta_vs_clean"] = record["team_reward_mean"] - clean_reward
        record["reward_degradation_vs_clean"] = clean_reward - record["team_reward_mean"]


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute mean of an empty value list.")
    return sum(values) / len(values)


def _mean_or_none(values: list[float]) -> float | None:
    return _mean(values) if values else None


def _aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["name"], []).append(record)

    aggregates = []
    for name, group in grouped.items():
        aggregates.append(
            {
                "name": name,
                "label": group[0]["label"],
                "num_seeds": len(group),
                "team_reward_mean_across_seeds": _mean(
                    [item["team_reward_mean"] for item in group]
                ),
                "reward_degradation_mean_across_seeds": _mean_or_none(
                    [
                        item["reward_degradation_vs_clean"]
                        for item in group
                        if item["reward_degradation_vs_clean"] is not None
                    ]
                ),
                "final_min_distance_sum_mean_across_seeds": _mean(
                    [item["final_min_distance_sum_mean"] for item in group]
                ),
                "final_unique_landmarks_mean_across_seeds": _mean(
                    [item["final_unique_landmarks_mean"] for item in group]
                ),
                "collision_pair_events_mean_across_seeds": _mean_or_none(
                    [
                        item.get("collision_pair_events_mean")
                        for item in group
                        if item.get("collision_pair_events_mean") is not None
                    ]
                ),
                "collision_step_rate_mean_across_seeds": _mean_or_none(
                    [
                        item.get("collision_step_rate_mean")
                        for item in group
                        if item.get("collision_step_rate_mean") is not None
                    ]
                ),
            }
        )
    return aggregates


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train clean and poisoned RMAPPO checkpoints across seeds."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Training seeds to run.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Optional total-timesteps override for every run.",
    )
    parser.add_argument(
        "--evaluation-episodes",
        type=int,
        default=32,
        help="Evaluation episodes after each training run.",
    )
    parser.add_argument(
        "--attack-probability",
        type=float,
        default=None,
        help="Optional poison probability override for attack configs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "seed_sweep_p01_results",
        help="Training output root for seed-sweep checkpoints and summaries.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = args.output_dir / "generated_configs"
    partial_path = args.output_dir / "seed_sweep_partial.json"
    if partial_path.exists():
        records = json.loads(partial_path.read_text(encoding="utf-8")).get("records", [])
    else:
        records = []

    for seed in args.seeds:
        for spec in DEFAULT_EXPERIMENTS:
            if any(
                record["seed"] == seed and record["name"] == spec.name
                for record in records
            ):
                continue
            config_path = _write_config_variant(
                source_config=spec.config_path,
                seed=seed,
                output_dir=args.output_dir,
                total_timesteps=args.total_timesteps,
                evaluation_episodes=args.evaluation_episodes,
                attack_probability=args.attack_probability,
                config_dir=config_dir,
            )
            result = train_clean_baseline(config_path)
            records.append(_record_from_training(spec.name, spec.label, seed, result))
            _add_deltas(records)

            partial_path.write_text(
                json.dumps({"records": records}, indent=2),
                encoding="utf-8",
            )

    _add_deltas(records)
    aggregates = _aggregate(records)
    summary = {
        "seeds": args.seeds,
        "total_timesteps": args.total_timesteps,
        "evaluation_episodes": args.evaluation_episodes,
        "attack_probability": args.attack_probability,
        "records": records,
        "aggregates": aggregates,
    }

    summary_json = args.output_dir / "seed_sweep_summary.json"
    records_csv = args.output_dir / "seed_sweep_records.csv"
    aggregates_csv = args.output_dir / "seed_sweep_aggregates.csv"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(records_csv, records)
    _write_csv(aggregates_csv, aggregates)

    print(
        json.dumps(
            {
                "summary_json": str(summary_json),
                "records_csv": str(records_csv),
                "aggregates_csv": str(aggregates_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
