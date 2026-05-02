from __future__ import annotations

import argparse
import csv
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


RUN_METADATA = {
    "clean_rmappo": ("clean_rmappo", "Clean RMAPPO", 0),
    "random_action_poison_rmappo": ("random_action_poison", "Random Action", 1),
    "targeted_action_poison_rmappo": ("targeted_action_poison", "Targeted Action", 2),
    "kl_targeted_action_poison_rmappo": (
        "kl_targeted_action_poison",
        "KL Targeted Action",
        3,
    ),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute mean of an empty value list.")
    return sum(values) / len(values)


def _mean_or_none(values: list[float]) -> float | None:
    return _mean(values) if values else None


def _record_from_summary(
    *,
    summary: dict[str, Any],
    sweep_dir: Path,
    config_path: Path,
    checkpoint_root: Path,
    evaluation_path: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    name, label, _ = RUN_METADATA[config.training.run_name]
    attack_summary = summary.get("attack_summary") or {}
    return {
        "name": name,
        "label": label,
        "seed": config.seed,
        "episodes": summary["episodes"],
        "team_reward_mean": summary["team_reward_mean"],
        "team_reward_std": summary["team_reward_std"],
        "episode_length_mean": summary.get("episode_length_mean"),
        "final_min_distance_sum_mean": summary["final_min_distance_sum_mean"],
        "final_unique_landmarks_mean": summary["final_unique_landmarks_mean"],
        "max_unique_landmarks_mean": summary["max_unique_landmarks_mean"],
        "collision_pair_events_mean": summary.get("collision_pair_events_mean"),
        "collision_step_rate_mean": summary.get("collision_step_rate_mean"),
        "final_collision_pairs_mean": summary.get("final_collision_pairs_mean"),
        "max_collision_pairs_mean": summary.get("max_collision_pairs_mean"),
        "poisoned_action_count": attack_summary.get("poisoned_action_count"),
        "mean_effective_probability": attack_summary.get("mean_effective_probability"),
        "checkpoint_root": _relative(checkpoint_root),
        "evaluation_path": _relative(evaluation_path),
        "requested_poison_probability": (
            config.attack.probability if config.attack.enabled else None
        ),
    }


def _add_deltas(records: list[dict[str, Any]]) -> None:
    clean_by_seed = {
        record["seed"]: record["team_reward_mean"]
        for record in records
        if record["name"] == "clean_rmappo"
    }
    clean_collisions_by_seed = {
        record["seed"]: record.get("collision_pair_events_mean")
        for record in records
        if record["name"] == "clean_rmappo"
    }
    for record in records:
        clean_reward = clean_by_seed.get(record["seed"])
        if clean_reward is None:
            record["reward_delta_vs_clean"] = None
            record["reward_degradation_vs_clean"] = None
        else:
            record["reward_delta_vs_clean"] = record["team_reward_mean"] - clean_reward
            record["reward_degradation_vs_clean"] = clean_reward - record["team_reward_mean"]

        clean_collisions = clean_collisions_by_seed.get(record["seed"])
        collisions = record.get("collision_pair_events_mean")
        if clean_collisions is None or collisions is None:
            record["collision_pair_events_delta_vs_clean"] = None
        else:
            record["collision_pair_events_delta_vs_clean"] = collisions - clean_collisions


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
                "collision_pair_events_delta_mean_across_seeds": _mean_or_none(
                    [
                        item.get("collision_pair_events_delta_vs_clean")
                        for item in group
                        if item.get("collision_pair_events_delta_vs_clean") is not None
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


def refresh_sweep(sweep_dir: Path, *, episodes: int) -> dict[str, Any]:
    sweep_dir = sweep_dir.resolve()
    config_dir = sweep_dir / "generated_configs"
    eval_dir = sweep_dir / "eval"
    checkpoint_dir = sweep_dir / "checkpoints"
    eval_dir.mkdir(parents=True, exist_ok=True)

    records = []
    config_paths = sorted(
        config_dir.glob("*.yaml"),
        key=lambda path: (
            load_config(path).seed,
            RUN_METADATA[load_config(path).training.run_name][2],
        ),
    )
    for config_path in config_paths:
        config = load_config(config_path)
        if config.training.run_name not in RUN_METADATA:
            continue
        checkpoint_root = checkpoint_dir / f"{config.training.run_name}_seed{config.seed}"
        evaluation_path = eval_dir / f"{config.training.run_name}_seed{config.seed}_summary.json"
        previous_summary = (
            _load_json(evaluation_path)
            if evaluation_path.exists()
            else {}
        )
        summary = evaluate_saved_model(
            model_path=checkpoint_root,
            config=config,
            episodes=episodes,
        )
        if "attack_summary" in previous_summary:
            summary["attack_summary"] = previous_summary["attack_summary"]
        evaluation_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        records.append(
            _record_from_summary(
                summary=summary,
                sweep_dir=sweep_dir,
                config_path=config_path,
                checkpoint_root=checkpoint_root,
                evaluation_path=evaluation_path,
            )
        )

    _add_deltas(records)
    aggregates = _aggregate(records)

    existing_summary_path = sweep_dir / "seed_sweep_summary.json"
    existing_summary = (
        _load_json(existing_summary_path)
        if existing_summary_path.exists()
        else {}
    )
    summary_payload = {
        "seeds": sorted({record["seed"] for record in records}),
        "total_timesteps": existing_summary.get("total_timesteps"),
        "evaluation_episodes": episodes,
        "attack_probability": existing_summary.get("attack_probability"),
        "note": existing_summary.get("note"),
        "records": records,
        "aggregates": aggregates,
    }
    existing_summary_path.write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    _write_csv(sweep_dir / "seed_sweep_records.csv", records)
    _write_csv(sweep_dir / "seed_sweep_aggregates.csv", aggregates)
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate saved seed-sweep checkpoints and refresh metric tables."
    )
    parser.add_argument(
        "sweep_dirs",
        type=Path,
        nargs="+",
        help="Seed-sweep result directories to refresh.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=32,
        help="Evaluation episodes per checkpoint.",
    )
    args = parser.parse_args()

    outputs = {}
    for sweep_dir in args.sweep_dirs:
        summary = refresh_sweep(sweep_dir, episodes=args.episodes)
        outputs[str(sweep_dir)] = {
            "num_records": len(summary["records"]),
            "seeds": summary["seeds"],
        }
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
