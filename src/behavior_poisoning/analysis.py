from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any

from behavior_poisoning.config import ExperimentConfig, load_config
from behavior_poisoning.evaluate import evaluate_saved_model


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "probability_comparison_results"


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    label: str
    config_path: Path
    model_path: Path
    summary_path: Path
    order: int


DEFAULT_EXPERIMENTS = [
    ExperimentSpec(
        name="clean_rmappo",
        label="Clean RMAPPO",
        config_path=REPO_ROOT / "configs" / "clean_mappo.yaml",
        model_path=REPO_ROOT / "seed_sweep_p01_results" / "checkpoints" / "clean_rmappo_seed7",
        summary_path=REPO_ROOT
        / "seed_sweep_p01_results"
        / "eval"
        / "clean_rmappo_seed7_summary.json",
        order=0,
    ),
    ExperimentSpec(
        name="random_action_poison",
        label="Random Action",
        config_path=REPO_ROOT / "configs" / "random_action_poisoning_mappo.yaml",
        model_path=REPO_ROOT
        / "seed_sweep_p01_results"
        / "checkpoints"
        / "random_action_poison_rmappo_seed7",
        summary_path=REPO_ROOT
        / "seed_sweep_p01_results"
        / "eval"
        / "random_action_poison_rmappo_seed7_summary.json",
        order=1,
    ),
    ExperimentSpec(
        name="targeted_action_poison",
        label="Targeted Action",
        config_path=REPO_ROOT / "configs" / "targeted_action_poisoning_mappo.yaml",
        model_path=REPO_ROOT
        / "seed_sweep_p01_results"
        / "checkpoints"
        / "targeted_action_poison_rmappo_seed7",
        summary_path=REPO_ROOT
        / "seed_sweep_p01_results"
        / "eval"
        / "targeted_action_poison_rmappo_seed7_summary.json",
        order=2,
    ),
    ExperimentSpec(
        name="kl_targeted_action_poison",
        label="KL Targeted Action",
        config_path=REPO_ROOT
        / "configs"
        / "kl_constrained_targeted_action_mappo.yaml",
        model_path=REPO_ROOT
        / "seed_sweep_p01_results"
        / "checkpoints"
        / "kl_targeted_action_poison_rmappo_seed7",
        summary_path=REPO_ROOT
        / "seed_sweep_p01_results"
        / "eval"
        / "kl_targeted_action_poison_rmappo_seed7_summary.json",
        order=3,
    ),
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _disable_attack(config: ExperimentConfig) -> ExperimentConfig:
    config.attack.enabled = False
    return config


def _metric(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    return None if value is None else float(value)


def _attack_summary_value(summary: dict[str, Any], key: str) -> float | int | None:
    attack_summary = summary.get("attack_summary") or {}
    value = attack_summary.get(key)
    return None if value is None else value


def build_comparison_records(
    summaries: dict[str, dict[str, Any]],
    specs: list[ExperimentSpec] | None = None,
    *,
    clean_name: str = "clean_rmappo",
) -> list[dict[str, Any]]:
    specs = specs or DEFAULT_EXPERIMENTS
    clean_reward = _metric(summaries[clean_name], "team_reward_mean")
    records = []

    for spec in sorted(specs, key=lambda item: item.order):
        summary = summaries[spec.name]
        config = load_config(spec.config_path)
        reward_mean = _metric(summary, "team_reward_mean")
        final_distance = _metric(summary, "final_min_distance_sum_mean")
        record = {
            "name": spec.name,
            "label": spec.label,
            "algorithm": config.training.algorithm,
            "attack_enabled": config.attack.enabled,
            "attack_mode": config.attack.mode,
            "attack_probability": config.attack.probability,
            "target_action": config.attack.target_action,
            "kl_budget": config.attack.kl_budget,
            "episodes": summary.get("episodes"),
            "team_reward_mean": reward_mean,
            "team_reward_std": _metric(summary, "team_reward_std"),
            "final_min_distance_sum_mean": final_distance,
            "final_unique_landmarks_mean": _metric(
                summary,
                "final_unique_landmarks_mean",
            ),
            "max_unique_landmarks_mean": _metric(summary, "max_unique_landmarks_mean"),
            "collision_pair_events_mean": _metric(
                summary,
                "collision_pair_events_mean",
            ),
            "collision_step_rate_mean": _metric(summary, "collision_step_rate_mean"),
            "final_collision_pairs_mean": _metric(
                summary,
                "final_collision_pairs_mean",
            ),
            "max_collision_pairs_mean": _metric(summary, "max_collision_pairs_mean"),
            "poisoned_action_count": _attack_summary_value(
                summary,
                "poisoned_action_count",
            ),
            "mean_effective_probability": _attack_summary_value(
                summary,
                "mean_effective_probability",
            ),
        }
        if clean_reward is not None and reward_mean is not None:
            record["reward_delta_vs_clean"] = reward_mean - clean_reward
            record["reward_degradation_vs_clean"] = clean_reward - reward_mean
        else:
            record["reward_delta_vs_clean"] = None
            record["reward_degradation_vs_clean"] = None
        records.append(record)

    return records


def write_records(records: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison_summary.json"
    csv_path = output_dir / "comparison_summary.csv"

    json_path.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    fieldnames = list(records[0].keys()) if records else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return {
        "comparison_json": str(json_path),
        "comparison_csv": str(csv_path),
    }


def write_plots(records: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, str] = {}

    def barplot(metric: str, filename: str, ylabel: str) -> None:
        usable_records = [record for record in records if record.get(metric) is not None]
        labels = [record["label"] for record in usable_records]
        values = [record[metric] for record in usable_records]
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths[metric] = str(path)

    barplot("team_reward_mean", "team_reward_mean.png", "Mean team reward")
    barplot(
        "reward_degradation_vs_clean",
        "reward_degradation_vs_clean.png",
        "Reward degradation vs clean",
    )
    barplot(
        "final_min_distance_sum_mean",
        "final_min_distance_sum_mean.png",
        "Final landmark min-distance sum",
    )
    barplot(
        "collision_pair_events_mean",
        "collision_pair_events_mean.png",
        "Mean collision pair events per episode",
    )
    return plot_paths


def evaluate_persistence(
    specs: list[ExperimentSpec] | None = None,
    *,
    episodes: int,
    seed_offset: int = 50_000,
    output_dir: Path = DEFAULT_ANALYSIS_DIR,
) -> dict[str, dict[str, Any]]:
    specs = specs or DEFAULT_EXPERIMENTS
    persistence_dir = output_dir / "persistence"
    persistence_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}

    for spec in specs:
        if not spec.model_path.exists():
            continue
        config = _disable_attack(load_config(spec.config_path))
        summary = evaluate_saved_model(
            model_path=spec.model_path,
            config=config,
            episodes=episodes,
            seed_offset=seed_offset,
        )
        path = persistence_dir / f"{spec.name}_summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries[spec.name] = summary

    return summaries


def analyze_experiments(
    *,
    specs: list[ExperimentSpec] | None = None,
    output_dir: Path = DEFAULT_ANALYSIS_DIR,
    refresh_persistence: bool = False,
    persistence_episodes: int = 8,
    write_plot_files: bool = True,
) -> dict[str, Any]:
    specs = specs or DEFAULT_EXPERIMENTS
    summaries = {
        spec.name: _load_json(spec.summary_path)
        for spec in specs
        if spec.summary_path.exists()
    }
    missing_summaries = [
        str(spec.summary_path) for spec in specs if spec.name not in summaries
    ]
    if missing_summaries:
        raise FileNotFoundError(
            "Missing evaluation summaries: " + ", ".join(missing_summaries)
        )

    records = build_comparison_records(summaries, specs)
    outputs = write_records(records, output_dir)

    if write_plot_files:
        outputs["plots"] = write_plots(records, output_dir)

    persistence_records: list[dict[str, Any]] | None = None
    if refresh_persistence:
        persistence_summaries = evaluate_persistence(
            specs,
            episodes=persistence_episodes,
            output_dir=output_dir,
        )
        persistence_records = build_comparison_records(persistence_summaries, specs)
        outputs.update(
            {
                f"persistence_{key}": value
                for key, value in write_records(
                    persistence_records,
                    output_dir / "persistence",
                ).items()
            }
        )
        if write_plot_files:
            outputs["persistence_plots"] = write_plots(
                persistence_records,
                output_dir / "persistence",
            )

    result = {
        "records": records,
        "outputs": outputs,
    }
    if persistence_records is not None:
        result["persistence_records"] = persistence_records
    return result
