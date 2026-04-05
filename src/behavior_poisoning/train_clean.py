from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from behavior_poisoning.config import ExperimentConfig, load_config, save_config_snapshot
from behavior_poisoning.envs import make_vectorized_env
from behavior_poisoning.evaluate import evaluate_saved_model, write_evaluation_summary
from behavior_poisoning.mappo_clean import train_mappo_clean_baseline


def _ensure_output_dirs(config: ExperimentConfig) -> dict[str, Path]:
    root = Path(config.output_dir)
    paths = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "eval": root / "eval",
        "logs": root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _train_shared_ppo_baseline(
    config: ExperimentConfig,
) -> dict[str, Any]:
    set_random_seed(config.seed)
    paths = _ensure_output_dirs(config)
    run_stem = f"{config.training.run_name}_seed{config.seed}"

    config_snapshot_path = paths["checkpoints"] / f"{run_stem}_config.yaml"
    model_path = paths["checkpoints"] / f"{run_stem}.zip"
    evaluation_path = paths["eval"] / f"{run_stem}_summary.json"

    save_config_snapshot(config, config_snapshot_path)

    env = make_vectorized_env(config)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.training.learning_rate,
        n_steps=config.training.n_steps,
        batch_size=config.training.batch_size,
        n_epochs=config.training.n_epochs,
        gamma=config.training.gamma,
        gae_lambda=config.training.gae_lambda,
        clip_range=config.training.clip_range,
        ent_coef=config.training.ent_coef,
        vf_coef=config.training.vf_coef,
        policy_kwargs={"net_arch": config.training.net_arch},
        tensorboard_log=str(paths["logs"]),
        verbose=1,
        device=config.training.device,
    )

    model.learn(total_timesteps=config.training.total_timesteps, progress_bar=False)
    model.save(str(model_path))
    env.close()

    summary = evaluate_saved_model(model_path=model_path, config=config)
    write_evaluation_summary(summary, evaluation_path)

    return {
        "config": asdict(config),
        "model_path": str(model_path),
        "evaluation_path": str(evaluation_path),
        "config_snapshot_path": str(config_snapshot_path),
        "evaluation_summary": summary,
    }


def train_clean_baseline(
    config_path: str | Path,
    *,
    total_timesteps: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    if total_timesteps is not None:
        config.training.total_timesteps = total_timesteps

    algorithm = config.training.algorithm.lower()
    if algorithm in {"rmappo", "mappo"}:
        return train_mappo_clean_baseline(
            config_path=config_path,
            total_timesteps=total_timesteps,
        )
    if algorithm == "ppo":
        return _train_shared_ppo_baseline(config)

    raise ValueError(
        f"Unsupported clean baseline algorithm '{config.training.algorithm}'."
    )
