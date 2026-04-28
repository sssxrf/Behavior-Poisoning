from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvironmentConfig:
    num_agents: int = 3
    num_landmarks: int = 3
    local_ratio: float = 0.5
    max_cycles: int = 25
    continuous_actions: bool = False


@dataclass
class TrainingConfig:
    algorithm: str = "rmappo"
    run_name: str = "clean_rmappo"
    total_timesteps: int = 2_500_000

    evaluation_episodes: int = 32
    deterministic_eval: bool = True
    device: str = "cpu"

    num_rollout_threads: int = 16
    num_eval_rollout_threads: int = 4
    num_training_threads: int = 1
    save_interval: int = 25
    log_interval: int = 5
    eval_interval: int = 25
    use_eval: bool = True
    use_linear_lr_decay: bool = False
    share_policy: bool = True
    use_centralized_v: bool = True
    use_agent_indicator: bool = True

    hidden_size: int = 64
    recurrent_N: int = 1
    data_chunk_length: int = 10
    lr: float = 7e-4
    critic_lr: float = 7e-4
    ppo_epoch: int = 10
    num_mini_batch: int = 1
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 1.0
    max_grad_norm: float = 10.0
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Legacy shared-PPO fields kept for fallback comparisons.
    num_vec_envs: int = 8
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 1024
    n_epochs: int = 10
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    net_arch: list[int] = field(default_factory=lambda: [128, 128])


@dataclass
class AttackConfig:
    enabled: bool = False
    mode: str = "none"
    compromised_agent: int = 0
    probability: float = 0.0
    target_action: int = 0
    kl_budget: float | None = None


@dataclass
class ExperimentConfig:
    project_name: str = "behavior-poisoning"
    seed: int = 7
    output_dir: str = "results"
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)


def _dataclass_from_dict(cls, raw: dict[str, Any] | None):
    raw = raw or {}
    allowed = {field.name for field in fields(cls)}
    filtered = {key: value for key, value in raw.items() if key in allowed}
    return cls(**filtered)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ExperimentConfig(
        project_name=raw.get("project_name", "behavior-poisoning"),
        seed=raw.get("seed", 7),
        output_dir=raw.get("output_dir", "results"),
        env=_dataclass_from_dict(EnvironmentConfig, raw.get("env")),
        training=_dataclass_from_dict(TrainingConfig, raw.get("training")),
        attack=_dataclass_from_dict(AttackConfig, raw.get("attack")),
    )


def save_config_snapshot(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
