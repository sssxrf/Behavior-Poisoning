from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean, pstdev
import sys
import time
import types
from typing import Any

import numpy as np
import torch

from behavior_poisoning.config import (
    AttackConfig,
    ExperimentConfig,
    load_config,
    save_config_snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = REPO_ROOT / ".mappo_upstream"


class SimpleSummaryWriter:
    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.scalars: list[dict[str, Any]] = []

    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: dict[str, Any],
        global_step: int,
    ) -> None:
        normalized = {}
        for key, value in tag_scalar_dict.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    normalized[key] = float(value.detach().cpu().item())
                else:
                    normalized[key] = value.detach().cpu().tolist()
            elif isinstance(value, np.ndarray):
                normalized[key] = value.tolist()
            elif isinstance(value, np.generic):
                normalized[key] = float(value)
            else:
                normalized[key] = value
        self.scalars.append(
            {
                "tag": main_tag,
                "step": int(global_step),
                "values": normalized,
            }
        )

    def export_scalars_to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.scalars, indent=2), encoding="utf-8")

    def close(self) -> None:
        return None


class AgentIndicatorMPEEnv:
    def __init__(self, args):
        _prepare_upstream_imports()

        from gym.spaces import Box
        from onpolicy.envs.mpe.MPE_env import MPEEnv

        self.env = MPEEnv(args)
        self.num_agents = args.num_agents
        self.action_space = self.env.action_space
        self.observation_space = []

        for base_space in self.env.observation_space:
            obs_dim = int(base_space.shape[0]) + self.num_agents
            box = Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )
            self.observation_space.append(box)

        shared_dim = self.observation_space[0].shape[0] * self.num_agents
        self.share_observation_space = [
            Box(
                low=-np.inf,
                high=np.inf,
                shape=(shared_dim,),
                dtype=np.float32,
            )
            for _ in range(self.num_agents)
        ]

    def _augment(self, observations) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32)
        augmented = []
        for agent_idx, observation in enumerate(observations):
            indicator = np.zeros(self.num_agents, dtype=np.float32)
            indicator[agent_idx] = 1.0
            augmented.append(
                np.concatenate([observation, indicator]).astype(np.float32, copy=False)
            )
        return np.asarray(augmented, dtype=np.float32)

    def seed(self, seed: int | None = None):
        return self.env.seed(seed)

    def reset(self):
        return self._augment(self.env.reset())

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        return self._augment(obs), rewards, dones, infos

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()
        return None

    @property
    def world(self):
        return self.env.world

    def __getattr__(self, item):
        return getattr(self.env, item)


SUPPORTED_ACTION_POISONING_MODES = {"random_action", "targeted_action"}


def _validate_action_poisoning(
    *,
    num_agents: int,
    compromised_agent: int,
    probability: float,
) -> None:
    if compromised_agent < 0 or compromised_agent >= num_agents:
        raise ValueError(
            f"compromised_agent must be in [0, {num_agents - 1}], "
            f"got {compromised_agent}."
        )
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}.")


def _validate_kl_budget(kl_budget: float | None) -> None:
    if kl_budget is not None and kl_budget < 0.0:
        raise ValueError(f"kl_budget must be non-negative or null, got {kl_budget}.")


def _validate_target_action(action_space, target_action: int) -> None:
    if target_action < 0 or target_action >= action_space.n:
        raise ValueError(
            f"target_action must be in [0, {action_space.n - 1}], "
            f"got {target_action}."
        )


def _normalized_probs(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probs, dtype=np.float64), 1e-8, None)
    return clipped / clipped.sum()


def _kl_divergence(q_probs: np.ndarray, p_probs: np.ndarray) -> float:
    q = _normalized_probs(q_probs)
    p = _normalized_probs(p_probs)
    return float(np.sum(q * (np.log(q) - np.log(p))))


def _attack_action_distribution(
    *,
    mode: str,
    action_space,
    target_action: int,
) -> np.ndarray:
    if action_space.__class__.__name__ != "Discrete":
        raise NotImplementedError(
            "Action poisoning currently supports Discrete actions only."
        )
    if mode == "random_action":
        return np.full(action_space.n, 1.0 / action_space.n, dtype=np.float64)
    if mode == "targeted_action":
        _validate_target_action(action_space, target_action)
        distribution = np.zeros(action_space.n, dtype=np.float64)
        distribution[target_action] = 1.0
        return distribution
    raise ValueError(
        f"Unsupported action poisoning mode '{mode}'. "
        f"Expected one of {sorted(SUPPORTED_ACTION_POISONING_MODES)}."
    )


def _kl_constrained_probability(
    *,
    clean_probs: np.ndarray,
    attack_probs: np.ndarray,
    requested_probability: float,
    kl_budget: float | None,
) -> float:
    _validate_action_poisoning(
        num_agents=1,
        compromised_agent=0,
        probability=requested_probability,
    )
    _validate_kl_budget(kl_budget)
    if requested_probability == 0.0 or kl_budget is None:
        return requested_probability

    clean = _normalized_probs(clean_probs)
    attack = _normalized_probs(attack_probs)

    def mixture_kl(alpha: float) -> float:
        mixed = (1.0 - alpha) * clean + alpha * attack
        return _kl_divergence(mixed, clean)

    if mixture_kl(requested_probability) <= kl_budget:
        return requested_probability

    low = 0.0
    high = requested_probability
    for _ in range(32):
        mid = (low + high) / 2.0
        if mixture_kl(mid) <= kl_budget:
            low = mid
        else:
            high = mid
    return low


def _poison_discrete_actions(
    actions,
    action_spaces,
    *,
    compromised_agent: int,
    probability: float,
    rng: np.random.Generator,
    mode: str = "random_action",
    target_action: int = 0,
) -> tuple[np.ndarray, bool]:
    action_array = np.asarray(actions, dtype=np.float32).copy()
    _validate_action_poisoning(
        num_agents=len(action_spaces),
        compromised_agent=compromised_agent,
        probability=probability,
    )
    if mode not in SUPPORTED_ACTION_POISONING_MODES:
        raise ValueError(
            f"Unsupported action poisoning mode '{mode}'. "
            f"Expected one of {sorted(SUPPORTED_ACTION_POISONING_MODES)}."
        )

    if probability == 0.0 or rng.random() >= probability:
        return action_array, False

    action_space = action_spaces[compromised_agent]
    if action_space.__class__.__name__ != "Discrete":
        raise NotImplementedError(
            "Action poisoning currently supports Discrete actions only."
        )

    if mode == "random_action":
        poisoned_action = int(rng.integers(action_space.n))
    else:
        _validate_target_action(action_space, target_action)
        poisoned_action = target_action

    action_array[compromised_agent] = np.eye(
        action_space.n,
        dtype=np.float32,
    )[poisoned_action]
    return action_array, True


def _poison_discrete_action_batch(
    actions: np.ndarray,
    clean_action_probs: np.ndarray,
    action_spaces,
    attack: AttackConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    poisoned_actions = np.asarray(actions, dtype=np.int64).copy()
    if not attack.enabled:
        return poisoned_actions, {"poisoned_count": 0, "mean_effective_probability": 0.0}

    _validate_action_poisoning(
        num_agents=len(action_spaces),
        compromised_agent=attack.compromised_agent,
        probability=attack.probability,
    )
    _validate_kl_budget(attack.kl_budget)

    agent_idx = attack.compromised_agent
    action_space = action_spaces[agent_idx]
    attack_distribution = _attack_action_distribution(
        mode=attack.mode,
        action_space=action_space,
        target_action=attack.target_action,
    )

    poisoned_count = 0
    effective_probabilities: list[float] = []
    for env_idx in range(poisoned_actions.shape[0]):
        clean_probs = clean_action_probs[env_idx, agent_idx]
        effective_probability = _kl_constrained_probability(
            clean_probs=clean_probs,
            attack_probs=attack_distribution,
            requested_probability=attack.probability,
            kl_budget=attack.kl_budget,
        )
        effective_probabilities.append(effective_probability)
        if rng.random() >= effective_probability:
            continue

        if attack.mode == "random_action":
            replacement = int(rng.integers(action_space.n))
        else:
            replacement = attack.target_action
        poisoned_actions[env_idx, agent_idx, 0] = replacement
        poisoned_count += 1

    return poisoned_actions, {
        "poisoned_count": poisoned_count,
        "mean_effective_probability": mean(effective_probabilities)
        if effective_probabilities
        else 0.0,
    }


def _should_apply_action_poisoning(config: ExperimentConfig | None) -> bool:
    if config is None or not config.attack.enabled:
        return False
    if config.attack.mode not in SUPPORTED_ACTION_POISONING_MODES:
        raise ValueError(
            "MAPPO training currently supports attack.mode in "
            f"{sorted(SUPPORTED_ACTION_POISONING_MODES)}; "
            f"got '{config.attack.mode}'."
        )
    return True


def _prepare_upstream_imports() -> None:
    if not UPSTREAM_ROOT.exists():
        raise FileNotFoundError(
            f"Official MAPPO repo not found at {UPSTREAM_ROOT}. "
            "Clone marlbenchmark/on-policy into .mappo_upstream first."
        )
    if str(UPSTREAM_ROOT) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_ROOT))

    import tensorboardX

    tensorboardX.SummaryWriter = SimpleSummaryWriter


def _build_mappo_args(config: ExperimentConfig):
    _prepare_upstream_imports()

    from onpolicy.config import get_config
    from onpolicy.scripts.train.train_mpe import parse_args

    parser = get_config()
    args = parse_args([], parser)

    args.env_name = "MPE"
    args.scenario_name = "simple_spread"
    args.num_agents = config.env.num_agents
    args.num_landmarks = config.env.num_landmarks
    args.episode_length = config.env.max_cycles
    args.algorithm_name = config.training.algorithm.lower()
    args.experiment_name = config.training.run_name
    args.user_name = config.project_name
    args.seed = config.seed
    args.num_env_steps = config.training.total_timesteps
    args.n_rollout_threads = config.training.num_rollout_threads
    args.n_eval_rollout_threads = config.training.num_eval_rollout_threads
    args.n_training_threads = config.training.num_training_threads
    args.hidden_size = config.training.hidden_size
    args.recurrent_N = config.training.recurrent_N
    args.data_chunk_length = config.training.data_chunk_length
    args.lr = config.training.lr
    args.critic_lr = config.training.critic_lr
    args.ppo_epoch = config.training.ppo_epoch
    args.num_mini_batch = config.training.num_mini_batch
    args.clip_param = config.training.clip_param
    args.entropy_coef = config.training.entropy_coef
    args.value_loss_coef = config.training.value_loss_coef
    args.max_grad_norm = config.training.max_grad_norm
    args.gamma = config.training.gamma
    args.gae_lambda = config.training.gae_lambda
    args.use_eval = config.training.use_eval
    args.eval_interval = config.training.eval_interval
    args.eval_episodes = config.training.evaluation_episodes
    args.log_interval = config.training.log_interval
    args.save_interval = config.training.save_interval
    args.use_linear_lr_decay = config.training.use_linear_lr_decay
    args.share_policy = config.training.share_policy
    args.use_centralized_V = config.training.use_centralized_v
    args.use_agent_indicator = config.training.use_agent_indicator
    args.use_wandb = False
    args.model_dir = None
    args.use_render = False

    if args.algorithm_name == "rmappo":
        args.use_recurrent_policy = True
        args.use_naive_recurrent_policy = False
    elif args.algorithm_name == "mappo":
        args.use_recurrent_policy = False
        args.use_naive_recurrent_policy = False
    else:
        raise ValueError(
            "MAPPO backend only supports training.algorithm in {'mappo', 'rmappo'}."
        )

    requested_device = config.training.device.lower()
    if requested_device == "cpu":
        args.cuda = False
    elif requested_device == "cuda":
        args.cuda = True
    else:
        args.cuda = torch.cuda.is_available()

    return args


def _make_dummy_vec_env(
    args,
    *,
    num_threads: int,
    seed_base: int,
):
    _prepare_upstream_imports()

    from onpolicy.envs.env_wrappers import DummyVecEnv

    def get_env_fn(rank: int):
        def init_env():
            if getattr(args, "use_agent_indicator", False):
                env = AgentIndicatorMPEEnv(args)
            else:
                from onpolicy.envs.mpe.MPE_env import MPEEnv

                env = MPEEnv(args)
            env.seed(seed_base + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(rank) for rank in range(num_threads)])


def _actor_discrete_action_probs(
    actor,
    obs,
    rnn_states,
    masks,
) -> np.ndarray:
    from onpolicy.algorithms.utils.util import check

    obs = check(obs).to(**actor.tpdv)
    rnn_states = check(rnn_states).to(**actor.tpdv)
    masks = check(masks).to(**actor.tpdv)

    actor_features = actor.base(obs)
    if actor._use_naive_recurrent_policy or actor._use_recurrent_policy:
        actor_features, _ = actor.rnn(actor_features, rnn_states, masks)
    return actor.act.get_probs(actor_features).detach().cpu().numpy()


def _action_log_probs_from_probs(
    action_probs: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    selected_probs = np.take_along_axis(
        action_probs,
        np.asarray(actions, dtype=np.int64),
        axis=2,
    )
    return np.log(np.clip(selected_probs, 1e-8, None)).astype(np.float32)


def _ensure_runner_attack_state(runner) -> None:
    if hasattr(runner, "_action_poisoning_rng"):
        return
    runner._action_poisoning_rng = np.random.default_rng(runner.all_args.seed + 9137)
    runner.action_poisoning_stats = {
        "poisoned_count": 0,
        "steps": 0,
        "effective_probabilities": [],
    }


def _record_runner_attack_stats(runner, stats: dict[str, Any]) -> None:
    runner.action_poisoning_stats["poisoned_count"] += int(stats["poisoned_count"])
    runner.action_poisoning_stats["steps"] += 1
    runner.action_poisoning_stats["effective_probabilities"].append(
        float(stats["mean_effective_probability"])
    )


def _runner_attack_summary(runner) -> dict[str, Any] | None:
    if not hasattr(runner, "action_poisoning_stats"):
        return None
    effective_probabilities = runner.action_poisoning_stats["effective_probabilities"]
    return {
        "poisoned_action_count": runner.action_poisoning_stats["poisoned_count"],
        "collection_steps": runner.action_poisoning_stats["steps"],
        "mean_effective_probability": mean(effective_probabilities)
        if effective_probabilities
        else 0.0,
    }


@torch.no_grad()
def _collect_shared_with_action_poisoning(self, step):
    self.trainer.prep_rollout()
    flat_share_obs = np.concatenate(self.buffer.share_obs[step])
    flat_obs = np.concatenate(self.buffer.obs[step])
    flat_rnn_states = np.concatenate(self.buffer.rnn_states[step])
    flat_rnn_states_critic = np.concatenate(self.buffer.rnn_states_critic[step])
    flat_masks = np.concatenate(self.buffer.masks[step])

    action_probs = _actor_discrete_action_probs(
        self.trainer.policy.actor,
        flat_obs,
        flat_rnn_states,
        flat_masks,
    )
    value, action, action_log_prob, rnn_states, rnn_states_critic = (
        self.trainer.policy.get_actions(
            flat_share_obs,
            flat_obs,
            flat_rnn_states,
            flat_rnn_states_critic,
            flat_masks,
        )
    )

    values = np.array(np.split(value.detach().cpu().numpy(), self.n_rollout_threads))
    actions = np.array(np.split(action.detach().cpu().numpy(), self.n_rollout_threads))
    action_probs = np.array(np.split(action_probs, self.n_rollout_threads))
    rnn_states = np.array(
        np.split(rnn_states.detach().cpu().numpy(), self.n_rollout_threads)
    )
    rnn_states_critic = np.array(
        np.split(rnn_states_critic.detach().cpu().numpy(), self.n_rollout_threads)
    )

    _ensure_runner_attack_state(self)
    actions, stats = _poison_discrete_action_batch(
        actions,
        action_probs,
        self.envs.action_space,
        self.attack_config,
        self._action_poisoning_rng,
    )
    _record_runner_attack_stats(self, stats)
    action_log_probs = _action_log_probs_from_probs(action_probs, actions)

    if self.envs.action_space[0].__class__.__name__ == "Discrete":
        actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
    else:
        raise NotImplementedError("Action poisoning currently supports Discrete actions only.")

    return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env


@torch.no_grad()
def _collect_separated_with_action_poisoning(self, step):
    values = []
    actions = []
    action_probs = []
    rnn_states = []
    rnn_states_critic = []

    for agent_id in range(self.num_agents):
        self.trainer[agent_id].prep_rollout()
        obs = self.buffer[agent_id].obs[step]
        actor_rnn_states = self.buffer[agent_id].rnn_states[step]
        masks = self.buffer[agent_id].masks[step]
        agent_probs = _actor_discrete_action_probs(
            self.trainer[agent_id].policy.actor,
            obs,
            actor_rnn_states,
            masks,
        )
        value, action, _, rnn_state, rnn_state_critic = (
            self.trainer[agent_id].policy.get_actions(
                self.buffer[agent_id].share_obs[step],
                obs,
                actor_rnn_states,
                self.buffer[agent_id].rnn_states_critic[step],
                masks,
            )
        )
        values.append(value.detach().cpu().numpy())
        actions.append(action.detach().cpu().numpy())
        action_probs.append(agent_probs)
        rnn_states.append(rnn_state.detach().cpu().numpy())
        rnn_states_critic.append(rnn_state_critic.detach().cpu().numpy())

    values = np.array(values).transpose(1, 0, 2)
    actions = np.array(actions).transpose(1, 0, 2)
    action_probs = np.array(action_probs).transpose(1, 0, 2)
    rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
    rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

    _ensure_runner_attack_state(self)
    actions, stats = _poison_discrete_action_batch(
        actions,
        action_probs,
        self.envs.action_space,
        self.attack_config,
        self._action_poisoning_rng,
    )
    _record_runner_attack_stats(self, stats)
    action_log_probs = _action_log_probs_from_probs(action_probs, actions)

    actions_env = []
    for env_idx in range(self.n_rollout_threads):
        one_hot_action_env = []
        for agent_idx in range(self.num_agents):
            action_space = self.envs.action_space[agent_idx]
            if action_space.__class__.__name__ != "Discrete":
                raise NotImplementedError(
                    "Action poisoning currently supports Discrete actions only."
                )
            action_idx = int(actions[env_idx, agent_idx, 0])
            one_hot_action_env.append(np.eye(action_space.n)[action_idx])
        actions_env.append(one_hot_action_env)

    return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env


def _ensure_output_dirs(config: ExperimentConfig) -> dict[str, Path]:
    root = Path(config.output_dir)
    run_stem = f"{config.training.run_name}_seed{config.seed}"
    checkpoint_root = root / "checkpoints" / run_stem

    paths = {
        "root": root,
        "checkpoint_root": checkpoint_root,
        "checkpoints": root / "checkpoints",
        "eval": root / "eval",
        "logs": checkpoint_root / "logs",
        "models": checkpoint_root / "models",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _occupied_landmarks(env) -> int:
    occupied = 0
    for landmark in env.world.landmarks:
        distances = [
            float(np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos))))
            for agent in env.world.agents
        ]
        if min(distances) < 0.1:
            occupied += 1
    return occupied


def _unique_landmarks(env) -> int:
    assigned_landmarks = []
    for agent in env.world.agents:
        distances = [
            float(np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos))))
            for landmark in env.world.landmarks
        ]
        assigned_landmarks.append(int(np.argmin(distances)))
    return len(set(assigned_landmarks))


def _landmark_min_distance_sum(env) -> float:
    total = 0.0
    for landmark in env.world.landmarks:
        distances = [
            float(np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos))))
            for agent in env.world.agents
        ]
        total += min(distances)
    return total


def _resolve_model_dir(model_path: str | Path) -> Path:
    path = Path(model_path)
    if path.is_file():
        if path.name in {"actor.pt", "critic.pt"} or path.name.startswith("actor_agent"):
            return path.parent
        raise FileNotFoundError(
            f"Expected a checkpoint directory or actor.pt/critic.pt, got {path}."
        )
    if (path / "actor.pt").exists() or (path / "actor_agent0.pt").exists():
        return path
    if (path / "models" / "actor.pt").exists() or (path / "models" / "actor_agent0.pt").exists():
        return path / "models"
    raise FileNotFoundError(
        f"Could not find actor.pt under {path}. Pass the checkpoint root or models directory."
    )


def train_mappo_clean_baseline(
    config_path: str | Path,
    *,
    total_timesteps: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    if total_timesteps is not None:
        config.training.total_timesteps = total_timesteps

    paths = _ensure_output_dirs(config)
    config_snapshot_path = paths["checkpoint_root"] / "config.yaml"
    evaluation_path = (
        paths["eval"] / f"{config.training.run_name}_seed{config.seed}_summary.json"
    )

    save_config_snapshot(config, config_snapshot_path)
    args = _build_mappo_args(config)

    device = torch.device(
        "cuda:0" if args.cuda and torch.cuda.is_available() else "cpu"
    )
    torch.set_num_threads(args.n_training_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    _prepare_upstream_imports()
    if args.share_policy:
        from onpolicy.runner.shared.mpe_runner import MPERunner
    else:
        from onpolicy.runner.separated.mpe_runner import MPERunner

    envs = _make_dummy_vec_env(
        args,
        num_threads=args.n_rollout_threads,
        seed_base=args.seed,
    )
    eval_envs = (
        _make_dummy_vec_env(
            args,
            num_threads=args.n_eval_rollout_threads,
            seed_base=args.seed * 50_000,
        )
        if args.use_eval
        else None
    )

    runner = MPERunner(
        {
            "all_args": args,
            "envs": envs,
            "eval_envs": eval_envs,
            "num_agents": args.num_agents,
            "device": device,
            "run_dir": paths["checkpoint_root"],
        }
    )
    if _should_apply_action_poisoning(config):
        runner.attack_config = config.attack
        if args.share_policy:
            runner.collect = types.MethodType(
                _collect_shared_with_action_poisoning,
                runner,
            )
        else:
            runner.collect = types.MethodType(
                _collect_separated_with_action_poisoning,
                runner,
            )

    try:
        runner.run()
    finally:
        envs.close()
        if eval_envs is not None and eval_envs is not envs:
            eval_envs.close()
        if not runner.use_wandb:
            runner.writter.export_scalars_to_json(Path(runner.log_dir) / "summary.json")
            runner.writter.close()

    attack_summary = _runner_attack_summary(runner)
    summary = evaluate_mappo_model(model_path=paths["models"], config=config)
    if attack_summary is not None:
        summary["attack_summary"] = attack_summary
    evaluation_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "config": asdict(config),
        "checkpoint_root": str(paths["checkpoint_root"]),
        "model_path": str(paths["models"]),
        "evaluation_path": str(evaluation_path),
        "config_snapshot_path": str(config_snapshot_path),
        "attack_summary": attack_summary,
        "evaluation_summary": summary,
    }


def evaluate_mappo_model(
    *,
    model_path: str | Path,
    config: ExperimentConfig,
    episodes: int | None = None,
    deterministic: bool | None = None,
    seed_offset: int = 10_000,
    render_mode: str | None = None,
    step_delay: float = 0.0,
) -> dict[str, Any]:
    args = _build_mappo_args(config)
    episodes = episodes or config.training.evaluation_episodes
    deterministic = (
        config.training.deterministic_eval
        if deterministic is None
        else deterministic
    )
    model_dir = _resolve_model_dir(model_path)

    _prepare_upstream_imports()
    from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
    if getattr(args, "use_agent_indicator", False):
        env = AgentIndicatorMPEEnv(args)
    else:
        from onpolicy.envs.mpe.MPE_env import MPEEnv

        env = MPEEnv(args)
    if args.share_policy:
        share_obs_space = (
            env.share_observation_space[0]
            if args.use_centralized_V
            else env.observation_space[0]
        )
        policies: list[R_MAPPOPolicy] = [
            R_MAPPOPolicy(
                args,
                env.observation_space[0],
                share_obs_space,
                env.action_space[0],
            )
        ]
        policies[0].actor.load_state_dict(
            torch.load(model_dir / "actor.pt", map_location="cpu", weights_only=True)
        )
        policies[0].critic.load_state_dict(
            torch.load(model_dir / "critic.pt", map_location="cpu", weights_only=True)
        )
    else:
        policies = []
        for agent_idx in range(args.num_agents):
            share_obs_space = (
                env.share_observation_space[agent_idx]
                if args.use_centralized_V
                else env.observation_space[agent_idx]
            )
            policy = R_MAPPOPolicy(
                args,
                env.observation_space[agent_idx],
                share_obs_space,
                env.action_space[agent_idx],
            )
            policy.actor.load_state_dict(
                torch.load(
                    model_dir / f"actor_agent{agent_idx}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            policy.critic.load_state_dict(
                torch.load(
                    model_dir / f"critic_agent{agent_idx}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            policies.append(policy)

    for policy in policies:
        policy.actor.eval()
        policy.critic.eval()

    episode_team_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_final_occupied_landmarks: list[int] = []
    episode_max_occupied_landmarks: list[int] = []
    episode_final_unique_landmarks: list[int] = []
    episode_max_unique_landmarks: list[int] = []
    episode_final_min_distance_sum: list[float] = []
    per_agent_rewards: dict[str, list[float]] = defaultdict(list)

    for episode_idx in range(episodes):
        env.seed(config.seed + seed_offset + episode_idx)
        obs = np.asarray(env.reset(), dtype=np.float32)
        rnn_states = np.zeros(
            (args.num_agents, args.recurrent_N, args.hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((args.num_agents, 1), dtype=np.float32)
        team_reward = 0.0
        steps = 0
        max_occupied_landmarks = 0
        max_unique_landmarks = 0
        agent_reward_totals: dict[str, float] = defaultdict(float)

        if render_mode is not None:
            env.render(render_mode)
            if step_delay > 0:
                time.sleep(step_delay)

        for _ in range(args.episode_length):
            if args.share_policy:
                with torch.no_grad():
                    actions, next_rnn_states = policies[0].act(
                        obs,
                        rnn_states,
                        masks,
                        deterministic=deterministic,
                    )

                rnn_states = next_rnn_states.detach().cpu().numpy()
                actions_np = actions.detach().cpu().numpy()
                if env.action_space[0].__class__.__name__ != "Discrete":
                    raise NotImplementedError("Only discrete MPE actions are supported here.")
                actions_env = np.squeeze(np.eye(env.action_space[0].n)[actions_np], axis=1)
            else:
                actions_env = []
                next_states = np.zeros_like(rnn_states)
                for agent_idx, policy in enumerate(policies):
                    with torch.no_grad():
                        action, next_rnn_state = policy.act(
                            obs[agent_idx : agent_idx + 1],
                            rnn_states[agent_idx : agent_idx + 1],
                            masks[agent_idx : agent_idx + 1],
                            deterministic=deterministic,
                        )
                    next_states[agent_idx : agent_idx + 1] = (
                        next_rnn_state.detach().cpu().numpy()
                    )
                    action_np = int(action.detach().cpu().numpy().item())
                    if env.action_space[agent_idx].__class__.__name__ != "Discrete":
                        raise NotImplementedError(
                            "Only discrete MPE actions are supported here."
                        )
                    actions_env.append(np.eye(env.action_space[agent_idx].n)[action_np])
                rnn_states = next_states

            obs, rewards, dones, infos = env.step(actions_env)
            obs = np.asarray(obs, dtype=np.float32)
            rewards_np = np.asarray(rewards, dtype=np.float32).reshape(-1)
            dones_np = np.asarray(dones, dtype=bool).reshape(-1)

            if render_mode is not None:
                env.render(render_mode)
                if step_delay > 0:
                    time.sleep(step_delay)

            steps += 1
            team_reward += float(rewards_np.sum())
            occupied_landmarks = _occupied_landmarks(env)
            unique_landmarks = _unique_landmarks(env)
            max_occupied_landmarks = max(max_occupied_landmarks, occupied_landmarks)
            max_unique_landmarks = max(max_unique_landmarks, unique_landmarks)

            for agent_idx, reward in enumerate(rewards_np):
                agent_reward_totals[f"agent_{agent_idx}"] += float(reward)

            masks = np.ones((args.num_agents, 1), dtype=np.float32)
            if np.all(dones_np):
                masks[:] = 0.0
                break

        episode_team_rewards.append(team_reward)
        episode_lengths.append(steps)
        episode_final_occupied_landmarks.append(_occupied_landmarks(env))
        episode_max_occupied_landmarks.append(max_occupied_landmarks)
        episode_final_unique_landmarks.append(_unique_landmarks(env))
        episode_max_unique_landmarks.append(max_unique_landmarks)
        episode_final_min_distance_sum.append(_landmark_min_distance_sum(env))
        for agent, reward_total in agent_reward_totals.items():
            per_agent_rewards[agent].append(reward_total)

    if hasattr(env, "close"):
        env.close()

    return {
        "model_path": str(model_dir),
        "episodes": episodes,
        "deterministic": deterministic,
        "team_reward_mean": mean(episode_team_rewards),
        "team_reward_std": pstdev(episode_team_rewards)
        if len(episode_team_rewards) > 1
        else 0.0,
        "episode_length_mean": mean(episode_lengths),
        "episode_length_std": pstdev(episode_lengths)
        if len(episode_lengths) > 1
        else 0.0,
        "team_rewards": episode_team_rewards,
        "episode_lengths": episode_lengths,
        "per_agent_reward_mean": {
            agent: mean(rewards) for agent, rewards in per_agent_rewards.items()
        },
        "final_occupied_landmarks_mean": mean(episode_final_occupied_landmarks),
        "max_occupied_landmarks_mean": mean(episode_max_occupied_landmarks),
        "final_occupied_landmarks": episode_final_occupied_landmarks,
        "max_occupied_landmarks": episode_max_occupied_landmarks,
        "final_unique_landmarks_mean": mean(episode_final_unique_landmarks),
        "max_unique_landmarks_mean": mean(episode_max_unique_landmarks),
        "final_unique_landmarks": episode_final_unique_landmarks,
        "max_unique_landmarks": episode_max_unique_landmarks,
        "final_min_distance_sum_mean": mean(episode_final_min_distance_sum),
        "final_min_distance_sum": episode_final_min_distance_sum,
    }
