from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean, pstdev
import sys
import time
from typing import Any

import numpy as np
import torch

from behavior_poisoning.config import ExperimentConfig, load_config, save_config_snapshot


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


def _make_dummy_vec_env(args, *, num_threads: int, seed_base: int):
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

    try:
        runner.run()
    finally:
        envs.close()
        if eval_envs is not None and eval_envs is not envs:
            eval_envs.close()
        if not runner.use_wandb:
            runner.writter.export_scalars_to_json(Path(runner.log_dir) / "summary.json")
            runner.writter.close()

    summary = evaluate_mappo_model(model_path=paths["models"], config=config)
    evaluation_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "config": asdict(config),
        "checkpoint_root": str(paths["checkpoint_root"]),
        "model_path": str(paths["models"]),
        "evaluation_path": str(evaluation_path),
        "config_snapshot_path": str(config_snapshot_path),
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
