from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, pstdev
import time
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from behavior_poisoning.config import ExperimentConfig
from behavior_poisoning.envs import make_parallel_env
from behavior_poisoning.mappo_clean import evaluate_mappo_model


def _to_action(value: Any):
    array = np.asarray(value)
    if array.size == 1:
        return int(array.item())
    return array


def _occupied_landmarks(env) -> int | None:
    try:
        unwrapped = env.unwrapped
        if not hasattr(unwrapped, "scenario") or not hasattr(unwrapped, "world"):
            return None
        benchmark = unwrapped.scenario.benchmark_data(
            unwrapped.world.agents[0],
            unwrapped.world,
        )
        return int(benchmark[3])
    except Exception:
        return None


def _evaluate_ppo_model(
    *,
    model_path: str | Path,
    config: ExperimentConfig,
    episodes: int | None = None,
    deterministic: bool | None = None,
    seed_offset: int = 10_000,
    render_mode: str | None = None,
    step_delay: float = 0.0,
) -> dict[str, Any]:
    model_path = Path(model_path)
    episodes = episodes or config.training.evaluation_episodes
    deterministic = (
        config.training.deterministic_eval
        if deterministic is None
        else deterministic
    )

    model = PPO.load(str(model_path))
    env = make_parallel_env(config.env, render_mode=render_mode)

    episode_team_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_final_occupied_landmarks: list[int] = []
    episode_max_occupied_landmarks: list[int] = []
    per_agent_rewards: dict[str, list[float]] = defaultdict(list)

    for episode_idx in range(episodes):
        observations, _ = env.reset(seed=config.seed + seed_offset + episode_idx)
        if render_mode is not None:
            env.render()
            if step_delay > 0:
                time.sleep(step_delay)
        team_reward = 0.0
        agent_reward_totals: dict[str, float] = defaultdict(float)
        steps = 0
        max_occupied_landmarks = 0

        while env.agents:
            actions = {
                agent: _to_action(
                    model.predict(obs, deterministic=deterministic)[0]
                )
                for agent, obs in observations.items()
            }
            observations, rewards, terminations, truncations, _ = env.step(actions)
            if render_mode is not None:
                env.render()
                if step_delay > 0:
                    time.sleep(step_delay)
            team_reward += float(sum(rewards.values()))
            steps += 1
            occupied_landmarks = _occupied_landmarks(env)
            if occupied_landmarks is not None:
                max_occupied_landmarks = max(
                    max_occupied_landmarks,
                    occupied_landmarks,
                )

            for agent, reward in rewards.items():
                agent_reward_totals[agent] += float(reward)

            if all(terminations.values()) or all(truncations.values()):
                break

        episode_team_rewards.append(team_reward)
        episode_lengths.append(steps)
        final_occupied_landmarks = _occupied_landmarks(env)
        if final_occupied_landmarks is not None:
            episode_final_occupied_landmarks.append(final_occupied_landmarks)
            episode_max_occupied_landmarks.append(max_occupied_landmarks)

        for agent, reward_total in agent_reward_totals.items():
            per_agent_rewards[agent].append(reward_total)

    env.close()

    return {
        "model_path": str(model_path),
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
        "final_occupied_landmarks_mean": mean(episode_final_occupied_landmarks)
        if episode_final_occupied_landmarks
        else None,
        "max_occupied_landmarks_mean": mean(episode_max_occupied_landmarks)
        if episode_max_occupied_landmarks
        else None,
        "final_occupied_landmarks": episode_final_occupied_landmarks,
        "max_occupied_landmarks": episode_max_occupied_landmarks,
    }


def evaluate_saved_model(
    *,
    model_path: str | Path,
    config: ExperimentConfig,
    episodes: int | None = None,
    deterministic: bool | None = None,
    seed_offset: int = 10_000,
    render_mode: str | None = None,
    step_delay: float = 0.0,
) -> dict[str, Any]:
    algorithm = config.training.algorithm.lower()
    if algorithm in {"rmappo", "mappo"}:
        return evaluate_mappo_model(
            model_path=model_path,
            config=config,
            episodes=episodes,
            deterministic=deterministic,
            seed_offset=seed_offset,
            render_mode=render_mode,
            step_delay=step_delay,
        )
    if algorithm == "ppo":
        return _evaluate_ppo_model(
            model_path=model_path,
            config=config,
            episodes=episodes,
            deterministic=deterministic,
            seed_offset=seed_offset,
            render_mode=render_mode,
            step_delay=step_delay,
        )

    raise ValueError(f"Unsupported clean baseline algorithm '{config.training.algorithm}'.")


def write_evaluation_summary(summary: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
