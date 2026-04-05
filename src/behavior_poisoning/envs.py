from __future__ import annotations

from pettingzoo.mpe import simple_spread_v3
import supersuit as ss

from behavior_poisoning.config import ExperimentConfig, EnvironmentConfig


def make_parallel_env(
    env_config: EnvironmentConfig,
    *,
    seed: int | None = None,
    render_mode: str | None = None,
):
    env = simple_spread_v3.parallel_env(
        N=env_config.num_agents,
        local_ratio=env_config.local_ratio,
        max_cycles=env_config.max_cycles,
        continuous_actions=env_config.continuous_actions,
        render_mode=render_mode,
    )
    env.reset(seed=seed)
    return env


def make_vectorized_env(config: ExperimentConfig):
    env = make_parallel_env(config.env, seed=config.seed)
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        config.training.num_vec_envs,
        num_cpus=0,
        base_class="stable_baselines3",
    )
    return env
