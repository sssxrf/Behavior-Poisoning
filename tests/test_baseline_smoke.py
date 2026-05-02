from __future__ import annotations

from pathlib import Path

import numpy as np
from gym.spaces import Discrete

from behavior_poisoning.config import load_config
from behavior_poisoning.evaluate import evaluate_saved_model
import behavior_poisoning.evaluate as evaluate_module
from behavior_poisoning.mappo_clean import (
    _agent_collision_pair_count,
    _kl_constrained_probability,
    _kl_divergence,
    _poison_discrete_actions,
)
import behavior_poisoning.train_clean as train_clean_module


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_ROOT = REPO_ROOT / "configs"


class _FakeState:
    def __init__(self, position):
        self.p_pos = np.asarray(position, dtype=np.float32)


class _FakeAgent:
    def __init__(self, position, *, size=0.15, collide=True):
        self.state = _FakeState(position)
        self.size = size
        self.collide = collide


class _FakeWorld:
    def __init__(self, agents):
        self.agents = agents


class _FakeEnv:
    def __init__(self, agents):
        self.world = _FakeWorld(agents)


def test_clean_ppo_config_declares_ppo_algorithm() -> None:
    config = load_config(CONFIGS_ROOT / "clean_ppo.yaml")
    assert config.training.algorithm == "ppo"


def test_clean_mappo_config_declares_rmappo_algorithm() -> None:
    config = load_config(CONFIGS_ROOT / "clean_mappo.yaml")
    assert config.training.algorithm == "rmappo"


def test_agent_collision_pair_count_counts_unique_pairs() -> None:
    env = _FakeEnv(
        [
            _FakeAgent([0.0, 0.0]),
            _FakeAgent([0.1, 0.0]),
            _FakeAgent([0.8, 0.0]),
        ]
    )

    assert _agent_collision_pair_count(env) == 1


def test_random_action_poisoning_config_enables_random_attack() -> None:
    config = load_config(CONFIGS_ROOT / "random_action_poisoning_mappo.yaml")
    assert config.training.algorithm == "rmappo"
    assert config.training.run_name == "random_action_poison_rmappo"
    assert config.attack.enabled is True
    assert config.attack.mode == "random_action"
    assert config.attack.compromised_agent == 0
    assert config.attack.probability == 0.1


def test_targeted_action_poisoning_config_enables_targeted_attack() -> None:
    config = load_config(CONFIGS_ROOT / "targeted_action_poisoning_mappo.yaml")
    assert config.training.algorithm == "rmappo"
    assert config.training.run_name == "targeted_action_poison_rmappo"
    assert config.attack.enabled is True
    assert config.attack.mode == "targeted_action"
    assert config.attack.compromised_agent == 0
    assert config.attack.probability == 0.1
    assert config.attack.target_action == 0


def test_kl_constrained_targeted_action_config_sets_budget() -> None:
    config = load_config(CONFIGS_ROOT / "kl_constrained_targeted_action_mappo.yaml")
    assert config.training.algorithm == "rmappo"
    assert config.training.run_name == "kl_targeted_action_poison_rmappo"
    assert config.attack.enabled is True
    assert config.attack.mode == "targeted_action"
    assert config.attack.probability == 0.1
    assert config.attack.target_action == 0
    assert config.attack.kl_budget == 0.02


def test_kl_budget_caps_intervention_probability() -> None:
    clean_probs = np.array([0.05, 0.6, 0.2, 0.1, 0.05])
    attack_probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

    effective_probability = _kl_constrained_probability(
        clean_probs=clean_probs,
        attack_probs=attack_probs,
        requested_probability=0.8,
        kl_budget=0.02,
    )

    attacked_probs = (
        (1.0 - effective_probability) * clean_probs
        + effective_probability * attack_probs
    )
    assert 0.0 < effective_probability < 0.8
    assert _kl_divergence(attacked_probs, clean_probs) <= 0.020001


def test_random_action_poisoning_replaces_only_compromised_agent() -> None:
    actions = np.eye(5, dtype=np.float32)[[0, 1, 2]]
    rng = np.random.default_rng(7)

    poisoned, did_poison = _poison_discrete_actions(
        actions,
        [Discrete(5), Discrete(5), Discrete(5)],
        compromised_agent=1,
        probability=1.0,
        rng=rng,
    )

    assert did_poison is True
    np.testing.assert_array_equal(poisoned[0], actions[0])
    np.testing.assert_array_equal(poisoned[2], actions[2])
    assert poisoned[1].sum() == 1.0
    assert poisoned[1].argmax() != actions[1].argmax()


def test_targeted_action_poisoning_forces_configured_action() -> None:
    actions = np.eye(5, dtype=np.float32)[[0, 1, 2]]
    rng = np.random.default_rng(7)

    poisoned, did_poison = _poison_discrete_actions(
        actions,
        [Discrete(5), Discrete(5), Discrete(5)],
        mode="targeted_action",
        compromised_agent=2,
        probability=1.0,
        target_action=4,
        rng=rng,
    )

    assert did_poison is True
    np.testing.assert_array_equal(poisoned[0], actions[0])
    np.testing.assert_array_equal(poisoned[1], actions[1])
    np.testing.assert_array_equal(poisoned[2], np.eye(5, dtype=np.float32)[4])


def test_targeted_action_poisoning_rejects_invalid_target_action() -> None:
    actions = np.eye(5, dtype=np.float32)[[0, 1, 2]]
    rng = np.random.default_rng(7)

    try:
        _poison_discrete_actions(
            actions,
            [Discrete(5), Discrete(5), Discrete(5)],
            mode="targeted_action",
            compromised_agent=2,
            probability=1.0,
            target_action=5,
            rng=rng,
        )
    except ValueError as exc:
        assert "target_action" in str(exc)
    else:
        raise AssertionError("Expected invalid target_action to raise ValueError")


def test_random_action_poisoning_zero_probability_leaves_actions_unchanged() -> None:
    actions = np.eye(5, dtype=np.float32)[[0, 1, 2]]
    rng = np.random.default_rng(7)

    poisoned, did_poison = _poison_discrete_actions(
        actions,
        [Discrete(5), Discrete(5), Discrete(5)],
        compromised_agent=1,
        probability=0.0,
        rng=rng,
    )

    assert did_poison is False
    np.testing.assert_array_equal(poisoned, actions)


def test_train_clean_dispatches_ppo_configs_to_ppo_backend(monkeypatch) -> None:
    expected = {"backend": "ppo"}

    def fake_ppo_backend(config):
        assert config.training.algorithm == "ppo"
        return expected

    def fail_mappo_backend(*args, **kwargs):
        raise AssertionError("MAPPO backend should not be selected for clean_ppo.yaml")

    monkeypatch.setattr(train_clean_module, "_train_shared_ppo_baseline", fake_ppo_backend)
    monkeypatch.setattr(train_clean_module, "train_mappo_clean_baseline", fail_mappo_backend)

    result = train_clean_module.train_clean_baseline(CONFIGS_ROOT / "clean_ppo.yaml")
    assert result == expected


def test_train_clean_dispatches_mappo_configs_to_mappo_backend(monkeypatch) -> None:
    expected = {"backend": "rmappo"}

    def fake_mappo_backend(*, config_path, total_timesteps=None):
        assert Path(config_path).name == "clean_mappo.yaml"
        assert total_timesteps is None
        return expected

    def fail_ppo_backend(*args, **kwargs):
        raise AssertionError("PPO backend should not be selected for clean_mappo.yaml")

    monkeypatch.setattr(train_clean_module, "train_mappo_clean_baseline", fake_mappo_backend)
    monkeypatch.setattr(train_clean_module, "_train_shared_ppo_baseline", fail_ppo_backend)

    result = train_clean_module.train_clean_baseline(CONFIGS_ROOT / "clean_mappo.yaml")
    assert result == expected


def test_evaluate_saved_model_dispatches_ppo_configs_to_ppo_backend(monkeypatch) -> None:
    expected = {"backend": "ppo_eval"}

    def fake_ppo_eval(**kwargs):
        assert kwargs["config"].training.algorithm == "ppo"
        return expected

    def fail_mappo_eval(*args, **kwargs):
        raise AssertionError("MAPPO evaluation should not be selected for clean_ppo.yaml")

    monkeypatch.setattr(evaluate_module, "_evaluate_ppo_model", fake_ppo_eval)
    monkeypatch.setattr(evaluate_module, "evaluate_mappo_model", fail_mappo_eval)

    config = load_config(CONFIGS_ROOT / "clean_ppo.yaml")
    result = evaluate_saved_model(model_path=REPO_ROOT / "dummy.zip", config=config)
    assert result == expected


def test_evaluate_saved_model_dispatches_mappo_configs_to_mappo_backend(monkeypatch) -> None:
    expected = {"backend": "rmappo_eval"}

    def fake_mappo_eval(**kwargs):
        assert kwargs["config"].training.algorithm == "rmappo"
        return expected

    def fail_ppo_eval(*args, **kwargs):
        raise AssertionError("PPO evaluation should not be selected for clean_mappo.yaml")

    monkeypatch.setattr(evaluate_module, "evaluate_mappo_model", fake_mappo_eval)
    monkeypatch.setattr(evaluate_module, "_evaluate_ppo_model", fail_ppo_eval)

    config = load_config(CONFIGS_ROOT / "clean_mappo.yaml")
    result = evaluate_saved_model(model_path=REPO_ROOT / "dummy", config=config)
    assert result == expected
