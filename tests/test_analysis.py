from __future__ import annotations

from pathlib import Path

from behavior_poisoning.analysis import (
    ExperimentSpec,
    build_comparison_records,
    write_records,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_ROOT = REPO_ROOT / "configs"


def test_build_comparison_records_computes_degradation_against_clean() -> None:
    specs = [
        ExperimentSpec(
            name="clean_rmappo",
            label="Clean",
            config_path=CONFIGS_ROOT / "clean_mappo.yaml",
            model_path=REPO_ROOT / "unused-clean",
            summary_path=REPO_ROOT / "unused-clean.json",
            order=0,
        ),
        ExperimentSpec(
            name="targeted_action_poison",
            label="Targeted",
            config_path=CONFIGS_ROOT / "targeted_action_poisoning_mappo.yaml",
            model_path=REPO_ROOT / "unused-targeted",
            summary_path=REPO_ROOT / "unused-targeted.json",
            order=1,
        ),
    ]
    summaries = {
        "clean_rmappo": {
            "episodes": 4,
            "team_reward_mean": -100.0,
            "team_reward_std": 5.0,
        },
        "targeted_action_poison": {
            "episodes": 4,
            "team_reward_mean": -140.0,
            "team_reward_std": 7.0,
            "attack_summary": {
                "poisoned_action_count": 12,
                "mean_effective_probability": 0.1,
            },
        },
    }

    records = build_comparison_records(summaries, specs)

    assert records[0]["reward_degradation_vs_clean"] == 0.0
    assert records[1]["reward_delta_vs_clean"] == -40.0
    assert records[1]["reward_degradation_vs_clean"] == 40.0
    assert records[1]["attack_mode"] == "targeted_action"
    assert records[1]["poisoned_action_count"] == 12


def test_write_records_creates_json_and_csv(tmp_path) -> None:
    records = [
        {
            "name": "clean",
            "label": "Clean",
            "team_reward_mean": -100.0,
        }
    ]

    outputs = write_records(records, tmp_path)

    assert Path(outputs["comparison_json"]).exists()
    assert Path(outputs["comparison_csv"]).exists()
