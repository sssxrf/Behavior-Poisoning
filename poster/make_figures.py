from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

ATTACKS = [
    ("random", "Random", "#2E86AB", "random_action_poison"),
    ("targeted", "Targeted no-op", "#F18F01", "targeted_action_poison"),
    ("kl", "KL-targeted", "#7B2CBF", "kl_targeted_action_poison"),
]
PS = ["0.1", "0.2", "0.3"]
SEEDS = ["7", "13", "21", "31", "37"]
SWEEP_DIRS = {
    "0.1": ROOT / "seed_sweep_p01_results",
    "0.2": ROOT / "seed_sweep_p02_results",
    "0.3": ROOT / "seed_sweep_p03_results",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def _seed_degradation() -> dict[str, dict[str, list[float]]]:
    files = {
        "0.1": ROOT / "probability_comparison_results" / "p01_hypothesis_by_seed.csv",
        "0.2": ROOT / "probability_comparison_results" / "p02_hypothesis_by_seed.csv",
        "0.3": ROOT / "probability_comparison_results" / "p03_hypothesis_by_seed.csv",
    }
    values: dict[str, dict[str, list[float]]] = {}
    for p, path in files.items():
        values[p] = {}
        rows = _read_csv(path)
        for key, _, _, _ in ATTACKS:
            values[p][key] = [_float(row, f"{key}_degradation") for row in rows]
    return values


def degradation_by_probability() -> None:
    rows = _read_csv(ROOT / "probability_comparison_results" / "probability_summary.csv")
    by_p = {row["requested_poison_probability"]: row for row in rows}
    seed_values = _seed_degradation()

    x = np.arange(len(PS))
    width = 0.24

    fig, ax = plt.subplots(figsize=(13.2, 7.0))
    for offset, (key, label, color, _) in zip([-width, 0, width], ATTACKS):
        means = [_float(by_p[p], f"{key}_mean_degradation") for p in PS]
        bars = ax.bar(
            x + offset,
            means,
            width,
            label=label,
            color=color,
            edgecolor="#101820",
            linewidth=0.6,
        )
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in means], fontsize=13, padding=4)
        for idx, p in enumerate(PS):
            jitter = np.linspace(-0.06, 0.06, len(seed_values[p][key]))
            ax.scatter(
                np.full(len(seed_values[p][key]), x[idx] + offset) + jitter,
                seed_values[p][key],
                s=34,
                color="#101820",
                alpha=0.68,
                zorder=5,
                linewidths=0,
            )

    ax.axhline(0, color="#101820", linewidth=1.4)
    ax.set_xticks(x, [f"p = {p}" for p in PS], fontsize=17)
    ax.set_ylabel(r"Reward degradation $D = R_{clean} - R_{poisoned}$", fontsize=17)
    ax.set_title("No reliable degradation trend across poison probability", fontsize=20, weight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper right", frameon=False, fontsize=15)
    ax.tick_params(axis="y", labelsize=15)
    _save(fig, "degradation_by_probability")


def seed_heatmap() -> None:
    rows = []
    for p, path in [
        ("0.1", ROOT / "probability_comparison_results" / "p01_hypothesis_by_seed.csv"),
        ("0.2", ROOT / "probability_comparison_results" / "p02_hypothesis_by_seed.csv"),
        ("0.3", ROOT / "probability_comparison_results" / "p03_hypothesis_by_seed.csv"),
    ]:
        by_seed = {row["seed"]: row for row in _read_csv(path)}
        for seed in SEEDS:
            row = by_seed[seed]
            rows.append(
                (
                    f"p={p.replace('0.', '.') } s{seed}",
                    [
                        _float(row, "random_degradation"),
                        _float(row, "targeted_degradation"),
                        _float(row, "kl_degradation"),
                    ],
                )
            )

    labels = [row[0] for row in rows]
    data = np.array([row[1] for row in rows])
    vmax = max(abs(float(data.min())), abs(float(data.max())))

    fig, ax = plt.subplots(figsize=(9.1, 11.3))
    im = ax.imshow(
        data,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
        aspect="auto",
    )
    ax.set_xticks(np.arange(3), ["Random", "Targeted\nno-op", "KL-targeted"], fontsize=15)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=12)
    ax.set_title("Matched-seed degradation is highly variable", fontsize=19, weight="bold")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                color="white" if abs(value) > vmax * 0.46 else "#101820",
                fontsize=11,
                weight="bold",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.052, pad=0.035)
    cbar.set_label("D, higher = more degradation", fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    _save(fig, "seed_heatmap")


def coverage_mechanism() -> None:
    dist_delta: dict[str, dict[str, list[float]]] = {p: {} for p in PS}
    collision_delta: dict[str, dict[str, list[float]]] = {p: {} for p in PS}

    for p, sweep_dir in SWEEP_DIRS.items():
        records = _read_csv(sweep_dir / "seed_sweep_records.csv")
        clean_by_seed = {
            row["seed"]: row
            for row in records
            if row["name"] == "clean_rmappo"
        }
        for _, _, _, record_name in ATTACKS:
            dist_delta[p][record_name] = []
            collision_delta[p][record_name] = []
            for row in records:
                if row["name"] != record_name:
                    continue
                clean = clean_by_seed[row["seed"]]
                dist_delta[p][record_name].append(
                    _float(row, "final_min_distance_sum_mean")
                    - _float(clean, "final_min_distance_sum_mean")
                )
                collision_delta[p][record_name].append(
                    _float(row, "collision_pair_events_mean")
                    - _float(clean, "collision_pair_events_mean")
                )

    x = np.arange(len(PS))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.2), sharex=True)
    panels = [
        (
            axes[0],
            dist_delta,
            r"$\Delta$Distance = Distance$_{poisoned}$ - Distance$_{clean}$",
            "positive = worse coverage",
        ),
        (
            axes[1],
            collision_delta,
            r"$\Delta$Collisions = Collisions$_{poisoned}$ - Collisions$_{clean}$",
            "positive = more collisions",
        ),
    ]

    for ax, values, title, ylabel in panels:
        for offset, (_, label, color, record_name) in zip([-width, 0, width], ATTACKS):
            means = [float(np.mean(values[p][record_name])) for p in PS]
            ax.bar(
                x + offset,
                means,
                width,
                label=label,
                color=color,
                edgecolor="#101820",
                linewidth=0.55,
            )
            for idx, p in enumerate(PS):
                seed_vals = values[p][record_name]
                jitter = np.linspace(-0.045, 0.045, len(seed_vals))
                ax.scatter(
                    np.full(len(seed_vals), x[idx] + offset) + jitter,
                    seed_vals,
                    s=22,
                    color="#101820",
                    alpha=0.62,
                    zorder=5,
                    linewidths=0,
                )
        ax.axhline(0, color="#101820", linewidth=1.2)
        ax.set_xticks(x, [f"p = {p}" for p in PS], fontsize=13)
        ax.set_title(title, fontsize=14.5, weight="bold")
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", alpha=0.22)

    axes[1].legend(loc="upper right", frameon=False, fontsize=11.5)
    fig.suptitle("Coverage and collision changes are noisy across seeds", fontsize=19, weight="bold")
    fig.tight_layout()
    _save(fig, "coverage_mechanism")


def main() -> None:
    degradation_by_probability()
    seed_heatmap()
    coverage_mechanism()


if __name__ == "__main__":
    main()
