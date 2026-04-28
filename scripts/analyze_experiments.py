from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from behavior_poisoning.analysis import analyze_experiments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare clean and poisoned MAPPO experiment summaries."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "analysis",
        help="Directory for comparison tables and plots.",
    )
    parser.add_argument(
        "--refresh-persistence",
        action="store_true",
        help="Re-evaluate checkpoints with attacks disabled before summarizing persistence.",
    )
    parser.add_argument(
        "--persistence-episodes",
        type=int,
        default=8,
        help="Episodes per checkpoint when --refresh-persistence is used.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write tables only.",
    )
    args = parser.parse_args()

    result = analyze_experiments(
        output_dir=args.output_dir,
        refresh_persistence=args.refresh_persistence,
        persistence_episodes=args.persistence_episodes,
        write_plot_files=not args.no_plots,
    )
    print(json.dumps(result["outputs"], indent=2))


if __name__ == "__main__":
    main()
