#!/usr/bin/env python
"""Compare IRT ability vs raw accuracy within runs and across datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scfm_irt.analysis.cross_dataset_compare import write_cross_dataset_comparison
from scfm_irt.analysis.report import _plot_ability_and_accuracy_forest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["runs/ranking_immune_human_v1", "runs/ranking_pbmc3k_v2"],
        help="run directories to compare",
    )
    parser.add_argument(
        "--out",
        default="runs/comparisons/irt_vs_accuracy",
        help="output directory for cross-run comparison",
    )
    args = parser.parse_args()

    run_dirs = {Path(r).name: Path(r) for r in args.runs}
    for path in run_dirs.values():
        if not (path / "ability_ranking.csv").exists():
            raise FileNotFoundError(f"missing ability_ranking.csv in {path}")
        ability = pd.read_csv(path / "ability_ranking.csv")
        resp = pd.read_parquet(path / "responses.parquet")
        _plot_ability_and_accuracy_forest(ability, resp, path.name, path)
        print(f"[compare] updated {path / 'ability_forest.png'}")

    report = write_cross_dataset_comparison(run_dirs, args.out)
    print(f"[compare] wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
