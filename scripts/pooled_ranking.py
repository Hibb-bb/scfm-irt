#!/usr/bin/env python
"""Pooled IRT ranking across multiple completed ranking runs (model union, masked fit).

    PYTHONPATH=. python scripts/pooled_ranking.py
    PYTHONPATH=. python scripts/pooled_ranking.py --config sim/conf/sim.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scfm_irt.analysis.report import _plot_ability_and_accuracy_forest
from scfm_irt.irt.fit_pyirt import _fit_mfvi
from scfm_irt.response.schema import validate_response_table


def load_responses(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "responses.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    return pd.read_parquet(path)


def concat_union(run_dirs: dict[str, Path]) -> pd.DataFrame:
    """Concatenate per-dataset runs; keep all models (ragged union)."""
    frames = [load_responses(path) for path in run_dirs.values()]
    out = pd.concat(frames, ignore_index=True)
    dup = out.duplicated(subset=["model_id", "cell_id"]).sum()
    if dup:
        raise ValueError(f"{dup} duplicated (model_id, cell_id) rows across runs")
    return validate_response_table(out, strict=False)


def to_matrix_with_mask(resp: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Dense Y and observation mask W for ragged long tables."""
    model_ids = sorted(resp["model_id"].unique())
    cell_ids = sorted(resp["cell_id"].unique())
    wide = resp.pivot_table(
        index="model_id",
        columns="cell_id",
        values="correct",
        aggfunc="first",
    ).reindex(index=model_ids, columns=cell_ids)
    W = wide.notna().to_numpy(dtype=float)
    Y = wide.fillna(0.0).to_numpy(dtype=float)
    return Y, W, model_ids, cell_ids


def model_coverage(resp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id, grp in resp.groupby("model_id", sort=True):
        rows.append(
            {
                "model_id": model_id,
                "n_responses": len(grp),
                "n_cells": grp["cell_id"].nunique(),
                "n_datasets": grp["dataset"].nunique(),
                "datasets": ",".join(sorted(grp["dataset"].unique())),
                "accuracy": float(grp["correct"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("model_id").reset_index(drop=True)


def _df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [
        "| " + " | ".join(str(row[c]) for c in cols) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="sim/conf/sim.yaml")
    parser.add_argument("--out", default="runs/pooled_3dataset_v1")
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with (REPO / args.config).open() as f:
        cfg = yaml.safe_load(f)

    run_dirs = {
        "lung_atlas": REPO / cfg["lung_run"],
        "pbmc3k": REPO / cfg["pbmc_run"],
        "immune_human": REPO / cfg["immune_run"],
    }
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[pooled] loading runs:")
    for name, path in run_dirs.items():
        models = sorted(load_responses(path)["model_id"].unique())
        print(f"  {name}: {path.name} ({len(models)} models)")

    resp = concat_union(run_dirs)
    coverage = model_coverage(resp)
    coverage.to_csv(out_dir / "model_coverage.csv", index=False)

    Y, W, model_ids, cell_ids = to_matrix_with_mask(resp)
    n_obs = int(W.sum())
    print(
        f"[pooled] union fit: {len(model_ids)} models x {len(cell_ids)} cells "
        f"({n_obs} observed responses)"
    )

    fit = _fit_mfvi(
        Y,
        model_ids,
        cell_ids,
        model="1pl",
        seed=args.seed,
        n_steps=args.n_steps,
        hierarchical=True,
        weights=W,
        item_granularity="cell",
        engine="pooled_3dataset",
    )

    ability = fit.ability_table()
    ability = ability.merge(coverage[["model_id", "n_responses", "n_cells", "n_datasets", "datasets"]], on="model_id")
    ability.to_csv(out_dir / "ability_ranking.csv", index=False)
    fit.item_table().to_csv(out_dir / "item_params.csv", index=False)
    resp.to_parquet(out_dir / "responses.parquet", index=False)

    run_name = "pooled_3dataset (lung + pbmc + immune)"
    _plot_ability_and_accuracy_forest(ability, resp, run_name, out_dir)

    lines = [
        f"# {run_name}",
        "",
        f"- runs: `{cfg['lung_run']}`, `{cfg['pbmc_run']}`, `{cfg['immune_run']}`",
        f"- models: {len(model_ids)} (union; missing pairs masked in likelihood)",
        f"- cells: {len(cell_ids)}",
        f"- observed responses: {n_obs}",
        f"- IRT: 1PL Bayesian VI, {args.n_steps} steps",
        "",
        "## Ability ranking",
        "",
        _df_to_md_table(ability),
        "",
        "## Coverage",
        "",
        _df_to_md_table(coverage),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines))

    print(f"[pooled] wrote {out_dir}/ability_ranking.csv")
    print(ability[["model_id", "ability", "sd", "n_cells", "datasets"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
