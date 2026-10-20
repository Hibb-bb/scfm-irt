#!/usr/bin/env python
"""Create a tiny synthetic AnnData for the smoke path (no external download).

    python scripts/make_smoke_data.py

Writes `data/raw/smoke_toy.h5ad` with a handful of cell types, donors, and genes — enough to
exercise splits, DevEmbedder, probe, response matrix, and IRT without pulling Zheng68k.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_smoke_adata(
    n_cells: int = 400,
    n_genes: int = 200,
    n_types: int = 5,
    n_donors: int = 8,
    seed: int = 0,
):
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(seed)
    types = np.array([f"type_{i}" for i in range(n_types)])
    donors = np.array([f"donor_{i}" for i in range(n_donors)])

    # Uneven class sizes (mild imbalance).
    weights = np.linspace(1.0, 0.3, n_types)
    weights = weights / weights.sum()
    cell_type = rng.choice(types, size=n_cells, p=weights)
    donor = rng.choice(donors, size=n_cells)

    # Type-specific expression + donor noise so a linear probe has something to fit.
    type_means = rng.normal(size=(n_types, n_genes))
    type_idx = np.array([int(t.split("_")[1]) for t in cell_type])
    rate = np.exp(0.5 * type_means[type_idx] + 0.1 * rng.normal(size=(n_cells, n_genes)))
    X = rng.poisson(rate).astype(np.float32)

    obs = pd.DataFrame(
        {
            "celltype": cell_type,
            "donor": donor,
            "batch": np.array([f"batch_{int(d.split('_')[1]) % 3}" for d in donor]),
        },
        index=[f"cell_{i:04d}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene_{i:04d}" for i in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/smoke_toy.h5ad")
    parser.add_argument("--n-cells", type=int, default=400)
    parser.add_argument("--n-genes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    adata = make_smoke_adata(n_cells=args.n_cells, n_genes=args.n_genes, seed=args.seed)
    adata.write_h5ad(out)
    print(f"[smoke-data] wrote {out}  ({adata.n_obs} cells x {adata.n_vars} genes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
