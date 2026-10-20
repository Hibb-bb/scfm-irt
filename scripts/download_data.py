#!/usr/bin/env python
"""Download registered datasets to `data/raw/` and verify them.

python scripts/download_data.py --dataset pbmc3k
python scripts/download_data.py --dataset smoke_toy
python scripts/download_data.py --all --family annotation
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

from scfm_irt.data.h5ad_compat import harmonize_var_names, sanitize_adata, write_h5ad_compat
from scfm_irt.data.registry import get_spec, list_datasets


def download(name: str, force: bool = False) -> None:
    spec = get_spec(name)
    dest = Path(spec.path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        print(f"[download] {name}: already at {dest}")
        return

    if name == "smoke_toy":
        import runpy

        script = Path(__file__).resolve().parent / "make_smoke_data.py"
        runpy.run_path(str(script), run_name="__not_main__")
        # Call the writer directly for a clean path.
        from make_smoke_data import make_smoke_adata  # type: ignore

        adata = make_smoke_adata()
        adata.write_h5ad(dest)
        print(f"[download] wrote {dest}  ({adata.n_obs} x {adata.n_vars})")
        return

    if name == "pbmc3k":
        _download_pbmc3k(dest)
        return

    if name == "immune_human":
        _download_immune_human(dest)
        return

    if name == "immune_human_full":
        _download_immune_human_full(dest)
        return

    if name == "lung_atlas":
        _download_lung_atlas(dest)
        return

    if name == "oasis":
        _download_oasis(dest)
        return

    if name == "human_thymus":
        _download_human_thymus(dest)
        return

    if spec.url is None:
        raise ValueError(f"{name}: no url registered; add one in data/registry.py")

    print(f"[download] {name}: GET {spec.url}")
    tmp = dest.with_suffix(dest.suffix + ".partial")
    urllib.request.urlretrieve(spec.url, tmp)  # noqa: S310 — curated registry URLs only
    tmp.replace(dest)
    print(f"[download] wrote {dest}")


def _download_pbmc3k(dest: Path) -> None:
    """10x PBMC 3k via scanpy, with cluster labels as cell types."""
    import scanpy as sc

    print("[download] pbmc3k: scanpy.datasets.pbmc3k_processed() for labels + expression")
    # Processed has louvain labels; expression is already filtered/normalized in some
    # scanpy versions — prefer raw counts from pbmc3k() merged with labels.
    raw = sc.datasets.pbmc3k()
    proc = sc.datasets.pbmc3k_processed()

    if "gene_symbols" in raw.var.columns:
        raw.var_names = raw.var["gene_symbols"].astype(str)
        raw.var_names_make_unique()

    # Barcode alignment: processed often drops a few cells.
    common = raw.obs_names.intersection(proc.obs_names)
    if len(common) < 500:
        # Fall back to processed matrix alone.
        adata = proc.copy()
        label_col = "louvain" if "louvain" in adata.obs else adata.obs.columns[0]
        adata.obs["celltype"] = adata.obs[label_col].astype(str)
    else:
        adata = raw[common].copy()
        label_col = "louvain" if "louvain" in proc.obs else proc.obs.columns[0]
        adata.obs["celltype"] = proc.obs.loc[common, label_col].astype(str)

    adata.obs["donor"] = [f"donor_{i % 8}" for i in range(adata.n_obs)]
    adata.obs["batch"] = [f"batch_{i % 3}" for i in range(adata.n_obs)]
    write_h5ad_compat(adata, dest)
    print(f"[download] wrote {dest}  ({adata.n_obs} x {adata.n_vars})")


def _urlretrieve(url: str, dest: Path) -> None:
    """Download with a User-Agent (scverse/Figshare often block bare urllib)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (scfm-irt)"})
    with urllib.request.urlopen(req, timeout=600) as resp, dest.open("wb") as out:  # noqa: S310
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)


def _download_immune_human(dest: Path) -> None:
    """scIB immune HVG subset via scverse example-data mirror."""
    import scanpy as sc

    url = get_spec("immune_human").url
    assert url is not None
    print(f"[download] immune_human: {url}")
    tmp = dest.with_suffix(".partial.h5ad")
    _urlretrieve(url, tmp)
    adata = sc.read_h5ad(tmp)
    tmp.unlink(missing_ok=True)
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"]
    if "sample_ID" not in adata.obs.columns and "donor" in adata.obs.columns:
        adata.obs["sample_ID"] = adata.obs["donor"].astype(str)
    write_h5ad_compat(sanitize_adata(adata), dest)
    print(f"[download] wrote {dest}  ({adata.n_obs} x {adata.n_vars})")


def _download_immune_human_full(dest: Path) -> None:
    """Full-transcriptome scIB immune atlas (Immune_ALL_human.h5ad).

    Same cells as immune_human (HVG subset); required for SCimilarity (>=5k genes)
    and TranscriptFormer (Ensembl gene mapping).
    """
    import scanpy as sc

    url = "https://exampledata.scverse.org/scvi-tools/Immune_ALL_human.h5ad"
    print(f"[download] immune_human_full: {url}")
    tmp = dest.with_suffix(".partial.h5ad")
    _urlretrieve(url, tmp)
    adata = sc.read_h5ad(tmp)
    tmp.unlink(missing_ok=True)
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"]
    if "sample_ID" not in adata.obs.columns and "donor" in adata.obs.columns:
        adata.obs["sample_ID"] = adata.obs["donor"].astype(str)
    write_h5ad_compat(sanitize_adata(adata), dest)
    print(f"[download] wrote {dest}  ({adata.n_obs} x {adata.n_vars})")


def _resolve_oasis_source() -> Path | None:
    """Locate a staged OASIS h5ad (controlled access — no public URL)."""
    env = os.environ.get("OASIS_H5AD_PATH", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
        raise FileNotFoundError(f"OASIS_H5AD_PATH set but not found: {p}")

    repo = Path(__file__).resolve().parents[1]
    for pattern in (
        "data/staging/oasis*.h5ad",
        "data/staging/OASIS*.h5ad",
        "data/raw/oasis_full.h5ad",
    ):
        matches = sorted(repo.glob(pattern))
        if matches:
            return matches[0]
    return None


def _normalize_oasis_adata(adata, *, max_cells: int = 50_000, seed: int = 0):
    """Map OASIS obs columns and subsample for ranking."""
    import numpy as np
    import scanpy as sc

    label_candidates = (
        "cell_type",
        "cell_type_L2",
        "L2",
        "manual_cell_type",
        "celltype_l2",
        "celltype",
        "final_annotation",
    )
    label_col = next((c for c in label_candidates if c in adata.obs.columns), None)
    if label_col is None:
        raise KeyError(
            "oasis: no cell-type column found in .obs; expected one of "
            f"{label_candidates}. Got: {list(adata.obs.columns)}"
        )
    adata.obs["cell_type"] = adata.obs[label_col].astype(str)

    donor_candidates = ("donor", "donor_id", "subject_id", "Subject", "sample_id", "sample_ID")
    donor_col = next((c for c in donor_candidates if c in adata.obs.columns), None)
    if donor_col is None:
        adata.obs["donor"] = [f"donor_{i % 235}" for i in range(adata.n_obs)]
    else:
        adata.obs["donor"] = adata.obs[donor_col].astype(str)

    batch_candidates = ("batch", "library_id", "sample_id", "sample_ID", "orig.ident")
    batch_col = next((c for c in batch_candidates if c in adata.obs.columns), None)
    adata.obs["batch"] = (
        adata.obs[batch_col].astype(str) if batch_col else adata.obs["donor"].astype(str)
    )

    if "counts" in adata.layers:
        adata.X = adata.layers["counts"]
    elif "raw_counts" in adata.layers:
        adata.X = adata.layers["raw_counts"]

    n_take = min(max_cells, adata.n_obs)
    if n_take < adata.n_obs:
        rng = np.random.default_rng(seed)
        groups = adata.obs["cell_type"].astype(str)
        picks: list[int] = []
        for ct in groups.unique():
            idx = np.flatnonzero(groups.to_numpy() == ct)
            n_ct = max(1, int(round(len(idx) * n_take / adata.n_obs)))
            n_ct = min(n_ct, len(idx))
            picks.extend(rng.choice(idx, size=n_ct, replace=False).tolist())
        if len(picks) > n_take:
            picks = rng.choice(np.array(picks), size=n_take, replace=False).tolist()
        elif len(picks) < n_take:
            remaining = np.setdiff1d(np.arange(adata.n_obs), np.array(picks))
            extra = min(n_take - len(picks), len(remaining))
            if extra:
                picks.extend(rng.choice(remaining, size=extra, replace=False).tolist())
        adata = adata[picks].copy()
        print(f"[download] oasis: stratified subsample -> {adata.n_obs} cells")

    adata = harmonize_var_names(adata)
    return sanitize_adata(adata)


def _download_oasis(dest: Path, *, max_cells: int = 50_000, seed: int = 0) -> None:
    """Prepare OASIS from a staged h5ad (NBDC/JGA controlled access).

    OASIS does not provide a public processed matrix. After NBDC approval (hum0197) or receiving
    an author h5ad, place it at ``data/staging/oasis.h5ad`` or set ``OASIS_H5AD_PATH``.
    """
    import scanpy as sc

    source = _resolve_oasis_source()
    if source is None:
        raise RuntimeError(
            "oasis: no staged h5ad found.\n"
            "OASIS expression data requires controlled access:\n"
            "  • NBDC hum0197: https://humandbs.dbcls.jp/en/hum0197-v25\n"
            "  • JGA JGAS000783 / JGAD000925\n"
            "  • JOB browser: https://japan-omics.jp/downloads\n"
            "After download, stage the processed h5ad at data/staging/oasis.h5ad\n"
            "or set OASIS_H5AD_PATH=/path/to/oasis.h5ad and re-run:\n"
            "  python scripts/download_data.py --dataset oasis"
        )

    print(f"[download] oasis: normalizing staged file {source}")
    adata = sc.read_h5ad(source)
    adata = _normalize_oasis_adata(adata, max_cells=max_cells, seed=seed)
    write_h5ad_compat(adata, dest)
    print(
        f"[download] wrote {dest}  ({adata.n_obs} x {adata.n_vars}, "
        f"{adata.obs['cell_type'].nunique()} types, {adata.obs['donor'].nunique()} donors)"
    )


def _download_human_thymus(dest: Path, *, max_cells: int = 50_000, seed: int = 0) -> None:
    """Developing human thymus atlas (Yayon et al. Nature 2024) via CELLxGENE Census."""
    try:
        import cellxgene_census
    except ImportError as exc:
        raise ImportError(
            "human_thymus download requires cellxgene-census. "
            "Run: python scripts/install_census_deps.py"
        ) from exc
    import numpy as np
    import scanpy as sc

    census_version = "2025-11-08"
    dataset_id = "29244e1d-02e6-4133-b411-516ef7474638"
    repo = Path(__file__).resolve().parents[1]
    staging = repo / "data" / "staging" / f"{dataset_id}.h5ad"
    staging.parent.mkdir(parents=True, exist_ok=True)

    if not staging.exists():
        print(f"[download] human_thymus: fetching source h5ad -> {staging}")
        cellxgene_census.download_source_h5ad(
            dataset_id, str(staging), census_version=census_version, progress_bar=True
        )
    else:
        print(f"[download] human_thymus: reusing staged source {staging}")

    print("[download] human_thymus: stratified subsample from backed h5ad")
    adata = sc.read_h5ad(staging, backed="r")
    obs = adata.obs.copy()
    label_col = next(
        (c for c in ("cell_type", "celltype", "author_cell_type") if c in obs.columns),
        None,
    )
    if label_col is None:
        raise KeyError(f"human_thymus: no cell-type column in {list(obs.columns)}")
    obs["cell_type"] = obs[label_col].astype(str)

    n_total = adata.n_obs
    n_take = min(max_cells, n_total)
    rng = np.random.default_rng(seed)
    groups = obs["cell_type"]
    pick_idx: list[int] = []
    for ct in groups.unique():
        idx = np.flatnonzero(groups.to_numpy() == ct)
        n_ct = max(1, int(round(len(idx) * n_take / n_total)))
        n_ct = min(n_ct, len(idx))
        pick_idx.extend(rng.choice(idx, size=n_ct, replace=False).tolist())
    if len(pick_idx) > n_take:
        pick_idx = rng.choice(np.array(pick_idx), size=n_take, replace=False).tolist()
    elif len(pick_idx) < n_take:
        remaining = np.setdiff1d(np.arange(n_total), np.array(pick_idx))
        extra = min(n_take - len(pick_idx), len(remaining))
        if extra:
            pick_idx.extend(rng.choice(remaining, size=extra, replace=False).tolist())

    subset = adata[pick_idx].to_memory()
    adata.file.close()

    subset.obs["cell_type"] = subset.obs[label_col].astype(str)
    donor_col = next((c for c in ("donor_id", "donor", "subject_id") if c in subset.obs.columns), None)
    subset.obs["donor"] = (
        subset.obs[donor_col].astype(str) if donor_col else subset.obs_names.astype(str)
    )
    batch_col = next((c for c in ("dataset_id", "batch", "sample_id") if c in subset.obs.columns), None)
    subset.obs["batch"] = (
        subset.obs[batch_col].astype(str) if batch_col else subset.obs["donor"].astype(str)
    )

    if "counts" in subset.layers:
        subset.X = subset.layers["counts"]
    elif "counts" not in subset.layers:
        subset.layers["counts"] = subset.X.copy()
    subset = harmonize_var_names(subset)
    write_h5ad_compat(sanitize_adata(subset), dest)
    print(
        f"[download] wrote {dest}  ({subset.n_obs} x {subset.n_vars}, "
        f"{subset.obs['cell_type'].nunique()} types, {subset.obs['donor'].nunique()} donors)"
    )


def _download_lung_atlas(dest: Path, *, max_cells: int = 50_000, seed: int = 0) -> None:
    """HLCA lung subsample via CELLxGENE Census (metadata-first to avoid OOM)."""
    try:
        import cellxgene_census
    except ImportError as exc:
        raise ImportError(
            "lung_atlas download requires cellxgene-census. "
            "Run: python scripts/install_census_deps.py"
        ) from exc
    import numpy as np

    census_version = "2025-11-08"
    filter_expr = "is_primary_data == True and tissue_general == 'lung'"
    print(f"[download] lung_atlas: querying obs from Census ({census_version})")
    with cellxgene_census.open_soma(census_version=census_version) as census:
        obs = cellxgene_census.get_obs(
            census,
            "Homo sapiens",
            value_filter=filter_expr,
            column_names=["soma_joinid", "cell_type", "donor_id", "dataset_id"],
        )
        n_total = len(obs)
        n_take = min(max_cells, n_total)
        rng = np.random.default_rng(seed)
        pick = rng.choice(obs["soma_joinid"].to_numpy(), size=n_take, replace=False)
        print(f"[download] lung_atlas: fetching {n_take}/{n_total} cells x full genes")
        adata = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            obs_coords=pick.tolist(),
            obs_column_names=["cell_type", "donor_id", "dataset_id"],
        )

    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
    adata.obs["batch"] = adata.obs.get("dataset_id", "batch").astype(str)
    adata.obs["donor"] = adata.obs.get("donor_id", adata.obs_names).astype(str)

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    adata = harmonize_var_names(adata)
    write_h5ad_compat(sanitize_adata(adata), dest)
    print(
        f"[download] wrote {dest}  ({adata.n_obs} x {adata.n_vars}, "
        f"{adata.obs['cell_type'].nunique()} types)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--family", choices=["annotation", "atlas"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    names = list_datasets(args.family) if args.all else args.dataset
    if not names:
        parser.error("pass --dataset NAME (repeatable) or --all")

    # Allow importing sibling scripts when running as a file.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    for name in names:
        print(f"[download] {name}")
        download(name, force=args.force)
        try:
            from scfm_irt.data.describe import describe_by_name

            info = describe_by_name(name)
            print(f"  n_cells={info['n_cells']} n_types={info['n_cell_types']}")
        except Exception as exc:
            print(f"  describe deferred: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
