"""Build correctness tensor with release-date metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import scanpy as sc
import yaml

REPO = Path(__file__).resolve().parents[2]


def load_release_dates() -> dict:
    path = Path(__file__).resolve().parent / "release_dates.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def _model_dates(dates_cfg: dict) -> dict[str, pd.Timestamp]:
    out = {}
    for mid, info in dates_cfg.get("models", {}).items():
        out[mid] = pd.Timestamp(info["release_date"])
    return out


def _lung_substudy_dates(dataset_ids: list[str], dates_cfg: dict) -> dict[str, pd.Timestamp]:
    proxy = dates_cfg.get("lung_substudy_proxy", {})
    start = pd.Timestamp(proxy.get("start", "2018-01-01"))
    end = pd.Timestamp(proxy.get("end", "2023-05-01"))
    ids = sorted(dataset_ids)
    if len(ids) <= 1:
        return {ids[0]: end} if ids else {}
    span = (end - start).days
    step = span / max(len(ids) - 1, 1)
    return {did: start + pd.Timedelta(days=int(round(i * step))) for i, did in enumerate(ids)}


def load_responses_from_run(run_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(run_dir / "responses.parquet")


def attach_cell_metadata(resp: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Add source_dataset, release_date, true_label from h5ad where available."""
    dates_cfg = load_release_dates()
    ds_dates = {
        k: pd.Timestamp(v["release_date"]) for k, v in dates_cfg.get("datasets", {}).items()
    }
    out = resp.copy()
    out["source_dataset"] = dataset_name
    out["true_label"] = out["cell_type"]

    h5ad_path = REPO / "data" / "raw" / f"{dataset_name}.h5ad"
    if h5ad_path.exists() and dataset_name == "lung_atlas":
        adata = sc.read_h5ad(h5ad_path, backed="r")
        obs = adata.obs.copy()
        obs.index = obs.index.astype(str)
        cell_to_ds = obs["dataset_id"].astype(str).to_dict() if "dataset_id" in obs.columns else {}
        sub_dates = _lung_substudy_dates(sorted(set(cell_to_ds.values())), dates_cfg)
        bare = out["cell_id"].str.replace(f"{dataset_name}:", "", regex=False)
        out["sub_study"] = bare.map(lambda c: cell_to_ds.get(c, "unknown"))
        out["release_date"] = out["sub_study"].map(sub_dates).fillna(ds_dates[dataset_name])
    else:
        out["sub_study"] = dataset_name
        out["release_date"] = ds_dates.get(dataset_name, pd.Timestamp("2020-01-01"))

    model_dates = _model_dates(dates_cfg)
    out["model_release_date"] = out["model_id"].map(model_dates)
    return out


def build_correctness_tensor(
    *,
    lung_run: Path | None = None,
    pbmc_run: Path | None = None,
    immune_run: Path | None = None,
    oasis_run: Path | None = None,
    human_thymus_run: Path | None = None,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Assemble tidy correctness table from completed ranking runs."""
    lung_run = lung_run or REPO / "runs" / "ranking_lung_atlas_v3_partial"
    pbmc_run = pbmc_run or REPO / "runs" / "ranking_pbmc3k_v2"

    run_specs: list[tuple[str, Path | None]] = [
        ("lung_atlas", lung_run),
        ("pbmc3k", pbmc_run),
        ("immune_human", immune_run),
        ("oasis", oasis_run),
        ("human_thymus", human_thymus_run),
    ]

    specs: list[tuple[str, Path]] = []
    for ds_name, run_dir in run_specs:
        if run_dir is not None and (Path(run_dir) / "responses.parquet").exists():
            specs.append((ds_name, Path(run_dir)))

    if not specs:
        raise FileNotFoundError("No responses.parquet found for any configured ranking run")

    shared_models: set[str] | None = None
    for _, run_dir in specs:
        models = set(load_responses_from_run(run_dir)["model_id"].unique())
        shared_models = models if shared_models is None else shared_models & models

    if not shared_models:
        raise ValueError("No shared models across configured ranking runs")

    frames: list[pd.DataFrame] = []
    for ds_name, run_dir in specs:
        r = load_responses_from_run(run_dir)
        r = r.loc[r["model_id"].isin(shared_models)]
        frames.append(attach_cell_metadata(r, ds_name))

    out = pd.concat(frames, ignore_index=True)
    out["release_date"] = pd.to_datetime(out["release_date"])
    out["model_release_date"] = pd.to_datetime(out["model_release_date"])

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lung-run", default="runs/ranking_lung_atlas_v3_partial")
    parser.add_argument("--pbmc-run", default="runs/ranking_pbmc3k_v2")
    parser.add_argument("--immune-run", default=None)
    parser.add_argument("--oasis-run", default=None)
    parser.add_argument("--human-thymus-run", default=None)
    parser.add_argument("--out", default="sim/data/correctness.parquet")
    args = parser.parse_args()
    df = build_correctness_tensor(
        lung_run=REPO / args.lung_run,
        pbmc_run=REPO / args.pbmc_run,
        immune_run=REPO / args.immune_run if args.immune_run else None,
        oasis_run=REPO / args.oasis_run if args.oasis_run else None,
        human_thymus_run=REPO / args.human_thymus_run if args.human_thymus_run else None,
        out_path=REPO / args.out,
    )
    print(f"wrote {len(df)} rows, models={df.model_id.nunique()}, cells={df.cell_id.nunique()}")
