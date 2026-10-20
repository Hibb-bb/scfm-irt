"""End-to-end orchestration: config in, run directory out.

The only place the six modules are wired together. Note what this function does *not* do: it
never imports an embedder. It reads cached embedding artifacts produced by
`scripts/extract_embeddings.py`, which runs separately in each model's isolated environment
(plan section 2.1).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scfm_irt.analysis.loo_calibration import write_loo_calibration_outputs
from scfm_irt.analysis.report import write_report, write_run_provenance
from scfm_irt.config import RunConfig
from scfm_irt.data.loaders import load_dataset
from scfm_irt.data.registry import get_spec
from scfm_irt.data.splits import make_split
from scfm_irt.embedders.base import load_embedding
from scfm_irt.irt.fit import fit_irt
from scfm_irt.probe.logreg import fit_probe, probe_summary, score_cells
from scfm_irt.response.build import build_response_table, make_cell_id


def run_pipeline(cfg: RunConfig) -> None:
    """Run data -> cached embeddings -> probe -> response table -> IRT -> report.

    Steps, each delegating to its module:
      1. resolve datasets from the registry and build splits (`data`)
      2. load cached embeddings for every (embedder, dataset, split) — fail loudly if a cache
         entry is missing rather than recomputing, since recompute needs the heavy env
      3. fit a probe per (embedder, dataset) on train cells, score test cells (`probe`)
      4. assemble responses.parquet (`response`)
      5. fit IRT (`irt`)
      6. emit tables, figures, and the summary report (`analysis`)

    Writes the resolved config and library versions into `cfg.run_dir` for reproducibility.
    """
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_provenance(cfg, run_dir)

    score_frames: list[pd.DataFrame] = []
    probe_logs: list[dict] = []

    for dataset_name in cfg.datasets:
        spec = get_spec(dataset_name)
        adata = load_dataset(spec)
        split = make_split(
            adata,
            spec,
            scheme=cfg.split.scheme,
            seed=cfg.seed,
            test_frac=cfg.split.test_frac,
        )
        labels = np.asarray(adata.obs[spec.label_key].astype(str))
        obs_names = np.asarray(adata.obs_names)

        for embedder_name in cfg.embedders:
            emb_all, cell_ids, meta = _load_cached_embedding(
                cfg, embedder_name, dataset_name, split.scheme
            )
            id_to_row = {str(c): i for i, c in enumerate(cell_ids)}

            train_rows = np.array([id_to_row[str(c)] for c in split.train_ids])
            test_rows = np.array([id_to_row[str(c)] for c in split.test_ids])
            # Map obs positions for labels via obs_names
            name_to_obs = {str(n): i for i, n in enumerate(obs_names)}
            y_train = labels[np.array([name_to_obs[str(c)] for c in split.train_ids])]
            y_test = labels[np.array([name_to_obs[str(c)] for c in split.test_ids])]

            probe = fit_probe(
                emb_all[train_rows],
                y_train,
                C=cfg.probe.C,
                penalty=cfg.probe.penalty,
                seed=cfg.seed,
                max_iter=cfg.probe.max_iter,
            )
            scores = score_cells(probe, emb_all[test_rows], y_test)
            summary = probe_summary(scores)
            probe_logs.append(
                {
                    "embedder": embedder_name,
                    "dataset": dataset_name,
                    "C": probe.C,
                    **summary,
                }
            )

            frame = scores.copy()
            frame["model_id"] = embedder_name
            frame["dataset"] = dataset_name
            frame["split_scheme"] = split.scheme
            frame["cell_id"] = [make_cell_id(dataset_name, str(c)) for c in split.test_ids]
            frame["cell_type"] = y_test
            score_frames.append(
                frame[
                    [
                        "model_id",
                        "dataset",
                        "split_scheme",
                        "cell_id",
                        "cell_type",
                        "correct",
                        "y_true",
                        "y_pred",
                        "p_true",
                    ]
                ]
            )
            del meta  # used via load for validation

    resp = build_response_table(
        [
            f[
                [
                    "model_id",
                    "dataset",
                    "split_scheme",
                    "cell_id",
                    "cell_type",
                    "correct",
                ]
            ]
            for f in score_frames
        ]
    )
    resp.to_parquet(run_dir / "responses.parquet", index=False)
    pd.DataFrame(probe_logs).to_csv(run_dir / "probe_summaries.csv", index=False)

    fit = fit_irt(
        resp,
        model=cfg.irt.model,
        method=cfg.irt.method,
        item_granularity=cfg.irt.item_granularity,
        item_covariates=cfg.irt.item_covariates,
        seed=cfg.seed,
        n_steps=cfg.irt.n_steps,
    )

    loo_paths: dict[str, Path] | None = None
    if cfg.irt.loo_calibration and len(fit.model_ids) >= 3:
        print(
            f"[pipeline] LOO calibration ({len(fit.model_ids)} models, "
            f"{cfg.irt.loo_n_steps} steps each holdout)..."
        )
        loo_paths = write_loo_calibration_outputs(
            resp,
            fit,
            run_dir,
            n_steps=cfg.irt.loo_n_steps,
            seed=cfg.seed,
        )
        loo_ability = pd.read_csv(loo_paths["ability"])
        print("[pipeline] LOO ability ranking:")
        print(loo_ability[["rank", "model_id", "ability", "accuracy"]].to_string(index=False))

    report = write_report(fit, resp, cfg, run_dir, loo_paths=loo_paths)
    fit.ability_table().to_csv(run_dir / "ability_ranking.csv", index=False)
    (run_dir / "irt_diagnostics.json").write_text(
        json.dumps({k: v for k, v in fit.diagnostics.items() if k != "ll_history"}, indent=2) + "\n"
    )
    print(f"[pipeline] wrote report -> {report}")
    print(fit.ability_table().to_string(index=False))


def build_responses(cfg: RunConfig) -> pd.DataFrame:
    """Rebuild probe response table from cached embeddings (no embedder imports)."""
    score_frames: list[pd.DataFrame] = []

    for dataset_name in cfg.datasets:
        spec = get_spec(dataset_name)
        adata = load_dataset(spec)
        split = make_split(
            adata,
            spec,
            scheme=cfg.split.scheme,
            seed=cfg.seed,
            test_frac=cfg.split.test_frac,
        )
        labels = np.asarray(adata.obs[spec.label_key].astype(str))
        obs_names = np.asarray(adata.obs_names)

        for embedder_name in cfg.embedders:
            emb_all, cell_ids, _meta = _load_cached_embedding(
                cfg, embedder_name, dataset_name, split.scheme
            )
            id_to_row = {str(c): i for i, c in enumerate(cell_ids)}

            train_rows = np.array([id_to_row[str(c)] for c in split.train_ids])
            test_rows = np.array([id_to_row[str(c)] for c in split.test_ids])
            name_to_obs = {str(n): i for i, n in enumerate(obs_names)}
            y_train = labels[np.array([name_to_obs[str(c)] for c in split.train_ids])]
            y_test = labels[np.array([name_to_obs[str(c)] for c in split.test_ids])]

            probe = fit_probe(
                emb_all[train_rows],
                y_train,
                C=cfg.probe.C,
                penalty=cfg.probe.penalty,
                seed=cfg.seed,
                max_iter=cfg.probe.max_iter,
            )
            scores = score_cells(probe, emb_all[test_rows], y_test)

            frame = scores.copy()
            frame["model_id"] = embedder_name
            frame["dataset"] = dataset_name
            frame["split_scheme"] = split.scheme
            frame["cell_id"] = [make_cell_id(dataset_name, str(c)) for c in split.test_ids]
            frame["cell_type"] = y_test
            score_frames.append(
                frame[
                    [
                        "model_id",
                        "dataset",
                        "split_scheme",
                        "cell_id",
                        "cell_type",
                        "correct",
                        "y_true",
                        "y_pred",
                        "p_true",
                    ]
                ]
            )

    return build_response_table(
        [
            f[
                [
                    "model_id",
                    "dataset",
                    "split_scheme",
                    "cell_id",
                    "cell_type",
                    "correct",
                ]
            ]
            for f in score_frames
        ]
    )


def _load_cached_embedding(
    cfg: RunConfig,
    embedder_name: str,
    dataset_name: str,
    split_scheme: str,
):
    """Find a matching cache entry; fail with a precise extract_embeddings invocation if missing."""
    cache_root = Path(cfg.cache_dir) / embedder_name / dataset_name
    if not cache_root.exists():
        _missing(embedder_name, dataset_name, split_scheme, cfg)

    candidates = sorted(cache_root.glob("*.npz"))
    for npz in candidates:
        try:
            emb, cell_ids, meta = load_embedding(npz)
        except Exception:
            continue
        if (
            meta.embedder == embedder_name
            and meta.dataset == dataset_name
            and meta.split_scheme == split_scheme
            and meta.split_seed == cfg.seed
        ):
            return emb, cell_ids, meta

    _missing(embedder_name, dataset_name, split_scheme, cfg)


def _missing(embedder: str, dataset: str, split: str, cfg: RunConfig) -> None:
    raise FileNotFoundError(
        f"missing embedding cache for embedder={embedder!r} dataset={dataset!r} "
        f"split={split!r} seed={cfg.seed}. Produce it with:\n"
        f"  python scripts/extract_embeddings.py --embedder {embedder} "
        f"--dataset {dataset} --split {split} --seed {cfg.seed} "
        f"--cache-dir {cfg.cache_dir}"
    )
