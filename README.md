# scfm-irt

Item Response Theory (IRT) evaluation of single-cell embedding models.

Each **embedding model is a respondent** with a latent ability `θ`, each **test cell is an
item** with a latent difficulty `b`, and the response is whether a frozen linear probe on that
model's embedding annotated that cell correctly. Fitting IRT over the resulting binary matrix
gives a model ranking *with uncertainty*, plus an estimate of what makes a cell hard to
annotate.

Full statistical spec: [docs/MATH.md](docs/MATH.md).
Architecture, module contracts, and roadmap: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

> **Status: smoke path working.** Phases 0–3 and the `mle_autodiff` estimator are implemented.
> The synthetic toy dataset + DevEmbedder path runs end-to-end (`configs/smoke_dev.yaml`,
> `slurm/smoke_dev.slurm`). Real foundation-model embedders and the Bayesian (`py-irt`)
> estimator are still TODO — see [docs/ROADMAP.md](docs/ROADMAP.md).

## The one boundary that matters

```
   HEAVY / GPU / one isolated env per model
   data ──▶ embedders ──▶ [cached embeddings on disk]
                              │  ← artifact boundary: the only thing crossing it
   LIGHT / CPU / a single env │
   probe ──▶ response matrix ──▶ irt ──▶ analysis
```

Foundation models (scGPT, Geneformer, UCE, SCimilarity, Cell-JEPA, …) have mutually
conflicting dependency stacks. They are never co-installed and never imported by the analysis
half — embedding extraction runs as a separate job in its own env and writes a cached artifact
(`envs/env-embedder-template.yml`). Everything downstream reads only those artifacts.

## Install

```bash
pip install -e ".[sc,dev]"
```

`pip install -e ".[irt]"` adds the Bayesian estimators (py-irt / numpyro / torch); the package
imports and the analysis path run without them.

## Quick start

```bash
# 1. install (analysis + scanpy for the smoke path)
pip install -e ".[sc,dev]"

# 2. build the tiny toy dataset, extract DevEmbedder caches, fit IRT
python scripts/make_smoke_data.py
python scripts/extract_embeddings.py --embedder dev_pca --dataset smoke_toy --split stratified --device cpu
python scripts/extract_embeddings.py --embedder dev_random --dataset smoke_toy --split stratified --device cpu
scfm-irt run --config configs/smoke_dev.yaml
```

Or on Quest:

```bash
mkdir -p logs
sbatch slurm/smoke_dev.slurm
```

A run is fully declared by its config — datasets, split scheme, embedders, probe settings, IRT
model and item granularity — and writes tables, figures, and a summary report under
`runs/<run_name>/`.

## Layout

```
src/scfm_irt/
  data/        dataset registry, loaders, splits, label harmonization
  embedders/   CellEmbedder protocol + one module per model (heavy side)
  probe/       multinomial logistic-regression probe
  response/    the tidy response table + matrix pivot (the bridge between halves)
  irt/         response models, synthetic recovery, three estimators
  analysis/    ability / difficulty / information / Wright map / fit stats / report
configs/       run configs (YAML)
scripts/       download_data.py  make_smoke_data.py  extract_embeddings.py  run_pipeline.py
tests/         synthetic recovery, orientation, split leakage
slurm/         smoke_dev.slurm  extract_embeddings.slurm  run_pipeline.slurm
envs/          isolated conda envs — one per embedder
docs/          MATH.md  IMPLEMENTATION_PLAN.md  ROADMAP.md
```

This uses a `src/` layout, with `configs/`, `scripts/`, and `tests/` at the repo root rather
than nested inside the package as sketched in the plan; module organization is otherwise as
specified.

## Two invariants worth memorizing

1. **Orientation.** `Y` is always `(P models, I cells)` — rows are respondents, columns are
   items. A transposition here silently corrupts every downstream number, so `to_matrix`
   asserts the shape and `tests/test_response_orientation.py` guards it.
2. **Regime.** Few respondents (`P` ≈ 5–15), many items (`I` ≈ 1e4–1e6). Abilities are precise;
   per-item difficulties are noisy and per-item discriminations more so. Prefer 1PL for the
   headline result, use hierarchical priors, and treat 2PL as exploratory
   ([MATH.md](docs/MATH.md) §10).

## Testing

```bash
pytest
```

The synthetic parameter-recovery test (`tests/test_synthetic_irt.py`, [MATH.md](docs/MATH.md)
§9) is the acceptance gate for the estimators — no real embedding model gets fit until it
passes.
