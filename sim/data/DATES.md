# Release dates (sources)

Collected for the online benchmark-maintenance simulation (`docs/SIMULATION_PLAN.md`).

## Models (11 embedders in `ranking_lung_atlas_v4`, excl. cellplm in rollout)

| model_id | release_date | source |
|---|---|---|
| dev_pca, dev_random | 2010-01-01 (proxy) | Internal baselines |
| geneformer | 2023-05-31 | [Nature 2023](https://doi.org/10.1038/s41586-023-06139-9) |
| uce | 2023-11-29 | [bioRxiv 2023.11.28.568918](https://doi.org/10.1101/2023.11.28.568918) |
| cellplm | 2023-10-05 | [bioRxiv 2023.10.03.560734](https://doi.org/10.1101/2023.10.03.560734) |
| scgpt | 2024-02-26 | [Nature Methods 2024](https://doi.org/10.1038/s41592-024-02201-0) |
| scimilarity | 2024-11-20 | [Nature 2024](https://doi.org/10.1038/s41586-024-08411-y) |
| geneformer_v2 | 2024-12-01 | HF Geneformer V2 weights (Dec 2024) |
| geneformer_v2_316m | 2024-12-01 | Same V2 generation |
| transcriptformer | 2025-04-29 | [bioRxiv 2025.04.25.650731](https://doi.org/10.1101/2025.04.25.650731) |
| state | 2025-06-23 | [Arc State launch](https://arcinstitute.org/news/virtual-cell-model-state); [SE-600M HF](https://huggingface.co/arcinstitute/SE-600M) |

## Datasets

| dataset | release_date | source |
|---|---|---|
| pbmc3k | 2017-01-16 | [Zheng et al. Nat Commun 2017](https://doi.org/10.1038/ncomms14049) |
| lung_atlas (HLCA) | 2023-06-08 | [Sikkema et al. Nat Med 2023](https://doi.org/10.1038/s41591-023-02327-2) |
| immune_human | 2022-04-01 (proxy) | scIB immune HVG subset (scverse mirror) |
| oasis | 2025-07-28 | [Edahiro et al. Nat Genet 2025](https://doi.org/10.1038/s41588-025-02266-3); NBDC hum0197 (controlled) |
| human_thymus | 2024-11-20 | [Yayon et al. Nature 2024](https://doi.org/10.1038/s41586-024-07944-6); CELLxGENE Census |

## Lung sub-studies (`dataset_id` in h5ad)

HLCA core uses 45 integrated source datasets (UUID `dataset_id`). Per-study publication dates
were not curated here; the simulation assigns **proxy dates** evenly spaced 2018-01-01 →
2023-05-01 ordered by `dataset_id` string (deterministic). Replace with GEO dates in
`release_dates.yaml` when available.

## Data availability checks (SIMULATION_PLAN §3)

1. **Per-cell correctness:** yes — `responses.parquet` has `correct` per (model, cell).
2. **Model release dates:** yes — table above (`drift_mode=real` supported).
3. **Shared models across datasets:** partial — pbmc3k v2 has 4 models overlapping lung v3
   (dev_pca, uce, scgpt, geneformer). M0 linking uses these four; full sim uses lung atlas.
4. **Sub-study dates:** proxy only for lung `dataset_id` (see above).
