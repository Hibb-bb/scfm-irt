# Online benchmark maintenance simulation — specification & results

This document describes a single-scenario rollout simulation intended for
downstream formatting (e.g. Word). **Primary numerical results:** `metrics.csv`.

---

## 1. Purpose

Simulate **Algorithm 1 (full maintenance)** on the real lung atlas benchmark as:
- new **embedding models** enter by publication/release date, and
- new **cells** enter by sub-study proxy release dates.

Drift mode: **real** (oracle abilities from joint Rasch fit; no synthetic field rise).

---

## 2. Scenario (B only)

**Rollout mode:** `models_and_items`

- **Models:** non-baseline embedders activate when `model_release_date <= round_date`.
  Baseline models (`dev_pca`, `dev_random`) are excluded from the active field.
- **Cells:** a cell enters the available pool when its sub-study proxy `release_date <= round_date`.
- **Simulation clock starts** at the first embedder release (geneformer), not at the first cell date.
- **Round 0 bank:** all cells in the pool are seeded at once (`seed_full_pool=true`).

---

## 3. Maintenance algorithm (full arm)

Each round:

1. **Ask:** each active model answers anchors + routed trusted/provisional items (`k_ask=22000`, `n_exp=8`; when `k_ask >= |bank|`, all bank items are queried).
2. **Fit abilities:** MLE on anchor + trusted answers only (provisional excluded).
3. **Recalibrate:** Kalman filter update on difficulties for all asked non-anchor items.
4. **Promote:** provisional → trusted when uncertainty `P <= p_star`.
5. **Acquire:** up to `kappa` new cells/round — hardest real cells from pool, then synthetic frontier items.
6. **Retire:**
   - **PI rule:** score each non-anchor cell by mean Fisher information among the top `0.5` fraction of models (by fitted θ); drop the lowest `5%` each round.
   - Anchors are never retired.

**Retirement (plain language):** remove the worst-discriminating cells for the current top-half of models; each round culls the bottom X% by that score.

---

## 4. Configuration

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `rollout_n_rounds` | 11 | Number of simulation rounds |
| `bank_budget` | 400 | Initial bank seed size (cap only if retire_cap) |
| `seed_full_pool` | True | Seed entire pool at round 0 |
| `rollout_datasets` | ['lung_atlas', 'pbmc3k', 'immune_human'] | Datasets included in rollout |
| `kappa` | 0 | Cells acquired per round |
| `k_ask` | 22000 | Items asked per model per round |
| `n_exp` | 8 | Provisional items in each model's query |
| `n_anchors` | 15 | Fixed anchor cells (difficulty pinned) |
| `retire_v` | 0.01 | Retire if v_t(b) below this (v_threshold mode) |
| `retire_mode` | bottom_pct_top_half | Retirement policy |
| `retire_top_model_frac` | 0.5 | Top model fraction for PI retirement score |
| `retire_bottom_pct` | 0.05 | Bottom fraction of cells retired each round (PI mode) |
| `retire_cap` | False | Enforce bank_budget via cap drops |
| `field_window_top_frac` | 0.5 | Top fraction of models for μ/σ field window |
| `field_window_top_k` | None | Top-k models for μ/σ field window (if no fraction) |
| `p_star` | 0.25 | Promote provisional when P <= this |
| `start_at_first_model` | True | Start timeline at first embedder |
| `baseline_models` | ['dev_pca', 'dev_random'] | Excluded from active model field |
| `lung_run` | runs/ranking_lung_atlas_v5 | Ranking run for responses |
| `seed` | 0 | RNG seed |

**Excluded models:** cellplm

---

## 5. Models in oracle fit

| model_id | theta_star | theta_sd | release_date |
|----------|------------|----------|--------------|
| geneformer_v2_316m | 0.9426 | 0.1570 | 2024-12-01 |
| geneformer_v2 | 0.8674 | 0.1561 | 2024-12-01 |
| uce | 0.6848 | 0.1331 | 2023-11-29 |
| scgpt | 0.4351 | 0.1602 | 2024-02-26 |
| langcell | 0.2717 | 0.1475 | n/a |
| dev_pca | 0.0708 | 0.1442 | 2010-01-01 |
| state | 0.0522 | 0.1115 | 2025-06-23 |
| geneformer | -0.1585 | 0.1106 | 2023-05-31 |
| scprint2 | -0.3114 | 0.1179 | n/a |
| scprint | -0.4699 | 0.1193 | n/a |
| scprint_large_v1 | -0.5859 | 0.1187 | n/a |
| scprint_medium_v1 | -0.7102 | 0.0974 | n/a |
| scprint_small_v1 | -1.0887 | 0.1074 | n/a |

---

## 6. Output files

| File | Description |
|------|-------------|
| `metrics.csv` | **Main results** — one row per round (full maintenance arm) |
| `SIMULATION_SPEC.md` | This file |
| `rollout_report.pdf` | Optional visual report (single scenario) |

---

## 7. metrics.csv column dictionary

| Column | Description |
|--------|-------------|
| `round` | Round index (0-based) |
| `round_date` | Simulated calendar date for this round |
| `n_active_models` | Embedders active this round |
| `n_new_models` | Models newly activated this round |
| `new_models` | Comma-separated model IDs activated |
| `n_items_available` | Cells in the release pool |
| `n_items_new` | Cells newly entering pool this round |
| `bank_size` | Cells currently in the maintained bank |
| `n_acquired` | Cells added to bank this round |
| `n_acquired_pool` | Real cells pulled from pool |
| `n_acquired_synth` | Synthetic provisional cells created |
| `n_retired` | Total cells removed this round |
| `n_retired_value` | Removed for low v_t(b) |
| `n_retired_cap` | Removed to enforce bank_budget |
| `n_promoted` | Provisional → trusted promotions |
| `mu_oracle` | Mean oracle θ* over active embedders |
| `mu_hat` | Estimated field mean from fitted abilities |
| `s_hat` | Estimated field spread |
| `bank_value` | Sum of item values v_t(b) over bank |
| `bank_value_per_item` | bank_value / bank_size |
| `ranking_spearman` | Spearman ρ between θ̂ and oracle (active models) |
| `ranking_spearman_incumbent` | ρ among models active previous round |
| `seed_mean_value` | Mean v_t(b_seed) for initial bank difficulties |

---

## 8. Per-round results (from metrics.csv)

| round | round_date | n_active_models | n_new_models | new_models | n_items_available | n_items_new | bank_size | n_acquired | n_retired | n_retired_value | n_retired_cap | mu_oracle | ranking_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2023-05-31 | 1 | 1 | geneformer | 21275 | 21275 | 20212 | 0 | 1063 | 1063 | 0 | -0.159 | n/a |
| 1 | 2023-08-14 | 1 | 0 |  | 21275 | 0 | 19202 | 0 | 1010 | 1010 | 0 | -0.159 | n/a |
| 2 | 2023-10-28 | 1 | 0 |  | 21275 | 0 | 18242 | 0 | 960 | 960 | 0 | -0.159 | n/a |
| 3 | 2024-01-12 | 2 | 1 | uce | 21275 | 0 | 17330 | 0 | 912 | 912 | 0 | 0.685 | n/a |
| 4 | 2024-03-27 | 3 | 1 | scgpt | 21275 | 0 | 16464 | 0 | 866 | 866 | 0 | 0.560 | 1.000 |
| 5 | 2024-06-11 | 3 | 0 |  | 21275 | 0 | 15641 | 0 | 823 | 823 | 0 | 0.560 | 1.000 |
| 6 | 2024-08-25 | 3 | 0 |  | 21275 | 0 | 14859 | 0 | 782 | 782 | 0 | 0.560 | 1.000 |
| 7 | 2024-11-08 | 3 | 0 |  | 21275 | 0 | 14116 | 0 | 743 | 743 | 0 | 0.560 | 1.000 |
| 8 | 2025-01-23 | 5 | 2 | geneformer_v2,geneformer_v2_316m | 21275 | 0 | 13410 | 0 | 706 | 706 | 0 | 0.832 | 1.000 |
| 9 | 2025-04-08 | 5 | 0 |  | 21275 | 0 | 12740 | 0 | 670 | 670 | 0 | 0.832 | 1.000 |
| 10 | 2025-06-23 | 6 | 1 | state | 21275 | 0 | 12103 | 0 | 637 | 637 | 0 | 0.832 | 1.000 |

---

## 9. Summary statistics

- Simulation span: `2023-05-31 00:00:00` → `2025-06-23 00:00:00` (11 rounds)
- Final active embedders: 6
- Final cell pool size: 21275
- Final bank size: 12103 (budget 400)
- Total cells retired: 9172 (value: 9172, cap: 0)
- Total cells acquired: 0
- Field mean μ: 0.832 (started -0.159)
- Final ranking Spearman ρ: 1.000

---

## 10. Model launch rounds

- Round 0 (2023-05-31): geneformer
- Round 3 (2024-01-12): uce
- Round 4 (2024-03-27): scgpt
- Round 8 (2025-01-23): geneformer_v2,geneformer_v2_316m
- Round 10 (2025-06-23): state

---

## 11. Notes for document generation

- Do **not** compare to Scenario A (`models_only`); this run uses Scenario B only.
- If all cells are in the pool by the sim start date, `n_items_new` is 0 after round 0;
  model rollout is still the main temporal signal.
- Ranking ρ requires ≥3 active embedders; early rounds may show `n/a`.
- cellplm excluded from fit by default ( suspect calibration).

## 12. Regenerate

```bash
PYTHONPATH=. python sim/run_rollout_report.py --exclude-models cellplm
PYTHONPATH=. python sim/run_rollout_report.py --report-only  # from existing metrics.csv
```
