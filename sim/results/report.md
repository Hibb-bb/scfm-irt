# Online maintenance simulation

- drift_mode: **real**
- models: 10
- lung items: 10021

## M0 linking

```
   dataset    b_mean    b_std  n_items
lung_atlas -0.666623 1.723834    10021
    pbmc3k  0.574118 0.416311      527
```

## Outputs

- `fig2_value_and_ranking.png`
- `fig_diagnostics.png` (μ drift, seed staleness, bank size)
- `diagnostics.md` + `diagnostics_final_summary.csv`
- `fig3_survival_threshold.png` (if sweep enabled)
- `metrics.csv`
- `fig_bank_budget_sweep.png` (if `--bank-budget-sweep` set)

- equal_bank_size: **True**
