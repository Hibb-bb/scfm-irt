# Simulation diagnostics

## Empirical field drift (μ_oracle slope vs round)

```
        arm  empirical_r  mu_oracle_start  mu_oracle_end
      fixed     0.326968         0.096907       4.674466
       full     0.326968         0.096907       4.674466
recalibrate     0.326968         0.096907       4.674466
```

If `empirical_r ≈ 0`, the real model rollout has little capability drift —
fixed ≥ full on ranking is expected; the theorem's decay regime is not active.

## Final round (controlled metrics)

```
        arm  round  bank_size  bank_value  bank_value_per_item  ranking_spearman  seed_mean_value  n_rankable_models
      fixed     14        400   14.701171             0.036753          0.966402         0.036523                 10
recalibrate     14        400   15.199356             0.037998          0.956954         0.050700                 10
       full     14        400   64.963985             0.162410          0.806249         0.050972                 10
```

Compare `bank_value_per_item` (not raw ΣV) across equal-sized banks.
Compare `ranking_spearman` (size-free) for the paper's claim.
Compare `seed_mean_value`: do round-0 items go stale under the final field?
