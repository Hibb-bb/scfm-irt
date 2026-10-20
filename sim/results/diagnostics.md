# Simulation diagnostics

## Empirical field drift (μ_oracle slope vs round)

```
        arm  empirical_r  mu_oracle_start  mu_oracle_end
      fixed     0.016669        -0.529037  -2.220446e-17
       full     0.016669        -0.529037  -2.220446e-17
recalibrate     0.016669        -0.529037  -2.220446e-17
```

If `empirical_r ≈ 0`, the real model rollout has little capability drift —
fixed ≥ full on ranking is expected; the theorem's decay regime is not active.

## Final round (controlled metrics)

```
        arm  round  bank_size  bank_value  bank_value_per_item  ranking_spearman  seed_mean_value  n_rankable_models
      fixed     14        400   46.664831             0.116662          0.733333         0.116049                 10
recalibrate     14        400   43.871870             0.109680          0.769697         0.110562                 10
       full     14        400   44.874139             0.112185          0.769697         0.110562                 10
```

Compare `bank_value_per_item` (not raw ΣV) across equal-sized banks.
Compare `ranking_spearman` (size-free) for the paper's claim.
Compare `seed_mean_value`: do round-0 items go stale under the final field?
