# Online benchmark-maintenance simulation

Implements **Algorithm 1** from `docs/SIMULATION_PLAN.md` on real scFM-IRT classification data.

```bash
# From repo root
PYTHONPATH=. python sim/data/build_tensor.py
PYTHONPATH=. python sim/run_sim.py
PYTHONPATH=. python sim/run_sim.py --drift-mode real
```

## Layout

- `data/release_dates.yaml` — model & dataset publication dates
- `data/build_tensor.py` — correctness parquet with release metadata
- `irt/ops.py` — Fit, Route, Filter, Value, Acquire
- `irt/rasch.py` — joint oracle calibration
- `engine/` — field drift, schedule, maintenance loop
- `results/` — figures and metrics (generated)

## Milestones

| Step | Output |
|---|---|
| M0 | `results/m0_linking_summary.csv` |
| M1 | `results/theta_star.csv`, `difficulty_star.csv` |
| M2 | `results/fig2_value_and_ranking.png`, `metrics.parquet` |
| M3 | `results/fig3_survival_threshold.png` |
