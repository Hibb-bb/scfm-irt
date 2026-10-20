# Rollout simulation — PI retirement (bottom 5%, top-50% models)

**Retirement protocol:** each round, score every non-anchor **cell** by mean Rasch Fisher
information among the **top 50% of models** (by fitted θ); retire the **bottom 5%** by that score.

**Other config:** `retire_cap=false`, `k_ask=10021` (full bank), `field_window_top_k=4`.

Excluded models: `cellplm`

## Key results (11 rounds)

| Round | Retired (5%) | Bank size | ρ |
|-------|--------------|-----------|---|
| 0–3 | 20 | 395 → 380 | — |
| 4–8 | 19 | 376 → 360 | 0.5 → 0.829 |
| 9–10 | 18 | 357 → **354** | 0.829 → **0.910** |

- **Total retirements:** 211 (steady ~5%/round; 0 cap-driven)
- **Final bank:** 354 cells (+ 15 anchors)
- **Final ρ:** 0.910

Compared with the prior absolute threshold rule (`retire_v=0.01`), this PI rule retires a **fixed fraction** each round rather than a large purge when the frontier jumps (old run: 149 at round 8).

## Outputs

`metrics.csv`, `model_rankings.csv`, `model_ranking.png`, `SIMULATION_SPEC.md`, `rollout_report.pdf`

## Tune X%

Set `retire_bottom_pct` in `sim/conf/sim.yaml` (default `0.05` = 5%).

## Regenerate

```bash
PYTHONPATH=. python3 sim/run_rollout_report.py
```
