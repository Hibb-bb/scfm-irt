# Survival curve — real lung atlas benchmark

Fitted field spread **s = 1.144** (std of joint-Rasch θ\*).
Bank size **N = 400**, write budget **κ = 15**.

## Observed drift (real rollout)

- Empirical **r ≈ 0.0167**/round
- Over 14 steps: **r·T ≈ 0.23** vs useful window **4s ≈ 4.58**
- Field crossed ~**5%** of one coverage window → out of decay regime

## Critical write rate

- **κ\* = r·N/(4s) ≈ 1.46** items/round
- Current κ = 15 → margin **κ/κ\* ≈ 10.3×**

The benchmark stays alive by adding <2 new items per round — trivially satisfied.

## Staleness horizon

- **r\* = 4sκ/N ≈ 0.172**/round
- Drift must exceed r\* before maintenance binding; observed r is **10.3×** below that

Cell-type classification won't need active maintenance until capability drifts ~**10×** faster than the real 10-model timeline.

## Synthetic validation (Fig 2 tuning)

For controlled drift demo: **r_tune = 4s/T ≈ 0.327**/round (r·T ≈ 4s).
