# Simulation Brief — Online Benchmark-Maintenance on a Real Classification Atlas

**Purpose of this file:** a self-contained implementation brief you can hand to a fresh Claude Code
session (e.g. on a remote cluster). It has no dependency on prior conversation. It specifies a
simulation that runs the *online item-bank maintenance* algorithm from the paper **"When Do
Benchmarks Go Stale? A Decay Law and an Online Maintenance Policy for Item Banks"** on a **real
single-cell cell-type-classification benchmark** (a human lung atlas built from several sub-studies,
plus pbmc3k and one other dataset), where **10 models have already been evaluated**.

Everything here is CPU-only and small (IRT fits + a discrete-round simulation loop). No GPU.

---

## 0. What we're testing (one paragraph)

The paper proves: as a field of models improves over time (mean ability rises at drift rate `r`),
each benchmark item's usefulness decays like `e^(−rt)` once the field passes it, so a **fixed**
benchmark goes stale. Their **Algorithm 1** keeps a benchmark near-optimal at every round by
holding a fixed **anchor set** to pin the ability scale, **recalibrating** item difficulties from
answers, **retiring** items that became too easy, and **acquiring** new hard items — and it stays
near-optimal **iff new hard items are written at least as fast as the field improves (`a ≥ r`,
equivalently `κ ≥ rN/(4s)`)**. This simulation reproduces that story on real classification data:
a fixed bank should decay while the maintained bank holds its ranking power, with a survival
threshold at `a/r = 1`.

---

## 1. Background you need from the paper (condensed, self-contained)

**Rasch/IRT model.** Model `j` has ability `θ_j`; item `i` has difficulty `b_i`. Probability model
`j` answers item `i` correctly:
```
P(U_ij = 1) = σ(θ_j − b_i),   σ(z) = 1/(1+e^−z)
```
An item's **information** about a model at ability θ is `I_b(θ) = σ(θ−b)·(1−σ(θ−b)) ≤ 1/4`,
maximized when `b = θ` and decaying like `e^(−|θ−b|)`. So an item is most useful for models near
its own difficulty and nearly useless for models that almost always pass or fail it.

**Moving field.** At round `t` the field of model abilities is `N(μ_t, s²)` with `μ_t = μ_0 + r·t`
(mean rises linearly at drift rate `r`, spread `s` fixed).

**Value & decay law.** An item's value = its information averaged over the current field,
`v_t(b) = E_{θ~N(μ_t,s²)}[I_b(θ)]`. Once the field rises above the item, `v_t(b) ≈ e^{s²/2}·e^{−(μ_t−b)} ∝ e^{−rt}`
— it halves every `ln2/r` rounds. Value peaks when `μ_t = b`. Useful items live in a sliding
window `W_t = [μ_t − 2s, μ_t + 2s]`.

**Objective — ranking error.** With total information `I_t(θ;B) = Σ_{i∈B} I_{b_i}(θ)`, the quantity
to minimize is the field-averaged ability variance
`Ψ_t(B) = ∫ ρ_t(θ) / I_t(θ;B) dθ` (small when the bank covers the field with information). NOT
"maximize total value" — that piles all items at one difficulty and leaves coverage gaps.

**Scale identifiability.** Only differences `θ_j − b_i` appear, so the scale is free up to
shift/stretch. A fixed **anchor set A** (kept every round, difficulties held fixed) pins it, making
`μ_t` comparable across rounds and `r` identifiable.

**Algorithm 1 (five operations, five steps per round).**
Operations: **Fit** (ability MLE, `argmax_θ Σ [u·logσ(θ−b) + (1−u)·log(1−σ(θ−b))] − θ²/2τ²`, Newton);
**Route(θ, S, m)** (the `m` items of nearest difficulty to a model — most informative);
**Filter** (Kalman update of an item's `(b̂, P)` from one logistic answer; process-noise `q` lets
difficulty track drift); **Value** (`I_b` integrated over the estimated field, by quadrature);
**Acquire(κ, ceiling)** (commission `κ` new items spaced across the new top band, each **provisional**
with large uncertainty `P₀`).
Each round `t`:
1. **Select & ask.** Each model gets anchors `A` + `k−n_exp` trusted items routed to its level +
   `n_exp` provisional items routed to its level. Observe answers.
2. **Estimate abilities & field.** Fit `θ_j` from **anchor + trusted answers only** (excluding
   provisional, so uncalibrated new items can't bias abilities). Then `μ̂_t`, `ŝ_t`.
3. **Recalibrate.** Filter-update the difficulty of every asked non-anchor item.
4. **Promote & acquire.** Move provisional items with uncertainty `P ≤ P⋆` to trusted; Acquire `κ`
   new provisional items at `μ̂_t + 2ŝ_t`.
5. **Retire & cap.** Drop non-anchor items with value `< v`; if still over budget `N`, drop the
   lowest-value non-anchors.

**Guarantee (Theorem 1).** With anchors fixing the scale, the gap to the one-round-best bank splits
into a calibration gap `O(1/√n_t)` (vanishes as items gather answers) + a coverage gap that stays
bounded **iff `a ≥ r`**, where the acquisition rate `a = κ/δ ≈ 4s·κ/N` (`δ ≈ N/(4s)` = bank density
in items per difficulty unit). Message: **writing hard items fast enough is the binding constraint.**

**"One pool, two jobs."** The same answers measure models (trusted items) and calibrate new items
(provisional items). The knob `n_exp` trades them: `n_exp=0` → sharp ability estimates but new items
never calibrate; larger `n_exp` → fast calibration at a small ranking cost.

**Reproduce these figures on real data:** Fig 2 (fixed & recalibrate-only decay; full maintenance
holds value + ranking), Fig 3 (steady-state ranking quality vs `a/r`; collapse below 1, plateau
above), and optionally Fig 4 (routing efficiency; the `n_exp` trade-off).

---

## 2. Mapping the paper onto the real classification benchmark

| Paper concept | This simulation |
|---|---|
| Item `i`, difficulty `b_i` | One **cell** (or a small cell-group) to classify; `b_i` from a Rasch fit |
| Model `j`, ability `θ_j` | One of the **10 evaluated models** |
| Answer `U_ij ∈ {0,1}` | **Did model j correctly classify cell i?** — already observed, no binarization needed |
| Item release time | The **source sub-study's release/publication date** (the atlas is a union of dated studies) |
| Anchor set `A` | A fixed subset of cells (spanning the difficulty range) held every round to pin the scale |
| Linking across datasets | A **joint Rasch fit** over shared models puts pbmc3k + other + lung-subset cells on ONE difficulty scale |
| Retirement | Easy studies' cells (e.g. pbmc3k band) stop discriminating as the field rises → dropped |
| Acquisition | New harder studies "arrive" by release date and enter the bank |

Classification is a **cleaner** fit than expression-prediction because correctness is intrinsically
binary — this is the ideal testbed for the algorithm.

---

## 3. Data the implementer has, and assumptions to CONFIRM with the user

**Available:**
- A **human lung atlas** for cell-type classification, composed of several smaller studies.
- **10 models** already evaluated on it (per-cell predicted label vs true label recoverable).
- **pbmc3k** and one other classification dataset.

**Confirm before building (these change the design):**
1. **Per-cell correctness recoverable?** Need `U_ij` = 1/0 per (model, cell), or at least
   per-(model, cell-group). If only aggregate accuracy per model exists, items must be coarser
   (cell-type × dataset groups) — still workable, fewer items.
2. **Do the 10 models have release/publication dates?** If yes → a **real** model-drift axis
   (order models by date; test whether `μ_t` actually rises). If no → impose a **synthetic** rising
   field (paper-style) over the real difficulties. Build the sim to support **both** (a `drift_mode`
   switch).
3. **Were the same 10 models run on pbmc3k + the other dataset** as on the lung atlas? Shared models
   are REQUIRED to link difficulties across datasets onto one scale. If not shared, run the whole
   sim **within the lung atlas alone** — its dated sub-studies already give both a difficulty spread
   and a release-date axis.
4. **Sub-study release dates** for the lung atlas components (and pbmc3k / other) — needed for the
   item-arrival schedule. If unknown, order studies by a proxy (a chosen sequence) and note it.

---

## 4. Getting the time axis (the one non-trivial modeling choice)

The paper's payload only bites under capability drift. Support two `drift_mode`s:

- **`drift_mode = "real"`** — order the 10 models by real release date; at round `t` the *active
  field* = models released by `t`. Measure whether `μ_t` rises ~linearly (a result in itself). Thin
  (10 models), so expect noisy `r`; report honestly.
- **`drift_mode = "synthetic"`** (paper-faithful; always implement this one) — real item
  difficulties `b_i` from the joint Rasch fit; at round `t` draw a field `θ ~ N(μ_0+rt, s²)` of `M`
  models and simulate answers `U ~ Bernoulli(σ(θ − b_i))`. The 10 real models still (a) calibrate
  `b_i` and (b) provide anchor difficulties. This mode is what enables the `κ`-vs-`r` sweep
  (Fig 3) because you can dial `r` and `κ` freely.

The **item** axis (arrival by real release date, retirement as the field passes) is real in both
modes.

---

## 5. Step-by-step simulation protocol

1. **Assemble the correctness tensor.** Build `U` (models × cells) of 0/1 correctness, with per-cell
   metadata `{source_dataset, release_date, true_label}`. Persist as a tidy parquet
   (`model, cell_id, correct, source_dataset, release_date`).
2. **Joint Rasch calibration (the linking step).** Fit a single Rasch model over ALL cells from ALL
   datasets that share the 10 models → oracle abilities `θ*` (10 models) and difficulties `b*`
   (every cell) on **one common scale**. Sanity-check: pbmc3k should sit in an easier band than the
   lung subsets. Use PyMC or numpyro (MCMC) for posteriors, or a fast penalized-MLE
   (`py-irt`/custom Newton) for speed; MCMC preferred for the difficulty CIs.
3. **Define rounds & schedules.** Choose a round grid over real time. Per round `t`: available item
   pool = cells whose `release_date ≤ t`; (real mode) active model field = models released by `t`.
4. **Run three arms** (Fig 2):
   - **fixed** — freeze the round-0 bank; never add/retire (may still re-fit abilities).
   - **recalibrate-only** — Filter-update difficulties each round, but never add/retire.
   - **full-maintenance** — Algorithm 1 (§1): anchors + Route + Fit-on-trusted + Filter + Retire
     low-value + Acquire hard, with budgets `N` (bank size), `κ` (writes/round), `k` (items asked
     per model), `n_exp` (exploration).
5. **Metrics per round:**
   - **Bank value** `V_t(B) = Σ_{i∈B} v_t(b_i)`.
   - **Ranking quality** = Spearman between each arm's estimated `θ̂` and the oracle `θ*` of the
     **current** field. (Full maintenance should hold; fixed/recalibrate should decay.)
6. **Survival-threshold sweep** (Fig 3, synthetic mode): grid over `κ` (or directly `a = 4sκ/N`)
   against `r`; plot steady-state ranking quality vs `a/r`. Expect collapse below 1, plateau above.
7. **(Optional) Efficiency** (Fig 4): matched-difficulty routing vs random; sweep `n_exp` for the
   calibration-speed vs ranking trade-off.
8. **Retirement demonstration:** log per-item value trajectories; show the pbmc3k/easy band decaying
   first and being retired as the field climbs — the concrete "field outgrew the ruler" instance.

---

## 6. Suggested module layout & file contracts

```
sim/
├── README.md
├── conf/sim.yaml               # rounds, budgets N/κ/k/n_exp, τ, q, P0, P*, drift_mode, r grid, seeds
├── data/
│   ├── build_tensor.py         # raw eval outputs -> correctness parquet (+ metadata)
│   └── correctness.parquet     # model, cell_id, correct, source_dataset, release_date, true_label
├── irt/
│   ├── rasch.py                # joint fit -> theta_star.csv, difficulty_star.csv (+ CIs)
│   └── ops.py                  # Fit, Route, Filter (Kalman), Value (quadrature), Acquire
├── engine/
│   ├── field.py                # drift_mode: "real" (model dates) | "synthetic" (N(mu0+rt, s^2))
│   ├── schedule.py             # item arrival by release_date; anchor set selection
│   └── maintain.py             # Algorithm 1 loop + the fixed / recalibrate-only arms
├── run_sim.py                  # orchestrates arms over rounds -> results parquet
├── analysis/
│   ├── figures.py              # Fig2 (value+ranking over rounds), Fig3 (a/r threshold), Fig4
│   └── retirement.py           # per-item value trajectories; easy-band decay demo
└── results/                    # metrics.parquet + figures/
```

**Key contracts / implementation notes:**
- `ops.Fit(D)` — Newton on `Σ(u − σ(θ−b)) = θ/τ²`; unique root.
- `ops.Route(θ, S, m)` — the `m` items with `|b_i − θ|` smallest.
- `ops.Filter(b̂, P; θ̂, u)` — Kalman logistic update: `p̂=σ(θ̂−b̂)`, `P̃=P+q`,
  `b̂' = b̂ + (P̃/(1+P̃·p̂(1−p̂)))·(p̂−u)`, `P' = (P̃⁻¹ + p̂(1−p̂))⁻¹`.
- `ops.Acquire(κ, c)` — `κ` new provisional items evenly spaced on `[c, c+a]`, `a ≈ 4sκ/N`, state
  `(b̂=c+((ℓ−½)/κ)·a, P=P₀)`, marked provisional.
- **Abilities use trusted answers only** (step 2); Filter (step 3) touches all asked non-anchors;
  anchors' difficulties are never filtered.
- Constants (`τ`, `q`, `P₀`, `P⋆`, retirement threshold `v`) live in `conf/sim.yaml`.
- **Determinism:** seed everything; the paper's environment forbids wall-clock/RNG-in-loop — pass a
  seed per round so runs are reproducible/resumable.

---

## 7. Milestones (do in order)

1. **M0 — Linking proof (notebook).** Build the correctness tensor for **pbmc3k + one lung subset**,
   run the **joint Rasch fit**, and show the two datasets land on a sensible shared difficulty scale
   (pbmc3k easier). This validates the single most load-bearing assumption before any loop code.
2. **M1 — Static IRT on the full atlas.** Joint fit over all 10 models × all cells → oracle
   `θ*`, `b*` with CIs. Confirms items are well-pinned (many cells → tight `b`; 10 models → note
   ability CIs and shaky item *discrimination*).
3. **M2 — Rounds + arms, synthetic drift.** Implement `field.py` (synthetic), `schedule.py`,
   `maintain.py`; run fixed vs recalibrate vs full-maintenance → **reproduce Fig 2** on real
   difficulties.
4. **M3 — Survival threshold.** Sweep `κ` vs `r` → **reproduce Fig 3** (`a/r = 1`).
5. **M4 — Real drift (if model dates exist).** Order models by release date; test whether `μ_t`
   rises; rerun arms with the real field.
6. **M5 — Retirement + write-up.** Retirement trajectories, `n_exp` efficiency, and a short results
   note.

---

## 8. Honest caveats to keep in the write-up

- **10 models is a thin field:** abilities are estimable but item *discrimination* and the drift
  rate `r` (real mode) are noisy. Report CIs; treat item-level difficulty as provisional.
- **Capability may not rise linearly/monotonically** on real models — testing that assumption is a
  legitimate finding, not a bug.
- **Anchors self-undermine** (paper §5): over long horizons anchors also get too easy; a full
  treatment refreshes them by linking new anchors to old before retiring old ones. Note it; only
  implement if you simulate very long horizons.
- **Linking needs shared models across datasets.** No overlap ⇒ can't co-calibrate ⇒ stay within the
  lung atlas's own dated sub-studies.

---

## 9. First question back to the user

Confirm the four items in §3 (per-cell correctness available? model release dates? same models across
datasets? sub-study release dates?). Then start at **M0** (the pbmc3k + lung-subset linking notebook)
— do not build the maintenance loop until the shared-scale linking is shown to work.

---

## 10. Controlled comparison + diagnostics (required before drawing conclusions)

The maintenance loop runs end-to-end, but **raw total bank value `V_t(B) = Σ_{i∈B} v_t(b_i)` is
confounded by bank size** when arms hold different numbers of items. The paper's Fig 2 holds `N`
fixed across arms so value reflects **coverage**, not count. Do not compare a 270-item full arm to a
45-item fixed arm on raw ΣV.

### 10.1 Equal-N arms (`equal_bank_size: true` / `--equal-bank-size`)

All three arms seed the **same** anchor set and the **same** `bank_budget` non-anchor items (drawn
from items available by the final round date). Fixed = frozen; recalibrate = filter-updates only;
full = Algorithm 1 with cap `N = bank_budget`. Then:

| Metric | Confound removed? | What it tests |
|--------|-------------------|---------------|
| `bank_value_per_item` = `V_t / \|B\|` | yes (size) | Per-item informativeness / coverage quality |
| `ranking_spearman` ρ | yes (size) | **Decisive metric** — paper Fig 2 right panel, Theorem 1 |
| raw `bank_value` | **no** | Only valid when `\|B\|` is equal across arms |

### 10.2 Per-round trajectories, not final snapshots

The paper's claim is that **fixed decays over rounds** while full holds. Plot ρ and
`bank_value_per_item` vs round for all arms. A single final-round row cannot show decay.

### 10.3 Field drift check (`fig_diagnostics.png`, `diagnostics_drift.csv`)

Plot `μ_oracle` (mean oracle θ\* over active models) and `μ̂_t` vs round. Fit the slope =
**empirical `r`**. If `r ≈ 0` (likely on the real 10-model rollout — a lumpy step function with
rounds 0–10 stuck at 2 dev baselines), items never fall below the field, nothing goes stale, and
**fixed ≥ full on ranking is expected**, not a refutation of the theorem. The synthetic-drift mode
(dial `r`) is where the mechanism demonstrates cleanly.

### 10.4 Seed-item staleness check

Track `seed_mean_value` = mean `v_t(b_i^seed)` where `b_i^seed` are the **initial** difficulties
of the round-0 bank (non-anchors). If this stays above `retire_v` at the final field, seed items
never went stale — maintenance has nothing to beat. Plot vs `retire_v` threshold.

### 10.5 Ranking validity

Require `n_rankable_models ≥ 3` before interpreting ρ. With 2 active models, Spearman is ±1 and
meaningless. The extended schedule (§ rounds to 2025) is needed for real-drift ranking.

### 10.6 What counts as "reproducing the paper"

- **Synthetic drift + equal N:** expect fixed ρ to decay, full to hold when `a ≥ r`; Fig 3 sweep
  locates the `a/r = 1` threshold.
- **Real drift + equal N:** primarily a **regime check** — is empirical `r` large enough and the
  rollout smooth enough for decay to bite? If not, report that honestly; do not headline raw ΣV
  wins driven by bank-size mismatch.

### 10.7 Survival curve (real benchmark placement)

Run `PYTHONPATH=. python sim/run_validation.py` after a real-drift diagnostic run. Computes:

- **κ\\* = r·N/(4s)** — critical items/round to stay on the survival curve
- **r\\* = 4sκ/N** — drift rate above which maintenance becomes binding at current κ
- **Synthetic Fig 2** with **r_tune = 4s/T** on real `b_i` (field sweeps one useful window)
- **Fig 3 sweep** of `a/r` with ranking gap `ρ_full − ρ_fixed`

The headline for real data is often: *this benchmark sits deep in the safe regime* (κ ≫ κ\\*,
r ≪ r\\*), not "maintenance wins."
