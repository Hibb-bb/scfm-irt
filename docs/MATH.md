# Mathematical Specification — IRT Evaluation of Cell Embedding Models

## 0. Purpose of this document

This document defines the statistical model behind the project. It is the source of
truth for *what* the code computes; the companion `IMPLEMENTATION_PLAN.md` defines *how*
the code is organized. The coding agent should treat the equations here as the
contract that the `irt` module must satisfy (validated against synthetic data — see §9).

---

## 1. The setup and the core reframing

We evaluate a set of **cell embedding models** on **cell-type annotation** with a frozen
linear probe (multinomial logistic regression, no fine-tuning). Each model produces an
embedding for every cell; a logistic-regression probe maps embeddings to cell-type
labels; each cell is then scored **correct (1) / incorrect (0)** per model.

We analyze the resulting binary outcomes with **Item Response Theory (IRT)** under the
following identification:

| IRT term            | Meaning in this project                                  | Count      |
|---------------------|----------------------------------------------------------|------------|
| **Respondent / subject** (has *ability* θ) | a cell embedding model (incl. the in-house Cell-JEPA) | small, `P` ≈ 5–15 |
| **Item** (has *difficulty* b) | a single test-set cell                          | large, `I` ≈ 1e4–1e6 |
| **Response** `y`    | 1 if the model's probe annotated that cell correctly     | `P × I`    |

The object every downstream computation depends on is the **response matrix**
`Y ∈ {0,1}^{P×I}`, where `Y[p,i] = 1` iff model `p` correctly classifies cell `i`.

> **Why annotation, and why binary items.** Cell-type annotation yields a clean binary
> correct/incorrect outcome per cell per model. That is exactly the dichotomous response
> IRT is built for, which is why annotation is the natural first task. Continuous tasks
> (e.g. perturbation-response regression) do **not** map onto dichotomous IRT without an
> extra response model, and are out of scope here.

### 1.1 A caveat baked into the interpretation
Item difficulty is estimated **relative to the respondent pool**. With only embedding
models in the pool, "a hard item" means "hard *for these embedding models*," not "hard in
some absolute sense" and not "requires more than raw gene features." Keep this in mind when
reading difficulty estimates; see §8.3 and §10 for what this does and does not license.

---

## 2. Notation

- `p = 1, …, P` indexes respondents (models); ability `θ_p ∈ ℝ`.
- `i = 1, …, I` indexes items (cells); difficulty `b_i ∈ ℝ`, discrimination `a_i > 0`
  (2PL+), lower asymptote / guessing `c_i ∈ [0,1]` (3PL+), upper asymptote `d_i ∈ [0,1]` (4PL).
- `y_{pi} ∈ {0,1}` is the observed response.
- `σ(x) = 1 / (1 + e^{-x})` is the logistic function.
- `φ(·)` is the standard normal pdf.
- Each cell carries covariates `x_i` (cell type, source dataset, #classes in that dataset,
  sequencing depth, etc.) used by the explanatory extension in §11.1.

---

## 3. Response models (item characteristic curves)

The probability of a correct response is a monotone function of the gap `θ_p − b_i`.

**1PL / Rasch** (default; most robust in this regime):
```
P(y_{pi}=1 | θ_p, b_i) = σ(θ_p − b_i)
```

**2PL** (adds per-item discrimination `a_i`):
```
P(y_{pi}=1 | θ_p, a_i, b_i) = σ(a_i (θ_p − b_i))
```

**3PL** (adds guessing floor `c_i`):
```
P(y_{pi}=1) = c_i + (1 − c_i) · σ(a_i (θ_p − b_i))
```

**4PL** (adds upper asymptote `d_i`; supported by py-irt):
```
P(y_{pi}=1) = c_i + (d_i − c_i) · σ(a_i (θ_p − b_i))
```

Guidance: **start with 1PL.** Move to 2PL only once 1PL is validated, because per-item
`a_i` is weakly identified here (§10). 3PL/4PL guessing/asymptote parameters are hard to
interpret for k-class annotation and are optional/experimental.

---

## 4. Likelihood

Assuming **local independence** (responses independent given the parameters — see §13 for
where this assumption is strained), the joint likelihood is
```
L = ∏_p ∏_i  P_{pi}^{y_{pi}} (1 − P_{pi})^{1 − y_{pi}}
```
and the log-likelihood the estimators actually optimize is
```
ℓ(Θ) = Σ_p Σ_i [ y_{pi} log P_{pi} + (1 − y_{pi}) log(1 − P_{pi}) ]
```
where `Θ` collects all ability and item parameters.

---

## 5. Identifiability / anchoring

The latent scale has no inherent origin or unit, so constraints are required:

- **1PL:** fix **location** only — e.g. `mean_p(θ_p) = 0`, *or* `Σ_i b_i = 0`, *or* anchor
  one parameter. The logistic's unit slope fixes the scale.
- **2PL+:** fix **location and scale** — e.g. impose `mean(θ)=0, var(θ)=1` on the person
  side, or anchor a reference item's `a,b`.
- **Bayesian route (recommended):** priors `θ_p ~ N(0,1)` and weakly-informative priors on
  item parameters softly fix location and scale, removing the need for hard constraints.
  This is how py-irt handles it.

The reporting layer must record which anchoring was used; abilities are only comparable
within a single anchored fit.

---

## 6. Estimation methods

The code should provide **two** estimators sharing one interface: a transparent reference
implementation and a scalable Bayesian one.

### 6.1 Joint Maximum Likelihood (JML) — reference only
Maximize `ℓ` over both `θ` and item parameters directly (autodiff, gradient descent).
Simple and transparent, good as a sanity/reference implementation, **but** JML item
estimates are known to be statistically inconsistent as `I → ∞` with fixed `P`. Use it for
validation and small runs, not as the headline estimator.

### 6.2 Marginal Maximum Likelihood (MML) + EM — classical
Treat `θ_p ~ N(0,1)` and integrate it out per respondent:
```
L_p = ∫  ∏_i P_{pi}(θ)^{y_{pi}} (1 − P_{pi}(θ))^{1 − y_{pi}} · φ(θ) dθ
```
approximated by Gauss–Hermite quadrature. **E-step:** posterior over `θ_p` on the quadrature
grid. **M-step:** maximize expected complete-data log-likelihood w.r.t. item parameters.
Abilities are then recovered as in §7.

> **Regime note.** Classical MML marginalizes the *persons* to shrink the parameter count.
> Here persons (models) are few and items are many, so the parameter explosion is on the
> **item** side. MML alone doesn't address that; the fix is priors/hierarchy on items
> (§6.3, §11.1). MML is included for completeness and cross-checking.

### 6.3 Bayesian variational inference — recommended default
Full Bayesian model with priors:
```
θ_p ~ N(0, 1)
b_i ~ N(μ_b, σ_b²)        # hierarchical prior regularizes the many noisy item difficulties
a_i ~ LogNormal(μ_a, σ_a²)  # 2PL only; positivity via log scale
(hyperpriors on μ_b, σ_b, μ_a, σ_a)
y_{pi} ~ Bernoulli(P_{pi})
```
Fit by stochastic variational inference (mean-field or low-rank). This yields **posterior
means and variances for every parameter** — crucial because with only `P ≈ 10` respondents
we need *uncertainty* on abilities, not just point estimates. `py-irt` implements exactly
this (1PL/2PL/4PL, vague or hierarchical priors, GPU-accelerated) and is the recommended
engine. `numpyro` (NUTS or SVI) is the escape hatch for custom structure (§11).

### 6.4 Equivalent GLMM view (useful for the numpyro implementation)
The Rasch model is a logistic mixed model with **crossed random effects**:
```
y_{pi} ~ Bernoulli( σ(θ_p − b_i) ),   θ_p ~ N(0, σ_θ²),   b_i ~ N(μ_b, σ_b²)
```
i.e. a model random effect crossed with a cell random effect. This framing makes the
hierarchical/explanatory extensions (§11) drop-in and is the cleanest thing to write in
numpyro.

---

## 7. Ability estimation (the headline output)

Given item parameters, estimate each model's ability:
- **MLE:** maximize `Σ_i [y_{pi} log P_{pi} + (1−y_{pi}) log(1−P_{pi})]` over `θ_p`.
- **MAP / EAP:** posterior mode / mean under `θ_p ~ N(0,1)`. **EAP is preferred** (stable,
  gives a variance). In the Bayesian fit (§6.3) abilities come out directly as posterior
  summaries.

Because each `θ_p` is informed by ~`I` responses (huge), ability point estimates are
**precise**; report posterior/interval widths anyway, since with few respondents the
*between-model* comparison is what carries uncertainty.

---

## 8. Information, discrimination, and item selection

### 8.1 Fisher information
For 2PL, the information an item contributes at ability level θ is
```
I_i(θ) = a_i² · P_i(θ) · (1 − P_i(θ))      # 1PL: set a_i = 1
```
Test information `I(θ) = Σ_i I_i(θ)`; the standard error of an ability estimate at θ is
`SE(θ) = 1 / sqrt(I(θ))`. Plot `I(θ)` to show **where on the ability scale the evaluation
is most precise**.

### 8.2 Discrimination = "informative eval item"
In 2PL, high-`a_i` cells are the ones that best separate strong from weak models. These are
the **most informative evaluation items** and directly formalize the earlier design goal of
wanting cells that spread models apart. The reporting layer should surface the highest- and
lowest-discrimination cells/types.

### 8.3 Difficulty spread and ceiling effects (why this matters for the pool)
If most cells are near-ceiling (every model gets them right), their difficulties pile up at
the bottom of the scale, item information is negligible there, and ability estimates become
poorly separated — everyone looks equally able because every item is trivial. The eval
therefore needs items spanning a **range of difficulty**: rare cell types, fine-grained
subtypes, and cross-dataset / cross-protocol / cross-species cells. The difficulty histogram
is a first-class diagnostic, not an afterthought.

---

## 9. Validation against synthetic data (mandatory)

Before touching real embeddings, the `irt` module must pass a **parameter-recovery test**:

1. Draw true `θ_p ~ N(0,1)` (P ≈ 10) and true `b_i` (and `a_i` for 2PL) from known priors.
2. Simulate `y_{pi} ~ Bernoulli(P_{pi})` for a grid of `I` (e.g. 1e3, 1e4, 1e5).
3. Fit each estimator; check recovered parameters correlate with truth (abilities: high
   correlation and calibrated intervals; difficulties: correlation improving with `I`).
4. Check calibration: posterior credible intervals contain the truth at nominal rates.

This test is the acceptance gate for the estimator (see roadmap Phase 4). Orientation bugs
in `Y` (persons↔items transposed) are the most common failure and will show up here as
scrambled recovery.

---

## 10. The few-respondents / many-items regime — consequences

This application inverts the usual IRT setting (normally many persons, moderate items).
With `P ≈ 10` and `I` in the tens of thousands to millions:

- **Abilities (`θ_p`)** are estimated from ~`I` responses each → **very precise.** Good, this
  is the quantity of interest.
- **Per-item parameters (`b_i`, and especially `a_i`)** are estimated from only ~`P`
  responses each → **noisy.** A per-cell difficulty from ~10 binary observations is barely
  identified; per-cell discrimination is worse.

**Mitigations (in rough order of preference):**
1. Prefer **1PL** for the headline result; treat 2PL as exploratory.
2. Use **hierarchical priors** on item parameters (§6.3) so noisy per-item estimates shrink
   toward the population.
3. Use the **explanatory model** (§11.1): predict difficulty from cell covariates, collapsing
   `I` free difficulties into a handful of coefficients — huge variance reduction plus a
   scientific readout ("which conditions are hard").
4. **Aggregate items** to `(dataset, cell_type)` granularity (§11.2), which both raises the
   per-item sample size and mitigates §13.

---

## 11. Extensions

### 11.1 Explanatory / hierarchical IRT (LLTM-style) — recommended
Model item difficulty as a function of cell covariates:
```
b_i = wᵀ x_i            (pure LLTM: difficulty fully explained by features)
b_i = wᵀ x_i + ε_i,  ε_i ~ N(0, σ_ε²)   (random-effects / partial pooling — preferred)
```
where `x_i` encodes cell type, source dataset, log #classes, sequencing depth, batch, etc.
Benefits: (a) drastically fewer effective parameters → stable in this regime; (b) directly
answers "**what makes a cell hard to annotate**," which is itself a result. Implement in
numpyro via the GLMM view (§6.4).

### 11.2 Aggregated / graded items (mitigates local independence)
Redefine an item as a `(dataset, cell_type)` group and model the count of correct cells:
```
k_{pg} ~ Binomial(n_g, P_{pg}),   P_{pg} = σ(θ_p − b_g)
```
`I` shrinks to the number of type×dataset groups (hundreds), per-item sample size rises, and
the correlation among same-type cells (§13) is absorbed. Offer this as an alternative
"item granularity" mode alongside per-cell.

### 11.3 Multidimensional IRT (dimensionality check)
Unidimensionality (§13) assumes a single "embedding quality" axis. If a model is strong on
immune cells but weak on neurons, responses are multidimensional. Check by (a) PCA/factor
analysis of standardized residuals, or (b) fitting a 2D MIRT
`P = σ(a_iᵀ θ_p − b_i)` and testing whether the second dimension explains meaningful
variance. Report as a robustness check, not a headline model.

---

## 12. Outputs the math layer must produce

For each fit (model spec + item granularity + IRT model), emit:
- **Ability table:** `θ̂_p` with credible/confidence intervals → the model ranking (primary
  deliverable).
- **Difficulty summaries:** distribution of `b̂_i`, aggregated by cell type and dataset →
  "hardest cells/types" tables.
- **Discrimination summaries (2PL):** highest/lowest `â_i` → most/least informative eval items.
- **Item–person (Wright) map:** abilities and difficulties on the shared latent scale.
- **ICC plots** for representative items; **test information curve** `I(θ)`.
- **Fit diagnostics** (§13): item and person infit/outfit or standardized-residual summaries.

---

## 13. Assumptions and how to check them

| Assumption | Statement | Risk here | Check / mitigation |
|---|---|---|---|
| **Unidimensionality** | one latent ability explains all responses | a model may excel on some tissues, fail others | residual PCA / 2D MIRT (§11.3) |
| **Local independence** | responses independent given parameters | cells of the same `(dataset, type)` are correlated (a model wrong on a rare type is wrong on *all* its cells) → **strained** | aggregate to type-level items (§11.2); testlet/group random effect; cluster-robust SEs |
| **Monotonicity** | P(correct) increases with ability | should hold | inspect empirical ICCs |
| **Correct link/model** | 1PL vs 2PL vs … | per-item `a`,`c` weakly identified (§10) | prefer 1PL; compare fit; hierarchical priors |
| **Parameter invariance** | item params stable across the respondent pool | difficulty is pool-relative (§1.1) | interpret accordingly; re-fit if the pool changes materially |

The local-independence issue is the most substantive; the aggregated-item mode (§11.2)
exists primarily to address it and should be run as a robustness comparison to the per-cell fit.
