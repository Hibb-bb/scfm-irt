# Implementation Plan & Development Roadmap

Companion to `MATH.md`. That document defines the statistics; this one defines the
software: architecture, module contracts, data schemas, tech stack, a phased roadmap with
acceptance criteria, and the gotchas most likely to sink the build.

Working package name (placeholder): **`scfm_irt`**.

---

## 1. Goal & scope

Build a reproducible pipeline that:
1. loads single-cell datasets with cell-type labels,
2. extracts frozen embeddings from a set of cell embedding models (existing foundation
   models **and** the in-house Cell-JEPA),
3. fits a logistic-regression probe per (model, dataset) — **no fine-tuning**,
4. scores each test cell correct/incorrect per model → a binary **response matrix**,
5. fits IRT models over that matrix to estimate **model abilities** and **cell
   difficulties/discriminations**, with uncertainty,
6. reports rankings and diagnostics.

Out of scope for v1: perturbation / State-Transition tasks, fine-tuning any embedder,
raw-feature baselines (see §12 — supported as a one-line config extension, not a v1 goal).

---

## 2. Architecture at a glance

Six modules and **one hard architectural boundary**:

```
            ┌─────────────── HEAVY / GPU / per-model envs ───────────────┐
            │  data ─▶ embedders ─▶ [cached embeddings on disk]          │
            └────────────────────────────┬──────────────────────────────┘
                                          │   (clean artifact boundary)
            ┌─────────────────────────────▼──────────────── LIGHT / CPU / one env ─┐
            │  probe ─▶ response-matrix ─▶ irt ─▶ analysis/reporting               │
            └─────────────────────────────────────────────────────────────────────┘
```

### 2.1 The boundary is non-negotiable
Each foundation model (scGPT, Geneformer, UCE, SCimilarity, CellPLM, scFoundation, CellFM,
Cell-JEPA, …) ships its **own, mutually conflicting** dependency stack and needs a GPU.
Do **not** attempt to import them into one environment. Instead:

- Embedding extraction runs **per model, in its own isolated environment** (separate venv /
  conda env / container), invoked as a subprocess or standalone job.
- Its only output is a **cached embedding artifact** on disk (see §5.1).
- Everything downstream (probe, response matrix, IRT, reporting) runs in **one lightweight
  CPU environment** and reads only those cached artifacts — it never imports an embedder.

This boundary is what keeps the analysis pipeline installable, fast, and testable while the
embedders stay quarantined. Treat cached embeddings + the response table as the contracts
between the two halves.

---

## 3. Module contracts

Interfaces are Python-typed sketches; the agent may refine signatures but must preserve the
inputs/outputs and the caching semantics.

### 3.1 `data`
Responsibilities: dataset registry, loading to `AnnData`, standardized splits, a `describe`
utility that reproduces the inventory (dims, #cell types, #donors/batches) as a sanity check.

```python
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: str                     # local h5ad after download
    label_key: str                # .obs column holding the ground-truth cell type
    batch_key: str | None         # .obs column for batch (if any)
    donor_key: str | None         # .obs column for donor (if any)
    notes: str = ""

def load_dataset(spec: DatasetSpec) -> AnnData: ...
def describe(adata: AnnData, spec: DatasetSpec) -> dict: ...   # dims, class counts, imbalance

def make_split(
    adata: AnnData, spec: DatasetSpec,
    scheme: Literal["stratified","cross_donor","cross_batch"],
    seed: int, test_frac: float = 0.2,
) -> Split:                       # returns train/test cell indices + provenance
    ...
```

Split policies:
- **stratified** — random, stratified by cell type (in-distribution; expect near-ceiling for
  common types).
- **cross_donor / cross_batch** — hold out whole donors/batches (harder; produces the
  difficulty spread IRT needs — see `MATH.md` §8.3).
- Cross-dataset transfer (train on A, test on B with harmonized labels) is a v2 extension.

**Label harmonization** across datasets (mapping disparate cell-type vocabularies to a
common ontology, e.g. Cell Ontology) lives here and is a known hazard — see §11.

### 3.2 `embedders` (heavy side)
One interface for every model; each concrete embedder wraps a model's official preprocessing.

```python
class CellEmbedder(Protocol):
    name: str
    requires_dataset_fit: bool     # True for scVI-style per-dataset training; False for zero-shot FMs
    def preprocess(self, adata: AnnData) -> Any: ...          # model-specific tokenization/normalization
    def embed(self, adata: AnnData) -> np.ndarray:            # returns (n_cells, d), row-aligned to adata
        ...
```

Rules:
- **Follow each model's official preprocessing/tokenization exactly** (do not invent it;
  wrong preprocessing silently produces garbage embeddings). Pin each model's own repo/deps
  inside its isolated env.
- `embed` output rows must stay aligned to `adata.obs_names`; persist the cell ids with the
  embedding so alignment survives caching.
- Provide a trivial **`DevEmbedder`** (PCA or random projection of log-normalized counts) with
  no heavy deps. Its purpose is to bring up the *entire* downstream pipeline before wrestling
  with real checkpoints — a smoke-test fixture, not an eval competitor. Whether any
  raw-feature embedder appears in the final evaluation is a separate decision (§12).

Candidate embedders to register (verify each one's current loading API against its own repo
at implementation time; capabilities and interfaces change): UCE and SCimilarity (zero-shot;
SCimilarity was trained with cell-type labels, so flag it as label-supervised for
interpretation), scGPT, Geneformer, CellPLM, scFoundation, CellFM, and the in-house
Cell-JEPA. `requires_dataset_fit=True` ones (e.g. scVI) must fit on **train cells only** to
avoid leakage.

### 3.3 `probe` (light side)
```python
def fit_probe(
    emb_train: np.ndarray, y_train: np.ndarray,
    C: float | None = None,        # None ⇒ pick by CV
    penalty: Literal["l2","l1"] = "l2",
    seed: int = 0,
) -> ProbeModel: ...

def score_cells(probe: ProbeModel, emb_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    # one row per test cell: y_true, y_pred, correct ∈ {0,1}, plus proba if useful
    ...
```
Use multinomial logistic regression (scikit-learn `LogisticRegression`, `lbfgs`/`saga`),
standardize features, select `C` by stratified CV on the train split. Deterministic given seed.

### 3.4 `response_matrix` (the bridge)
Assemble all per-cell scores into one **tidy long table** (the canonical artifact), plus a
pivot helper for estimators that want a dense/sparse matrix.

Long schema (`responses.parquet`) — one row per (model, cell):
```
model_id        str      # respondent / subject id
dataset         str
split_scheme    str
cell_id         str      # unique across datasets: f"{dataset}:{obs_name}"
cell_type       str      # for difficulty aggregation & explanatory covariates
correct         int8     # 0/1  ← the IRT response
# optional covariates for explanatory IRT (MATH.md §11.1):
n_classes       int16    # #cell types in that dataset
log_depth       float32  # log total counts for the cell
batch           str|None
```

```python
def build_response_table(score_frames: list[pd.DataFrame]) -> pd.DataFrame: ...
def to_matrix(resp: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    # returns Y with shape (P_models, I_cells) and the aligned (model_ids, cell_ids)
    ...
```

> **Orientation is the #1 bug.** Fix and document the convention once: **rows = models
> (subjects/respondents), columns = cells (items).** Assert `Y.shape == (len(model_ids),
> len(cell_ids))` everywhere. The synthetic recovery test (`MATH.md` §9) is what catches a
> flip.

### 3.5 `irt`
Two estimators behind one interface (see `MATH.md` §6):

```python
def fit_irt(
    resp: pd.DataFrame,
    model: Literal["1pl","2pl"] = "1pl",
    method: Literal["mle_autodiff","bayes_variational"] = "bayes_variational",
    item_granularity: Literal["cell","type_dataset"] = "cell",   # MATH.md §11.2
    item_covariates: list[str] | None = None,                    # enables explanatory model §11.1
    seed: int = 0,
) -> IRTFit:  # exposes abilities (mean+sd), item params (mean+sd), and fit diagnostics
    ...
```
- `mle_autodiff` — transparent PyTorch reference (JML/MML), for validation and small runs.
- `bayes_variational` — **default**; use **py-irt** (Pyro-based, scalable, 1PL/2PL/4PL, vague
  or hierarchical priors, returns latent means+variances). Use **numpyro** for the
  explanatory / GLMM / aggregated variants (`MATH.md` §6.4, §11).
- Must run the §9 synthetic recovery harness as a built-in self-test.

### 3.6 `analysis` / reporting
Produces every output in `MATH.md` §12: ability ranking with intervals, difficulty and
discrimination summaries (aggregated by cell type / dataset), Wright map, ICCs, test
information curve, item/person fit stats. Emits tidy tables (parquet/CSV) + figures (PNG/SVG)
and a short HTML/Markdown summary report per run.

---

## 4. Config system

Config-driven runs (YAML via Hydra or a typed `pydantic` settings object). One run declares
the cross product to evaluate.

```yaml
run_name: annotation_irt_v1
seed: 0
datasets: [zheng68k, pancreas, immune_human, lung_atlas]   # from the registry
split: {scheme: cross_donor, test_frac: 0.2}
embedders: [dev_pca, uce, scgpt, geneformer, cell_jepa]
probe: {penalty: l2, C: null}
irt:
  model: 1pl
  method: bayes_variational
  item_granularity: cell
  item_covariates: [n_classes, log_depth, cell_type, dataset]   # null ⇒ standard (non-explanatory)
outputs_dir: runs/${run_name}
```

Everything downstream keys off this; nothing is hard-coded.

---

## 5. Caching, artifacts, reproducibility

### 5.1 Embedding cache
Key each embedding artifact by a hash of `(embedder.name, embedder version, dataset, split,
preprocessing config)`. Store as `.h5`/`.npy` **with** the aligned `cell_id` array and a JSON
sidecar of the provenance. Embeddings are expensive → compute once, reuse across all probe /
IRT experiments. The downstream pipeline reads these and must never trigger a recompute.

### 5.2 Determinism
Single global seed threaded through splits, probe, and IRT. Log library versions and config
hash into every run directory. Same config + same seed ⇒ identical response table and fit.

---

## 6. Repository layout

```
scfm_irt/
  data/           registry.py  loaders.py  splits.py  harmonize.py  describe.py
  embedders/      base.py  registry.py  dev.py  uce.py  scgpt.py  geneformer.py  cell_jepa.py  ...
  probe/          logreg.py
  response/       build.py  schema.py
  irt/            models.py  fit_autodiff.py  fit_pyirt.py  fit_numpyro.py  synthetic.py
  analysis/       ability.py  difficulty.py  information.py  wright.py  fit_stats.py  report.py
  configs/        *.yaml
  scripts/        download_data.py  extract_embeddings.py  run_pipeline.py
  tests/          test_synthetic_irt.py  test_response_orientation.py  test_split_leakage.py ...
docs/             MATH.md  IMPLEMENTATION_PLAN.md
envs/             env-analysis.yml  env-scgpt.yml  env-geneformer.yml  ...   (isolated per embedder)
```

---

## 7. Tech stack

- **Single-cell:** `scanpy`, `anndata` (h5ad IO, preprocessing).
- **Probe / numerics:** `scikit-learn`, `numpy`, `pandas`, `pyarrow` (parquet).
- **IRT:** **`py-irt`** (primary, Pyro/PyTorch, scalable Bayesian, 1PL/2PL/4PL, hierarchical
  priors); **`numpyro`** (custom explanatory/GLMM/aggregated models, NUTS or SVI); a
  hand-written PyTorch 1PL/2PL as the reference/validation estimator. (`girth` is an optional
  frequentist cross-check.)
- **Embedders:** each in its own pinned env per its official repo (quarantined per §2.1).
- **Config/orchestration:** Hydra or pydantic-settings; plain scripts for extraction/runs.
- **Reporting:** `matplotlib`, and a simple Markdown/HTML template.

Pin versions at implementation time and record them per run; the analysis env must stay
importable independent of any embedder env.

---

## 8. Development roadmap (phased, with acceptance criteria)

**Phase 0 — Scaffolding.** Repo skeleton, config system, dataset registry seeded with the
inventory, the isolated-env story (§2.1) stubbed, `download_data.py`, and `describe`.
*Done when:* `describe` reproduces the inventory table (dims, #types, #donors) for ≥1 dataset.

**Phase 1 — Data + splits.** Loaders and all split schemes for a first small dataset (e.g.
Zheng68k or a PBMC set); preprocessing correct.
*Done when:* stratified and cross_donor splits produce disjoint, leakage-free indices (test in
§10) with class distributions logged.

**Phase 2 — Pipeline on `DevEmbedder`.** Bring the *entire* downstream path up end-to-end
using the dependency-free PCA/random embedder.
*Done when:* one command runs data → dev embeddings → probe → response table for a dataset
and writes a valid `responses.parquet`.

**Phase 3 — Probe + response matrix.** Finalize LR probe (CV over `C`), correctness scoring,
tidy-table assembly, and the `to_matrix` pivot with the orientation assertion.
*Done when:* response table validates against the schema and `to_matrix` returns
`(P, I)`-shaped `Y` with aligned ids; orientation test passes.

**Phase 4 — IRT core + synthetic validation.** Implement 1PL in both estimators; build and
pass the synthetic recovery harness (`MATH.md` §9); then 2PL; abilities with intervals.
*Done when:* recovery test passes (abilities recovered with calibrated intervals;
difficulties' correlation-with-truth rises with `I`) — **this is the gate before real models.**

**Phase 5 — One real embedder end-to-end.** Wire in one truly zero-shot model (UCE or
SCimilarity) in its isolated env; produce + cache embeddings; run the full pipeline.
*Done when:* a real model gets an ability estimate from cached embeddings, with the analysis
env never importing the embedder.

**Phase 6 — Scale out.** Register remaining embedders (incl. Cell-JEPA) and all annotation
datasets; assemble the full response matrix; full IRT fit.
*Done when:* a single config runs all models × all datasets and yields one ranking with
uncertainty.

**Phase 7 — Robustness / explanatory models.** Hierarchical + explanatory item difficulty
(numpyro, §11.1), aggregated `type_dataset` items (§11.2, addresses local independence),
dimensionality check (§11.3).
*Done when:* explanatory fit reports difficulty-vs-covariate effects and the aggregated-item
fit is available as a robustness comparison to the per-cell fit.

**Phase 8 — Reporting + polish.** All figures/tables from `MATH.md` §12, the per-run summary
report, and docs.
*Done when:* a run emits ranking + Wright map + difficulty/discrimination tables + fit
diagnostics + info curve, reproducibly from config.

---

## 9. Testing & validation strategy

- **Synthetic IRT recovery** (`test_synthetic_irt.py`) — the estimator's correctness gate
  (`MATH.md` §9). Non-negotiable, runs in CI.
- **Response-matrix orientation** (`test_response_orientation.py`) — construct a tiny known
  case and assert `Y` shape and that a deliberately strong/weak fake model lands high/low.
- **Split leakage** (`test_split_leakage.py`) — no `cell_id` (and, for cross_donor, no donor)
  appears in both train and test; embedders with `requires_dataset_fit` fit on train only.
- **Cache integrity** — same config+seed ⇒ identical response table hash; embedding artifact
  rows stay aligned to `cell_id`.
- **End-to-end smoke** — Phase 2 dev-embedder run wired into CI on a tiny subsampled dataset.

---

## 10. Known pitfalls / gotchas

1. **Embedder dependency hell** — the reason for the §2.1 boundary. Never co-install
   embedders; communicate only through cached artifacts.
2. **Response-matrix orientation** — persons↔items transposition silently corrupts every
   result; assert shape everywhere; catch via synthetic recovery.
3. **Local independence** (`MATH.md` §13) — same-type cells are correlated; run the aggregated
   `type_dataset` item mode as the robustness comparison.
4. **Few-respondents/many-items regime** (`MATH.md` §10) — prefer 1PL; use hierarchical/
   explanatory priors; don't over-read noisy per-cell `a_i`.
5. **Ceiling effects** (`MATH.md` §8.3) — in-distribution common types are near-trivial and
   don't separate models; deliberately include hard items (rare types, cross-donor/dataset).
6. **Label harmonization** — cell-type vocabularies differ across datasets; inconsistent
   mapping poisons both probe labels and difficulty aggregation. Centralize in
   `data/harmonize.py`; log the mapping.
7. **Duplicate datasets** — several pancreas/immune/PBMC sets recur across benchmarks under
   different names/preprocessing; dedupe before pooling or you double-count cells.
8. **Leakage via dataset-fit embedders** — fit scVI-style models on train cells only.
9. **Class imbalance** — heavily imbalanced sets (e.g. Zheng68k) affect probe training and
   accuracy interpretation; prefer macro-averaged reporting and stratified CV.
10. **Preprocessing mismatch** — each embedder needs its *own* official preprocessing; the
    wrong pipeline yields plausible-looking but meaningless embeddings.

---

## 11. Data notes (from the project inventory)

- Primary eval sets are the annotation benchmarks (Zheng68k, Pancreas/hPancreas, Multiple
  Sclerosis, PBMC variants, Cell Lines, MCA, DC, MHSP, MB Spatial, Liver cross, …).
- The atlas/integration sets (Immune human, Immune human+mouse, Lung, Pancreas-scIB, Immune
  atlas [46 types], Lung atlas [50 types], Heart atlas) carry cell-type labels and provide the
  **high-class-count, high-difficulty-spread** items IRT wants (`MATH.md` §8.3) — prioritize a
  few of these.
- The training corpora (STATE, Marson, X-Atlas/Orion, scJEPA) are for Cell-JEPA training;
  scJEPA's SE (164 cell types, thousands of donors) could double as a wide-spread eval set.
- Verify every matrix dimension against its source (the inventory's genes×cells convention has
  a few possibly-transposed entries) before relying on it.

---

## 12. Extension points

- **Raw-feature baseline as a respondent** — because a baseline is just another subject in the
  response matrix, adding LR-on-HVG or LR-on-PCA is a **config/registry change** (register it
  as an embedder), not an architecture change. Left out of v1 by design; enabling it is one line
  when/if the question shifts from "rank these embedders" to "do these embedders beat raw
  features."
- **New embedding models** — implement the `CellEmbedder` interface + an isolated env.
- **New tasks (State-Transition / perturbation)** — a different response model (continuous
  outcomes don't fit dichotomous IRT); a separate track, not a drop-in.
- **Cross-dataset transfer eval** — train probe on dataset A, test on harmonized B.
