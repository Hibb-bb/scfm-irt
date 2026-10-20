# Roadmap status

The phases and acceptance criteria are defined in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) section 8. This file tracks where the code
actually is. `TODO(phase-N)` markers in the source point back here:

```bash
grep -rn "TODO(phase" src/ scripts/
```

| Phase | What | Status |
|---|---|---|
| 0 | Scaffolding, config system, dataset registry, `describe` | done for smoke + `pbmc3k` download |
| 1 | Loaders, split schemes, label harmonization | done — harmonize is identity until `_LABEL_MAP` is populated |
| 2 | Full downstream path on `DevEmbedder` | done — `slurm/smoke_dev.slurm` |
| 3 | Probe + response matrix | done |
| 4 | IRT core + synthetic recovery — **the gate** | done — `mle_autodiff` + mean-field `bayes_variational` (py-irt optional on Py&lt;3.12) |
| 5 | One real embedder end-to-end | done — UCE on PBMC3k (`runs/uce_pbmc3k_v1`) |
| 6 | Scale out to all models x datasets | in progress — PBMC3k 4-model ranking done; atlas + SCimilarity queued |

| 7 | Explanatory / aggregated / MIRT robustness | aggregated path in autodiff; numpyro still TODO |
| 8 | Reporting and figures | minimal report; Wright/ICC/fit-stats still TODO |

## Open questions

- **Which datasets.** Prioritize a few high-class-count atlas sets — that is where the
  difficulty spread comes from. Without them, most items sit at the ceiling and abilities
  barely separate (MATH.md section 8.3).
- **Label harmonization scope.** Full Cell Ontology mapping is a real project on its own.
  Consider whether v1 can fit each dataset separately with per-dataset labels and pool only at
  the response-matrix level, which avoids the mapping entirely at the cost of losing
  cross-dataset difficulty comparisons.
- **SCimilarity's status.** It was pretrained with cell-type labels, so its ability is not
  comparable to the zero-shot models on equal footing. Decide whether it appears in the
  headline ranking or only in a supplementary one.
- **Raw-feature baseline.** Out of v1 by design, and one registry entry to add. Worth adding
  the moment anyone asks whether these embedders beat LR on HVGs — which they will.
