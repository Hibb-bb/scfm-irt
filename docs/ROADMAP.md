# Roadmap status

The phases and acceptance criteria are defined in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) section 8. This file tracks where the code
actually is. `TODO(phase-N)` markers in the source point back here:

```bash
grep -rn "TODO(phase" src/ scripts/
```

| Phase | What | Status |
|---|---|---|
| 0 | Scaffolding, config system, dataset registry, `describe` | skeleton — registry seeded with 4 datasets, paths/URLs and expected dims still to fill in |
| 1 | Loaders, split schemes, label harmonization | not started |
| 2 | Full downstream path on `DevEmbedder` | not started |
| 3 | Probe + response matrix | response schema, `to_matrix`, and the orientation guard done; probe not started |
| 4 | IRT core + synthetic recovery — **the gate** | response curves, information functions, and the simulator done; all three estimators stubbed |
| 5 | One real embedder end-to-end | not started |
| 6 | Scale out to all models x datasets | not started |
| 7 | Explanatory / aggregated / MIRT robustness | not started |
| 8 | Reporting and figures | not started |

## Suggested order

Phases 0–2 are the unglamorous part and everything else depends on them. The `DevEmbedder`
path exists specifically so the whole pipeline can be debugged before any checkpoint is
downloaded — bring that up end-to-end before touching a real model.

Phase 4 is the gate. Until synthetic recovery passes for both estimators, a real ability
ranking cannot be distinguished from a plausible-looking bug, and the most likely bug
(persons/items transposed) produces output that looks entirely reasonable.

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
