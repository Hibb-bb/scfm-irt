# Environments

One environment per embedder, plus one for everything else.

`env-analysis.yml` is the light side — probe, response matrix, IRT, analysis. It has no
embedder dependencies and must stay that way.

Every other file here is a heavy environment for a single model. They are mutually
incompatible by design (conflicting torch versions, conflicting transformers pins, model repos
installed from source), and nothing tries to reconcile them. They never need to coexist:

```bash
conda env create -f envs/env-scgpt.yml
conda run -n scfm-irt-scgpt python scripts/extract_embeddings.py --embedder scgpt --dataset zheng68k
```

That writes a cached artifact, and the analysis env picks it up from there. The artifact is the
only thing that crosses between them.

## Adding a model

1. Copy `env-embedder-template.yml`, pin against the model's *own* repo and its documented
   dependency versions.
2. Implement the `CellEmbedder` protocol, wrapping the model's official preprocessing —
   substituting a generic pipeline gives embeddings that look plausible and mean nothing.
3. Register a lazy factory in `src/scfm_irt/embedders/registry.py`.
4. Extract embeddings, then add the model to a run config. It is now a respondent.

Pin exact versions when the model actually works; record them per run. These repos change their
loading APIs, so verify against the source rather than against any example in this repo.
