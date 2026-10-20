"""scPRINT and scPRINT-2 — zero-shot large cell models (Cantini lab).

Runs inside `envs/env-scprint.yml` only. Uses the Python `Embedder` task rather than the
`scprint embed` CLI: the CLI appends a PCA + UMAP pass, which is wasted work here and blows up
memory on a 50k-cell atlas.

Setup that must happen before any job runs (network-bound, so on a login node):

    bash scripts/setup_embedder_venv.sh scprint     # includes lamin init + ontology populate
    python scripts/download_scprint_assets.py       # checkpoints from HuggingFace

Checkpoints live at https://huggingface.co/jkobject/scPRINT (both v1 and v2).

See https://github.com/cantinilab/scPRINT and https://github.com/cantinilab/scPRINT-2
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from scfm_irt.embedders.base import align_embedding_rows

if TYPE_CHECKING:
    from anndata import AnnData

HUMAN_ONTOLOGY = "NCBITaxon:9606"
EMB_KEY = "scprint_emb"
# scDataLoader replaces obs.index with UUIDs; carry the original barcode here so we can map
# embeddings back onto the response matrix afterwards.
BARCODE_COL = "_scfm_barcode"


def _patch_simpler_flash_use_flash_attn() -> None:
    """scprint passes ``use_flash_attn`` into FlashTransformer, which forwards it to MHA.

    simpler_flash 1.3.x MHA does not accept that kwarg unless flash-attn is installed.
    Map it to ``attn_type`` and drop it before layer construction.
    """
    import simpler_flash.flashformer as ff

    if getattr(ff, "_scfm_patched_use_flash_attn", False):
        return

    _orig_init = ff.FlashTransformer.__init__

    def _patched_init(self, *args, **kwargs):
        use_flash = kwargs.pop("use_flash_attn", None)
        if use_flash is True:
            kwargs.setdefault("attn_type", "flash")
        elif use_flash is False:
            kwargs.setdefault("attn_type", "normal")
        return _orig_init(self, *args, **kwargs)

    ff.FlashTransformer.__init__ = _patched_init  # type: ignore[method-assign]
    ff._scfm_patched_use_flash_attn = True


def _load_lightning_ckpt(cls, path: Path, **kwargs):  # noqa: ANN001, ANN202
    """Load a scPRINT Lightning checkpoint on clusters without flash-attn."""
    import inspect
    import os
    import tempfile

    import torch

    _patch_simpler_flash_use_flash_attn()
    allowed = set(inspect.signature(cls.__init__).parameters) - {"self"}
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = {
        key: value
        for key, value in (ckpt.get("hyper_parameters") or {}).items()
        if key in allowed
    }
    hp["transformer"] = "normal"
    ckpt["hyper_parameters"] = hp

    fd, tmp_name = tempfile.mkstemp(suffix=".ckpt", prefix="scprint_")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        torch.save(ckpt, tmp)
        return cls.load_from_checkpoint(
            str(tmp),
            precpt_gene_emb=None,
            transformer="normal",
            **kwargs,
        )
    finally:
        tmp.unlink(missing_ok=True)


class SCPRINTEmbedder:
    """scPRINT v1 (medium-v1.5): ~50M cells, human + mouse."""

    name = "scprint"
    version = "medium-v1.5"
    requires_dataset_fit = False
    label_supervised = False
    env = "envs/env-scprint.yml"
    ckpt_name = "medium-v1.5.ckpt"

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str = "cuda",
        batch_size: int = 32,
        max_len: int = 4000,
        work_dir: str | None = None,
    ):
        self.checkpoint = checkpoint
        self.device = device
        self.batch_size = batch_size
        self.max_len = max_len
        self.work_dir = Path(
            work_dir or os.environ.get("SCFM_SCPRINT_WORKDIR", "cache/scprint_workdir")
        ).resolve()

    def _ckpt_path(self) -> Path:
        path = (
            Path(self.checkpoint).resolve()
            if self.checkpoint
            else self.work_dir / self.ckpt_name
        )
        if not path.exists():
            raise FileNotFoundError(
                f"scPRINT checkpoint not found at {path}. "
                f"Run: python scripts/download_scprint_assets.py --model {self.name}"
            )
        return path

    def preprocess(self, adata: AnnData) -> AnnData:
        """Raw counts, human ontology term, and scDataLoader's own gene validation.

        scPRINT takes raw counts and aligns genes itself, so the only thing added here is the
        organism term it requires in `obs`. The `Preprocessor` is run with every cell-dropping
        filter disabled, because a QC filter here would silently shrink the response matrix.

        Idempotent: `extract_embeddings.py` calls this before `embed()`, and `embed()` calls it
        again on the result. A second pass with Ensembl-indexed var names and `is_symbol=True`
        drops every gene — that was the scPRINT family's failure mode on the cluster.
        """
        if self._already_preprocessed(adata):
            out = adata
            if BARCODE_COL not in out.obs.columns:
                out = out.copy()
                out.obs[BARCODE_COL] = out.obs_names.astype(str)
            return out

        out = adata.copy()
        for key in list(out.obsm.keys()):
            del out.obsm[key]

        totals = out.X.sum(axis=1)
        if hasattr(totals, "A1"):
            totals = totals.A1
        if float(np.asarray(totals).mean() or 0.0) <= 50:
            raise ValueError(
                "scPRINT expects raw counts in .X; this matrix looks normalized "
                "(mean total per cell <= 50)."
            )

        out.obs["organism_ontology_term_id"] = HUMAN_ONTOLOGY
        out.obs[BARCODE_COL] = out.obs_names.astype(str)
        out = _run_preprocessor(out, model_name=self.name)
        # scDataLoader hard-codes a UUID obs index (preprocess.py:424). Put the original barcodes
        # back so extract_embeddings caches the same cell_ids the probe pipeline expects.
        out.obs_names = out.obs[BARCODE_COL].astype(str)
        return out

    @staticmethod
    def _already_preprocessed(adata: AnnData) -> bool:
        """True once scDataLoader has run — `embed()` must not run it again."""
        return "unseen_genes" in adata.uns and "ensembl_gene_id" in adata.var.columns

    def fit(self, adata: AnnData) -> None:
        return None

    def _load_model(self):  # noqa: ANN202
        from scprint import scPrint

        return _load_lightning_ckpt(scPrint, self._ckpt_path())

    def _make_embedder_task(self):
        """Build scPRINT v1 Embedder with only kwargs this installed version accepts."""
        import torch
        from scprint.tasks import Embedder

        wanted: dict[str, Any] = {
            "batch_size": self.batch_size,
            "max_len": self.max_len,
            "num_workers": 4,
            "doplot": False,
            "doclass": False,
            "output_expression": "none",
            "dtype": torch.float16 if self.device.startswith("cuda") else torch.float32,
        }
        accepted = set(inspect.signature(Embedder.__init__).parameters)
        return Embedder(**{k: v for k, v in wanted.items() if k in accepted})

    def embed(self, adata: AnnData) -> np.ndarray:
        prepared = self.preprocess(adata)
        model = self._load_model()
        if self.device.startswith("cuda"):
            import torch

            model = model.to("cuda").half()

        embedder = self._make_embedder_task()
        result = embedder(model, prepared, cache=False)
        embedded = result[0] if isinstance(result, tuple) else result

        if EMB_KEY not in embedded.obsm:
            raise KeyError(
                f"expected obsm[{EMB_KEY!r}] from scPRINT; got {list(embedded.obsm)}"
            )
        emb = np.asarray(embedded.obsm[EMB_KEY], dtype=np.float32)
        if BARCODE_COL in embedded.obs.columns:
            got_ids = embedded.obs[BARCODE_COL].astype(str).to_numpy()
        else:
            got_ids = embedded.obs_names.to_numpy()
        return align_embedding_rows(
            emb,
            got_ids,
            adata.obs_names.to_numpy(),
            model_name=self.name,
        )


class SCPRINT2Embedder(SCPRINTEmbedder):
    """scPRINT-2 (small-v2): ~350M cells across 22k datasets and 16 species.

    Separate `name` so it is a separate respondent; `pipeline.py` keys the cache and the IRT
    `model_id` off `name`.
    """

    name = "scprint2"
    version = "small-v2"
    ckpt_name = "small-v2.ckpt"

    def _load_model(self):  # noqa: ANN202
        from scprint2 import scPRINT2

        return _load_lightning_ckpt(
            scPRINT2,
            self._ckpt_path(),
            gene_pos_file=None,
        )

    def embed(self, adata: AnnData) -> np.ndarray:
        from scprint2.tasks import Embedder

        prepared = self.preprocess(adata)
        model = self._load_model()
        model.organisms = [HUMAN_ONTOLOGY]
        if self.device.startswith("cuda"):
            import torch

            model = model.to("cuda")

        wanted: dict[str, Any] = {
            "batch_size": self.batch_size,
            "max_len": self.max_len,
            "num_workers": 4,
            "doplot": False,
            "doclass": False,
            "pred_embedding": ["all"],
            "use_knn": False,
        }
        accepted = set(inspect.signature(Embedder.__init__).parameters)
        embedder = Embedder(**{k: v for k, v in wanted.items() if k in accepted})
        embedded, _metrics = embedder(model, prepared)

        if EMB_KEY not in embedded.obsm:
            raise KeyError(
                f"expected obsm[{EMB_KEY!r}] from scPRINT-2; got {list(embedded.obsm)}"
            )
        emb = np.asarray(embedded.obsm[EMB_KEY], dtype=np.float32)
        if BARCODE_COL in embedded.obs.columns:
            got_ids = embedded.obs[BARCODE_COL].astype(str).to_numpy()
        else:
            got_ids = embedded.obs_names.to_numpy()
        return align_embedding_rows(
            emb,
            got_ids,
            adata.obs_names.to_numpy(),
            model_name=self.name,
        )


# --- v1 size ladder -----------------------------------------------------------------
#
# scPRINT publishes small/medium/large at the same v1 generation, which is unusual and useful:
# the three differ in capacity while holding the tokenizer, corpus, and training recipe fixed.
# That makes "does more capacity help on the cells IRT calls hard?" answerable within one
# family, instead of confounded with every other difference between two vendors' models.
#
# `scPrint.load_from_checkpoint` reads architecture hyperparameters out of the checkpoint, so
# the size difference needs no code beyond pointing at a different file.


class SCPRINTSmallV1Embedder(SCPRINTEmbedder):
    name = "scprint_small_v1"
    version = "small-v1"
    ckpt_name = "small-v1.ckpt"


class SCPRINTMediumV1Embedder(SCPRINTEmbedder):
    name = "scprint_medium_v1"
    version = "medium-v1"
    ckpt_name = "medium-v1.ckpt"


class SCPRINTLargeV1Embedder(SCPRINTEmbedder):
    name = "scprint_large_v1"
    version = "large-v1"
    ckpt_name = "large-v1.ckpt"

    def __init__(self, *args, **kwargs):
        # ~1.2 GB of weights; the default batch of 32 runs out of memory on a 40 GB A100.
        kwargs.setdefault("batch_size", 16)
        super().__init__(*args, **kwargs)


def _run_preprocessor(adata: AnnData, model_name: str) -> AnnData:
    """Run scDataLoader's `Preprocessor` with all cell-dropping filters turned off.

    Its defaults filter outlier cells (~15% on the upstream demo). Losing cells mid-pipeline
    would silently change which items each model answered, so every threshold that can drop a
    cell is relaxed, and anything still lost is caught by the row-alignment check afterwards.
    """
    from scdataloader import Preprocessor

    wanted: dict[str, Any] = {
        "filter_gene_by_counts": False,
        "filter_cell_by_counts": False,
        "min_nnz_genes": 0,
        "min_dataset_size": 1,
        # Default 10000 rejects pbmc3k (~19k ontology-mapped genes after symbol join, but the
        # check runs before zero-padding unseen genes — without this the whole family fails).
        "min_valid_genes_id": 1000,
        "maxdropamount": 1_000_000,
        "madoutlier": 1_000,
        "pct_mt_outlier": 100,
        "do_postp": False,
        "skip_validate": True,
        "is_symbol": True,
        "force_preprocess": True,
    }
    # Keyword names drift between scDataLoader releases; pass only what this one accepts.
    accepted = set(inspect.signature(Preprocessor.__init__).parameters)
    kwargs = {k: v for k, v in wanted.items() if k in accepted}

    out = Preprocessor(**kwargs)(adata.copy())
    if out.n_obs != adata.n_obs:
        raise RuntimeError(
            f"{model_name}: scDataLoader Preprocessor dropped "
            f"{adata.n_obs - out.n_obs}/{adata.n_obs} cells despite filters being disabled; "
            "the response matrix would no longer be comparable across models."
        )
    return out
