"""UCE (Universal Cell Embeddings) — zero-shot foundation model.

Runs inside the UCE env only. Invokes UCE via a small driver script (not raw CLI flags)
because UCE's argparse uses `type=bool`, where `--filter False` is parsed as True.

Assets must be pre-downloaded with `scripts/download_uce_assets.py` — UCE's own Figshare
helper treats empty files as cache hits.

See https://github.com/snap-stanford/UCE
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData


class UCEEmbedder:
    name = "uce"
    version = "4layer"
    requires_dataset_fit = False
    label_supervised = False
    env = "envs/env-uce.yml"

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str = "cuda",
        species: str = "human",
        batch_size: int = 25,
        nlayers: int = 4,
        work_dir: str | None = None,
    ):
        self.checkpoint = checkpoint
        self.device = device
        self.species = species
        self.batch_size = batch_size
        self.nlayers = nlayers
        self.work_dir = Path(
            work_dir or __import__("os").environ.get("SCFM_UCE_WORKDIR", "cache/uce_workdir")
        ).resolve()

    def preprocess(self, adata: AnnData) -> AnnData:
        """UCE wants raw counts in `.X` and gene symbols in `.var_names` — do not log-normalize."""
        out = adata.copy()
        for key in list(out.obsm.keys()):
            del out.obsm[key]
        return out

    def fit(self, adata: AnnData) -> None:
        return None

    def _uce_root(self) -> Path:
        root = self.work_dir / "UCE"
        if not (root / "eval_single_anndata.py").exists():
            raise FileNotFoundError(
                f"UCE checkout missing at {root}. Clone it or run the SLURM setup first."
            )
        return root

    def _require_assets(self, uce_root: Path) -> dict[str, Path]:
        """Fail fast with a clear message if assets are missing/empty."""
        paths = {
            "spec_chrom_csv_path": uce_root / "model_files" / "species_chrom.csv",
            "offset_pkl_path": uce_root / "model_files" / "species_offsets.pkl",
            "token_file": uce_root / "model_files" / "all_tokens.torch",
            "protein_embeddings_dir": uce_root / "model_files" / "protein_embeddings",
            "model_loc": Path(self.checkpoint)
            if self.checkpoint
            else uce_root / "model_files" / "4layer_model.torch",
        }
        # Per-file floors — species_offsets.pkl is legitimately ~139 bytes.
        min_bytes = {
            "spec_chrom_csv_path": 1_000_000,
            "offset_pkl_path": 50,
            "token_file": 2_000_000_000,
            "model_loc": 2_000_000_000,
        }
        missing = []
        for key, path in paths.items():
            if key == "protein_embeddings_dir":
                if not path.is_dir() or not any(path.rglob("*.pt")):
                    missing.append(f"{key}={path} (no .pt files)")
            else:
                floor = min_bytes.get(key, 1000)
                if (not path.exists()) or path.stat().st_size < floor:
                    size = path.stat().st_size if path.exists() else 0
                    missing.append(f"{key}={path} (size={size}, need>={floor})")
        if missing:
            raise FileNotFoundError(
                "UCE model assets missing or empty:\n  - "
                + "\n  - ".join(missing)
                + "\nRun: python scripts/download_uce_assets.py "
                f"--uce-dir {uce_root}"
            )
        return paths

    def embed(self, adata: AnnData) -> np.ndarray:
        """Run UCE evaluation and return (n_cells, d) aligned to `adata.obs_names`."""
        import anndata as ad

        uce_root = self._uce_root()
        assets = self._require_assets(uce_root)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="uce_in_", dir=str(self.work_dir)) as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "input.h5ad"
            out_dir = tmp_path / "out"
            out_dir.mkdir()

            to_write = adata.copy()
            to_write.obs_names = to_write.obs_names.astype(str)
            to_write.var_names = to_write.var_names.astype(str)
            to_write.var.index.name = None
            for frame in (to_write.obs, to_write.var):
                for col in frame.columns:
                    frame[col] = frame[col].astype(str).astype(object)
            try:
                ad.settings.allow_write_nullable_strings = True
            except Exception:
                pass
            to_write.write_h5ad(in_path)

            # Driver avoids UCE's `type=bool` argparse bug (`--filter False` -> True).
            driver = tmp_path / "run_uce_driver.py"
            driver.write_text(
                textwrap.dedent(
                    f"""\
                    import sys
                    from pathlib import Path
                    sys.path.insert(0, {str(uce_root)!r})
                    from accelerate import Accelerator
                    from evaluate import AnndataProcessor

                    class Args:
                        adata_path = {str(in_path)!r}
                        dir = {str(out_dir) + "/"!r}
                        species = {self.species!r}
                        filter = False
                        skip = False
                        model_loc = {str(assets["model_loc"])!r}
                        batch_size = {int(self.batch_size)}
                        pad_length = 1536
                        pad_token_idx = 0
                        chrom_token_left_idx = 1
                        chrom_token_right_idx = 2
                        cls_token_idx = 3
                        CHROM_TOKEN_OFFSET = 143574
                        sample_size = 1024
                        CXG = True
                        nlayers = {int(self.nlayers)}
                        output_dim = 1280
                        d_hid = 5120
                        token_dim = 5120
                        multi_gpu = False
                        spec_chrom_csv_path = {str(assets["spec_chrom_csv_path"])!r}
                        token_file = {str(assets["token_file"])!r}
                        protein_embeddings_dir = {str(assets["protein_embeddings_dir"]) + "/"!r}
                        offset_pkl_path = {str(assets["offset_pkl_path"])!r}

                    args = Args()
                    Path(args.dir).mkdir(parents=True, exist_ok=True)
                    accelerator = Accelerator(project_dir=args.dir)
                    processor = AnndataProcessor(args, accelerator)
                    processor.preprocess_anndata()
                    processor.generate_idxs()
                    processor.run_evaluation()
                    """
                )
            )

            proc = subprocess.run(
                ["python", str(driver)],
                cwd=str(uce_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "UCE embedding failed.\n"
                    f"cmd stdout:\n{proc.stdout[-4000:]}\n"
                    f"cmd stderr:\n{proc.stderr[-4000:]}\n"
                    f"Assets dir: {uce_root / 'model_files'}"
                )

            outs = sorted(out_dir.rglob("*.h5ad"))
            # Prefer the final evaluated file (not *_proc.h5ad alone if both exist).
            preferred = [p for p in outs if not p.name.endswith("_proc.h5ad")]
            pick = preferred[-1] if preferred else (outs[-1] if outs else None)
            if pick is None:
                raise FileNotFoundError(f"UCE produced no h5ad under {out_dir}")

            result = ad.read_h5ad(pick)
            if "X_uce" not in result.obsm:
                raise KeyError(f"UCE output missing obsm['X_uce']; keys={list(result.obsm)}")

            emb = np.asarray(result.obsm["X_uce"], dtype=np.float32)
            if list(map(str, result.obs_names)) != list(map(str, adata.obs_names)):
                idx = {str(n): i for i, n in enumerate(result.obs_names)}
                missing = [str(n) for n in adata.obs_names if str(n) not in idx]
                if missing:
                    note = self.work_dir / "last_filter_note.json"
                    note.write_text(
                        json.dumps(
                            {
                                "n_input": int(adata.n_obs),
                                "n_embedded": int(len(idx)),
                                "n_missing": len(missing),
                                "missing_head": missing[:20],
                            },
                            indent=2,
                        )
                    )
                    raise RuntimeError(
                        f"UCE dropped {len(missing)}/{adata.n_obs} cells despite filter=False. "
                        f"See {note}"
                    )
                aligned = np.empty((adata.n_obs, emb.shape[1]), dtype=np.float32)
                for i, name in enumerate(map(str, adata.obs_names)):
                    aligned[i] = emb[idx[name]]
                emb = aligned

            if emb.shape[0] != adata.n_obs:
                raise ValueError(f"UCE returned {emb.shape[0]} rows for {adata.n_obs} cells")
            return emb
