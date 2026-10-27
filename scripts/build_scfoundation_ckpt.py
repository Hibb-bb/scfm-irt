#!/usr/bin/env python
"""Rebuild scFoundation's `models.ckpt` from the public Hugging Face re-upload.

The official checkpoint (`models.ckpt`, keys `cell`, `gene`, `rde`) is only distributed through a
SharePoint link that needs a manual download. The same three weight sets are re-uploaded under
Apache-2.0 by PerturbLab (`perturblab/scfoundation-{cell,gene,rde}`), each as a plain PyTorch state
dict (`model.pt`) plus a flat `config.json`. This script downloads them and writes a `models.ckpt`
in the layout `scFoundation/model/load.py::load_model_frommmf` reads, so the official
`get_embedding.py` and our adapter run unchanged:

    ckpt[key] = {"config": {"model": "mae_autobin",
                            "model_config": {"mae_autobin": {...}},
                            "dataset_config": {"rnaseq": {}}},
                 "state_dict": {"model.<param>": tensor, ...}}

Safety: the state dicts are loaded with `torch.load(weights_only=True)`, which refuses any pickled
object other than tensors and plain containers, so nothing in the downloaded files can execute.

Fidelity checks: `seq_len` is read from the positional-embedding table (`max_seq_len + 1` rows) and
must be 19266 (19264 genes + 2 depth tokens); every state dict is loaded into the official model
class with `strict=True`, so any missing, extra or mis-shaped weight fails here, not at embed time.

    .venv-scfoundation/bin/python scripts/build_scfoundation_ckpt.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KEYS = ("cell", "gene", "rde")
HF = "https://huggingface.co/perturblab/scfoundation-{key}/resolve/main/{file}"


def fetch(key: str, file: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[scf] downloading {key}/{file}", flush=True)
    urllib.request.urlretrieve(HF.format(key=key, file=file), tmp)
    tmp.rename(dest)
    return dest


def build_entry(key: str, raw: Path) -> dict:
    import torch

    cfg = json.loads(fetch(key, "config.json", raw / f"config_{key}.json").read_text())
    sd = torch.load(fetch(key, "model.pt", raw / f"model_{key}.pt"), map_location="cpu", weights_only=True)
    if not isinstance(sd, dict) or "pos_emb.weight" not in sd:
        raise SystemExit(f"[scf] {key}: not a scFoundation state dict")
    seq_len = int(sd["pos_emb.weight"].shape[0]) - 1
    if seq_len != 19266:
        raise SystemExit(f"[scf] {key}: positional table implies seq_len={seq_len}, expected 19266")
    if cfg.get("model_type") != "mae_autobin":
        raise SystemExit(f"[scf] {key}: unexpected model_type {cfg.get('model_type')}")

    def sub(prefix: str) -> dict:
        return {
            "hidden_dim": cfg[f"{prefix}_hidden_dim"],
            "depth": cfg[f"{prefix}_depth"],
            "heads": cfg[f"{prefix}_heads"],
            "dim_head": cfg[f"{prefix}_dim_head"],
            "module_type": cfg[f"{prefix}_module_type"],
            "ff_dropout": cfg.get("ff_dropout", 0.0),
            "attn_dropout": cfg.get("attn_dropout", 0.0),
        }

    model_cfg = {
        "model": "mae_autobin",
        "n_class": 104,  # stored by MaeAutobin but unused in its forward pass
        "seq_len": seq_len,
        "encoder": sub("encoder"),
        "decoder": sub("decoder"),
        "bin_alpha": cfg["bin_alpha"],
        "bin_num": cfg["bin_num"],
        "pad_token_id": cfg["pad_token_id"],
        "mask_token_id": cfg["mask_token_id"],
    }
    return {
        "config": {"model": "mae_autobin", "model_config": {"mae_autobin": model_cfg}, "dataset_config": {"rnaseq": {}}},
        "state_dict": OrderedDict((f"model.{k}", v) for k, v in sd.items()),
    }


def verify(entry: dict, repo_model: Path) -> None:
    """Instantiate the official model from the converted config and load the weights strictly."""
    sys.path.insert(0, str(repo_model))
    from load import convertconfig  # noqa: E402  (official scFoundation code)
    from pretrainmodels.select_model import select_model  # noqa: E402

    conv = convertconfig(entry)
    model = select_model(conv["config"])
    model.load_state_dict(conv["model_state_dict"], strict=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    work = Path(os.environ.get("SCFM_SCFOUNDATION_WORKDIR", REPO / "cache" / "scfoundation_workdir"))
    ap.add_argument("--work-dir", type=Path, default=work)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch

    repo_model = args.work_dir / "scFoundation" / "model"
    out = repo_model / "models" / "models.ckpt"
    if out.exists() and not args.force:
        print(f"[scf] {out} exists (use --force to rebuild)")
        return
    raw = args.work_dir / "perturblab"
    ckpt = {}
    for key in KEYS:
        ckpt[key] = build_entry(key, raw)
        verify(ckpt[key], repo_model)
        print(f"[scf] {key}: {len(ckpt[key]['state_dict'])} tensors, strict load ok", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    print(f"[scf] wrote {out} ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
