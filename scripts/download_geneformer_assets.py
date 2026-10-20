#!/usr/bin/env python
"""Download the Geneformer package + one model checkpoint from Hugging Face.

Uses stdlib urllib + the HF tree API (no pip / huggingface_hub at runtime).

    python scripts/download_geneformer_assets.py
    python scripts/download_geneformer_assets.py --out cache/geneformer_workdir
    python scripts/download_geneformer_assets.py --model Geneformer-V2-104M

After download you should have:
  cache/geneformer_workdir/Geneformer/          # installable package (pip install .)
  cache/geneformer_workdir/Geneformer/<model>/  # e.g. Geneformer-V1-10M

Model weights are opt-in per `--model` and land side by side, so V1 and V2 can coexist in one
workdir and be selected by the corresponding embedder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ID = "ctheodoris/Geneformer"
MODEL_DIRNAME = "Geneformer-V1-10M"
HF_RESOLVE = f"https://huggingface.co/{REPO_ID}/resolve/main"
HF_TREE = f"https://huggingface.co/api/models/{REPO_ID}/tree/main"
# Package code + dictionaries, always. Model weights are opt-in per `--model` so a V1 run never
# pays for the much larger V2 checkpoints (and vice versa).
BASE_PREFIXES = (
    "geneformer/",
    "setup.py",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "MANIFEST.in",
)

# Only these model directories may be requested; anything else is a typo, not a new model.
KNOWN_MODELS = (
    "Geneformer-V1-10M",
    "Geneformer-V2-104M",
    "Geneformer-V2-316M",
)

# Mutated by main() to include the requested model directory.
ALLOW_PREFIXES: tuple[str, ...] = (*BASE_PREFIXES, f"{MODEL_DIRNAME}/")


def _allowed(rel: str) -> bool:
    return any(rel == p.rstrip("/") or rel.startswith(p) for p in ALLOW_PREFIXES)


def _api_get(url: str, retries: int = 5) -> list:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; scfm-irt/0.0.1)"},
    )
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(min(attempt * 5, 30))
    raise RuntimeError(f"HF API failed for {url}: {last}") from last


def _list_files(prefix: str = "") -> list[tuple[str, int]]:
    """Recursively list repo files under `prefix` as (path, size_bytes)."""
    out: list[tuple[str, int]] = []
    url = HF_TREE if not prefix else f"{HF_TREE}/{prefix}"
    for entry in _api_get(url):
        path = entry["path"]
        if entry["type"] == "directory":
            if _allowed(path + "/"):
                out.extend(_list_files(path))
        elif entry["type"] == "file" and _allowed(path):
            out.append((path, int(entry.get("size") or 0)))
    return out


def _download_one(rel: str, dest: Path, min_bytes: int = 1, retries: int = 5) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return
    if dest.exists():
        dest.unlink()
    tmp = dest.with_suffix(dest.suffix + ".partial")
    url = f"{HF_RESOLVE}/{rel}"
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; scfm-irt/0.0.1)"},
            )
            with urllib.request.urlopen(req, timeout=3600) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out, length=16 * 1024 * 1024)
            size = tmp.stat().st_size
            if size < min_bytes:
                raise OSError(f"undersized {rel}: {size} bytes")
            tmp.replace(dest)
            print(f"[ok] {rel} ({size:,} bytes)")
            return
        except Exception as exc:
            last = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            time.sleep(min(attempt * 10, 60))
    raise RuntimeError(f"download failed for {rel}: {last}") from last


def _model_ok(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    has_config = (model_dir / "config.json").exists()
    has_weights = any(
        (model_dir / name).exists() and (model_dir / name).stat().st_size > 1_000_000
        for name in ("pytorch_model.bin", "model.safetensors")
    )
    return has_config and has_weights


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="cache/geneformer_workdir",
        help="Parent directory; package lands in <out>/Geneformer",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--model",
        default=MODEL_DIRNAME,
        choices=KNOWN_MODELS,
        help="Model subdirectory to download and require (default: Geneformer-V1-10M)",
    )
    args = parser.parse_args()

    global ALLOW_PREFIXES
    ALLOW_PREFIXES = (*BASE_PREFIXES, f"{args.model}/")

    out = Path(args.out).resolve()
    repo_dir = out / "Geneformer"
    model_dir = repo_dir / args.model
    out.mkdir(parents=True, exist_ok=True)

    if _model_ok(model_dir) and (repo_dir / "geneformer").is_dir() and not args.force:
        total = sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file())
        print(f"[ok] Geneformer assets already at {repo_dir}")
        print(f"[ok] model {model_dir} ({total:,} bytes)")
        return 0

    print(f"[list] {REPO_ID}")
    files = _list_files()
    if not files:
        raise SystemExit("no files matched ALLOW_PREFIXES — check HF repo layout")

    print(f"[get] {len(files)} files -> {repo_dir}")
    for rel, expected in sorted(files):
        dest = repo_dir / rel
        min_bytes = max(1, int(expected * 0.5)) if expected > 0 else 1
        _download_one(rel, dest, min_bytes=min_bytes)

    if not (repo_dir / "geneformer").is_dir():
        raise SystemExit(f"package tree missing under {repo_dir}/geneformer")
    if not _model_ok(model_dir):
        raise SystemExit(f"model incomplete at {model_dir}")

    total = sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file())
    print(f"[done] {repo_dir}")
    print(f"[done] {model_dir}: {total:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
