#!/usr/bin/env python
"""Download and validate UCE model assets.

UCE's own Figshare helper is broken in two ways for us:
  1. It hits figshare.com URLs that AWS WAF challenges (HTTP 202, empty body).
  2. It treats *any* existing path as a cache hit, including 0-byte failed downloads.

This script uses the working `ndownloader.figshare.com` endpoints, checks sizes, retries,
and extracts protein_embeddings.tar.gz.

    python scripts/download_uce_assets.py --uce-dir cache/uce_workdir/UCE
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

# Use ndownloader.figshare.com — figshare.com/ndownloader triggers AWS WAF challenges
# that return Content-Length: 0 (HTTP 202). Sizes from the Figshare API (article 24320806).
ASSETS: list[tuple[str, str, int]] = [
    (
        "model_files/species_chrom.csv",
        "https://ndownloader.figshare.com/files/42706558",
        1_000_000,
    ),
    (
        "model_files/species_offsets.pkl",
        "https://ndownloader.figshare.com/files/42706555",
        100,
    ),
    (
        "model_files/protein_embeddings.tar.gz",
        "https://ndownloader.figshare.com/files/42715213",
        2_000_000_000,
    ),
    (
        "model_files/all_tokens.torch",
        "https://ndownloader.figshare.com/files/42706585",
        2_000_000_000,
    ),
    (
        "model_files/4layer_model.torch",
        "https://ndownloader.figshare.com/files/42706576",
        2_000_000_000,
    ),
]


def _is_bad(path: Path, min_bytes: int) -> bool:
    return (not path.exists()) or path.stat().st_size < min_bytes


def download_one(url: str, dest: Path, min_bytes: int, retries: int = 5) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not _is_bad(dest, min_bytes):
        print(f"[ok] {dest} ({dest.stat().st_size:,} bytes)")
        return

    if dest.exists():
        print(f"[purge] undersized {dest} ({dest.stat().st_size} bytes)")
        dest.unlink()

    tmp = dest.with_suffix(dest.suffix + ".partial")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"[get] {dest.name} attempt {attempt}/{retries}")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; scfm-irt/0.0.1)"},
            )
            with urllib.request.urlopen(req, timeout=3600) as resp, open(tmp, "wb") as out:
                print(f"  -> {resp.geturl()[:80]}... status={resp.status}")
                shutil.copyfileobj(resp, out, length=16 * 1024 * 1024)
            size = tmp.stat().st_size
            if size < min_bytes:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"got {size} bytes, need >= {min_bytes}")
            tmp.replace(dest)
            print(f"[ok] {dest} ({size:,} bytes)")
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"[warn] attempt {attempt} failed: {exc}")
            tmp.unlink(missing_ok=True)
            time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"failed to download {dest}: {last_err}")


def extract_protein_embeddings(uce_dir: Path) -> None:
    tar_path = uce_dir / "model_files" / "protein_embeddings.tar.gz"
    pe_dir = uce_dir / "model_files" / "protein_embeddings"
    if pe_dir.is_dir() and any(pe_dir.rglob("*.pt")):
        n = len(list(pe_dir.rglob("*.pt")))
        print(f"[ok] {pe_dir} ({n} .pt files)")
        return
    if pe_dir.exists():
        shutil.rmtree(pe_dir)
    print(f"[extract] {tar_path}")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=tar_path.parent)
    pts = list((uce_dir / "model_files").rglob("*.pt"))
    # Prefer files under protein_embeddings/
    if not pe_dir.is_dir() or not any(pe_dir.rglob("*.pt")):
        if not pts:
            raise RuntimeError("protein_embeddings.tar.gz extracted but no .pt files found")
    print(f"[ok] protein embeddings ready ({len(list(pe_dir.rglob('*.pt')))} .pt under {pe_dir})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uce-dir", default="cache/uce_workdir/UCE")
    args = parser.parse_args()
    uce_dir = Path(args.uce_dir).resolve()
    if not (uce_dir / "eval_single_anndata.py").exists():
        raise SystemExit(f"UCE checkout missing at {uce_dir}")

    # Drop empty leftovers from earlier WAF-blocked attempts.
    mf = uce_dir / "model_files"
    if mf.is_dir():
        for p in mf.rglob("*"):
            if p.is_file() and p.stat().st_size == 0:
                print(f"[purge] empty {p}")
                p.unlink()

    for rel, url, min_bytes in ASSETS:
        download_one(url, uce_dir / rel, min_bytes)
    extract_protein_embeddings(uce_dir)
    print("[done] UCE assets ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
