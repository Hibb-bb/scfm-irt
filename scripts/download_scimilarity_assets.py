#!/usr/bin/env python
"""Download SCimilarity v1.1 model weights from Zenodo (~30 GB tarball).

    python scripts/download_scimilarity_assets.py
    python scripts/download_scimilarity_assets.py --out cache/scimilarity_workdir
"""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

ZENODO_URL = "https://zenodo.org/records/10685499/files/model_v1.1.tar.gz"
EXPECTED_MD5 = "546251b7c435f3b1dbe38e2e420ad57f"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(out_dir: Path, *, force: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "model_v1.1"
    marker = model_dir / ".download_complete"

    if marker.exists() and not force:
        print(f"[ok] SCimilarity model already at {model_dir}")
        return model_dir

    archive = out_dir / "model_v1.1.tar.gz"
    if not archive.exists() or force:
        print(f"[download] SCimilarity model from Zenodo (~30 GB) -> {archive}")
        tmp = archive.with_suffix(".partial")
        urllib.request.urlretrieve(ZENODO_URL, tmp)  # noqa: S310
        tmp.replace(archive)

    digest = _md5(archive)
    if digest != EXPECTED_MD5:
        raise RuntimeError(f"MD5 mismatch for {archive}: got {digest}, expected {EXPECTED_MD5}")

    print(f"[extract] {archive} -> {out_dir}")
    if model_dir.exists() and force:
        import shutil

        shutil.rmtree(model_dir)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=out_dir, filter="data")

    # Zenodo tarball may unpack to model_v1.1/ or nested layout — normalize.
    if not model_dir.exists():
        candidates = [p for p in out_dir.iterdir() if p.is_dir() and "model" in p.name.lower()]
        if len(candidates) == 1:
            candidates[0].rename(model_dir)
        else:
            raise FileNotFoundError(f"could not locate extracted model dir under {out_dir}")

    marker.write_text("ok\n")
    print(f"[ok] SCimilarity ready at {model_dir}")
    return model_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="cache/scimilarity_workdir")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download(Path(args.out), force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
