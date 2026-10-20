#!/usr/bin/env python
"""Clone the scFoundation repo and validate a manually-staged checkpoint.

The code is fetched automatically. The weights are not, and cannot be: the authors distribute
`models.ckpt` only through a SharePoint share, and the Figshare and Hugging Face mirrors both
return 404 (upstream issue #76). SharePoint requires an interactive browser session, so this
script prints the URL and verifies the file once you have put it in place, rather than
pretending it can download it.

    python scripts/download_scfoundation_assets.py
    python scripts/download_scfoundation_assets.py --out cache/scfoundation_workdir

Expected layout afterwards:
  cache/scfoundation_workdir/scFoundation/model/get_embedding.py
  cache/scfoundation_workdir/scFoundation/model/models/models.ckpt   <- staged by hand

See https://github.com/biomap-research/scFoundation
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/biomap-research/scFoundation.git"
CKPT_URL = (
    "https://hopebio2020.sharepoint.com/:f:/s/PublicSharedfiles/"
    "EmUQnvZMETlDvoCaBduCNeIBQArcOrd8T8iEpiGofFZ9CQ?e=3SpPZU"
)
# The real checkpoint is ~1.6 GB; anything much smaller is an HTML error page.
MIN_CKPT_BYTES = 100_000_000


def _clone(repo_dir: Path) -> None:
    if (repo_dir / "model" / "get_embedding.py").exists():
        print(f"[ok] repo already at {repo_dir}")
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)]
    print(f"[get] {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"git clone failed (exit {proc.returncode})")
    if not (repo_dir / "model" / "get_embedding.py").exists():
        raise SystemExit(f"clone succeeded but model/get_embedding.py missing under {repo_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="cache/scfoundation_workdir", help="Parent workdir")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a models.ckpt staged elsewhere; copied into the repo layout",
    )
    args = parser.parse_args()

    out = Path(args.out).resolve()
    repo_dir = out / "scFoundation"
    _clone(repo_dir)

    models_dir = repo_dir / "model" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt = models_dir / "models.ckpt"

    if args.checkpoint:
        src = Path(args.checkpoint).resolve()
        if not src.exists():
            raise SystemExit(f"--checkpoint {src} does not exist")
        if src != ckpt:
            import shutil

            print(f"[copy] {src} -> {ckpt}")
            shutil.copy2(src, ckpt)

    if not ckpt.exists():
        print()
        print("[action required] scFoundation checkpoint is not staged.")
        print(f"  1. Open: {CKPT_URL}")
        print("  2. Download the file named 'models.ckpt' (~1.6 GB)")
        print(f"  3. Place it at: {ckpt}")
        print("     (or re-run this script with --checkpoint /path/to/models.ckpt)")
        print()
        return 1

    size = ckpt.stat().st_size
    if size < MIN_CKPT_BYTES:
        raise SystemExit(
            f"{ckpt} is only {size:,} bytes (expected >= {MIN_CKPT_BYTES:,}); "
            "the download was probably truncated or is an error page."
        )

    print(f"[done] repo   {repo_dir}")
    print(f"[done] ckpt   {ckpt} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
