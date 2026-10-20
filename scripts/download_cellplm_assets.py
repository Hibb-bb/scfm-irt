#!/usr/bin/env python
"""Download CellPLM pretrained checkpoint files from the official Dropbox share.

The public Dropbox folder is distributed as a single zip (~2.3 GB). This script
downloads it once, caches the archive, and extracts only the requested checkpoint
prefix from the embedded ``ckpt/`` directory.

    python scripts/download_cellplm_assets.py
    python scripts/download_cellplm_assets.py --out cache/cellplm_workdir/ckpt
    python scripts/download_cellplm_assets.py --prefix 20231027_85M

Expected files per prefix:
  {prefix}.config.json
  {prefix}.best.ckpt

Dropbox folder: https://www.dropbox.com/scl/fo/i5rmxgtqzg7iykt2e9uqm/h
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_PREFIX = "20230926_85M"
# Official shared-folder zip (rlkey from CellPLM README).
DROPBOX_FOLDER_ZIP = (
    "https://www.dropbox.com/scl/fo/i5rmxgtqzg7iykt2e9uqm/h"
    "?rlkey=o8hi0xads9ol07o48jdityzv1&dl=1"
)
BUNDLE_NAME = "cellplm_dropbox_bundle.zip"


def _required_files(out: Path, prefix: str) -> tuple[Path, ...]:
    return (out / f"{prefix}.config.json", out / f"{prefix}.best.ckpt")


def _complete(out: Path, prefix: str) -> bool:
    return all(p.exists() and p.stat().st_size > 0 for p in _required_files(out, prefix))


def _download_bundle(bundle_path: Path, *, force: bool = False) -> None:
    if bundle_path.exists() and not force:
        try:
            with zipfile.ZipFile(bundle_path) as zf:
                if any(n.startswith("ckpt/") for n in zf.namelist()):
                    print(f"[cellplm] using cached bundle {bundle_path}")
                    return
        except zipfile.BadZipFile:
            print(f"[cellplm] removing corrupt bundle {bundle_path}")
            bundle_path.unlink(missing_ok=True)

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = bundle_path.with_suffix(".partial")
    print(f"[cellplm] downloading Dropbox bundle (~2.3 GB) -> {bundle_path}")

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = 100.0 * done / total_size
        mb = done / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r[cellplm] {mb:.0f}/{total_mb:.0f} MiB ({pct:.1f}%)", end="", flush=True)

    urllib.request.urlretrieve(DROPBOX_FOLDER_ZIP, tmp, reporthook=_report)  # noqa: S310
    print()
    tmp.replace(bundle_path)
    print(f"[cellplm] saved bundle {bundle_path} ({bundle_path.stat().st_size / (1024**3):.2f} GiB)")


def _extract_prefix(bundle_path: Path, out: Path, prefix: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    wanted = {f"ckpt/{prefix}.config.json", f"ckpt/{prefix}.best.ckpt"}
    with zipfile.ZipFile(bundle_path) as zf:
        names = set(zf.namelist())
        missing = sorted(wanted - names)
        if missing:
            ckpt_names = sorted(n for n in names if n.startswith("ckpt/"))
            raise FileNotFoundError(
                f"prefix {prefix!r} not found in bundle; missing {missing}. "
                f"Available ckpt entries: {ckpt_names}"
            )
        for member in wanted:
            dest = out / Path(member).name
            print(f"[cellplm] extracting {member} -> {dest}")
            with zf.open(member) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def download(
    out_dir: Path,
    *,
    prefix: str = DEFAULT_PREFIX,
    force: bool = False,
    bundle_dir: Path | None = None,
) -> Path:
    out_dir = out_dir.resolve()
    cache_root = (bundle_dir or out_dir.parent).resolve()
    bundle_path = cache_root / BUNDLE_NAME

    if _complete(out_dir, prefix) and not force:
        sizes = {p.name: p.stat().st_size for p in _required_files(out_dir, prefix)}
        print(f"[ok] CellPLM checkpoint already at {out_dir}: {sizes}")
        return out_dir

    _download_bundle(bundle_path, force=force)
    _extract_prefix(bundle_path, out_dir, prefix)

    if not _complete(out_dir, prefix):
        raise RuntimeError(f"extraction failed; expected files missing under {out_dir}")

    sizes = {p.name: p.stat().st_size for p in _required_files(out_dir, prefix)}
    print(f"[ok] CellPLM checkpoint ready at {out_dir}: {sizes}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="cache/cellplm_workdir/ckpt",
        help="Directory for checkpoint files",
    )
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--bundle-dir",
        default=None,
        help="Where to cache the Dropbox zip (default: parent of --out)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract")
    args = parser.parse_args()

    download(
        Path(args.out),
        prefix=args.prefix,
        force=args.force,
        bundle_dir=Path(args.bundle_dir) if args.bundle_dir else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
