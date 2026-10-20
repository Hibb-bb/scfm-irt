#!/usr/bin/env python
"""Download the scGPT whole-human checkpoint (Google Drive via gdown).

Official folder:
  https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y

    python scripts/download_scgpt_assets.py
    python scripts/download_scgpt_assets.py --out cache/scgpt_workdir/scGPT_human

Expected files after download: args.json, best_model.pt, vocab.json
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# whole-human (recommended default)
FOLDER_ID = "1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
REQUIRED = ("args.json", "best_model.pt", "vocab.json")


def _complete(path: Path) -> bool:
    return all((path / f).exists() and (path / f).stat().st_size > 0 for f in REQUIRED)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="cache/scgpt_workdir/scGPT_human",
        help="Destination directory for the checkpoint folder",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if _complete(out) and not args.force:
        sizes = {f: (out / f).stat().st_size for f in REQUIRED}
        print(f"[ok] checkpoint already at {out}: {sizes}")
        return 0

    try:
        import gdown
    except ImportError as exc:
        raise SystemExit(
            "gdown is required: pip install gdown\n"
            f"(original import error: {exc})"
        ) from exc

    # Download into a temp sibling, then move — gdown --folder creates a named subdir.
    staging = out.parent / "_scgpt_download_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
    print(f"[get] {url} -> {staging}")
    gdown.download_folder(url=url, output=str(staging), quiet=False, use_cookies=False)

    # gdown may nest as staging/scGPT_human/... or dump files directly into staging.
    candidates = [staging]
    candidates.extend(p for p in staging.iterdir() if p.is_dir())
    source = None
    for cand in candidates:
        if _complete(cand):
            source = cand
            break
        # Sometimes files land one level deeper with arbitrary folder name.
        for sub in cand.rglob("best_model.pt"):
            parent = sub.parent
            if _complete(parent):
                source = parent
                break
        if source:
            break

    if source is None:
        found = list(staging.rglob("*"))
        raise SystemExit(
            f"download finished but required files {REQUIRED} not found under {staging}. "
            f"Saw {len(found)} paths."
        )

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(source, out)
    shutil.rmtree(staging, ignore_errors=True)

    if not _complete(out):
        raise SystemExit(f"copy incomplete at {out}")
    sizes = {f: (out / f).stat().st_size for f in REQUIRED}
    print(f"[done] {out}: {sizes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
