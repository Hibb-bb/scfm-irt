#!/usr/bin/env python
"""Download the LangCell checkpoint modules (Google Drive via gdown).

Official folder:
  https://drive.google.com/drive/folders/1cuhVG9v0YoAnjW-t_WMpQQguajumCBTp

    python scripts/download_langcell_assets.py
    python scripts/download_langcell_assets.py --out cache/langcell_workdir

The checkpoint is split into five modules (cell_bert, text_bert, cell_proj, text_proj,
ctm_head). Only `cell_bert` is needed here — we use the cell encoder's [CLS] output and never
touch the text tower — but the whole folder is fetched since it downloads as one unit.

See https://github.com/PharMolix/LangCell
"""

from __future__ import annotations

import argparse
from pathlib import Path

FOLDER_ID = "1cuhVG9v0YoAnjW-t_WMpQQguajumCBTp"
# The only module this embedder actually loads.
REQUIRED_SUBDIR = "cell_bert"
REQUIRED_FILES = ("config.json",)


def _complete(path: Path) -> bool:
    cell_bert = path / REQUIRED_SUBDIR
    if not cell_bert.is_dir():
        return False
    if not all((cell_bert / f).exists() for f in REQUIRED_FILES):
        return False
    return any(
        (cell_bert / name).exists() and (cell_bert / name).stat().st_size > 1_000_000
        for name in ("pytorch_model.bin", "model.safetensors")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="cache/langcell_workdir", help="Parent workdir")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.out).resolve() / "langcell"
    out.parent.mkdir(parents=True, exist_ok=True)

    if _complete(out) and not args.force:
        print(f"[ok] LangCell checkpoint already at {out}")
        return 0

    try:
        import gdown
    except ImportError as exc:
        raise SystemExit(
            "gdown is required: pip install gdown\n"
            f"(original import error: {exc})"
        ) from exc

    url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"

    # List first, then download only cell_bert. The shared folder also contains the text tower
    # and a multi-hundred-file demo dataset we never touch; pulling all of it burns through
    # Google's anonymous rate limit long before reaching the weights we actually need.
    print(f"[list] {url}")
    try:
        listing = gdown.download_folder(
            url=url, skip_download=True, quiet=True, use_cookies=False
        )
    except Exception as exc:
        raise SystemExit(_rate_limit_message(exc, url, out)) from exc

    wanted = [f for f in listing if f"/{REQUIRED_SUBDIR}/" in f"/{f.path}"]
    if not wanted:
        sample = sorted({str(f.path).split("/")[0] for f in listing})[:10]
        raise SystemExit(
            f"no files under {REQUIRED_SUBDIR}/ in the shared folder; top level: {sample}"
        )

    dest = out / REQUIRED_SUBDIR
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[get] {len(wanted)} file(s) -> {dest}")

    for entry in wanted:
        target = dest / Path(entry.path).name
        if target.exists() and target.stat().st_size > 0:
            print(f"[skip] {target.name} (already present)")
            continue
        try:
            gdown.download(
                id=entry.id, output=str(target), quiet=False, use_cookies=False, resume=True
            )
        except Exception as exc:
            raise SystemExit(_rate_limit_message(exc, url, out)) from exc

    if not _complete(out):
        present = sorted(p.name for p in dest.glob("*"))
        raise SystemExit(
            f"download incomplete at {dest}; got {present}. "
            f"Need {REQUIRED_FILES} plus a weights file."
        )

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"[done] {out}: {total:,} bytes")
    return 0


def _rate_limit_message(exc: Exception, url: str, out: Path) -> str:
    """Explain a gdown failure as the throttle it usually is, not a permissions problem."""
    return (
        f"gdown failed: {exc}\n\n"
        "Google Drive rate-limits anonymous downloads of popular folders, and reports it as a\n"
        "permissions error even though the folder is public. Options:\n"
        "  1. Wait and re-run this script; already-downloaded files are skipped.\n"
        f"  2. Download the folder in a browser from {url}\n"
        f"     and place its 'cell_bert' directory at {out / REQUIRED_SUBDIR}\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
