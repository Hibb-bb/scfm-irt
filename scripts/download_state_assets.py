#!/usr/bin/env python
"""Download Arc Institute State SE-600M weights from Hugging Face.

Uses stdlib urllib against the HF resolve endpoint (no huggingface_hub needed).

    python scripts/download_state_assets.py
    python scripts/download_state_assets.py --out cache/state_workdir/SE-600M

See https://huggingface.co/arcinstitute/SE-600M
"""

from __future__ import annotations

import argparse
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ID = "arcinstitute/SE-600M"
HF_RESOLVE = f"https://huggingface.co/{REPO_ID}/resolve/main"

# Files needed for `state emb transform --model-folder ...`.
FILES = {
    "se600m_epoch16.ckpt": 10_000_000_000,
    "protein_embeddings.pt": 300_000_000,
    "config.yaml": 1_000,
}


def _download(rel: str, dest: Path, min_bytes: int, retries: int = 5) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    url = f"{HF_RESOLVE}/{rel}"
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; scfm-irt/0.0.1)"},
            )
            with urllib.request.urlopen(req, timeout=7200) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out, length=16 * 1024 * 1024)
            size = tmp.stat().st_size
            if size < min_bytes:
                raise OSError(f"undersized {rel}: {size} bytes (expected >= {min_bytes})")
            tmp.replace(dest)
            print(f"[ok] {rel} ({size:,} bytes)")
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            tmp.unlink(missing_ok=True)
            time.sleep(min(attempt * 15, 90))
    raise RuntimeError(f"download failed for {rel}: {last}") from last


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="cache/state_workdir/SE-600M",
        help="Directory for SE-600M checkpoint files",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    for rel, min_bytes in FILES.items():
        dest = out / rel
        if dest.exists() and dest.stat().st_size >= min_bytes and not args.force:
            print(f"[ok] {rel} already present ({dest.stat().st_size:,} bytes)")
            continue
        if args.force:
            dest.unlink(missing_ok=True)
        print(f"[get] {REPO_ID}/{rel}")
        _download(rel, dest, min_bytes)

    ckpts = sorted(out.glob("*.ckpt"))
    if not ckpts:
        raise SystemExit(f"no .ckpt files under {out}")
    print(f"[done] model folder ready: {out} ({len(ckpts)} checkpoint(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
