#!/usr/bin/env python
"""Download scPRINT / scPRINT-2 checkpoints from Hugging Face.

Both model generations are published in the same repo (`jkobject/scPRINT`), so one script
covers them. Uses stdlib urllib against the HF resolve endpoint, matching
`download_geneformer_assets.py` — no huggingface_hub needed at runtime.

    python scripts/download_scprint_assets.py                 # both
    python scripts/download_scprint_assets.py --model scprint
    python scripts/download_scprint_assets.py --out cache/scprint_workdir

See https://huggingface.co/jkobject/scPRINT
"""

from __future__ import annotations

import argparse
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ID = "jkobject/scPRINT"
HF_RESOLVE = f"https://huggingface.co/{REPO_ID}/resolve/main"

# embedder name -> checkpoint filename in the HF repo.
CHECKPOINTS = {
    "scprint": "medium-v1.5.ckpt",
    "scprint2": "small-v2.ckpt",
    # v1 size ladder: same tokenizer and corpus, three capacities.
    "scprint_small_v1": "small-v1.ckpt",
    "scprint_medium_v1": "medium-v1.ckpt",
    "scprint_large_v1": "large-v1.ckpt",
}
# Real checkpoints are hundreds of MB; below this we got an error page, not weights.
MIN_BYTES = 50_000_000


def _download(rel: str, dest: Path, retries: int = 5) -> None:
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
            with urllib.request.urlopen(req, timeout=3600) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out, length=16 * 1024 * 1024)
            size = tmp.stat().st_size
            if size < MIN_BYTES:
                raise OSError(f"undersized {rel}: {size} bytes")
            tmp.replace(dest)
            print(f"[ok] {rel} ({size:,} bytes)")
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            tmp.unlink(missing_ok=True)
            time.sleep(min(attempt * 10, 60))
    raise RuntimeError(f"download failed for {rel}: {last}") from last


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="cache/scprint_workdir", help="Destination directory")
    parser.add_argument(
        "--model",
        default="all",
        choices=[*sorted(CHECKPOINTS), "all"],
        help="Which checkpoint to fetch (default: all)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    wanted = sorted(CHECKPOINTS) if args.model == "all" else [args.model]
    for model in wanted:
        rel = CHECKPOINTS[model]
        dest = out / rel
        if dest.exists() and dest.stat().st_size >= MIN_BYTES and not args.force:
            print(f"[ok] {model} already at {dest} ({dest.stat().st_size:,} bytes)")
            continue
        print(f"[get] {model}: {rel}")
        _download(rel, dest)

    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
