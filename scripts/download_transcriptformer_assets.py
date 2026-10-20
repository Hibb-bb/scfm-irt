#!/usr/bin/env python
"""Download TranscriptFormer model weights via the package's own CLI.

Must run inside the TranscriptFormer venv (`.venv-transcriptformer`), since it shells out to
the `transcriptformer download` entrypoint, which pulls artifacts from the CZI S3 bucket.

    python scripts/download_transcriptformer_assets.py
    python scripts/download_transcriptformer_assets.py --model tf_exemplar
    python scripts/download_transcriptformer_assets.py --out cache/tf_workdir

After download you should have:
  cache/tf_workdir/checkpoints/tf_sapiens/

Only human data is in scope here, so TF-Sapiens is the default; the cross-species checkpoints
are much larger and buy nothing for our datasets.

See https://github.com/czi-ai/transcriptformer
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

# CLI slugs use hyphens, checkpoint directories use underscores.
MODELS = {
    "tf_sapiens": "tf-sapiens",
    "tf_exemplar": "tf-exemplar",
    "tf_metazoa": "tf-metazoa",
}
# The weights are multi-GB; anything smaller means a truncated or failed download.
MIN_TOTAL_BYTES = 500_000_000


def _dir_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="cache/tf_workdir", help="Parent workdir")
    parser.add_argument("--model", default="tf_sapiens", choices=sorted(MODELS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    ckpt_root = out / "checkpoints"
    model_dir = ckpt_root / args.model
    out.mkdir(parents=True, exist_ok=True)

    total = _dir_bytes(model_dir)
    if total >= MIN_TOTAL_BYTES and not args.force:
        print(f"[ok] {args.model} already at {model_dir} ({total:,} bytes)")
        return 0

    if args.force and model_dir.exists():
        shutil.rmtree(model_dir)

    if shutil.which("transcriptformer") is None:
        raise SystemExit(
            "transcriptformer CLI not found on PATH.\n"
            "Run this inside the TranscriptFormer venv:\n"
            "  bash scripts/setup_embedder_venv.sh transcriptformer\n"
            "  source .venv-transcriptformer/bin/activate"
        )

    cmd = [
        "transcriptformer",
        "download",
        MODELS[args.model],
        "--checkpoint-dir",
        str(ckpt_root),
    ]
    print(f"[get] {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"transcriptformer download failed (exit {proc.returncode})")

    total = _dir_bytes(model_dir)
    if total < MIN_TOTAL_BYTES:
        raise SystemExit(
            f"{model_dir} is only {total:,} bytes after download; expected "
            f">= {MIN_TOTAL_BYTES:,}. Re-run with --force."
        )

    print(f"[done] {model_dir}: {total:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
