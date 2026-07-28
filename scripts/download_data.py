#!/usr/bin/env python
"""Download registered datasets to `data/raw/` and verify them.

    python scripts/download_data.py --dataset zheng68k
    python scripts/download_data.py --all --family atlas

After downloading, run `describe` and compare against the spec's expected dims. Several
benchmark sets circulate in transposed or differently-preprocessed copies, so a download that
"worked" is not the same as a download that is usable.
"""

from __future__ import annotations

import argparse

from scfm_irt.data.registry import get_spec, list_datasets


def download(name: str, force: bool = False) -> None:
    spec = get_spec(name)
    if spec.url is None:
        raise ValueError(f"{name}: no url registered; add one in data/registry.py")
    raise NotImplementedError("TODO(phase-0): fetch spec.url -> spec.path, checksum, verify")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--family", choices=["annotation", "atlas"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    names = list_datasets(args.family) if args.all else args.dataset
    if not names:
        parser.error("pass --dataset NAME (repeatable) or --all")

    for name in names:
        print(f"[download] {name}")
        download(name, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
