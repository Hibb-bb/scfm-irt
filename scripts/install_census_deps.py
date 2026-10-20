#!/usr/bin/env python
"""Install cellxgene-census deps offline (Quest PyPI downloads often truncate)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

WHEEL_DIR = Path(__file__).resolve().parent.parent / "cache" / "pip_wheels"


def main() -> int:
    WHEEL_DIR.mkdir(parents=True, exist_ok=True)
    pkgs = ["tiledbsoma==2.3.0", "s3fs==2024.12.0", "cellxgene-census==1.18.0"]
    for pkg in pkgs:
        print(f"[census-install] download {pkg}")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                pkg,
                "-d",
                str(WHEEL_DIR),
                "--no-cache-dir",
            ],
            check=True,
        )
    print(f"[census-install] install from {WHEEL_DIR}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            f"--find-links={WHEEL_DIR}",
            *pkgs,
        ],
        check=True,
    )
    import cellxgene_census  # noqa: F401

    print("[census-install] ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
