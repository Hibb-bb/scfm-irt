"""Command-line entry point: `scfm-irt <command>`."""

from __future__ import annotations

import argparse
import sys

from scfm_irt import __version__
from scfm_irt.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scfm-irt", description=__doc__)
    parser.add_argument("--version", action="version", version=f"scfm-irt {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the full pipeline from a config")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--dry-run", action="store_true", help="validate the config and exit")

    p_describe = sub.add_parser("describe", help="print the inventory table for a dataset")
    p_describe.add_argument("dataset")

    args = parser.parse_args(argv)

    if args.command == "run":
        cfg = load_config(args.config)
        print(f"run_name={cfg.run_name}  config_hash={cfg.config_hash()}  seed={cfg.seed}")
        print(f"datasets={cfg.datasets}")
        print(f"embedders={cfg.embedders}")
        print(f"irt={cfg.irt.model}/{cfg.irt.method}  granularity={cfg.irt.item_granularity}")
        if args.dry_run:
            return 0
        from scfm_irt.pipeline import run_pipeline

        run_pipeline(cfg)
        return 0

    if args.command == "describe":
        from scfm_irt.data.describe import describe_by_name

        for key, value in describe_by_name(args.dataset).items():
            print(f"{key}: {value}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
