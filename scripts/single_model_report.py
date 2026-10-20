#!/usr/bin/env python
"""Generate a single-model item difficulty report from an existing IRT run.

Rebuilds the response table from cached embeddings (if responses.parquet is missing),
reloads item difficulties from the run, and writes accuracy-vs-difficulty analysis.

    python scripts/single_model_report.py --run runs/ranking_pbmc3k_v2 --model uce
    python scripts/single_model_report.py --run runs/ranking_pbmc3k_v2 --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scfm_irt.analysis.model_report import write_single_model_report
from scfm_irt.analysis.run_io import load_fit_from_run, load_responses
from scfm_irt.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Existing run directory (e.g. runs/ranking_pbmc3k_v2)")
    parser.add_argument("--model", help="Model/respondent id (e.g. uce)")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate reports for every embedder listed in the run config",
    )
    args = parser.parse_args()

    if not args.all and not args.model:
        parser.error("pass --model NAME or --all")

    run_dir = Path(args.run).resolve()
    cfg = load_config(run_dir / "config.yaml")
    models = list(cfg.embedders) if args.all else [args.model]
    for model in models:
        if model not in cfg.embedders:
            raise SystemExit(
                f"model {model!r} not in run embedders {cfg.embedders}. "
                "Pick one of those or re-run IRT with that model included."
            )

    fit = load_fit_from_run(run_dir)
    resp = load_responses(run_dir, cfg)

    for model in models:
        report = write_single_model_report(
            run_dir,
            model,
            fit,
            resp,
            source_run_name=cfg.run_name,
        )
        print(f"[report] {model} -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
