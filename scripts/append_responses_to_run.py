#!/usr/bin/env python
"""Append probe responses for new embedders to an existing run directory.

    PYTHONPATH=. python scripts/append_responses_to_run.py \
        --run runs/ranking_immune_human_v5 \
        --config configs/ranking_immune_human_scim_tf_v1.yaml
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from scfm_irt.config import load_config
from scfm_irt.pipeline import build_responses
from scfm_irt.response.schema import validate_response_table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="existing run directory")
    parser.add_argument("--config", required=True, help="config listing only new embedders")
    parser.add_argument("--backup", action="store_true", help="copy responses.parquet before merge")
    args = parser.parse_args()

    run_dir = Path(args.run)
    resp_path = run_dir / "responses.parquet"
    if not resp_path.exists():
        raise FileNotFoundError(f"missing {resp_path}")

    cfg = load_config(args.config)
    new_resp = build_responses(cfg)
    old_resp = pd.read_parquet(resp_path)

    overlap = set(cfg.embedders) & set(old_resp["model_id"].unique())
    if overlap:
        print(f"[append] replacing existing rows for: {sorted(overlap)}")
        old_resp = old_resp[~old_resp["model_id"].isin(cfg.embedders)]

    combined = validate_response_table(
        pd.concat([old_resp, new_resp], ignore_index=True),
        strict=False,
    )

    if args.backup:
        backup = resp_path.with_name("responses.before_append.parquet")
        shutil.copy2(resp_path, backup)
        print(f"[append] backup -> {backup}")

    combined.to_parquet(resp_path, index=False)
    print(
        f"[append] wrote {resp_path}  "
        f"({len(combined)} rows, {combined['model_id'].nunique()} models, "
        f"{combined['cell_id'].nunique()} cells)"
    )
    for model_id in cfg.embedders:
        n = int((combined["model_id"] == model_id).sum())
        print(f"  {model_id}: {n} responses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
