"""Load artifacts from a completed pipeline run directory."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from scfm_irt.config import RunConfig
from scfm_irt.irt.models import IRTFit
from scfm_irt.pipeline import build_responses


def load_fit_from_run(run_dir: Path) -> IRTFit:
    cfg_dict = yaml.safe_load((run_dir / "config.yaml").read_text())
    irt_cfg = cfg_dict["irt"]
    ability = pd.read_csv(run_dir / "ability_ranking.csv")
    items = pd.read_csv(run_dir / "item_params.csv")
    diag = json.loads((run_dir / "irt_diagnostics.json").read_text())

    # Match `to_matrix` row order (sorted model_id strings).
    ability = ability.sort_values("model_id").reset_index(drop=True)

    return IRTFit(
        model=irt_cfg["model"],
        method=irt_cfg["method"],
        item_granularity=irt_cfg["item_granularity"],
        model_ids=ability["model_id"].tolist(),
        item_ids=items["item_id"].tolist(),
        ability_mean=ability["ability"].to_numpy(dtype=float),
        ability_sd=ability["sd"].to_numpy(dtype=float),
        difficulty_mean=items["difficulty"].to_numpy(dtype=float),
        difficulty_sd=items["difficulty_sd"].to_numpy(dtype=float),
        discrimination_mean=None,
        discrimination_sd=None,
        anchoring="priors (theta ~ N(0,1), hierarchical item priors)",
        diagnostics=diag,
    )


def load_responses(run_dir: Path, cfg: RunConfig) -> pd.DataFrame:
    resp_path = run_dir / "responses.parquet"
    if resp_path.exists():
        return pd.read_parquet(resp_path)
    resp = build_responses(cfg)
    resp.to_parquet(resp_path, index=False)
    return resp
