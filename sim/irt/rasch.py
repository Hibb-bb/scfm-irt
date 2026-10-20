"""Joint Rasch calibration for oracle θ* and b*."""

from __future__ import annotations

import pandas as pd

from scfm_irt.irt.fit_pyirt import _fit_mfvi
from scfm_irt.irt.models import IRTFit
from scfm_irt.response.build import to_matrix


def fit_joint_rasch(resp: pd.DataFrame, *, n_steps: int = 800, seed: int = 0) -> IRTFit:
    """Fit 1PL on full correctness tensor."""
    Y, model_ids, item_ids = to_matrix(resp)
    return _fit_mfvi(
        Y.astype(float),
        list(model_ids),
        list(item_ids),
        model="1pl",
        seed=seed,
        n_steps=n_steps,
        hierarchical=True,
        weights=None,
        item_granularity="cell",
        engine="sim_rasch",
    )


def oracle_params(fit: IRTFit) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return oracle ability and difficulty tables."""
    theta = pd.DataFrame(
        {
            "model_id": fit.model_ids,
            "theta_star": fit.ability_mean.astype(float),
            "theta_sd": fit.ability_sd.astype(float),
        }
    )
    diff = pd.DataFrame(
        {
            "item_id": fit.item_ids,
            "b_star": fit.difficulty_mean.astype(float),
            "b_sd": fit.difficulty_sd.astype(float),
        }
    )
    return theta, diff


def linking_summary(
    resp: pd.DataFrame,
    *,
    datasets: list[str],
    seed: int = 0,
) -> pd.DataFrame:
    """M0: show mean difficulty by dataset on a shared scale."""
    fit = fit_joint_rasch(resp.loc[resp["dataset"].isin(datasets)], seed=seed)
    _, diff = oracle_params(fit)
    diff = diff.rename(columns={"item_id": "cell_id"})
    meta = resp[["cell_id", "dataset"]].drop_duplicates()
    merged = meta.merge(diff, on="cell_id", how="left")
    return (
        merged.groupby("dataset", as_index=False)
        .agg(b_mean=("b_star", "mean"), b_std=("b_star", "std"), n_items=("cell_id", "nunique"))
        .sort_values("b_mean")
    )
