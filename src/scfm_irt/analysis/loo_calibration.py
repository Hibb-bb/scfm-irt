"""Leave-one-model-out calibration for item difficulties and model abilities.

When scoring model *m*, item difficulties are fit on all other models only; ability for *m*
is then estimated with those fixed difficulties (MLE under 1PL). This avoids leakage from
*m*'s own responses into the difficulty bank used to score it.

The joint (full-pool) fit remains useful as a reference scale and for truly new models that
were not in the calibration pool.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scfm_irt.analysis.subsample_sim import mle_theta_1pl_fixed_b
from scfm_irt.irt.fit_pyirt import _fit_mfvi
from scfm_irt.irt.models import IRTFit
from scfm_irt.response.build import to_matrix


def align_difficulty(b_loo: np.ndarray, b_ref: np.ndarray) -> np.ndarray:
    """Shift LOO difficulties onto the reference scale (1PL location indeterminacy)."""
    b_loo = np.asarray(b_loo, dtype=float)
    b_ref = np.asarray(b_ref, dtype=float)
    return b_loo - float(np.mean(b_loo - b_ref))


def loo_calibrate_banks(
    resp: pd.DataFrame,
    fit: IRTFit,
    *,
    n_steps: int = 400,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, IRTFit]]:
    """Fit item difficulties on P−1 models for each holdout; compare to the full-fit bank."""
    Y, model_ids, item_ids = to_matrix(resp)
    if list(model_ids) != list(fit.model_ids):
        raise ValueError("model_ids mismatch")
    if len(model_ids) < 3:
        raise ValueError("LOO calibration needs ≥3 models")

    b_full = fit.difficulty_mean.astype(float)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    loo_fits: dict[str, IRTFit] = {}

    for holdout_idx, holdout_id in enumerate(model_ids):
        cal_idx = [i for i in range(len(model_ids)) if i != holdout_idx]
        cal_ids = [model_ids[i] for i in cal_idx]
        Y_cal = Y[cal_idx, :]

        fit_loo = _fit_mfvi(
            Y_cal.astype(float),
            cal_ids,
            list(item_ids),
            model=fit.model,
            seed=int(rng.integers(0, 2**31 - 1)),
            n_steps=n_steps,
            hierarchical=True,
            weights=None,
            item_granularity=fit.item_granularity,
            engine="mfvi_loo_cal",
        )
        loo_fits[holdout_id] = fit_loo
        b_loo = fit_loo.difficulty_mean.astype(float)
        b_loo_aligned = align_difficulty(b_loo, b_full)
        diff = b_loo_aligned - b_full
        rows.append(
            {
                "holdout_model": holdout_id,
                "n_calibration_models": len(cal_ids),
                "calibration_models": ",".join(cal_ids),
                "difficulty_pearson_r": float(np.corrcoef(b_full, b_loo_aligned)[0, 1]),
                "difficulty_spearman_r": float(
                    pd.Series(b_full).corr(pd.Series(b_loo_aligned), method="spearman")
                ),
                "difficulty_rmse": float(np.sqrt(np.mean(diff**2))),
                "difficulty_mae": float(np.mean(np.abs(diff))),
                "difficulty_mean_shift": float(np.mean(b_loo - b_full)),
            }
        )

    return pd.DataFrame(rows), loo_fits


def loo_ability_table(
    resp: pd.DataFrame,
    fit: IRTFit,
    loo_fits: dict[str, IRTFit],
) -> pd.DataFrame:
    """Ability for each model using difficulties calibrated without that model."""
    Y, model_ids, _item_ids = to_matrix(resp)
    b_full = fit.difficulty_mean.astype(float)
    rows: list[dict] = []

    for holdout_idx, holdout_id in enumerate(model_ids):
        b_loo = align_difficulty(loo_fits[holdout_id].difficulty_mean.astype(float), b_full)
        y_holdout = Y[holdout_idx, :].astype(float)
        theta_loo = mle_theta_1pl_fixed_b(y_holdout, b_loo)
        acc = float(y_holdout.mean())
        rows.append(
            {
                "model_id": holdout_id,
                "ability": theta_loo,
                "ability_joint": float(fit.ability_mean[holdout_idx]),
                "accuracy": acc,
                "n_calibration_models": len(model_ids) - 1,
                "calibration_excludes": holdout_id,
            }
        )

    out = pd.DataFrame(rows)
    out["rank"] = out["ability"].rank(ascending=False, method="min").astype(int)
    out["rank_joint"] = out["ability_joint"].rank(ascending=False, method="min").astype(int)
    out["rank_accuracy"] = out["accuracy"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("rank").reset_index(drop=True)


def loo_item_params_summary(
    fit: IRTFit,
    loo_fits: dict[str, IRTFit],
) -> pd.DataFrame:
    """Per-item difficulty: joint fit vs mean/std of LOO banks (each excludes one model)."""
    b_full = fit.difficulty_mean.astype(float)
    stack = np.stack(
        [
            align_difficulty(f.difficulty_mean.astype(float), b_full)
            for f in loo_fits.values()
        ],
        axis=0,
    )
    b_loo_mean = stack.mean(axis=0)
    b_loo_std = stack.std(axis=0)
    return pd.DataFrame(
        {
            "item_id": fit.item_ids,
            "difficulty_joint": b_full,
            "difficulty_loo_mean": b_loo_mean,
            "difficulty_loo_std": b_loo_std,
            "difficulty_joint_sd": fit.difficulty_sd.astype(float),
        }
    )


def write_loo_calibration_outputs(
    resp: pd.DataFrame,
    fit: IRTFit,
    out_dir: str | Path,
    *,
    n_steps: int = 400,
    seed: int = 0,
) -> dict[str, Path]:
    """Write LOO difficulty stability, abilities, and item summaries into `out_dir`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stability, loo_fits = loo_calibrate_banks(resp, fit, n_steps=n_steps, seed=seed)
    ability_loo = loo_ability_table(resp, fit, loo_fits)
    items_loo = loo_item_params_summary(fit, loo_fits)

    paths = {
        "stability": out / "loo_difficulty_stability.csv",
        "ability": out / "ability_ranking_loo.csv",
        "items": out / "item_params_loo_summary.csv",
    }
    stability.to_csv(paths["stability"], index=False)
    ability_loo.to_csv(paths["ability"], index=False)
    items_loo.to_csv(paths["items"], index=False)

    # Per-holdout difficulty banks (for audit / adaptive sim reuse).
    loo_dir = out / "loo_item_banks"
    loo_dir.mkdir(exist_ok=True)
    b_full = fit.difficulty_mean.astype(float)
    for holdout_id, fit_loo in loo_fits.items():
        b_aligned = align_difficulty(fit_loo.difficulty_mean.astype(float), b_full)
        pd.DataFrame({"item_id": fit.item_ids, "difficulty": b_aligned}).to_csv(
            loo_dir / f"difficulty_excluding_{holdout_id}.csv",
            index=False,
        )

    return paths
