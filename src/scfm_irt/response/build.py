"""Assembling per-model scores into the response table, and pivoting it to a matrix.

**Orientation convention, fixed once and asserted everywhere: `Y` is `(P models, I cells)` —
rows are respondents, columns are items.** A persons/items transposition produces a fit that
runs cleanly and is entirely wrong, which is why `to_matrix` asserts its own output shape and
why the synthetic recovery test exists (plan section 10.2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scfm_irt.response.schema import validate_response_table


def make_cell_id(dataset: str, obs_name: str) -> str:
    """Globally unique item id. Bare `obs_name` collides across datasets."""
    return f"{dataset}:{obs_name}"


def build_response_table(score_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-(model, dataset) score frames into the canonical long table.

    Each input frame comes from `probe.score_cells` and must already carry model_id, dataset,
    split_scheme, cell_id, cell_type, and correct. Validated before returning.
    """
    if not score_frames:
        raise ValueError("no score frames to assemble")

    resp = pd.concat(score_frames, ignore_index=True)
    return validate_response_table(resp)


def to_matrix(
    resp: pd.DataFrame,
    fill_value: int = 0,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Pivot the long table to a dense response matrix.

    Returns `(Y, model_ids, cell_ids)` with `Y.shape == (len(model_ids), len(cell_ids))` —
    rows are models (respondents), columns are cells (items).

    `fill_value` is used only for missing (model, cell) pairs, which `validate_response_table`
    rejects by default; it exists for the deliberately ragged case.
    """
    model_ids = sorted(resp["model_id"].unique())
    cell_ids = sorted(resp["cell_id"].unique())

    wide = (
        resp.pivot_table(
            index="model_id",
            columns="cell_id",
            values="correct",
            aggfunc="first",
        )
        .reindex(index=model_ids, columns=cell_ids)
        .fillna(fill_value)
    )

    Y = wide.to_numpy(dtype=np.int8)

    # The assertion that matters. If this ever fires, nothing downstream is meaningful.
    assert Y.shape == (len(model_ids), len(cell_ids)), (
        f"orientation violated: Y has shape {Y.shape}, expected "
        f"({len(model_ids)}, {len(cell_ids)}) = (n_models, n_cells)"
    )
    return Y, list(model_ids), list(cell_ids)


def aggregate_items(resp: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-cell items to `(dataset, cell_type)` groups.

    Produces the binomial form of MATH.md section 11.2: per (model, group), the number correct
    out of the group size. This is the robustness mode for local independence — cells of the
    same type are correlated (a model wrong on a rare type is wrong on *all* its cells), and
    grouping absorbs that while raising the per-item sample size from ~P to ~P x n_group.

    Returns columns: model_id, dataset, cell_type, group_id, n_correct, n_total.
    """
    required = {"model_id", "dataset", "cell_type", "correct"}
    missing = required - set(resp.columns)
    if missing:
        raise ValueError(f"cannot aggregate, missing columns: {sorted(missing)}")

    out = (
        resp.groupby(["model_id", "dataset", "cell_type"], observed=True)["correct"]
        .agg(n_correct="sum", n_total="size")
        .reset_index()
    )
    out["group_id"] = out["dataset"] + "|" + out["cell_type"]
    return out
