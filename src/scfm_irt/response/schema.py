"""Schema for `responses.parquet`, the canonical artifact of the whole pipeline.

One row per (model, cell). Everything after this point — every IRT fit, every figure — is a
function of this table, so it is validated rather than trusted.
"""

from __future__ import annotations

import pandas as pd

# column -> pandas dtype
RESPONSE_SCHEMA: dict[str, str] = {
    "model_id": "object",  # respondent id
    "dataset": "object",
    "split_scheme": "object",
    "cell_id": "object",  # item id, unique across datasets: f"{dataset}:{obs_name}"
    "cell_type": "object",  # for difficulty aggregation and as an explanatory covariate
    "correct": "int8",  # 0/1 — the IRT response
}

# Optional covariates for the explanatory model (MATH.md section 11.1).
OPTIONAL_COLUMNS: dict[str, str] = {
    "n_classes": "int16",  # number of cell types in that dataset
    "log_depth": "float32",  # log total counts for the cell
    "batch": "object",
    "donor": "object",
}


def validate_response_table(resp: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Check required columns, dtypes, the 0/1 domain, and the rectangular structure.

    Returns the table with dtypes coerced to the schema. Raises on anything that would make an
    IRT fit quietly wrong: a missing column, a `correct` value outside {0,1}, a duplicated
    (model_id, cell_id) pair, or NaNs in a required column.

    With `strict`, also requires the table to be rectangular — every model responded to every
    cell. Ragged tables are fittable in principle (IRT handles missingness) but here they
    usually mean a model's run failed partway, which should be noticed rather than absorbed.
    """
    missing = [c for c in RESPONSE_SCHEMA if c not in resp.columns]
    if missing:
        raise ValueError(f"response table is missing required columns: {missing}")

    for col in RESPONSE_SCHEMA:
        if resp[col].isna().any():
            raise ValueError(f"column {col!r} contains NaN")

    out = resp.copy()
    for col, dtype in RESPONSE_SCHEMA.items():
        out[col] = out[col].astype(dtype)
    for col, dtype in OPTIONAL_COLUMNS.items():
        if col in out.columns and out[col].notna().all():
            out[col] = out[col].astype(dtype)

    bad = set(out["correct"].unique()) - {0, 1}
    if bad:
        raise ValueError(f"`correct` must be 0/1, found {sorted(bad)}")

    dup = out.duplicated(subset=["model_id", "cell_id"]).sum()
    if dup:
        raise ValueError(f"{dup} duplicated (model_id, cell_id) rows")

    if strict:
        n_models = out["model_id"].nunique()
        n_cells = out["cell_id"].nunique()
        if len(out) != n_models * n_cells:
            raise ValueError(
                f"table is ragged: {len(out)} rows for {n_models} models x {n_cells} cells "
                f"(= {n_models * n_cells} expected). Pass strict=False to allow missingness."
            )

    return out
