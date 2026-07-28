"""Per-run summary report: tidy tables, figures, and a short Markdown/HTML page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scfm_irt.config import RunConfig
from scfm_irt.irt.models import IRTFit


def write_report(fit: IRTFit, resp: pd.DataFrame, cfg: RunConfig, out_dir: str | Path) -> Path:
    """Emit every output of MATH.md section 12 into `out_dir` and return the report path.

    Tables (parquet + CSV): ability ranking with intervals, difficulty summaries by cell type
    and dataset, discrimination extremes for 2PL, item/person fit statistics.
    Figures (PNG + SVG): ability forest plot, difficulty histogram, Wright map, representative
    ICCs, test information curve.

    The report must state, in text and not only in metadata: the anchoring used (abilities are
    comparable only within one anchored fit), that difficulty is relative to this respondent
    pool, which models are label-supervised, and the config hash and library versions.
    """
    raise NotImplementedError("TODO(phase-8): render tables + figures + summary page")


def write_run_provenance(cfg: RunConfig, out_dir: str | Path) -> None:
    """Write the resolved config, its hash, and library versions into the run directory."""
    raise NotImplementedError("TODO(phase-0): dump config + pip freeze subset")
