"""Per-run summary report: tidy tables, figures, and a short Markdown/HTML page."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scfm_irt.config import RunConfig
from scfm_irt.irt.models import IRTFit


def write_report(
    fit: IRTFit,
    resp: pd.DataFrame,
    cfg: RunConfig,
    out_dir: str | Path,
    *,
    loo_paths: dict[str, Path] | None = None,
) -> Path:
    """Emit core outputs into `out_dir` and return the report path.

    Phase-8 will add Wright maps, ICC plots, and fit stats. For the smoke path we write the
    ability ranking, item table, response table, and a short Markdown summary.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ability = fit.ability_table()
    items = fit.item_table()
    ability.to_csv(out / "ability_ranking_joint.csv", index=False)
    ability.to_parquet(out / "ability_ranking_joint.parquet", index=False)
    # Back-compat alias: joint-fit ranking (includes holdout in difficulty bank).
    ability.to_csv(out / "ability_ranking.csv", index=False)
    ability.to_parquet(out / "ability_ranking.parquet", index=False)
    items.to_csv(out / "item_params.csv", index=False)
    items.to_parquet(out / "item_params.parquet", index=False)
    resp.to_parquet(out / "responses.parquet", index=False)

    loo_note = ""
    if loo_paths and (loo_paths.get("ability") or Path(out / "ability_ranking_loo.csv")).exists():
        loo_path = loo_paths.get("ability") if loo_paths else out / "ability_ranking_loo.csv"
        ability_loo = pd.read_csv(loo_path)
        try:
            _plot_ability_and_accuracy_forest(
                ability_loo.rename(columns={"ability": "ability"}),
                resp,
                f"{cfg.run_name} (LOO-calibrated θ)",
                out,
                stem="ability_forest_loo",
            )
        except Exception as exc:  # pragma: no cover
            (out / "ability_forest_loo_error.txt").write_text(str(exc))
        loo_note = (
            "- **Headline abilities** use leave-one-model-out calibration: item difficulties "
            "for model *m* are fit without *m*'s responses (`ability_ranking_loo.csv`).\n"
            "- Joint-fit abilities (`ability_ranking_joint.csv`) include all models in the "
            "difficulty bank and are kept for comparison only.\n"
        )

    # Forest plot: joint IRT ability vs raw probe accuracy.
    try:
        _plot_ability_and_accuracy_forest(ability, resp, cfg.run_name, out)
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        (out / "ability_forest_error.txt").write_text(str(exc))

    report = out / "report.md"
    header = "| " + " | ".join(ability.columns) + " |"
    sep = "| " + " | ".join("---" for _ in ability.columns) + " |"
    ranking_lines = [header, sep]
    for _, row in ability.iterrows():
        cells = []
        for c in ability.columns:
            val = row[c]
            cells.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        ranking_lines.append("| " + " | ".join(cells) + " |")
    lines = [
        f"# {cfg.run_name}",
        "",
        f"- config hash: `{cfg.config_hash()}`",
        f"- IRT: `{fit.model}` / `{fit.method}` / granularity=`{fit.item_granularity}`",
        f"- anchoring: **{fit.anchoring}** (abilities comparable only within this fit)",
        "- difficulty is relative to this respondent pool, not absolute biology",
        f"- n_models={len(fit.model_ids)}, n_items={len(fit.item_ids)}, n_responses={len(resp)}",
        "",
    ]
    if loo_note:
        lines.extend([loo_note.rstrip(), ""])
    lines.extend(
        [
            "## Ability ranking (joint fit)",
            "",
            *ranking_lines,
            "",
            "## Notes",
            "",
            f"- uncertainty: {fit.diagnostics.get('uncertainty', 'see diagnostics')}",
            "",
        ]
    )
    report.write_text("\n".join(lines))
    return report


def _accuracy_table(resp: pd.DataFrame) -> pd.DataFrame:
    """Per-model raw probe accuracy on the IRT test items."""
    rows = []
    for model_id, grp in resp.groupby("model_id", sort=False):
        n = len(grp)
        acc = float(grp["correct"].mean())
        se = float(np.sqrt(acc * (1.0 - acc) / n)) if n else 0.0
        rows.append(
            {
                "model_id": model_id,
                "accuracy": acc,
                "sd": se,
                "lower": acc - 1.96 * se,
                "upper": acc + 1.96 * se,
                "n_items": n,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("accuracy", ascending=False).reset_index(drop=True)


def _plot_ability_and_accuracy_forest(
    ability: pd.DataFrame,
    resp: pd.DataFrame,
    run_name: str,
    out: Path,
    *,
    stem: str = "ability_forest",
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    accuracy = _accuracy_table(resp)
    # Keep model order aligned with IRT ability ranking.
    model_order = ability["model_id"].tolist()
    accuracy = accuracy.set_index("model_id").loc[model_order].reset_index()

    n_models = len(model_order)
    fig_h = max(2.8, 0.45 * n_models)
    fig, (ax_theta, ax_acc) = plt.subplots(
        1,
        2,
        figsize=(10.5, fig_h),
        sharey=True,
        gridspec_kw={"width_ratios": [1.1, 1.0], "wspace": 0.08},
        constrained_layout=True,
    )
    y = np.arange(n_models)

    ax_theta.errorbar(
        ability["ability"],
        y,
        xerr=1.96 * ability["sd"],
        fmt="o",
        color="#1f4e79",
        ecolor="#7a8a99",
        capsize=3,
    )
    ax_theta.axvline(0.0, color="#bbbbbb", lw=1)
    ax_theta.set_xlabel("ability (θ)")
    ax_theta.set_title("IRT ability")
    ax_theta.invert_yaxis()
    ax_theta.set_yticks(y)
    ax_theta.set_yticklabels(model_order)

    ax_acc.errorbar(
        accuracy["accuracy"],
        y,
        xerr=1.96 * accuracy["sd"],
        fmt="o",
        color="#1f4e79",
        ecolor="#7a8a99",
        capsize=3,
    )
    acc_min = float(accuracy["lower"].min())
    acc_max = float(accuracy["upper"].max())
    pad = max(0.01, 0.08 * (acc_max - acc_min))
    ax_acc.set_xlim(max(0.0, acc_min - pad), min(1.0, acc_max + pad))
    ax_acc.set_xlabel("raw probe accuracy")
    ax_acc.set_title("Test-set accuracy")
    ax_acc.tick_params(axis="y", labelleft=False)

    fig.suptitle(f"{run_name}: model abilities vs raw accuracy", fontsize=12)
    fig.savefig(out / f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    if stem == "ability_forest":
        accuracy.to_csv(out / "accuracy_ranking.csv", index=False)


def write_run_provenance(cfg: RunConfig, out_dir: str | Path) -> None:
    """Write the resolved config, its hash, and library versions into the run directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=True))
    (out / "config_hash.txt").write_text(cfg.config_hash() + "\n")

    versions = {
        "python": sys.version.split()[0],
        "scfm_irt": _pkg_version("scfm_irt"),
        "numpy": _pkg_version("numpy"),
        "pandas": _pkg_version("pandas"),
        "sklearn": _pkg_version("sklearn"),
        "anndata": _pkg_version("anndata"),
        "scanpy": _pkg_version("scanpy"),
        "torch": _pkg_version("torch"),
        "py_irt": _pkg_version("py_irt"),
    }
    (out / "versions.json").write_text(json.dumps(versions, indent=2, sort_keys=True) + "\n")


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        dist = {"sklearn": "scikit-learn", "py_irt": "py-irt"}.get(name, name)
        return version(dist)
    except Exception:
        try:
            mod = __import__(name)
            return getattr(mod, "__version__", "unknown")
        except ImportError:
            return None
        except Exception:
            return None
