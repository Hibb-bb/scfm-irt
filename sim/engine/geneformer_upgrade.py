"""Geneformer v1 → v2 upgrade case study on disjoint cell pools."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.irt.ops import fit_ability


@dataclass(frozen=True)
class GeneformerUpgradeConfig:
    enabled: bool = True
    v1_model: str = "geneformer"
    v2_model: str = "geneformer_v2"


def _fit_on_items(
    item_ids: list[str],
    answers: dict[str, float],
    b_by_item: dict[str, float],
    *,
    tau: float,
) -> float:
    u_list, b_list = [], []
    for iid in item_ids:
        if iid not in answers or iid not in b_by_item:
            continue
        u_list.append(answers[iid])
        b_list.append(b_by_item[iid])
    if not u_list:
        return float("nan")
    return fit_ability(np.array(u_list), np.array(b_list), tau=tau)


def evaluate_upgrade_round(
    *,
    round_idx: int,
    round_date,
    cfg: GeneformerUpgradeConfig,
    anchor_ids: list[str],
    asked: dict[str, dict[str, float]],
    bank_before,
    retired_ids: list[str],
    bank_after,
    theta_hat_full: dict[str, float],
    oracle_theta: dict[str, float],
    tau: float,
) -> dict | None:
    """Compare ability gain under disjoint vs anchor-only vs full-bank estimation."""
    v1, v2 = cfg.v1_model, cfg.v2_model
    if v2 not in asked or v1 not in asked:
        return None

    b_before = {iid: it.b_hat for iid, it in bank_before.items.items()}
    b_after = {iid: it.b_hat for iid, it in bank_after.items.items()}
    b_all = {**b_before, **b_after}

    anchors = [i for i in anchor_ids if i in b_all]
    s1_legacy = [i for i in retired_ids if i not in anchor_ids]
    s2_frontier = [i for i in bank_after.all_ids() if not bank_after.items[i].anchor]

    theta_v1_legacy = _fit_on_items(s1_legacy, asked[v1], b_all, tau=tau)
    theta_v2_frontier = _fit_on_items(s2_frontier, asked[v2], b_all, tau=tau)
    theta_v1_anchor = _fit_on_items(anchors, asked[v1], b_all, tau=tau)
    theta_v2_anchor = _fit_on_items(anchors, asked[v2], b_all, tau=tau)

    theta_v1_full = theta_hat_full.get(v1, float("nan"))
    theta_v2_full = theta_hat_full.get(v2, float("nan"))
    oracle_v1 = oracle_theta.get(v1, float("nan"))
    oracle_v2 = oracle_theta.get(v2, float("nan"))

    delta_oracle = oracle_v2 - oracle_v1
    delta_disjoint = theta_v2_frontier - theta_v1_legacy
    delta_anchor = theta_v2_anchor - theta_v1_anchor
    delta_full = theta_v2_full - theta_v1_full

    overlap = len(set(s1_legacy) & set(s2_frontier))
    return {
        "round": round_idx,
        "round_date": round_date,
        "v1_model": v1,
        "v2_model": v2,
        "n_s1_legacy": len(s1_legacy),
        "n_s2_frontier": len(s2_frontier),
        "n_anchors": len(anchors),
        "s1_s2_overlap": overlap,
        "theta_v1_legacy": theta_v1_legacy,
        "theta_v2_frontier": theta_v2_frontier,
        "theta_v1_anchor": theta_v1_anchor,
        "theta_v2_anchor": theta_v2_anchor,
        "theta_v1_full": theta_v1_full,
        "theta_v2_full": theta_v2_full,
        "oracle_v1": oracle_v1,
        "oracle_v2": oracle_v2,
        "delta_oracle": delta_oracle,
        "delta_disjoint": delta_disjoint,
        "delta_anchor": delta_anchor,
        "delta_full": delta_full,
        "err_disjoint": delta_disjoint - delta_oracle,
        "err_anchor": delta_anchor - delta_oracle,
        "err_full": delta_full - delta_oracle,
    }
