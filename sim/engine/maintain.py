"""Algorithm 1: online benchmark maintenance (fixed / recalibrate / full)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sim.engine.geneformer_upgrade import GeneformerUpgradeConfig, evaluate_upgrade_round
from sim.engine.field import estimate_field, field_at_round, init_synthetic_offsets
from sim.engine.schedule import (
    build_rounds,
    first_embedder_release,
    items_available_as_of,
    items_available_by_round,
    select_anchors,
)
from sim.irt.ops import (
    acquire_items,
    bank_value,
    filter_update,
    fit_ability,
    item_discrimination_top_models,
    item_value,
    retire_bottom_pct_ids,
    route_items,
    top_model_thetas,
)

ArmName = Literal["fixed", "recalibrate", "full"]


@dataclass
class ItemState:
    item_id: str
    b_hat: float
    p_var: float
    provisional: bool = False
    anchor: bool = False


@dataclass
class SimConfig:
    drift_mode: str = "synthetic"
    n_rounds: int = 12
    bank_budget: int = 500
    k_ask: int = 50
    n_exp: int = 10
    kappa: int = 20
    tau: float = 1.0
    q: float = 0.05
    p0: float = 2.0
    p_star: float = 0.25
    retire_v: float = 0.01
    n_anchors: int = 20
    mu0: float = 0.0
    r: float = 0.15
    s: float = 1.0
    n_synthetic: int = 10
    seed: int = 0
    equal_bank_size: bool = False
    rollout_mode: str = "models_and_items"  # models_only | models_and_items
    exclude_models: tuple[str, ...] = ()
    baseline_models: tuple[str, ...] = ("dev_pca", "dev_random")
    start_at_first_model: bool = False
    retire_cap: bool = True
    # Retirement policy: v_threshold (paper) or bottom_pct_top_half (PI protocol).
    retire_mode: str = "v_threshold"
    retire_top_model_frac: float = 0.5
    retire_bottom_pct: float = 0.05
    # When set, μ̂/ŝ for item value, retirement, and acquisition use only the top-k
    # highest-ability models (frontier window) instead of the full active pool.
    field_window_top_k: int | None = None
    # Alternative to top-k: use the top fraction of models (e.g. 0.5 = top half).
    field_window_top_frac: float | None = None
    # Seed the entire available pool into the bank at round 0 (no bank_budget cap on seed).
    seed_full_pool: bool = False
    # Geneformer v1→v2 upgrade case study at v2 release round.
    geneformer_upgrade: bool = False
    geneformer_v1_model: str = "geneformer"
    geneformer_v2_model: str = "geneformer_v2"


@dataclass
class BankState:
    items: dict[str, ItemState] = field(default_factory=dict)
    acq_counter: int = 0

    def trusted_ids(self) -> list[str]:
        return [iid for iid, it in self.items.items() if not it.provisional and not it.anchor]

    def all_ids(self) -> list[str]:
        return list(self.items.keys())


def _init_bank(
    anchor_ids: list[str],
    avail_ids: list[str],
    b_init: dict[str, float],
    cfg: SimConfig,
) -> BankState:
    bank = BankState()
    for iid in anchor_ids:
        bank.items[iid] = ItemState(
            item_id=iid, b_hat=b_init[iid], p_var=0.0, provisional=False, anchor=True
        )
    rest = [i for i in avail_ids if i not in bank.items]
    if cfg.seed_full_pool:
        seed_ids = rest
    else:
        seed_ids = rest[: max(0, cfg.bank_budget - len(bank.items))]
    for iid in seed_ids:
        bank.items[iid] = ItemState(
            item_id=iid, b_hat=b_init[iid], p_var=cfg.p0, provisional=False, anchor=False
        )
    return bank


def _observe(
    model_id: str,
    item_ids: list[str],
    U: pd.Series,
    *,
    drift_mode: str,
    theta: float,
    b_true: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for iid in item_ids:
        if drift_mode == "real":
            key = (model_id, iid)
            if key in U.index:
                out[iid] = float(U.loc[key])
            else:
                p = 1.0 / (1.0 + np.exp(-(theta - b_true[iid])))
                out[iid] = float(rng.random() < p)
        else:
            p = 1.0 / (1.0 + np.exp(-(theta - b_true[iid])))
            out[iid] = float(rng.random() < p)
    return out


def _ranking_quality(
    theta_hat: dict[str, float],
    theta_ref: dict[str, float],
    active: list[str],
) -> float:
    common = [m for m in active if m in theta_hat and m in theta_ref]
    if len(common) < 3:
        return float("nan")
    a = [theta_hat[m] for m in common]
    b = [theta_ref[m] for m in common]
    rho, _ = spearmanr(a, b)
    return float(rho)


def _incumbent_rank_shift(
    theta_hat: dict[str, float],
    theta_ref: dict[str, float],
    incumbent: list[str],
) -> float:
    """Mean absolute rank change among incumbent models (oracle order)."""
    common = [m for m in incumbent if m in theta_hat and m in theta_ref]
    if len(common) < 2:
        return float("nan")
    ref_order = sorted(common, key=lambda m: theta_ref[m], reverse=True)
    hat_order = sorted(common, key=lambda m: theta_hat[m], reverse=True)
    ref_rank = {m: i for i, m in enumerate(ref_order)}
    hat_rank = {m: i for i, m in enumerate(hat_order)}
    return float(np.mean([abs(ref_rank[m] - hat_rank[m]) for m in common]))


def _filter_models(
    resp: pd.DataFrame,
    theta_star: pd.DataFrame,
    exclude: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not exclude:
        return resp, theta_star
    ex = set(exclude)
    resp = resp.loc[~resp["model_id"].isin(ex)].copy()
    theta_star = theta_star.loc[~theta_star["model_id"].isin(ex)].copy()
    return resp, theta_star


def _select_items(
    bank: BankState,
    anchor_ids: list[str],
    avail: set[str],
    theta_guess: float,
    cfg: SimConfig,
) -> list[str]:
    eligible = [i for i in bank.all_ids() if i in avail]
    if cfg.k_ask >= len(eligible):
        return eligible

    pick: list[str] = [i for i in anchor_ids if i in bank.items and i in avail]
    trusted = [i for i in bank.trusted_ids() if i in avail]
    prov = [i for i in bank.all_ids() if bank.items[i].provisional and i in avail]

    n_trusted = max(0, cfg.k_ask - cfg.n_exp - len(pick))
    if trusted and n_trusted > 0:
        t_arr = np.asarray(trusted)
        t_b = np.array([bank.items[i].b_hat for i in trusted])
        idx = route_items(theta_guess, t_arr, t_b, n_trusted)
        pick.extend([trusted[int(i)] for i in idx])
    if prov and cfg.n_exp > 0:
        p_b = np.array([bank.items[i].b_hat for i in prov])
        p_arr = np.asarray(prov)
        idx = route_items(theta_guess, p_arr, p_b, cfg.n_exp)
        pick.extend([prov[int(i)] for i in idx])
    return pick


def run_arm(
    arm: ArmName,
    resp: pd.DataFrame,
    theta_star: pd.DataFrame,
    diff_star: pd.DataFrame,
    cfg: SimConfig,
    *,
    anchor_ids: list[str] | None = None,
    seed_avail_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resp, theta_star = _filter_models(resp, theta_star, cfg.exclude_models)
    rng = np.random.default_rng(cfg.seed + (0 if cfg.equal_bank_size else hash(arm) % 10000))
    U = resp.set_index(["model_id", "cell_id"])["correct"]
    item_dataset: dict[str, str] = {}
    if "dataset" in resp.columns:
        item_dataset = resp.drop_duplicates("cell_id").set_index("cell_id")["dataset"].to_dict()
    item_release = resp.drop_duplicates("cell_id").set_index("cell_id")["release_date"]
    model_release = resp.drop_duplicates("model_id").set_index("model_id")["model_release_date"]

    diff_star = diff_star.copy()
    if "cell_id" not in diff_star.columns:
        diff_star = diff_star.rename(columns={"item_id": "cell_id"})
    b_init = diff_star.set_index("cell_id")["b_star"].to_dict()
    oracle_theta = theta_star.set_index("model_id")["theta_star"].to_dict()
    all_models = list(theta_star["model_id"])
    item_ids = diff_star["cell_id"].to_numpy()

    rounds_end = pd.Timestamp(item_release.max())
    if cfg.drift_mode == "real":
        rounds_end = max(rounds_end, pd.Timestamp(model_release.max()))

    baseline = set(cfg.baseline_models)
    sim_start: pd.Timestamp | None = None
    if cfg.start_at_first_model and cfg.drift_mode == "real":
        sim_start = first_embedder_release(model_release, baseline)

    rounds = build_rounds(
        item_release,
        n_rounds=cfg.n_rounds,
        end_date=rounds_end,
        start_date=sim_start,
    )
    avail_by_round = items_available_by_round(item_ids, item_release, rounds)
    accumulated_avail: list[str] = []
    if sim_start is not None:
        accumulated_avail = items_available_as_of(item_ids, item_release, sim_start)

    if anchor_ids is None:
        anchor_ids = select_anchors(
            item_ids, diff_star["b_star"].to_numpy(), n_anchors=cfg.n_anchors, rng=rng
        )
    if cfg.seed_full_pool:
        init_avail = list(item_ids)
    elif cfg.start_at_first_model and accumulated_avail:
        init_avail = accumulated_avail
    elif seed_avail_ids is not None:
        init_avail = seed_avail_ids
    elif cfg.drift_mode == "synthetic" or cfg.rollout_mode == "models_only":
        init_avail = list(item_ids)
    else:
        init_avail = avail_by_round[0]
    bank = _init_bank(anchor_ids, init_avail, b_init, cfg)
    seed_item_b: dict[str, float] = {
        iid: bank.items[iid].b_hat for iid in bank.items if iid not in anchor_ids
    }
    synth_offsets = (
        init_synthetic_offsets(rng, cfg.n_synthetic, cfg.s) if cfg.drift_mode == "synthetic" else None
    )
    rows: list[dict] = []
    model_rows: list[dict] = []
    upgrade_rows: list[dict] = []
    gf_cfg = GeneformerUpgradeConfig(
        enabled=cfg.geneformer_upgrade,
        v1_model=cfg.geneformer_v1_model,
        v2_model=cfg.geneformer_v2_model,
    )
    upgrade_done = False
    prev_active: set[str] = set()
    prev_avail_count = 0

    for rd in rounds:
        if cfg.drift_mode == "synthetic":
            avail = set(item_ids)
        elif cfg.rollout_mode == "models_only" and cfg.start_at_first_model and accumulated_avail:
            avail = set(accumulated_avail)
        elif cfg.rollout_mode == "models_only":
            avail = set(item_ids)
        else:
            avail = set(avail_by_round[rd.round_idx])
        fld = field_at_round(
            drift_mode=cfg.drift_mode,
            round_idx=rd.round_idx,
            round_date=rd.round_date,
            mu0=cfg.mu0,
            r=cfg.r,
            s=cfg.s,
            all_model_ids=all_models,
            model_release=model_release,
            oracle_theta=oracle_theta,
            rng=rng,
            n_synthetic=cfg.n_synthetic,
            synth_offsets=synth_offsets,
            baseline_models=baseline if cfg.start_at_first_model else None,
            field_window_top_k=cfg.field_window_top_k,
            field_window_top_frac=cfg.field_window_top_frac,
        )

        asked: dict[str, dict[str, float]] = {}
        for mid in fld.active_model_ids:
            theta_guess = fld.thetas.get(mid, fld.mu)
            pick = _select_items(bank, anchor_ids, avail, theta_guess, cfg)
            asked[mid] = _observe(
                mid,
                pick,
                U,
                drift_mode=cfg.drift_mode,
                theta=theta_guess,
                b_true=b_init,
                rng=rng,
            )

        theta_hat: dict[str, float] = {}
        for mid in fld.active_model_ids:
            u_list, b_list = [], []
            for iid, u in asked[mid].items():
                if iid not in bank.items:
                    continue
                it = bank.items[iid]
                if it.anchor or not it.provisional:
                    u_list.append(u)
                    b_list.append(it.b_hat)
            theta_hat[mid] = fit_ability(np.array(u_list), np.array(b_list), tau=cfg.tau)

        mu_hat, s_hat = estimate_field(
            theta_hat, top_k=cfg.field_window_top_k, top_frac=cfg.field_window_top_frac
        )
        if not theta_hat:
            mu_hat, s_hat = fld.mu, fld.s
        mu_oracle_pool = float(np.mean(list(fld.thetas.values()))) if fld.thetas else fld.mu
        n_active = len(theta_hat) if theta_hat else len(fld.thetas)
        if cfg.field_window_top_frac and 0 < cfg.field_window_top_frac < 1:
            n_field_window = max(1, int(np.ceil(n_active * cfg.field_window_top_frac)))
        elif cfg.field_window_top_k and theta_hat:
            n_field_window = min(cfg.field_window_top_k, len(theta_hat))
        else:
            n_field_window = n_active

        if arm in ("recalibrate", "full"):
            for mid, answers in asked.items():
                th = theta_hat.get(mid, mu_hat)
                for iid, u in answers.items():
                    if iid not in bank.items or bank.items[iid].anchor:
                        continue
                    it = bank.items[iid]
                    b_new, p_new = filter_update(it.b_hat, it.p_var, theta_hat=th, u=u, q=cfg.q)
                    bank.items[iid] = ItemState(
                        item_id=iid,
                        b_hat=b_new,
                        p_var=p_new,
                        provisional=it.provisional,
                        anchor=False,
                    )

        n_retired_value = 0
        n_retired_cap = 0
        n_acquired_pool = 0
        n_acquired_synth = 0
        n_promoted = 0
        retired_by_dataset: dict[str, int] = {}

        active_set = set(fld.active_model_ids)
        new_models = sorted(active_set - prev_active)

        if arm == "full":
            bank_before_retire = BankState(
                items={iid: ItemState(**vars(it)) for iid, it in bank.items.items()},
                acq_counter=bank.acq_counter,
            )
            for iid, it in list(bank.items.items()):
                if it.provisional and it.p_var <= cfg.p_star:
                    bank.items[iid] = ItemState(
                        item_id=iid, b_hat=it.b_hat, p_var=it.p_var, provisional=False, anchor=False
                    )
                    n_promoted += 1

            in_bank = set(bank.items.keys())
            candidates = [i for i in avail if i not in in_bank]
            n_pull = min(cfg.kappa, len(candidates))
            if n_pull:
                cand_b = np.array([b_init[i] for i in candidates])
                hard_order = np.argsort(-cand_b)[:n_pull]
                for j in hard_order:
                    iid = candidates[int(j)]
                    bank.items[iid] = ItemState(
                        item_id=iid, b_hat=b_init[iid], p_var=cfg.p0, provisional=True, anchor=False
                    )
                n_acquired_pool = n_pull

            band = 4.0 * s_hat * cfg.kappa / max(cfg.bank_budget, 1)
            synth = acquire_items(
                max(0, cfg.kappa - n_pull),
                ceiling=mu_hat + 2 * s_hat,
                band_width=band,
                p0=cfg.p0,
                start_id=bank.acq_counter,
            )
            bank.acq_counter += len(synth)
            n_acquired_synth = len(synth)
            for rec in synth:
                bank.items[rec["item_id"]] = ItemState(**rec)

            if cfg.retire_mode == "bottom_pct_top_half":
                top_thetas = top_model_thetas(theta_hat, cfg.retire_top_model_frac)
                scored = [
                    (iid, item_discrimination_top_models(it.b_hat, top_thetas))
                    for iid, it in bank.items.items()
                    if not it.anchor
                ]
                to_drop = retire_bottom_pct_ids(scored, cfg.retire_bottom_pct)
            else:
                to_drop = [
                    iid
                    for iid, it in bank.items.items()
                    if not it.anchor and item_value(it.b_hat, mu_hat, s_hat) < cfg.retire_v
                ]
            retired_by_dataset = {}
            for iid in to_drop:
                ds = item_dataset.get(iid, "unknown")
                retired_by_dataset[ds] = retired_by_dataset.get(ds, 0) + 1
                del bank.items[iid]
            n_retired_value = len(to_drop)

            if (
                gf_cfg.enabled
                and not upgrade_done
                and cfg.geneformer_v2_model in new_models
            ):
                rec = evaluate_upgrade_round(
                    round_idx=rd.round_idx,
                    round_date=rd.round_date,
                    cfg=gf_cfg,
                    anchor_ids=anchor_ids,
                    asked=asked,
                    bank_before=bank_before_retire,
                    retired_ids=to_drop,
                    bank_after=bank,
                    theta_hat_full=theta_hat,
                    oracle_theta=fld.thetas,
                    tau=cfg.tau,
                )
                if rec is not None:
                    upgrade_rows.append(rec)
                    upgrade_done = True

            if cfg.retire_cap and len(bank.items) > cfg.bank_budget:
                scored = sorted(
                    ((iid, item_value(it.b_hat, mu_hat, s_hat)) for iid, it in bank.items.items() if not it.anchor),
                    key=lambda x: x[1],
                )
                n_over = len(bank.items) - cfg.bank_budget
                for iid, _ in scored[:n_over]:
                    del bank.items[iid]
                n_retired_cap = n_over

        b_vals = np.array([it.b_hat for it in bank.items.values()])
        v_t = bank_value(b_vals, mu_hat, s_hat) if len(b_vals) else 0.0
        n_bank = len(bank.items)
        rho = _ranking_quality(theta_hat, fld.thetas, fld.active_model_ids)
        n_rankable = sum(
            1
            for m in fld.active_model_ids
            if m in theta_hat and m in fld.thetas
        )
        seed_vals = [
            item_value(seed_item_b[iid], mu_hat, s_hat)
            for iid in seed_item_b
        ]
        seed_mean_value = float(np.mean(seed_vals)) if seed_vals else float("nan")

        incumbent = sorted(active_set & prev_active) if prev_active else []
        rho_incumbent = _ranking_quality(theta_hat, fld.thetas, incumbent)
        rank_shift_incumbent = _incumbent_rank_shift(theta_hat, fld.thetas, incumbent)
        n_items_new = max(0, len(avail) - prev_avail_count)

        rankable = [m for m in fld.active_model_ids if m in theta_hat and m in fld.thetas]
        hat_order = sorted(rankable, key=lambda m: theta_hat[m], reverse=True)
        oracle_order = sorted(rankable, key=lambda m: fld.thetas[m], reverse=True)
        hat_rank = {m: i + 1 for i, m in enumerate(hat_order)}
        oracle_rank = {m: i + 1 for i, m in enumerate(oracle_order)}
        for mid in rankable:
            model_rows.append(
                {
                    "round": rd.round_idx,
                    "round_date": rd.round_date,
                    "model_id": mid,
                    "theta_hat": theta_hat[mid],
                    "theta_star": fld.thetas[mid],
                    "rank_hat": hat_rank[mid],
                    "rank_oracle": oracle_rank[mid],
                    "n_active_models": len(rankable),
                }
            )

        bank_by_dataset = {}
        for iid in bank.items:
            ds = item_dataset.get(iid, "unknown")
            bank_by_dataset[ds] = bank_by_dataset.get(ds, 0) + 1

        row = {
                "arm": arm,
                "rollout_mode": cfg.rollout_mode,
                "round": rd.round_idx,
                "round_date": rd.round_date,
                "mu_hat": mu_hat,
                "mu_oracle": fld.mu,
                "mu_oracle_pool": mu_oracle_pool,
                "s_hat": s_hat,
                "field_window_top_k": cfg.field_window_top_k if cfg.field_window_top_k else 0,
                "field_window_top_frac": cfg.field_window_top_frac if cfg.field_window_top_frac else 0,
                "n_field_window_models": n_field_window,
                "bank_value": v_t,
                "bank_value_per_item": v_t / n_bank if n_bank else 0.0,
                "bank_size": n_bank,
                "ranking_spearman": rho,
                "ranking_spearman_incumbent": rho_incumbent,
                "incumbent_rank_shift": rank_shift_incumbent,
                "n_rankable_models": n_rankable,
                "seed_mean_value": seed_mean_value,
                "n_active_models": len(fld.active_model_ids),
                "n_new_models": len(new_models),
                "new_models": ",".join(new_models),
                "n_items_available": len(avail),
                "n_items_new": n_items_new,
                "n_acquired_pool": n_acquired_pool,
                "n_acquired_synth": n_acquired_synth,
                "n_acquired": n_acquired_pool + n_acquired_synth,
                "n_retired_value": n_retired_value,
                "n_retired_cap": n_retired_cap,
                "n_retired": n_retired_value + n_retired_cap,
                "n_promoted": n_promoted,
                "bank_budget": cfg.bank_budget,
                "equal_bank_size": cfg.equal_bank_size,
                "seed_full_pool": cfg.seed_full_pool,
            }
        if item_dataset:
            for ds in sorted(set(item_dataset.values())):
                row[f"bank_{ds}"] = bank_by_dataset.get(ds, 0)
                row[f"n_retired_{ds}"] = retired_by_dataset.get(ds, 0) if arm == "full" else 0
        rows.append(row)
        prev_active = active_set
        prev_avail_count = len(avail)

    return pd.DataFrame(rows), pd.DataFrame(model_rows), pd.DataFrame(upgrade_rows)


def run_all_arms(
    resp: pd.DataFrame,
    theta_star: pd.DataFrame,
    diff_star: pd.DataFrame,
    cfg: SimConfig,
) -> pd.DataFrame:
    diff_star = diff_star.copy()
    if "cell_id" not in diff_star.columns:
        diff_star = diff_star.rename(columns={"item_id": "cell_id"})
    item_ids = diff_star["cell_id"].to_numpy()
    item_release = resp.drop_duplicates("cell_id").set_index("cell_id")["release_date"]
    model_release = resp.drop_duplicates("model_id").set_index("model_id")["model_release_date"]

    anchor_ids: list[str] | None = None
    seed_avail_ids: list[str] | None = None
    if cfg.equal_bank_size:
        rng = np.random.default_rng(cfg.seed)
        anchor_ids = select_anchors(
            item_ids, diff_star["b_star"].to_numpy(), n_anchors=cfg.n_anchors, rng=rng
        )
        rounds_end = pd.Timestamp(item_release.max())
        if cfg.drift_mode == "real":
            rounds_end = max(rounds_end, pd.Timestamp(model_release.max()))
        rounds = build_rounds(item_release, n_rounds=cfg.n_rounds, end_date=rounds_end)
        avail_by_round = items_available_by_round(item_ids, item_release, rounds)
        seed_avail_ids = avail_by_round[rounds[-1].round_idx]

    frames = [
        run_arm(a, resp, theta_star, diff_star, cfg, anchor_ids=anchor_ids, seed_avail_ids=seed_avail_ids)[0]
        for a in ("fixed", "recalibrate", "full")
    ]
    return pd.concat(frames, ignore_index=True)
