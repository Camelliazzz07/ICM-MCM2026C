import argparse
import math
import numpy as np
import pandas as pd
from pathlib import Path


# -----------------------------
# Voting-rule utilities (match your infer_fan_votes.py intent)
# -----------------------------
def voting_method_for_season(season_id: int) -> str:
    """
    Based on the problem statement / history:
      - Seasons 1-2: rank-based combination
      - Seasons 3-27: percent-based combination
      - Seasons 28+: rank-based again, plus 'judges save' between bottom two
    """
    if season_id <= 2:
        return "rank"
    if 3 <= season_id <= 27:
        return "percent"
    return "rank_judges_save"


def soft_rank_np(x: np.ndarray, tau: float = 0.05) -> np.ndarray:
    """
    Differentiable-ish approximation of rank (1 = best), higher x means better.
    r_i = 1 + sum_j sigmoid((x_j - x_i)/tau)
    """
    x = x.reshape(-1)
    m = x.shape[0]
    # pairwise differences: j - i
    pair = (x.reshape(1, m) - x.reshape(m, 1)) / float(tau)
    sig = 1.0 / (1.0 + np.exp(-pair))
    # subtract ~0.5 to reduce self-comparison bias (mirror your torch version)
    return 1.0 + sig.sum(axis=1) - 0.5


def compute_badness_np(
    season_id: int,
    Jt: np.ndarray,
    vp: np.ndarray,
    weight_judge: float,
    rank_tau: float,
) -> np.ndarray:
    """
    Convention: larger badness => more likely to be eliminated (worse).
    """
    method = voting_method_for_season(int(season_id))

    if method == "percent":
        jp = Jt / (np.sum(Jt) + 1e-12)          # higher is better
        C = weight_judge * jp + (1 - weight_judge) * vp
        return -C                               # lower C => worse

    # rank-based
    rj = soft_rank_np(Jt, tau=rank_tau)         # 1 best, larger worse
    rv = soft_rank_np(vp, tau=rank_tau)
    return rj + rv                              # larger worse


# -----------------------------
# Sampling votes
# -----------------------------
def sample_votes_dirichlet_approx(p_mean: np.ndarray, p_sd: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Simple Dirichlet approximation from mean+sd (not exact; fallback).
    Not used by default.
    """
    p = np.clip(p_mean, eps, 1 - eps)
    v = np.clip(p_sd**2, eps, None)

    # For Dirichlet: Var[p_i] = (a_i (a0 - a_i)) / (a0^2 (a0 + 1))
    # Use method-of-moments: estimate a0 from average implied a0_i, then a_i = p_i * a0
    a0_list = []
    for pi, vi in zip(p, v):
        denom = vi
        num = pi * (1 - pi)
        if denom <= 0:
            continue
        a0 = num / denom - 1.0
        if np.isfinite(a0) and a0 > 2.0:
            a0_list.append(a0)
    if not a0_list:
        a0 = 50.0
    else:
        a0 = float(np.median(a0_list))
        a0 = max(a0, 5.0)

    alpha = np.clip(p * a0, eps, None)
    return np.random.dirichlet(alpha)


def sample_votes_logitnormal_indep(p_mean: np.ndarray, p_sd: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Practical logit-normal-ish sampler using mean+sd only:
      1) convert mean to logits
      2) set independent Gaussian noise scale from sd via local linearization:
         Var(p) ≈ (p(1-p))^2 Var(logit)
         => Var(logit) ≈ Var(p) / (p(1-p))^2
      3) sample logits, softmax
    This ignores cross-covariance (we don't have it), but works well in practice for MC propagation.
    """
    p = np.clip(p_mean, eps, 1 - eps)
    v = np.clip(p_sd**2, 0.0, None)

    # logit mean
    z_mu = np.log(p)  # we will softmax; log(p) is fine up to constant
    # local scale
    denom = (p * (1 - p)) ** 2
    denom = np.clip(denom, eps, None)
    z_var = v / denom
    z_sd = np.sqrt(np.clip(z_var, 0.0, None))

    z = z_mu + z_sd * np.random.randn(*z_mu.shape)

    # stabilize
    z = z - np.max(z)
    expz = np.exp(z)
    return expz / (np.sum(expz) + 1e-12)


# -----------------------------
# Week-level MC
# -----------------------------
def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p)]
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def run_mc_for_week(
    season_id: int,
    week: int,
    names: list,
    Jt: np.ndarray,
    vote_mean: np.ndarray,
    vote_sd: np.ndarray,
    true_elims: list,
    mc: int,
    weight_judge: float,
    rank_tau: float,
    rule: str,
    rng_seed: int,
    sampler: str,
):
    """
    Returns:
      probs: dict name -> (p_elim, p_bottom1, p_bottom2)
      week_metrics: dict of stability metrics
    """
    rng = np.random.RandomState(rng_seed)
    np_random_backup = np.random.get_state()
    np.random.set_state(rng.get_state())

    m = len(names)
    E = len(true_elims)

    # If no elimination label (some weeks), skip MC output
    if E == 0:
        np.random.set_state(np_random_backup)
        return None, {
            "season": season_id,
            "week": week,
            "method": rule,
            "n_active": m,
            "n_elim_true": 0,
            "mc": mc,
            "entropy_elim": np.nan,
            "top1_gap_elim": np.nan,
            "p_true_elim": np.nan,
        }

    elim_counts = np.zeros(m, dtype=int)
    bottom1_counts = np.zeros(m, dtype=int)
    bottom2_counts = np.zeros(m, dtype=int)

    # map true elim indices
    true_set = set(true_elims)

    for _ in range(mc):
        if sampler == "dirichlet":
            vp = sample_votes_dirichlet_approx(vote_mean, vote_sd)
        else:
            vp = sample_votes_logitnormal_indep(vote_mean, vote_sd)

        b = compute_badness_np(season_id, Jt, vp, weight_judge, rank_tau)

        # bottom ordering: worst first => descending badness
        order = np.argsort(-b)
        bottom1 = order[0]
        bottom2 = order[: min(2, m)]

        bottom1_counts[bottom1] += 1
        bottom2_counts[bottom2] += 1

        if rule == "rank_judges_save" and m >= 2:
            # Judges-save: eliminated is one of bottom2; choose lower judge_total among bottom2
            jvals = Jt[bottom2]
            elim_one = bottom2[int(np.argmin(jvals))]
            elim_idx = [elim_one]  # 1 eliminated
        else:
            # Standard: eliminate E worst by badness
            elim_idx = list(order[:E])

        for ei in elim_idx:
            elim_counts[ei] += 1

    p_elim = elim_counts / float(mc)
    p_b1 = bottom1_counts / float(mc)
    p_b2 = bottom2_counts / float(mc)

    # stability metrics for this week
    pe = p_elim.copy()
    pe_sum = pe.sum()
    if pe_sum > 0:
        pe = pe / pe_sum
    H = entropy(pe) if np.isfinite(pe).all() else np.nan
    top2 = np.sort(pe)[-2:] if m >= 2 else np.array([pe.max(), 0.0])
    top1_gap = float(top2[-1] - top2[-2]) if len(top2) >= 2 else np.nan

    # probability that the true eliminated set is exactly matched in a draw:
    # We approximate by sum over individuals when E==1; for E>1 exact set prob would need joint.
    if E == 1:
        true_i = list(true_set)[0]
        p_true = float(p_elim[true_i])
    else:
        # conservative proxy: average probability mass on true eliminated individuals
        p_true = float(np.mean([p_elim[i] for i in true_set])) if true_set else np.nan

    week_metrics = {
        "season": season_id,
        "week": week,
        "method": rule,
        "n_active": m,
        "n_elim_true": E,
        "mc": mc,
        "entropy_elim": H,
        "top1_gap_elim": top1_gap,
        "p_true_elim_proxy": p_true,
        "sampler": sampler,
    }

    probs = {
        names[i]: (float(p_elim[i]), float(p_b1[i]), float(p_b2[i]))
        for i in range(m)
    }

    np.random.set_state(np_random_backup)
    return probs, week_metrics


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", type=str, default="data/clean_long.csv")
    ap.add_argument("--votes_csv", type=str, default="output/bayes_vi_vote_estimates.csv")
    ap.add_argument("--out_dir", type=str, default="output")

    ap.add_argument("--kind", type=str, default="bayes",
                    help="prefix: bayes_vote_share, bayes_vote_share_sd")
    ap.add_argument("--mc", type=int, default=2000)
    ap.add_argument("--weight_judge", type=float, default=0.5)
    ap.add_argument("--rank_tau", type=float, default=0.05)
    ap.add_argument("--sampler", type=str, default="logitnormal",
                    choices=["logitnormal", "dirichlet"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = pd.read_csv(args.features_csv)
    votes = pd.read_csv(args.votes_csv)

    # normalize types
    feats["season"] = pd.to_numeric(feats["season"], errors="coerce").astype(int)
    feats["week"] = pd.to_numeric(feats["week"], errors="coerce").astype(int)
    votes["season"] = pd.to_numeric(votes["season"], errors="coerce").astype(int)
    votes["week"] = pd.to_numeric(votes["week"], errors="coerce").astype(int)

    mean_col = f"{args.kind}_vote_share"
    sd_col = f"{args.kind}_vote_share_sd"
    need = ["season", "week", "celebrity_name", mean_col, sd_col]
    for c in need:
        if c not in votes.columns:
            raise ValueError(f"votes_csv missing required col: {c}")

    df = feats.merge(
        votes[need],
        on=["season", "week", "celebrity_name"],
        how="left",
    )

    # active filter
    if "active" in df.columns:
        if df["active"].dtype != bool:
            df["active"] = df["active"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
        df = df[df["active"] == True].copy()

    # required
    if "judge_total" not in df.columns:
        raise ValueError("features_csv must contain judge_total")
    if "elim_after_week" not in df.columns:
        raise ValueError("features_csv must contain elim_after_week (official labels)")

    df["judge_total"] = pd.to_numeric(df["judge_total"], errors="coerce")
    df[mean_col] = pd.to_numeric(df[mean_col], errors="coerce")
    df[sd_col] = pd.to_numeric(df[sd_col], errors="coerce")

    # Drop rows where votes are missing (should not happen for active, but safe)
    df = df.dropna(subset=[mean_col, sd_col, "judge_total"])

    prob_rows = []
    week_rows = []

    # iterate season-week groups
    for (s, w), g in df.groupby(["season", "week"], sort=True):
        season_id = int(s)
        week = int(w)
        rule = voting_method_for_season(season_id)

        # stable ordering
        g = g.sort_values("celebrity_name")
        names = g["celebrity_name"].tolist()
        Jt = g["judge_total"].to_numpy(dtype=float)

        vote_mean = g[mean_col].to_numpy(dtype=float)
        vote_sd = g[sd_col].to_numpy(dtype=float)

        # renormalize mean votes within group (numerical safety)
        sm = vote_mean.sum()
        if sm > 0 and np.isfinite(sm):
            vote_mean = vote_mean / sm

        # true eliminated indices for this week (may be multiple)
        true_elims = g.index[g["elim_after_week"].astype(bool) == True].tolist()
        # convert to local positions
        idx_map = {idx: i for i, idx in enumerate(g.index.tolist())}
        true_pos = [idx_map[idx] for idx in true_elims if idx in idx_map]

        seed_week = int(args.seed + 100000 * season_id + 1000 * week)

        probs, wk_metrics = run_mc_for_week(
            season_id=season_id,
            week=week,
            names=names,
            Jt=Jt,
            vote_mean=vote_mean,
            vote_sd=vote_sd,
            true_elims=true_pos,
            mc=int(args.mc),
            weight_judge=float(args.weight_judge),
            rank_tau=float(args.rank_tau),
            rule=rule,
            rng_seed=seed_week,
            sampler=args.sampler,
        )

        week_rows.append(wk_metrics)

        if probs is None:
            continue

        for i, name in enumerate(names):
            p_elim, p_b1, p_b2 = probs[name]
            prob_rows.append({
                "season": season_id,
                "week": week,
                "method": rule,
                "celebrity_name": name,
                "judge_total": float(Jt[i]),
                "vote_mean": float(vote_mean[i]),
                "vote_sd": float(vote_sd[i]),
                "p_eliminated_mc": p_elim,
                "p_bottom1_mc": p_b1,
                "p_bottom2_mc": p_b2,
                "elim_after_week_true": bool(i in set(true_pos)),
            })

        # minimal progress print
        if (week % 2) == 0:
            print(f"[Season {season_id} Week {week}] n={len(names)} elim_true={len(true_pos)} rule={rule}")

    probs_df = pd.DataFrame(prob_rows)
    weeks_df = pd.DataFrame(week_rows)

    out_probs = out_dir / "mc_elimination_probs.csv"
    out_weeks = out_dir / "mc_week_metrics.csv"
    probs_df.to_csv(out_probs, index=False)
    weeks_df.to_csv(out_weeks, index=False)

    print("Saved:")
    print(" ", out_probs)
    print(" ", out_weeks)


if __name__ == "__main__":
    main()
