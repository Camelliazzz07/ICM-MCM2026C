# src/mc_elimination.py
# Monte Carlo elimination determinism / robustness analysis
#
# Usage (from project root):
#   python src/mc_elimination.py --features_csv data/clean_long.csv --votes_csv output/bayes_vi_vote_estimates.csv --out_dir output
#
# Inputs:
#   features_csv (clean_long.csv):
#     required columns: season, week, celebrity_name, judge_total, active
#     optional: elim_after_week (True/False), method (if already computed)
#
#   votes_csv (bayes_vi_vote_estimates.csv):
#     required columns: season, week, celebrity_name, bayes_vote_share
#     optional: bayes_vote_share_sd (recommended)
#
# Outputs (in out_dir):
#   mc_elimination_probs.csv
#   mc_week_metrics.csv

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Voting-rule utilities
# -----------------------------

def voting_method_for_season(season_id: int) -> str:
    """
    Based on the problem statement / history (as used in your infer script):
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
    x = x.astype(float)
    # pair[j,i] = (x_j - x_i)/tau
    pair = (x.reshape(1, -1) - x.reshape(-1, 1)) / max(tau, 1e-6)
    sig = 1.0 / (1.0 + np.exp(-pair))
    # subtract ~0.5 to reduce self-comparison bias, mirroring your torch code
    return 1.0 + sig.sum(axis=1) - 0.5


def compute_badness_np(
    season_id: int,
    Jt: np.ndarray,
    vp: np.ndarray,
    weight_judge: float,
    rank_tau: float,
) -> np.ndarray:
    """
    Larger badness => more likely eliminated (worse).
    Mirrors your infer_fan_votes.py convention.
    """
    method = voting_method_for_season(season_id)

    if method == "percent":
        jp = Jt / (np.sum(Jt) + 1e-12)  # higher better
        C = weight_judge * jp + (1.0 - weight_judge) * vp
        return -C  # lower C => worse, so badness = -C

    # rank-based
    rj = soft_rank_np(Jt, tau=rank_tau)  # 1 best, larger worse
    rv = soft_rank_np(vp, tau=rank_tau)
    return rj + rv


def pick_eliminated_index_from_badness(
    season_id: int,
    badness: np.ndarray,
    Jt: np.ndarray,
) -> int:
    """
    Return index (0..m-1) eliminated under rule proxy:
      - percent / rank: highest badness
      - rank_judges_save: pick bottom-2 by badness, then eliminate lower judge total among those two
    """
    method = voting_method_for_season(season_id)
    if method == "rank_judges_save" and len(badness) >= 2:
        bottom2 = np.argsort(-badness)[:2]  # worst two (highest badness)
        # Judges-save proxy: eliminated is the one with lower judge total among bottom2
        choice = bottom2[np.argmin(Jt[bottom2])]
        return int(choice)
    return int(np.argmax(badness))


# -----------------------------
# Vote sampling
# -----------------------------

def sample_votes_logit_normal(
    p_mean: np.ndarray,
    p_sd: Optional[np.ndarray],
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Sample a probability vector on simplex given mean & sd.
    We use a practical approximation:
      - convert mean shares to logits z = log(p)
      - add Gaussian noise scaled by p_sd / (p*(1-p)) (delta method, clipped)
      - softmax back to simplex

    If p_sd is None or missing, falls back to small noise to avoid degeneracy.
    """
    eps = 1e-9
    p = np.clip(p_mean.astype(float), eps, 1.0)
    p = p / np.sum(p)

    if p_sd is None:
        # mild noise
        sigma_z = np.full_like(p, 0.10)
    else:
        sd = np.clip(p_sd.astype(float), 0.0, 0.25)
        # delta method: Var(logit(p)) ≈ Var(p) / (p(1-p))^2
        denom = np.clip(p * (1.0 - p), 1e-3, None)
        sigma_z = np.clip(sd / denom, 0.02, 0.80)

    z = np.log(p)  # log-space (not logit), then softmax
    z_s = z + rng.randn(*z.shape) * sigma_z
    z_s = z_s - np.mean(z_s)  # remove location to avoid blow-up

    # softmax
    ex = np.exp(z_s - np.max(z_s))
    out = ex / np.sum(ex)
    return out


# -----------------------------
# Metrics
# -----------------------------

def entropy_of_probs(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return float("nan")
    return float(-np.sum(p * np.log(p + 1e-12)))


def top1_gap(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    if len(p) < 2:
        return float("nan")
    s = np.sort(p)[::-1]
    return float(s[0] - s[1])


# -----------------------------
# I/O helpers
# -----------------------------

def read_csv_stripped(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df


def normalize_bool_col(x) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ("1", "true", "t", "yes", "y")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_csv", type=str, required=True)
    parser.add_argument("--votes_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="output")

    parser.add_argument("--n_mc", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)

    # rule parameters (match infer script defaults)
    parser.add_argument("--weight_judge", type=float, default=0.5)
    parser.add_argument("--rank_tau", type=float, default=0.05)

    # filtering
    parser.add_argument("--min_active", type=int, default=3)
    parser.add_argument("--require_judge_total", action="store_true",
                        help="If set, only include active rows with non-missing judge_total")

    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    feat = read_csv_stripped(args.features_csv)
    votes = read_csv_stripped(args.votes_csv)

    # Standardize columns
    required_feat = {"season", "week", "celebrity_name", "active"}
    missing_feat = required_feat - set(feat.columns)
    if missing_feat:
        raise ValueError(f"features_csv missing columns: {missing_feat}")

    required_votes = {"season", "week", "celebrity_name", "bayes_vote_share"}
    missing_votes = required_votes - set(votes.columns)
    if missing_votes:
        raise ValueError(f"votes_csv missing columns: {missing_votes}")

    feat["season"] = pd.to_numeric(feat["season"], errors="coerce").astype(int)
    feat["week"] = pd.to_numeric(feat["week"], errors="coerce").astype(int)
    feat["active"] = feat["active"].apply(normalize_bool_col)

    if "judge_total" in feat.columns:
        feat["judge_total"] = pd.to_numeric(feat["judge_total"], errors="coerce")
    else:
        feat["judge_total"] = np.nan

    if "elim_after_week" in feat.columns:
        feat["elim_after_week"] = feat["elim_after_week"].apply(normalize_bool_col)
    else:
        feat["elim_after_week"] = False

    votes["season"] = pd.to_numeric(votes["season"], errors="coerce").astype(int)
    votes["week"] = pd.to_numeric(votes["week"], errors="coerce").astype(int)
    votes["bayes_vote_share"] = pd.to_numeric(votes["bayes_vote_share"], errors="coerce")
    if "bayes_vote_share_sd" in votes.columns:
        votes["bayes_vote_share_sd"] = pd.to_numeric(votes["bayes_vote_share_sd"], errors="coerce")
    else:
        votes["bayes_vote_share_sd"] = np.nan

    # Merge: we only care about rows that are active in features
    df = feat.merge(
        votes[["season", "week", "celebrity_name", "bayes_vote_share", "bayes_vote_share_sd"]],
        on=["season", "week", "celebrity_name"],
        how="left",
        validate="m:1",
    )

    # Compute method per season
    df["method"] = df["season"].apply(voting_method_for_season)

    # Build groups
    groups = df.groupby(["season", "week"], as_index=False)

    prob_rows = []
    week_rows = []

    n_groups_total = 0
    n_groups_used = 0

    for (season_id, week_id), sub in df.groupby(["season", "week"]):
        n_groups_total += 1

        sub = sub.copy()
        sub = sub[sub["active"] == True]
        if len(sub) < args.min_active:
            continue

        if args.require_judge_total:
            sub = sub[~sub["judge_total"].isna()]
            if len(sub) < args.min_active:
                continue

        # Need vote mean available
        sub = sub[~sub["bayes_vote_share"].isna()]
        if len(sub) < args.min_active:
            continue

        # Extract arrays
        names = sub["celebrity_name"].tolist()
        Jt = sub["judge_total"].to_numpy(dtype=float)
        # If judge_total missing, fill by group mean (weak) to keep MC running for determinism-only
        if np.any(np.isnan(Jt)):
            muJ = np.nanmean(Jt)
            if np.isnan(muJ):
                # if all missing, cannot use judge_total at all
                continue
            Jt = np.where(np.isnan(Jt), muJ, Jt)

        p_mean = sub["bayes_vote_share"].to_numpy(dtype=float)
        p_sd = sub["bayes_vote_share_sd"].to_numpy(dtype=float)
        if np.all(np.isnan(p_sd)):
            p_sd_use = None
        else:
            p_sd_use = np.where(np.isnan(p_sd), np.nanmedian(p_sd), p_sd)

        m = len(sub)

        # MC tally: which index eliminated
        counts = np.zeros(m, dtype=int)

        for _ in range(args.n_mc):
            vp = sample_votes_logit_normal(p_mean, p_sd_use, rng)
            bad = compute_badness_np(season_id, Jt, vp, args.weight_judge, args.rank_tau)
            e_idx = pick_eliminated_index_from_badness(season_id, bad, Jt)
            counts[e_idx] += 1

        probs = counts / float(args.n_mc)

        # Determine true elimination (only if exactly one elim_after_week in this group)
        elim_flags = sub["elim_after_week"].astype(bool).to_numpy()
        true_elim_idx = None
        if np.sum(elim_flags) == 1:
            true_elim_idx = int(np.where(elim_flags)[0][0])

        # Metrics
        ent = entropy_of_probs(probs)
        gap = top1_gap(probs)
        pmax = float(np.max(probs)) if len(probs) else float("nan")

        p_true = float(probs[true_elim_idx]) if true_elim_idx is not None else float("nan")

        # Write per-contestant probs
        for i in range(m):
            prob_rows.append({
                "season": int(season_id),
                "week": int(week_id),
                "method": voting_method_for_season(int(season_id)),
                "celebrity_name": names[i],
                "judge_total": float(Jt[i]),
                "bayes_vote_share": float(p_mean[i]),
                "bayes_vote_share_sd": float(p_sd[i]) if not np.isnan(p_sd[i]) else np.nan,
                "p_eliminated_mc": float(probs[i]),
                "elim_after_week_true": bool(elim_flags[i]),
            })

        # Write week metrics
        week_rows.append({
            "season": int(season_id),
            "week": int(week_id),
            "method": voting_method_for_season(int(season_id)),
            "n_active_used": int(m),
            "entropy_elim": float(ent),
            "top1_gap_elim": float(gap),
            "p_max_elim": float(pmax),
            "p_true_elim": float(p_true),
            "has_single_true_elim": bool(true_elim_idx is not None),
        })

        n_groups_used += 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_prob = pd.DataFrame(prob_rows)
    df_week = pd.DataFrame(week_rows)

    prob_out = out_dir / "mc_elimination_probs.csv"
    week_out = out_dir / "mc_week_metrics.csv"

    df_prob.to_csv(prob_out, index=False)
    df_week.to_csv(week_out, index=False)

    print("======== MC ELIMINATION DONE ========")
    print(f"Groups total (season-week in input): {n_groups_total}")
    print(f"Groups used (n_active>= {args.min_active} and vote_mean present): {n_groups_used}")
    print("Saved:")
    print(f"  {prob_out}")
    print(f"  {week_out}")

    # Quick overall summaries
    if len(df_week):
        print("\n======== QUICK SUMMARY ========")
        print("weeks:", len(df_week))
        print("p_max mean/median:", df_week["p_max_elim"].mean(), df_week["p_max_elim"].median())
        # normalized entropy determinism score (optional)
        H = df_week["entropy_elim"].to_numpy(dtype=float)
        n = df_week["n_active_used"].to_numpy(dtype=float)
        H_norm = H / np.log(np.clip(n, 2, None))
        D = 1.0 - H_norm
        print("D_entropy mean/median:", float(np.nanmean(D)), float(np.nanmedian(D)))
        print("strong-deterministic rate (p_max>=0.8):", float(np.mean(df_week["p_max_elim"] >= 0.8)))
        print("highly-sensitive rate (gap<0.05):", float(np.mean(df_week["top1_gap_elim"] < 0.05)))

        # true-elim diagnostic only on weeks with exactly 1 true elim
        df_true = df_week[df_week["has_single_true_elim"] == True].copy()
        if len(df_true):
            print("\n======== TRUE ELIM DIAGNOSTIC (single-elim weeks) ========")
            print("true-elim weeks:", len(df_true))
            print("p_true mean/median:", df_true["p_true_elim"].mean(), df_true["p_true_elim"].median())
            print("low-prob true elimination rate (p_true<=0.3):", float(np.mean(df_true["p_true_elim"] <= 0.3)))
        else:
            print("\n(no weeks with exactly one true elimination flag; p_true_elim is NaN)")

if __name__ == "__main__":
    main()
