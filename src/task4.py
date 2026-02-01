# -*- coding: utf-8 -*-
"""
Task 4: Proposed voting rule + evaluation metrics

Inputs:
  - data/clean_long.csv
  - output/bayes_vi_vote_estimates.csv

Outputs (in output/):
  - task4_weekly_elims_newrule.csv
  - task4_final_ranking_newrule.csv
  - task4_metrics_newrule.csv
  - task4_stability_newrule.csv

Run (from repo root):
  python src/task4.py

Optional args:
  python src/task4.py --alpha0 0.5 --k 1.0 --n_samples 200 --seed 42
"""

import os
import argparse
import numpy as np
import pandas as pd


# ---------------------------
# Utilities
# ---------------------------
def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """Strip spaces, lower, replace internal whitespace with underscore."""
    new_cols = []
    for c in df.columns:
        c2 = str(c).strip().lower()
        c2 = "_".join(c2.split())  # collapse all whitespace to underscores
        new_cols.append(c2)
    df = df.copy()
    df.columns = new_cols
    return df


def safe_mkdir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def spearman_corr(a: pd.Series, b: pd.Series) -> float:
    """Spearman correlation without scipy: corr(rank(a), rank(b))."""
    ra = a.rank(method="average")
    rb = b.rank(method="average")
    if ra.std(ddof=0) == 0 or rb.std(ddof=0) == 0:
        return np.nan
    return float(ra.corr(rb))


def certainty_weight(sd: pd.Series, k: float, eps: float = 1e-8) -> pd.Series:
    """
    certainty c = 1/(sd+eps)
    normalize within week by median: c_norm = c / median(c)
    g(c_norm) = min(1, c_norm/(c_norm + k))
    """
    c = 1.0 / (sd.astype(float).clip(lower=eps))
    med = np.median(c[~np.isnan(c)]) if np.any(~np.isnan(c)) else 1.0
    if med <= 0 or np.isnan(med):
        med = 1.0
    c_norm = c / med
    g = c_norm / (c_norm + k)
    g = np.minimum(1.0, np.maximum(0.0, g))
    return g


def compute_week_scores(df_week: pd.DataFrame, alpha0: float, k: float) -> pd.DataFrame:
    """
    df_week must include:
      - judge_total
      - bayes_vote_share, bayes_vote_share_sd
    returns df_week with:
      - judge_pct, fan_pct, alpha_i, combined_score
    """
    d = df_week.copy()

    # Judge percent
    jt = d["judge_total"].astype(float).fillna(0.0)
    jt_sum = jt.sum()
    d["judge_pct"] = (jt / jt_sum) if jt_sum > 0 else 0.0

    # Fan percent (re-normalize among active)
    fs = d["bayes_vote_share"].astype(float)
    fs = fs.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fs_sum = fs.sum()
    d["fan_pct"] = (fs / fs_sum) if fs_sum > 0 else 0.0

    # Certainty -> alpha_i
    sd = d["bayes_vote_share_sd"].astype(float)
    g = certainty_weight(sd=sd, k=k)
    d["alpha_i"] = alpha0 * g

    # Combined score
    d["combined_score"] = d["alpha_i"] * d["fan_pct"] + (1.0 - d["alpha_i"]) * d["judge_pct"]
    return d


# ---------------------------
# Main simulation: elimination + final ranking
# ---------------------------
def simulate_newrule(
    clean_long: pd.DataFrame,
    votes: pd.DataFrame,
    alpha0: float,
    k: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      weekly_elims: rows = season, week, predicted_elim_count, predicted_elims, actual_elims, exact_match, jaccard
      final_ranking: rows = season, celebrity_name, predicted_final_rank, predicted_final_score, actual_placement (if available)
    """
    # Merge
    df = clean_long.merge(
        votes,
        on=["season", "week", "celebrity_name"],
        how="left",
        validate="many_to_one",
    )

    # Some weeks may lack vote estimates due to preprocessing differences; keep but warn via metrics later.
    # Work only on active contestants for each (season, week)
    df["active"] = df["active"].astype(bool)

    # Determine per season last week in data
    season_last_week = df.groupby("season")["week"].max().to_dict()

    weekly_rows = []
    final_rows = []

    for season, d_season in df.groupby("season"):
        last_week = int(season_last_week[season])

        # weekly elimination simulation
        for week, d_week_all in d_season.groupby("week"):
            d_week = d_week_all[d_week_all["active"]].copy()
            if d_week.empty:
                continue

            # actual eliminations after this week (from clean_long)
            actual_elims = sorted(d_week_all.loc[d_week_all["elim_after_week"].astype(bool), "celebrity_name"].dropna().unique().tolist())
            actual_k = len(actual_elims)

            # compute new rule scores
            d_scored = compute_week_scores(d_week, alpha0=alpha0, k=k)

            # predicted eliminations: choose actual_k lowest combined_score
            if actual_k == 0:
                pred_elims = []
            else:
                d_sorted = d_scored.sort_values(["combined_score", "judge_pct", "fan_pct"], ascending=[True, True, True])
                pred_elims = sorted(d_sorted["celebrity_name"].head(actual_k).tolist())

            # consistency metrics
            set_a = set(actual_elims)
            set_p = set(pred_elims)
            exact = int(set_a == set_p)
            jacc = (len(set_a & set_p) / len(set_a | set_p)) if len(set_a | set_p) > 0 else 1.0

            weekly_rows.append(
                {
                    "season": season,
                    "week": week,
                    "n_active": int(d_week.shape[0]),
                    "actual_elim_count": actual_k,
                    "predicted_elim_count": len(pred_elims),
                    "actual_elims": ";".join(actual_elims),
                    "predicted_elims": ";".join(pred_elims),
                    "exact_match": exact,
                    "jaccard": jacc,
                    "missing_vote_estimates": int(d_week_all["bayes_vote_share"].isna().sum()),
                }
            )

        # final ranking simulation: use last week, rank all active contestants by combined_score (descending)
        d_last_all = d_season[d_season["week"] == last_week].copy()
        d_last = d_last_all[d_last_all["active"]].copy()
        if not d_last.empty:
            d_last_scored = compute_week_scores(d_last, alpha0=alpha0, k=k)
            d_last_scored = d_last_scored.sort_values(["combined_score", "judge_pct", "fan_pct"], ascending=[False, False, False]).copy()
            d_last_scored["predicted_final_rank"] = np.arange(1, d_last_scored.shape[0] + 1)

            # actual placement is season-level constant in your clean_long; take first non-null per celeb
            placement_map = (
                d_season.groupby("celebrity_name")["placement"]
                .apply(lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan)
                .to_dict()
            )

            for _, r in d_last_scored.iterrows():
                final_rows.append(
                    {
                        "season": season,
                        "celebrity_name": r["celebrity_name"],
                        "predicted_final_rank": int(r["predicted_final_rank"]),
                        "predicted_final_score": float(r["combined_score"]),
                        "predicted_alpha_i": float(r["alpha_i"]),
                        "predicted_judge_pct": float(r["judge_pct"]),
                        "predicted_fan_pct": float(r["fan_pct"]),
                        "actual_placement": placement_map.get(r["celebrity_name"], np.nan),
                    }
                )

    weekly_elims = pd.DataFrame(weekly_rows).sort_values(["season", "week"]).reset_index(drop=True)
    final_ranking = pd.DataFrame(final_rows).sort_values(["season", "predicted_final_rank"]).reset_index(drop=True)
    return weekly_elims, final_ranking


# ---------------------------
# Stability analysis via posterior sampling
# ---------------------------
def stability_analysis(
    clean_long: pd.DataFrame,
    votes: pd.DataFrame,
    alpha0: float,
    k: float,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """
    For each (season, week), sample fan shares ~ Normal(mean, sd), truncate at 0, renormalize,
    recompute combined scores, and estimate:
      - elimination probability for each contestant that week (when there is elimination)
    Output aggregated stability:
      - avg_week_entropy: mean entropy of elimination distribution (higher => less stable)
      - avg_top1_prob: mean probability that predicted #1 stays #1 in final week (rough proxy)
    """
    rng = np.random.default_rng(seed)

    df = clean_long.merge(votes, on=["season", "week", "celebrity_name"], how="left")
    df = df.copy()
    df["active"] = df["active"].astype(bool)

    season_last_week = df.groupby("season")["week"].max().to_dict()

    rows = []
    for season, d_season in df.groupby("season"):
        last_week = int(season_last_week[season])

        week_entropies = []
        # final week top1 stability
        top1_probs = []

        # --- weekly elimination stability ---
        for week, d_week_all in d_season.groupby("week"):
            d_week = d_week_all[d_week_all["active"]].copy()
            if d_week.empty:
                continue

            actual_elims = d_week_all.loc[d_week_all["elim_after_week"].astype(bool), "celebrity_name"].dropna().unique().tolist()
            elim_k = len(actual_elims)
            if elim_k == 0:
                continue

            # judge pct fixed
            jt = d_week["judge_total"].astype(float).fillna(0.0).to_numpy()
            jt_sum = jt.sum()
            judge_pct = (jt / jt_sum) if jt_sum > 0 else np.zeros_like(jt)

            mu = d_week["bayes_vote_share"].astype(float).fillna(0.0).to_numpy()
            sd = d_week["bayes_vote_share_sd"].astype(float).fillna(0.0).to_numpy()

            # alpha_i fixed per week based on sd
            g = certainty_weight(pd.Series(sd), k=k).to_numpy()
            alpha_i = alpha0 * g

            names = d_week["celebrity_name"].tolist()
            elim_counts = {nm: 0 for nm in names}

            for _ in range(n_samples):
                fs = rng.normal(loc=mu, scale=sd)
                fs = np.clip(fs, 0.0, None)
                s = fs.sum()
                fan_pct = (fs / s) if s > 0 else np.zeros_like(fs)

                combined = alpha_i * fan_pct + (1.0 - alpha_i) * judge_pct
                # lowest elim_k eliminated
                idx = np.argsort(combined)[:elim_k]
                for j in idx:
                    elim_counts[names[j]] += 1

            probs = np.array([elim_counts[nm] / n_samples for nm in names], dtype=float)
            # entropy of elimination distribution (normalized by log(n))
            p = probs[probs > 0]
            entropy = -np.sum(p * np.log(p)) / (np.log(len(names)) if len(names) > 1 else 1.0)
            week_entropies.append(entropy)

        # --- final week top-1 stability ---
        d_last_all = d_season[d_season["week"] == last_week].copy()
        d_last = d_last_all[d_last_all["active"]].copy()
        if not d_last.empty:
            jt = d_last["judge_total"].astype(float).fillna(0.0).to_numpy()
            jt_sum = jt.sum()
            judge_pct = (jt / jt_sum) if jt_sum > 0 else np.zeros_like(jt)

            mu = d_last["bayes_vote_share"].astype(float).fillna(0.0).to_numpy()
            sd = d_last["bayes_vote_share_sd"].astype(float).fillna(0.0).to_numpy()
            g = certainty_weight(pd.Series(sd), k=k).to_numpy()
            alpha_i = alpha0 * g
            names = d_last["celebrity_name"].tolist()

            top1_count = {nm: 0 for nm in names}

            for _ in range(n_samples):
                fs = rng.normal(loc=mu, scale=sd)
                fs = np.clip(fs, 0.0, None)
                s = fs.sum()
                fan_pct = (fs / s) if s > 0 else np.zeros_like(fs)
                combined = alpha_i * fan_pct + (1.0 - alpha_i) * judge_pct
                top1 = names[int(np.argmax(combined))]
                top1_count[top1] += 1

            # probability of most-likely winner
            best_p = max(top1_count.values()) / n_samples
            top1_probs.append(best_p)

        rows.append(
            {
                "season": season,
                "avg_week_entropy": float(np.mean(week_entropies)) if len(week_entropies) else np.nan,
                "avg_top1_prob_final": float(np.mean(top1_probs)) if len(top1_probs) else np.nan,
                "n_weeks_with_elim": int(len(week_entropies)),
            }
        )

    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)


# ---------------------------
# Metrics summary for Task 4
# ---------------------------
def build_metrics(weekly_elims: pd.DataFrame, final_ranking: pd.DataFrame, clean_long: pd.DataFrame) -> pd.DataFrame:
    """
    Consistency:
      - exact_match_rate
      - avg_jaccard

    Fairness proxies (final week):
      - spearman(judge_pct_final, predicted_rank_final) per season
      - "low_judge_winner_rate": fraction of seasons where predicted winner is in bottom 25% judge_pct_final

    Fan influence proxy (final week):
      - corr(fan_pct_final, combined_score_final) per season (higher => more fan-driven)
    """
    # consistency by season
    cons = weekly_elims.groupby("season").agg(
        weeks=("week", "nunique"),
        exact_match_rate=("exact_match", "mean"),
        avg_jaccard=("jaccard", "mean"),
        avg_missing_votes=("missing_vote_estimates", "mean"),
    ).reset_index()

    # fairness + fan influence from final_ranking (per season)
    fr = final_ranking.copy()
    if fr.empty:
        # return at least consistency
        cons["judge_spearman_final"] = np.nan
        cons["low_judge_winner"] = np.nan
        cons["fan_influence_corr_final"] = np.nan
        return cons

    fair_rows = []
    for season, d in fr.groupby("season"):
        # judge_spearman: judge_pct vs predicted_final_rank (lower rank is better, so use -rank)
        judge_s = spearman_corr(d["predicted_judge_pct"], -d["predicted_final_rank"].astype(float))

        # low judge winner: predicted rank 1 has judge_pct in bottom quartile?
        d_sorted_j = d.sort_values("predicted_judge_pct", ascending=True).reset_index(drop=True)
        q = max(1, int(np.ceil(0.25 * d_sorted_j.shape[0])))
        bottom_names = set(d_sorted_j["celebrity_name"].head(q).tolist())
        winner = d.loc[d["predicted_final_rank"] == 1, "celebrity_name"].iloc[0]
        low_judge_winner = int(winner in bottom_names)

        # fan influence proxy: corr(fan_pct, combined_score)
        if d["predicted_fan_pct"].std(ddof=0) == 0 or d["predicted_final_score"].std(ddof=0) == 0:
            fan_corr = np.nan
        else:
            fan_corr = float(d["predicted_fan_pct"].corr(d["predicted_final_score"]))

        fair_rows.append(
            {
                "season": season,
                "judge_spearman_final": judge_s,
                "low_judge_winner": low_judge_winner,
                "fan_influence_corr_final": fan_corr,
            }
        )

    fair = pd.DataFrame(fair_rows)
    out = cons.merge(fair, on="season", how="left")

    # overall summary row (season = "ALL")
    overall = {
        "season": "ALL",
        "weeks": int(cons["weeks"].sum()),
        "exact_match_rate": float(cons["exact_match_rate"].mean()),
        "avg_jaccard": float(cons["avg_jaccard"].mean()),
        "avg_missing_votes": float(cons["avg_missing_votes"].mean()),
        "judge_spearman_final": float(np.nanmean(fair["judge_spearman_final"])) if len(fair) else np.nan,
        "low_judge_winner": float(np.mean(fair["low_judge_winner"])) if len(fair) else np.nan,
        "fan_influence_corr_final": float(np.nanmean(fair["fan_influence_corr_final"])) if len(fair) else np.nan,
    }
    out = pd.concat([out, pd.DataFrame([overall])], ignore_index=True)
    return out


# ---------------------------
# Entry
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha0", type=float, default=0.5, help="Base fan weight (0~1). Default 0.5.")
    parser.add_argument("--k", type=float, default=1.0, help="Certainty damping factor in g(c)=c/(c+k). Default 1.0.")
    parser.add_argument("--n_samples", type=int, default=200, help="Posterior samples for stability analysis. Default 200.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stability analysis. Default 42.")
    args = parser.parse_args()

    root = os.getcwd()
    data_path = os.path.join(root, "data", "clean_long.csv")
    votes_path = os.path.join(root, "output", "bayes_vi_vote_estimates.csv")
    out_dir = os.path.join(root, "output")
    safe_mkdir(out_dir)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing file: {data_path}")
    if not os.path.exists(votes_path):
        raise FileNotFoundError(f"Missing file: {votes_path}")

    clean_long = pd.read_csv(data_path)
    clean_long = normalize_colnames(clean_long)

    # Rename key columns if needed (after normalization)
    # Expecting: celebrity_name, season, week, judge_total, active, elim_after_week, placement
    required_cols = ["celebrity_name", "season", "week", "judge_total", "active", "elim_after_week", "placement"]
    miss = [c for c in required_cols if c not in clean_long.columns]
    if miss:
        raise ValueError(f"clean_long.csv missing columns after normalization: {miss}\nColumns seen: {clean_long.columns.tolist()}")

    votes = pd.read_csv(votes_path)
    votes = normalize_colnames(votes)
    vote_req = ["season", "week", "celebrity_name", "bayes_vote_share", "bayes_vote_share_sd"]
    miss2 = [c for c in vote_req if c not in votes.columns]
    if miss2:
        raise ValueError(f"bayes_vi_vote_estimates.csv missing columns after normalization: {miss2}\nColumns seen: {votes.columns.tolist()}")

    # Ensure types
    clean_long["season"] = pd.to_numeric(clean_long["season"], errors="coerce").astype("Int64")
    clean_long["week"] = pd.to_numeric(clean_long["week"], errors="coerce").astype("Int64")
    votes["season"] = pd.to_numeric(votes["season"], errors="coerce").astype("Int64")
    votes["week"] = pd.to_numeric(votes["week"], errors="coerce").astype("Int64")

    # Drop rows missing keys
    clean_long = clean_long.dropna(subset=["season", "week", "celebrity_name"]).copy()
    votes = votes.dropna(subset=["season", "week", "celebrity_name"]).copy()

    # Run simulation
    weekly_elims, final_ranking = simulate_newrule(
        clean_long=clean_long,
        votes=votes,
        alpha0=float(args.alpha0),
        k=float(args.k),
    )

    # Metrics
    metrics = build_metrics(weekly_elims=weekly_elims, final_ranking=final_ranking, clean_long=clean_long)

    # Stability
    stability = stability_analysis(
        clean_long=clean_long,
        votes=votes,
        alpha0=float(args.alpha0),
        k=float(args.k),
        n_samples=int(args.n_samples),
        seed=int(args.seed),
    )

    # Save outputs
    weekly_path = os.path.join(out_dir, "task4_weekly_elims_newrule.csv")
    final_path = os.path.join(out_dir, "task4_final_ranking_newrule.csv")
    metrics_path = os.path.join(out_dir, "task4_metrics_newrule.csv")
    stab_path = os.path.join(out_dir, "task4_stability_newrule.csv")

    weekly_elims.to_csv(weekly_path, index=False)
    final_ranking.to_csv(final_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    stability.to_csv(stab_path, index=False)

    # Console summary (short)
    print("=== Task4 New Rule Finished ===")
    print(f"Saved: {weekly_path}")
    print(f"Saved: {final_path}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {stab_path}")

    overall = metrics.loc[metrics["season"] == "ALL"]
    if not overall.empty:
        r = overall.iloc[0].to_dict()
        print("\n--- Overall Summary (ALL seasons) ---")
        print(f"Exact match rate (weekly elim): {r.get('exact_match_rate', np.nan):.3f}")
        print(f"Avg Jaccard (weekly elim):      {r.get('avg_jaccard', np.nan):.3f}")
        print(f"Judge Spearman (final week):    {r.get('judge_spearman_final', np.nan):.3f}")
        print(f"Low-judge winner rate:          {r.get('low_judge_winner', np.nan):.3f}")
        print(f"Fan influence corr (final):     {r.get('fan_influence_corr_final', np.nan):.3f}")


if __name__ == "__main__":
    main()
