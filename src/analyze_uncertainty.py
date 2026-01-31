import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    x = x[m]; y = y[m]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", type=str, default="data/clean_long.csv",
                    help="clean_long.csv with season/week/celebrity_name/judge_total/active/n_active etc.")
    ap.add_argument("--votes_csv", type=str, default="output/bayes_vi_vote_estimates.csv",
                    help="bayes_vi_vote_estimates.csv from infer_fan_votes.py")
    ap.add_argument("--out_dir", type=str, default="output")
    ap.add_argument("--kind", type=str, default="bayes",
                    help="prefix in votes file: bayes_vote_share, bayes_vote_share_sd")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = pd.read_csv(args.features_csv)
    votes = pd.read_csv(args.votes_csv)

    # Normalize columns we need
    for c in ["season", "week"]:
        if c in feats.columns:
            feats[c] = pd.to_numeric(feats[c], errors="coerce").astype(int)
        if c in votes.columns:
            votes[c] = pd.to_numeric(votes[c], errors="coerce").astype(int)

    # votes columns
    mean_col = f"{args.kind}_vote_share"
    sd_col = f"{args.kind}_vote_share_sd"
    need_votes = {"season", "week", "celebrity_name", mean_col, sd_col}
    missing = [c for c in need_votes if c not in votes.columns]
    if missing:
        raise ValueError(f"votes_csv missing columns: {missing}. Got columns: {list(votes.columns)}")

    # Merge
    df = feats.merge(
        votes[["season", "week", "celebrity_name", mean_col, sd_col]],
        on=["season", "week", "celebrity_name"],
        how="left",
    )

    # Filter to active rows (where the show has contestants)
    if "active" in df.columns:
        # active may be bool or string
        if df["active"].dtype != bool:
            df["active"] = df["active"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
        df = df[df["active"] == True].copy()

    # Basic sanity
    df["vote_mean"] = pd.to_numeric(df[mean_col], errors="coerce")
    df["vote_sd"] = pd.to_numeric(df[sd_col], errors="coerce")

    # Derived uncertainty measures (choose one or more)
    # Absolute sd is most direct.
    df["unc_abs"] = df["vote_sd"]

    # Relative sd (guard small means)
    df["unc_rel"] = df["vote_sd"] / (df["vote_mean"].clip(lower=1e-6))

    # Logit sd proxy (rough): sd / (p(1-p)) – indicates how sensitive odds are
    p = df["vote_mean"].clip(1e-6, 1 - 1e-6)
    df["unc_logit_proxy"] = df["vote_sd"] / (p * (1 - p))

    # ----------------------------
    # Aggregations
    # ----------------------------
    by_week = (
        df.groupby(["season", "week"])
          .agg(
              n_active=("celebrity_name", "count"),
              mean_vote_sd=("unc_abs", "mean"),
              median_vote_sd=("unc_abs", "median"),
              mean_vote_mean=("vote_mean", "mean"),
          )
          .reset_index()
          .sort_values(["season", "week"])
    )

    by_season = (
        df.groupby(["season"])
          .agg(
              n_rows=("celebrity_name", "count"),
              avg_vote_sd=("unc_abs", "mean"),
              med_vote_sd=("unc_abs", "median"),
              avg_n_active=("n_active", "mean") if "n_active" in df.columns else ("vote_mean", "size"),
          )
          .reset_index()
          .sort_values("season")
    )

    # Top uncertain rows
    top_unc = (
        df[["season", "week", "celebrity_name", "judge_total", "n_active", "vote_mean", "vote_sd", "unc_logit_proxy"]]
        if all(c in df.columns for c in ["judge_total", "n_active"])
        else df[["season", "week", "celebrity_name", "vote_mean", "vote_sd", "unc_logit_proxy"]]
    )
    top_unc = top_unc.sort_values("vote_sd", ascending=False).head(50)

    # ----------------------------
    # Correlations within week (optional but useful)
    # ----------------------------
    corr_rows = []
    for (s, w), g in df.groupby(["season", "week"]):
        row = {"season": s, "week": w, "n": len(g)}
        if "judge_total" in g.columns:
            row["corr(sd, judge_total)"] = safe_corr(g["vote_sd"], g["judge_total"])
            row["corr(mean, judge_total)"] = safe_corr(g["vote_mean"], g["judge_total"])
        row["corr(sd, mean)"] = safe_corr(g["vote_sd"], g["vote_mean"])
        corr_rows.append(row)
    corr_df = pd.DataFrame(corr_rows).sort_values(["season", "week"])

    # Summary stats to print
    print("======== UNCERTAINTY SUMMARY ========")
    print("Active rows used:", len(df))
    print("Mean vote_sd:", float(df["vote_sd"].mean()))
    print("Median vote_sd:", float(df["vote_sd"].median()))
    print("90% quantile vote_sd:", float(df["vote_sd"].quantile(0.9)))
    print()

    if "judge_total" in df.columns:
        print("Overall corr(vote_sd, judge_total):", safe_corr(df["vote_sd"], df["judge_total"]))
    if "n_active" in df.columns:
        print("Overall corr(vote_sd, n_active):", safe_corr(df["vote_sd"], df["n_active"]))
    print()

    # Save CSVs
    by_week.to_csv(out_dir / "uncertainty_by_week.csv", index=False)
    by_season.to_csv(out_dir / "uncertainty_by_season.csv", index=False)
    top_unc.to_csv(out_dir / "uncertainty_top_rows.csv", index=False)
    corr_df.to_csv(out_dir / "uncertainty_within_week_correlations.csv", index=False)

    print("Saved:")
    print(" ", out_dir / "uncertainty_by_week.csv")
    print(" ", out_dir / "uncertainty_by_season.csv")
    print(" ", out_dir / "uncertainty_top_rows.csv")
    print(" ", out_dir / "uncertainty_within_week_correlations.csv")

if __name__ == "__main__":
    main()
