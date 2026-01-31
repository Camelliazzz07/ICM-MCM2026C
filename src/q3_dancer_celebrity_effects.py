import argparse
import os
import re
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

def _strip_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def _strip_strings(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def collapse_rare(series: pd.Series, min_count: int, other_label: str = "Other") -> pd.Series:
    vc = series.value_counts(dropna=False)
    keep = set(vc[vc >= min_count].index.tolist())
    return series.where(series.isin(keep), other=other_label)


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def prepare_dataframe(clean_long_path: str,
                      votes_path: str,
                      min_count_dancer: int = 20,
                      min_count_industry: int = 25,
                      min_count_region: int = 25) -> pd.DataFrame:
    df = pd.read_csv(clean_long_path)
    df = _strip_colnames(df)

    # Expected columns in clean_long (your preprocess already produced these)
    # celebrity_name, ballroom_partner, celebrity_industry, celebrity_homecountry/region,
    # celebrity_age_during_season, season, week, judge_total, active, n_active
    df = _strip_strings(df, [
        "celebrity_name", "ballroom_partner", "celebrity_industry",
        "celebrity_homecountry/region", "celebrity_homestate", "results"
    ])

    # types
    for c in ["season", "week"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # active sometimes is string
    if "active" in df.columns and df["active"].dtype == object:
        df["active"] = df["active"].astype(str).str.strip().str.lower().map({"true": True, "false": False})

    # read vote estimates
    v = pd.read_csv(votes_path)
    v = _strip_colnames(v)
    v = _strip_strings(v, ["celebrity_name"])
    for c in ["season", "week"]:
        if c in v.columns:
            v[c] = pd.to_numeric(v[c], errors="coerce").astype("Int64")

    # Accept a few common naming variants
    rename_map = {}
    if "bayes_vote_share" not in v.columns:
        # try common alternatives
        for cand in ["vote_share", "vote_mean", "fan_vote_share", "bayes_share"]:
            if cand in v.columns:
                rename_map[cand] = "bayes_vote_share"
                break
    if "bayes_vote_share_sd" not in v.columns:
        for cand in ["vote_share_sd", "vote_sd", "fan_vote_share_sd", "bayes_share_sd"]:
            if cand in v.columns:
                rename_map[cand] = "bayes_vote_share_sd"
                break
    if rename_map:
        v = v.rename(columns=rename_map)

    required_vote_cols = {"season", "week", "celebrity_name", "bayes_vote_share"}
    missing = required_vote_cols - set(v.columns)
    if missing:
        raise ValueError(f"votes file missing columns: {sorted(missing)}")

    # merge
    key = ["season", "week", "celebrity_name"]
    merged = df.merge(v[key + [c for c in ["bayes_vote_share", "bayes_vote_share_sd"] if c in v.columns]],
                      on=key, how="left")

    # keep rows where vote share exists (these are your active rows in practice)
    merged = merged[merged["bayes_vote_share"].notna()].copy()

    # numeric conversions
    merged["judge_total"] = pd.to_numeric(merged.get("judge_total"), errors="coerce")
    merged["n_active"] = pd.to_numeric(merged.get("n_active"), errors="coerce")
    merged["age"] = pd.to_numeric(merged.get("celebrity_age_during_season"), errors="coerce")

    merged = merged[merged["judge_total"].notna() & merged["age"].notna()].copy()

    # Derived outcomes
    merged["judge_sum_week"] = merged.groupby(["season", "week"])["judge_total"].transform("sum")
    merged["judge_percent"] = merged["judge_total"] / merged["judge_sum_week"]
    merged["y_vote"] = logit(merged["bayes_vote_share"].to_numpy())

    # feature engineering
    merged["celebrity_id"] = merged["season"].astype(int).astype(str) + "_" + merged["celebrity_name"].astype(str)

    # standardize age
    merged["z_age"] = (merged["age"] - merged["age"].mean()) / (merged["age"].std(ddof=0) + 1e-12)

    # collapse rare categories to improve stability / interpretability
    merged["dancer"] = collapse_rare(merged["ballroom_partner"], min_count=min_count_dancer)
    merged["industry"] = collapse_rare(merged["celebrity_industry"], min_count=min_count_industry)
    merged["home_region"] = collapse_rare(merged["celebrity_homecountry/region"], min_count=min_count_region)

    # drop degenerate rows
    merged = merged[np.isfinite(merged["judge_percent"]) & np.isfinite(merged["y_vote"])].copy()

    return merged


def fit_clustered_ols(formula: str, df: pd.DataFrame, y_name: str, cluster_col: str):
    # Cluster-robust covariance by celebrity_id (repeated weekly observations per celebrity)
    res = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col]})
    params = res.params.rename("coef").to_frame()
    params["se"] = res.bse
    params["p"] = res.pvalues
    params["y"] = y_name
    # Wald tests per term (ANOVA-like)
    term_tests = res.wald_test_terms().summary_frame().reset_index().rename(columns={"index": "term"})
    term_tests["y"] = y_name
    return res, params.reset_index().rename(columns={"index": "param"}), term_tests


def fit_clustered_wls(formula: str, df: pd.DataFrame, y_name: str, cluster_col: str, weight_col: str):
    w = pd.to_numeric(df[weight_col], errors="coerce").to_numpy()
    w = np.where(np.isfinite(w) & (w > 0), w, np.nan)
    # robust: fill missing weights with median
    med = np.nanmedian(w)
    w = np.where(np.isfinite(w), w, med)
    res = smf.wls(formula, data=df, weights=w).fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col]})
    params = res.params.rename("coef").to_frame()
    params["se"] = res.bse
    params["p"] = res.pvalues
    params["y"] = y_name
    term_tests = res.wald_test_terms().summary_frame().reset_index().rename(columns={"index": "term"})
    term_tests["y"] = y_name
    return res, params.reset_index().rename(columns={"index": "param"}), term_tests


def extract_category_effects(res, df: pd.DataFrame, cat_col: str, y_col: str, baseline: str = None):
    """
    Convert dummy-coded coefficients into an "effect per category" table,
    measured as difference vs baseline category (statsmodels baseline).
    """
    levels = sorted(df[cat_col].dropna().unique().tolist())
    # Identify baseline used by patsy/statsmodels: usually the first in sorted order.
    if baseline is None:
        baseline = levels[0] if levels else None

    # Build effects
    effects = []
    for lv in levels:
        if lv == baseline:
            eff = 0.0
            se = np.nan
            p = np.nan
        else:
            key = f"C({cat_col})[T.{lv}]"
            if key in res.params.index:
                eff = float(res.params[key])
                se = float(res.bse[key])
                p = float(res.pvalues[key])
            else:
                # dropped due to collinearity / rare categories
                eff, se, p = np.nan, np.nan, np.nan
        effects.append({"category": lv, "baseline": baseline, "effect_vs_baseline": eff, "se": se, "p": p})
    out = pd.DataFrame(effects)
    out["n_obs"] = df.groupby(cat_col)[y_col].size().reindex(levels).values
    return out.sort_values("effect_vs_baseline", ascending=False)


def mc_propagate_vote(df: pd.DataFrame, formula: str, cluster_col: str,
                      draws: int = 50, seed: int = 0):
    """
    Monte Carlo propagation for vote share uncertainty:
    - draw vote_share ~ Normal(mean, sd)
    - recompute y_vote
    - refit WLS
    Returns: dataframe of coefficients across draws.
    """
    rng = np.random.default_rng(seed)
    if "bayes_vote_share_sd" not in df.columns:
        raise ValueError("Need bayes_vote_share_sd for Monte Carlo propagation.")
    base = df.copy()
    coefs = []
    for k in range(draws):
        share = rng.normal(loc=base["bayes_vote_share"].to_numpy(),
                           scale=base["bayes_vote_share_sd"].to_numpy())
        share = np.clip(share, 1e-6, 1 - 1e-6)
        base["y_vote_draw"] = logit(share)
        # weights based on sd (approx)
        w = 1.0 / np.maximum(base["bayes_vote_share_sd"].to_numpy(), 1e-6) ** 2
        res = smf.wls(formula.replace("y_vote", "y_vote_draw"), data=base, weights=w)\
                .fit(cov_type="cluster", cov_kwds={"groups": base[cluster_col]})
        s = res.params.rename(k).to_frame().T
        coefs.append(s)
    out = pd.concat(coefs, axis=0).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_long", type=str, default="data/clean_long.csv")
    ap.add_argument("--votes", type=str, default="output/bayes_vi_vote_estimates.csv")
    ap.add_argument("--out_dir", type=str, default="output/q3")
    ap.add_argument("--min_count_dancer", type=int, default=20)
    ap.add_argument("--min_count_industry", type=int, default=25)
    ap.add_argument("--min_count_region", type=int, default=25)
    ap.add_argument("--mc_draws", type=int, default=0, help="Monte Carlo draws for vote uncertainty (0=off).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = prepare_dataframe(
        clean_long_path=args.clean_long,
        votes_path=args.votes,
        min_count_dancer=args.min_count_dancer,
        min_count_industry=args.min_count_industry,
        min_count_region=args.min_count_region
    )

    # Main formulas (keep them explainable in report)
    # - judge_percent: relative judges score within week
    # - y_vote: logit vote share
    # Cluster by celebrity_id to handle repeated measures across weeks.
    base_terms = "z_age + C(industry) + C(home_region) + C(dancer) + week + n_active"
    judge_formula = f"judge_percent ~ {base_terms}"
    vote_formula  = f"y_vote ~ {base_terms}"

    # Judge: clustered OLS
    judge_res, judge_params, judge_terms = fit_clustered_ols(
        judge_formula, df, y_name="judge_percent", cluster_col="celebrity_id"
    )

    # Vote: clustered WLS (if sd available), else clustered OLS
    if "bayes_vote_share_sd" in df.columns and df["bayes_vote_share_sd"].notna().any():
        df["vote_w"] = 1.0 / np.maximum(df["bayes_vote_share_sd"].to_numpy(), 1e-6) ** 2
        vote_res, vote_params, vote_terms = fit_clustered_wls(
            vote_formula, df, y_name="y_vote", cluster_col="celebrity_id", weight_col="vote_w"
        )
    else:
        vote_res, vote_params, vote_terms = fit_clustered_ols(
            vote_formula, df, y_name="y_vote", cluster_col="celebrity_id"
        )

    judge_params.to_csv(os.path.join(args.out_dir, "judge_model_params.csv"), index=False)
    vote_params.to_csv(os.path.join(args.out_dir, "vote_model_params.csv"), index=False)
    judge_terms.to_csv(os.path.join(args.out_dir, "judge_term_tests.csv"), index=False)
    vote_terms.to_csv(os.path.join(args.out_dir, "vote_term_tests.csv"), index=False)

    # Category effects tables
    dancer_j = extract_category_effects(judge_res, df, "dancer", "judge_percent")
    dancer_v = extract_category_effects(vote_res,  df, "dancer", "y_vote")
    dancer = dancer_j.rename(columns={"effect_vs_baseline": "effect_judge",
                                      "se": "se_judge", "p": "p_judge"})\
                    .merge(dancer_v.rename(columns={"effect_vs_baseline": "effect_vote",
                                                    "se": "se_vote", "p": "p_vote"})[
                              ["category", "effect_vote", "se_vote", "p_vote"]],
                           on="category", how="left")
    dancer = dancer.rename(columns={"category": "dancer"})
    dancer["effect_corr_judge_vs_vote"] = dancer[["effect_judge", "effect_vote"]].corr().iloc[0, 1]

    dancer.to_csv(os.path.join(args.out_dir, "dancer_effects.csv"), index=False)

    industry_j = extract_category_effects(judge_res, df, "industry", "judge_percent")
    industry_v = extract_category_effects(vote_res,  df, "industry", "y_vote")
    industry = industry_j.rename(columns={"category": "industry",
                                          "effect_vs_baseline": "effect_judge",
                                          "se": "se_judge", "p": "p_judge"})\
                         .merge(industry_v.rename(columns={"category": "industry",
                                                           "effect_vs_baseline": "effect_vote",
                                                           "se": "se_vote", "p": "p_vote"})[
                                   ["industry", "effect_vote", "se_vote", "p_vote"]],
                                on="industry", how="left")
    industry.to_csv(os.path.join(args.out_dir, "industry_effects.csv"), index=False)

    region_j = extract_category_effects(judge_res, df, "home_region", "judge_percent")
    region_v = extract_category_effects(vote_res,  df, "home_region", "y_vote")
    region = region_j.rename(columns={"category": "home_region",
                                      "effect_vs_baseline": "effect_judge",
                                      "se": "se_judge", "p": "p_judge"})\
                     .merge(region_v.rename(columns={"category": "home_region",
                                                     "effect_vs_baseline": "effect_vote",
                                                     "se": "se_vote", "p": "p_vote"})[
                               ["home_region", "effect_vote", "se_vote", "p_vote"]],
                            on="home_region", how="left")
    region.to_csv(os.path.join(args.out_dir, "home_region_effects.csv"), index=False)

    # Optional: Monte Carlo propagation for vote uncertainty
    if args.mc_draws and args.mc_draws > 0:
        coefs = mc_propagate_vote(df, vote_formula, cluster_col="celebrity_id",
                                  draws=args.mc_draws, seed=0)
        coefs.to_csv(os.path.join(args.out_dir, "vote_mc_coef_draws.csv"), index=False)

        # summarize
        summary = coefs.describe(percentiles=[0.05, 0.5, 0.95]).T.reset_index().rename(columns={"index": "param"})
        summary.to_csv(os.path.join(args.out_dir, "vote_mc_coef_summary.csv"), index=False)

    # minimal console output for sanity
    print("Saved to:", args.out_dir)
    print("N rows used:", len(df))
    print("Key term tests (judge):")
    print(judge_terms.sort_values("P>chi2")[["term", "chi2", "df constraint", "P>chi2"]].head(8).to_string(index=False))
    print("Key term tests (vote):")
    print(vote_terms.sort_values("P>chi2")[["term", "chi2", "df constraint", "P>chi2"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
