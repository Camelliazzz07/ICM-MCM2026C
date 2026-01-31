# 1) Merge clean_long (features) with bayes vote estimates on (season, week, celebrity_name)
# 2) Build dynamic "prior" features (celebrity/state historical mean vote share)
# 3) Fit a softmax regression model per (season, week) group:
#       v_hat = softmax(X beta)
#    by minimizing cross-entropy with soft targets (bayes_vote_share)
# 4) Save coefficients + per-row predictions

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# -------------------------
# IO + basic checks
# -------------------------

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # strip strings
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df


def ensure_columns(df: pd.DataFrame, cols, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


# -------------------------
# Feature engineering
# -------------------------

def build_priors_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df should already be merged and filtered to active rows with targets present.
    Builds:
      - celebrity_prior_mean_vote
      - state_prior_mean_vote (if celebrity_homestate exists)
      - judge_rank
      - judge_change
    Also sanitizes types and fills missing values conservatively.
    """
    df = df.copy()

    # types
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype(int)
    df["judge_total"] = pd.to_numeric(df["judge_total"], errors="coerce")

    # sort for time features
    df = df.sort_values(["season", "celebrity_name", "week"]).reset_index(drop=True)

    # --- celebrity prior mean vote: expanding mean up to previous week
    df["celebrity_prior_mean_vote"] = (
        df.groupby(["season", "celebrity_name"])["bayes_vote_share"]
          .expanding()
          .mean()
          .shift(1)
          .reset_index(level=[0, 1], drop=True)
    )

    # --- state prior mean vote (optional)
    if "celebrity_homestate" in df.columns:
        df["state_prior_mean_vote"] = (
            df.groupby(["season", "celebrity_homestate"])["bayes_vote_share"]
              .expanding()
              .mean()
              .shift(1)
              .reset_index(level=[0, 1], drop=True)
        )
    else:
        df["state_prior_mean_vote"] = np.nan

    # --- judge rank within (season, week): 1 = best
    df["judge_rank"] = (
        df.groupby(["season", "week"])["judge_total"]
          .rank(ascending=False, method="average")
    )

    # --- judge change week-to-week for each celebrity
    df["judge_change"] = (
        df.groupby(["season", "celebrity_name"])["judge_total"]
          .diff()
    )

    # Fill missing prior features:
    # If no history exists, use season-week mean vote share (or global mean)
    global_mean = float(df["bayes_vote_share"].mean())
    sw_mean = df.groupby(["season", "week"])["bayes_vote_share"].transform("mean")

    df["celebrity_prior_mean_vote"] = df["celebrity_prior_mean_vote"].fillna(sw_mean).fillna(global_mean)
    df["state_prior_mean_vote"] = df["state_prior_mean_vote"].fillna(sw_mean).fillna(global_mean)

    # judge_change missing for first week per celeb -> 0
    df["judge_change"] = df["judge_change"].fillna(0.0)

    # judge_total missing -> fill with season-week mean
    jt_sw_mean = df.groupby(["season", "week"])["judge_total"].transform("mean")
    df["judge_total"] = df["judge_total"].fillna(jt_sw_mean)

    # judge_rank missing -> fill with season-week mean rank
    jr_sw_mean = df.groupby(["season", "week"])["judge_rank"].transform("mean")
    df["judge_rank"] = df["judge_rank"].fillna(jr_sw_mean)

    return df


def standardize(train_df: pd.DataFrame, feature_cols):
    """
    z-score standardization for numeric stability.
    Returns standardized df + (mean, std) dicts.
    """
    df = train_df.copy()
    stats = {}
    for c in feature_cols:
        x = pd.to_numeric(df[c], errors="coerce").astype(float)
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x))
        if sd < 1e-8:
            sd = 1.0
        df[c] = (x - mu) / sd
        stats[c] = (mu, sd)
    return df, stats


# -------------------------
# Softmax regression training
# -------------------------

def fit_softmax_regression(
    df: pd.DataFrame,
    feature_cols,
    target_col="bayes_vote_share",
    weight_by_sd=True,
    lr=0.05,
    epochs=1200,
    l2=1e-3,
    seed=0,
    device="cpu",
):
    """
    Fits beta for v_hat = softmax(X beta) within each (season, week) group.
    Loss = mean_{groups} [ -sum_i v_i * log vhat_i ] + l2 * ||beta||^2
    Optionally weights each row by 1/(sd^2) to de-emphasize uncertain targets.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    k = len(feature_cols)
    beta = torch.zeros(k, requires_grad=True, device=device)
    opt = torch.optim.Adam([beta], lr=lr)

    # pre-extract group indices for speed
    groups = list(df.groupby(["season", "week"]).indices.items())

    # tensors cache
    X_all = torch.tensor(df[feature_cols].to_numpy(dtype=np.float32), device=device)
    y_all = torch.tensor(df[target_col].to_numpy(dtype=np.float32), device=device)

    if weight_by_sd and "bayes_vote_share_sd" in df.columns:
        sd = df["bayes_vote_share_sd"].to_numpy(dtype=np.float32)
        # avoid exploding weights: cap sd from below
        sd = np.clip(sd, 1e-3, None)
        w = 1.0 / (sd * sd)
        # normalize weights to mean 1
        w = w / (np.mean(w) + 1e-12)
        w_all = torch.tensor(w, device=device)
    else:
        w_all = torch.ones(len(df), device=device)

    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = torch.tensor(0.0, device=device)

        # iterate each (season, week) group (variable number of contestants)
        n_groups_used = 0
        for (_key, idx) in groups:
            idx = np.asarray(idx, dtype=np.int64)
            if idx.size < 2:
                continue

            Xg = X_all[idx, :]                     # [m, k]
            yg = y_all[idx]                        # [m]
            wg = w_all[idx]                        # [m]

            # skip if targets are bad
            if torch.isnan(yg).any():
                continue

            # renormalize soft target to sum 1 within group
            ysum = yg.sum()
            if ysum <= 0:
                continue
            yg = yg / ysum

            logits = Xg @ beta                     # [m]
            vhat = torch.softmax(logits, dim=0)     # [m]

            # weighted cross-entropy with soft labels
            ce = -(wg * yg * torch.log(vhat + 1e-12)).sum()
            # normalize by sum weights to keep scales comparable
            ce = ce / (wg.sum() + 1e-12)

            loss = loss + ce
            n_groups_used += 1

        if n_groups_used > 0:
            loss = loss / n_groups_used

        # L2 regularization
        loss = loss + l2 * (beta * beta).sum()

        loss.backward()
        opt.step()

        # light logging
        if ep in (1, 50, 100, 200, 400, 800, epochs):
            print(f"epoch {ep:4d} | loss={loss.item():.6f} | groups={n_groups_used}")

    return beta.detach().cpu().numpy()


def predict_softmax(df: pd.DataFrame, feature_cols, beta_np: np.ndarray) -> np.ndarray:
    X = df[feature_cols].to_numpy(dtype=np.float32)
    logits = X @ beta_np.reshape(-1, 1)  # [N,1]
    logits = logits.reshape(-1)

    # softmax must be applied within each (season, week) group, not globally
    vhat = np.full(len(df), np.nan, dtype=float)
    for (s, w), idx in df.groupby(["season", "week"]).indices.items():
        idx = np.asarray(idx, dtype=np.int64)
        lg = logits[idx]
        # stable softmax
        lg = lg - np.max(lg)
        ex = np.exp(lg)
        vhat[idx] = ex / (np.sum(ex) + 1e-12)
    return vhat


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_csv", type=str, required=True,
                        help="clean_long-like CSV with columns including season, week, celebrity_name, judge_total, active, etc.")
    parser.add_argument("--votes_csv", type=str, required=True,
                        help="bayes_vi_vote_estimates.csv with columns including season, week, celebrity_name, bayes_vote_share (and optionally sd)")
    parser.add_argument("--out_dir", type=str, default="output")

    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_weight_by_sd", action="store_true",
                        help="If set, do not weight loss by bayes_vote_share_sd.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_feat = load_csv(args.features_csv)
    df_vote = load_csv(args.votes_csv)

    ensure_columns(df_feat, ["season", "week", "celebrity_name"], "features_csv")
    ensure_columns(df_vote, ["season", "week", "celebrity_name"], "votes_csv")

    # Find which target column exists (vote file might include bayes or nonbayes)
    if "bayes_vote_share" not in df_vote.columns:
        # try fallback
        candidates = [c for c in df_vote.columns if c.endswith("_vote_share")]
        if not candidates:
            raise ValueError("votes_csv must contain bayes_vote_share or a *_vote_share column.")
        # pick first
        df_vote = df_vote.rename(columns={candidates[0]: "bayes_vote_share"})

    # Merge
    df = df_feat.merge(df_vote, on=["season", "week", "celebrity_name"], how="left")

    # Keep active
    if "active" in df.columns:
        # handle string "True"/"False"
        if df["active"].dtype == object:
            df["active"] = df["active"].astype(str).str.lower().isin(["true", "1", "yes"])
        df = df[df["active"] == True].copy()

    # Need targets
    df["bayes_vote_share"] = pd.to_numeric(df["bayes_vote_share"], errors="coerce")
    df = df[~df["bayes_vote_share"].isna()].copy()

    # Feature engineering
    df = build_priors_and_features(df)

    # Define feature columns (simple + defensible baseline)
    feature_cols = [
        "judge_total",
        "judge_rank",
        "judge_change",
        "celebrity_prior_mean_vote",
        "state_prior_mean_vote",
        "week",
    ]

    # Ensure columns exist (state_prior_mean_vote always created)
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Missing feature column {c} after engineering.")

    # Standardize features
    df_std, stats = standardize(df, feature_cols)

    # Fit
    beta = fit_softmax_regression(
        df_std,
        feature_cols=feature_cols,
        target_col="bayes_vote_share",
        weight_by_sd=(not args.no_weight_by_sd),
        lr=args.lr,
        epochs=args.epochs,
        l2=args.l2,
        seed=args.seed,
    )

    # Predict
    df_out = df_std.copy()
    df_out["vote_share_hat"] = predict_softmax(df_out, feature_cols, beta)

    # Save coefficients
    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "beta": beta,
        "orig_mean": [stats[c][0] for c in feature_cols],
        "orig_std": [stats[c][1] for c in feature_cols],
    })
    coef_path = out_dir / "vote_driver_softmax_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)

    # Save predictions (keep key cols + useful fields)
    keep_cols = ["season", "week", "celebrity_name", "bayes_vote_share", "vote_share_hat"]
    if "bayes_vote_share_sd" in df_out.columns:
        keep_cols.append("bayes_vote_share_sd")
    for c in feature_cols:
        keep_cols.append(c)

    pred_path = out_dir / "vote_driver_softmax_predictions.csv"
    df_out[keep_cols].to_csv(pred_path, index=False)

    # Simple diagnostics: average KL per (season, week)
    eps = 1e-12
    kls = []
    for (s, w), g in df_out.groupby(["season", "week"]):
        y = g["bayes_vote_share"].to_numpy(float)
        y = y / (y.sum() + eps)
        yh = g["vote_share_hat"].to_numpy(float)
        yh = yh / (yh.sum() + eps)
        kl = float(np.sum(y * (np.log(y + eps) - np.log(yh + eps))))
        kls.append(kl)
    print(f"Saved:\n  {coef_path}\n  {pred_path}")
    print(f"Avg KL(y || yhat) over groups: {np.mean(kls):.6f} (lower is better)")

    # Optional: show a quick ranking agreement measure
    # Spearman within each group (rough) - skip if group < 3
    try:
        from scipy.stats import spearmanr
        rs = []
        for (s, w), g in df_out.groupby(["season", "week"]):
            if len(g) < 3:
                continue
            r, _ = spearmanr(g["bayes_vote_share"], g["vote_share_hat"])
            if np.isfinite(r):
                rs.append(r)
        if rs:
            print(f"Avg Spearman corr (within week): {np.mean(rs):.3f} (higher is better)")
    except Exception:
        pass


if __name__ == "__main__":
    main()
