import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # strip strings
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df


def stable_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    x = x - np.max(x)
    ex = np.exp(x)
    s = np.sum(ex)
    if not np.isfinite(s) or s <= 0:
        # fallback uniform
        return np.ones_like(x) / max(len(x), 1)
    return ex / (s + 1e-12)


def group_softmax_scaled(df: pd.DataFrame, logits: np.ndarray, alpha: float) -> np.ndarray:
    vhat = np.full(len(df), np.nan, dtype=float)
    for (s, w), idx in df.groupby(["season", "week"]).indices.items():
        idx = np.asarray(idx, dtype=np.int64)
        lg = logits[idx] * alpha
        vhat[idx] = stable_softmax(lg)
    return vhat


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p / (p.sum() + 1e-12)
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def top1_name(df_g: pd.DataFrame, score_col: str) -> str:
    # returns celebrity_name of max score
    idx = df_g[score_col].astype(float).values.argmax()
    return str(df_g.iloc[idx]["celebrity_name"])


def compute_week_metrics(df: pd.DataFrame, target_col: str, pred_col: str) -> pd.DataFrame:
    rows = []
    for (s, w), g in df.groupby(["season", "week"]):
        y = g[target_col].to_numpy(dtype=float)
        y = y / (y.sum() + 1e-12)

        yh = g[pred_col].to_numpy(dtype=float)
        yh = yh / (yh.sum() + 1e-12)

        n = int(len(g))
        p_sorted = np.sort(yh)[::-1]
        pmax = float(p_sorted[0]) if n else np.nan
        gap = float(p_sorted[0] - p_sorted[1]) if n >= 2 else np.nan

        H = entropy(yh)
        H_norm = H / np.log(max(n, 2))
        D_entropy = 1.0 - H_norm

        rows.append({
            "season": int(s),
            "week": int(w),
            "n_active": n,
            "p_max_hat": pmax,
            "gap_top1_top2_hat": gap,
            "entropy_hat": H,
            "H_norm_hat": H_norm,
            "D_entropy_hat": D_entropy,
            "kl_y_yhat": kl_div(y, yh),
            "top1_hat": top1_name(g, pred_col),
        })
    return pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, required=True,
                        help="vote_driver_softmax_predictions.csv (must include standardized feature cols and bayes_vote_share)")
    parser.add_argument("--coef_csv", type=str, required=True,
                        help="vote_driver_softmax_coefficients.csv (feature, beta)")
    parser.add_argument("--out_dir", type=str, default="output/sensitivity")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.7, 1.0, 1.3],
                        help="logit scale multipliers to test (alpha=1 is baseline)")
    parser.add_argument("--target_col", type=str, default="bayes_vote_share",
                        help="soft target column name (default bayes_vote_share)")
    parser.add_argument("--write_scaled_predictions", action="store_true",
                        help="If set, write scaled predictions CSV for each alpha")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfp = load_csv(args.pred_csv)
    coef = load_csv(args.coef_csv)

    # sanity checks
    need_cols = {"season", "week", "celebrity_name", args.target_col}
    missing = [c for c in need_cols if c not in dfp.columns]
    if missing:
        raise ValueError(f"pred_csv missing columns: {missing}")

    if "feature" not in coef.columns or "beta" not in coef.columns:
        raise ValueError("coef_csv must contain columns: feature, beta")

    feature_cols = coef["feature"].astype(str).tolist()
    betas = coef["beta"].astype(float).to_numpy()

    for c in feature_cols:
        if c not in dfp.columns:
            raise ValueError(f"pred_csv missing standardized feature column: {c}")

    # base logits (alpha=1)
    X = dfp[feature_cols].to_numpy(dtype=float)
    logits = X @ betas.reshape(-1, 1)
    logits = logits.reshape(-1)

    # baseline predicted col name in file might exist; but we will recompute to be consistent
    base_col = "vote_share_hat_alpha_1.0"
    dfp[base_col] = group_softmax_scaled(dfp, logits, alpha=1.0)

    overall_rows = []

    # metrics for baseline (for stability comparisons)
    week_base = compute_week_metrics(dfp, target_col=args.target_col, pred_col=base_col)
    base_top1 = week_base.set_index(["season", "week"])["top1_hat"]

    for a in args.alphas:
        col = f"vote_share_hat_alpha_{a:.3f}"
        dfp[col] = group_softmax_scaled(dfp, logits, alpha=float(a))

        weekm = compute_week_metrics(dfp, target_col=args.target_col, pred_col=col)
        weekm["alpha"] = float(a)

        # top-1 stability vs baseline
        top1 = weekm.set_index(["season", "week"])["top1_hat"]
        aligned = top1.align(base_top1, join="inner")
        top1_stable = float((aligned[0] == aligned[1]).mean()) if len(aligned[0]) else np.nan

        # overall summary
        overall_rows.append({
            "alpha": float(a),
            "weeks": int(len(weekm)),
            "avg_KL(y||yhat)": float(np.nanmean(weekm["kl_y_yhat"])),
            "median_KL(y||yhat)": float(np.nanmedian(weekm["kl_y_yhat"])),
            "avg_p_max": float(np.nanmean(weekm["p_max_hat"])),
            "avg_D_entropy": float(np.nanmean(weekm["D_entropy_hat"])),
            "strong_deterministic_rate_pmax_ge_0.8": float(np.mean(weekm["p_max_hat"] >= 0.8)),
            "highly_sensitive_rate_gap_lt_0.05": float(np.mean(weekm["gap_top1_top2_hat"] < 0.05)),
            "top1_stability_vs_alpha1": top1_stable,
        })

        # write per-week metrics for this alpha
        weekm.to_csv(out_dir / f"sensitivity_week_metrics_alpha_{a:.3f}.csv", index=False)

        if args.write_scaled_predictions:
            keep = ["season", "week", "celebrity_name", args.target_col, col]
            dfp[keep].to_csv(out_dir / f"predictions_scaled_alpha_{a:.3f}.csv", index=False)

    overall = pd.DataFrame(overall_rows).sort_values("alpha").reset_index(drop=True)
    overall.to_csv(out_dir / "sensitivity_overall_summary.csv", index=False)

    # also a combined week file (stacked)
    combined = []
    for a in args.alphas:
        f = out_dir / f"sensitivity_week_metrics_alpha_{a:.3f}.csv"
        if f.exists():
            combined.append(pd.read_csv(f))
    if combined:
        pd.concat(combined, ignore_index=True).to_csv(out_dir / "sensitivity_week_metrics_all.csv", index=False)

    print("Wrote:")
    print(" -", out_dir / "sensitivity_overall_summary.csv")
    print(" -", out_dir / "sensitivity_week_metrics_all.csv")
    if args.write_scaled_predictions:
        print(" - predictions_scaled_alpha_*.csv")


if __name__ == "__main__":
    main()
