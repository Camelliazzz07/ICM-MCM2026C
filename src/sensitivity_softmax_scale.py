import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df


def stable_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    ex = np.exp(x)
    s = float(np.sum(ex))
    if (not np.isfinite(s)) or s <= 0:
        return np.ones_like(x) / max(len(x), 1)
    return ex / (s + 1e-12)


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


def top1_name(df_g: pd.DataFrame, score: np.ndarray) -> str:
    idx = int(np.argmax(score))
    return str(df_g.iloc[idx]["celebrity_name"])


def compute_week_metrics(df: pd.DataFrame, target_col: str, logits: np.ndarray, alpha: float) -> pd.DataFrame:
    rows = []
    # stable ordering over groups
    grouped = df.groupby(["season", "week"], sort=True)
    for gid, ((s, w), g) in enumerate(grouped):
        idx = g.index.to_numpy(dtype=np.int64)
        y = df.loc[idx, target_col].to_numpy(dtype=float)
        y = y / (y.sum() + 1e-12)

        lg = logits[idx] * float(alpha)
        yh = stable_softmax(lg)

        n = int(len(idx))
        p_sorted = np.sort(yh)[::-1]
        pmax = float(p_sorted[0]) if n else np.nan
        gap = float(p_sorted[0] - p_sorted[1]) if n >= 2 else np.nan

        H = entropy(yh)
        H_norm = H / np.log(max(n, 2))
        D_entropy = 1.0 - H_norm

        rows.append({
            "group_id": gid,
            "season": s,
            "week": w,
            "n_active": n,
            "p_max_hat": pmax,
            "gap_top1_top2_hat": gap,
            "entropy_hat": H,
            "H_norm_hat": H_norm,
            "D_entropy_hat": D_entropy,
            "kl_y_yhat": kl_div(y, yh),
            "top1_hat": top1_name(g, yh),
            "alpha": float(alpha),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, required=True)
    parser.add_argument("--coef_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="output/sensitivity")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.7, 1.0, 1.3])
    parser.add_argument("--target_col", type=str, default="bayes_vote_share")
    parser.add_argument("--write_scaled_predictions", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfp = load_csv(args.pred_csv)
    coef = load_csv(args.coef_csv)

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

    X = dfp[feature_cols].to_numpy(dtype=float)
    logits = (X @ betas.reshape(-1, 1)).reshape(-1)

    # baseline week metrics at alpha=1.0 for stability checks (by group_id)
    base_week = compute_week_metrics(dfp, args.target_col, logits, alpha=1.0)
    base_top1 = base_week.set_index("group_id")["top1_hat"]

    overall_rows = []
    all_weeks = []

    for a in args.alphas:
        weekm = compute_week_metrics(dfp, args.target_col, logits, alpha=float(a))
        all_weeks.append(weekm)

        # top-1 stability vs alpha=1.0 (by group_id)
        top1 = weekm.set_index("group_id")["top1_hat"]
        aligned = top1.align(base_top1, join="inner")
        top1_stable = float((aligned[0] == aligned[1]).mean()) if len(aligned[0]) else np.nan

        overall_rows.append({
            "alpha": float(a),
            "groups": int(len(weekm)),
            "avg_KL(y||yhat)": float(np.nanmean(weekm["kl_y_yhat"])),
            "median_KL(y||yhat)": float(np.nanmedian(weekm["kl_y_yhat"])),
            "avg_p_max": float(np.nanmean(weekm["p_max_hat"])),
            "avg_D_entropy": float(np.nanmean(weekm["D_entropy_hat"])),
            "top1_stability_vs_alpha1": top1_stable,
        })

        weekm.to_csv(out_dir / f"sensitivity_week_metrics_alpha_{a:.3f}.csv", index=False)

        if args.write_scaled_predictions:
            # write per-row scaled predictions (vote_share_hat) for this alpha
            vhat = np.full(len(dfp), np.nan, dtype=float)
            for (s, w), g in dfp.groupby(["season", "week"], sort=True):
                idx = g.index.to_numpy(dtype=np.int64)
                vhat[idx] = stable_softmax(logits[idx] * float(a))
            keep = ["season", "week", "celebrity_name", args.target_col]
            out = dfp[keep].copy()
            out[f"vote_share_hat_alpha_{a:.3f}"] = vhat
            out.to_csv(out_dir / f"predictions_scaled_alpha_{a:.3f}.csv", index=False)

    overall = pd.DataFrame(overall_rows).sort_values("alpha").reset_index(drop=True)
    overall.to_csv(out_dir / "sensitivity_overall_summary.csv", index=False)

    if all_weeks:
        pd.concat(all_weeks, ignore_index=True).to_csv(out_dir / "sensitivity_week_metrics_all.csv", index=False)

    print("Wrote:")
    print(" -", out_dir / "sensitivity_overall_summary.csv")
    print(" -", out_dir / "sensitivity_week_metrics_all.csv")


if __name__ == "__main__":
    main()
