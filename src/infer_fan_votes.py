# Run from project root
# 跑出来群里那个输出需要手动输入参数：
# python src/infer_fan_votes.py --seasons all --sigma_u 1.2 --epochs_bayes 2500 --post_samples 400
# 输入文件：../data/ProbC_Data.csv （即文件结构和本次commit一致
# 输出结果会生成在新output目录下：nonbayes_vote_estimates.csv, bayes_vi_vote_estimates.csv, model_summary.csv

import argparse
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import os
from pathlib import Path

# Project paths (robust no matter where you run)
SRC_DIR = Path(__file__).resolve().parent          # .../src
PROJ_DIR = SRC_DIR.parent                          # .../ICM-MCM2026C
DATA_DIR = PROJ_DIR / "data"
OUT_DIR_DEFAULT = PROJ_DIR / "output"

# -----------------------------
# Data utilities
# -----------------------------

def load_data(csv_name: str = "../data/ProbC_Data.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_name)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df


def find_score_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if re.match(r"week\d+_judge\d+_score", c)]


def build_judge_totals(df_season: pd.DataFrame, score_cols: List[str], max_week: int = 11) -> pd.DataFrame:
    week_totals = {}
    for w in range(1, max_week + 1):
        cols = [c for c in score_cols if c.startswith(f"week{w}_")]
        if not cols:
            continue

        # Avoid FutureWarning: convert to numeric directly
        vals = df_season[cols].apply(pd.to_numeric, errors="coerce")
        week_totals[w] = vals.sum(axis=1, min_count=1)  # NaN if all NaN

    return pd.DataFrame(week_totals, index=df_season.index)


def parse_elimination_week(df_season: pd.DataFrame) -> pd.Series:
    res = df_season["results"].astype(str)
    ew = res.str.extract(r"Eliminated Week (\d+)")[0].astype(float)
    return ew


def season_tensor(df_season: pd.DataFrame, score_cols: List[str], max_week: int = 11):
    jt = build_judge_totals(df_season, score_cols, max_week=max_week)

    # dataset sometimes uses 0 after elimination; treat as missing
    jt = jt.replace(0.0, np.nan)

    elim_week = parse_elimination_week(df_season)

    # mask out weeks after elimination
    for ridx, ew in elim_week.items():
        if not np.isnan(ew):
            ew_int = int(ew)
            for w in jt.columns:
                if w > ew_int:
                    jt.loc[ridx, w] = np.nan

    names = df_season["celebrity_name"].tolist()
    season_id = int(df_season["season"].iloc[0])
    return season_id, jt, elim_week, names


def build_elim_sets(elim_week: pd.Series, T: int) -> Dict[int, List[int]]:
    elim_sets = {t: [] for t in range(1, T + 1)}
    for i, ew in enumerate(elim_week.fillna(-1).astype(int).values):
        if 1 <= ew <= T:
            elim_sets[ew].append(i)
    return elim_sets


# -----------------------------
# Helpers
# -----------------------------

def safe_mean(x: torch.Tensor) -> torch.Tensor:
    """Mean that won't become NaN when x has 0 elements."""
    if x.numel() == 0:
        return torch.tensor(0.0, device=x.device)
    return x.mean()


# -----------------------------
# Model: Non-Bayesian optimization
# -----------------------------

@dataclass
class NonBayesConfig:
    weight_judge: float = 0.5
    smooth_lambda: float = 5.0
    lr: float = 0.05
    epochs: int = 1200
    seed: int = 0


def estimate_votes_nonbayes(
    J: torch.Tensor,
    active: torch.Tensor,
    elim_sets: Dict[int, List[int]],
    cfg: NonBayesConfig,
) -> Tuple[np.ndarray, Dict[int, int], float]:
    torch.manual_seed(cfg.seed)
    n, T = J.shape

    U = torch.zeros((n, T), requires_grad=True)
    opt = torch.optim.Adam([U], lr=cfg.lr)

    for _ in range(cfg.epochs):
        opt.zero_grad()
        loss = torch.tensor(0.0)

        for t in range(1, T + 1):
            mask = active[:, t - 1]
            idx = mask.nonzero(as_tuple=True)[0]
            if idx.numel() < 3:
                continue

            Jt = J[idx, t - 1]
            denom = Jt.sum() + 1e-8
            jp = Jt / denom

            ut = U[idx, t - 1]
            vp = torch.softmax(ut, dim=0)

            C = cfg.weight_judge * jp + (1 - cfg.weight_judge) * vp

            Es = [e for e in elim_sets.get(t, []) if e in idx.tolist()]
            if len(Es) == 0:
                continue

            Epos = torch.tensor([idx.tolist().index(e) for e in Es], dtype=torch.long)
            all_pos = torch.arange(idx.numel())
            keep = torch.ones(idx.numel(), dtype=torch.bool)
            keep[Epos] = False
            Kpos = all_pos[keep]

            Ce = C[Epos].unsqueeze(1)
            Ck = C[Kpos].unsqueeze(0)
            loss = loss + F.softplus(Ce - Ck).mean()

        # temporal smoothness regularization (robust)
        diff = U[:, 1:] - U[:, :-1]
        diff = diff.reshape(-1)
        loss = loss + cfg.smooth_lambda * safe_mean(diff * diff)

        loss.backward()
        opt.step()

    # outputs
    p_est = np.full((n, T), np.nan, dtype=float)
    pred_elim: Dict[int, int] = {}

    with torch.no_grad():
        for t in range(1, T + 1):
            mask = active[:, t - 1]
            idx = mask.nonzero(as_tuple=True)[0]
            if idx.numel() < 2:
                continue

            vp = torch.softmax(U[idx, t - 1], dim=0).cpu().numpy()
            p_est[idx.cpu().numpy(), t - 1] = vp

            Jt = J[idx, t - 1]
            jp = (Jt / (Jt.sum() + 1e-8)).cpu().numpy()
            C = cfg.weight_judge * jp + (1 - cfg.weight_judge) * vp
            pred_elim[t] = int(idx.cpu().numpy()[np.argmin(C)])

    correct, total = 0, 0
    for t, Es in elim_sets.items():
        if len(Es) == 0 or t not in pred_elim:
            continue
        total += 1
        if pred_elim[t] in Es:
            correct += 1

    acc = correct / total if total else float("nan")
    return p_est, pred_elim, acc


# -----------------------------
# Model: Bayesian VI
# -----------------------------

@dataclass
class BayesVIConfig:
    weight_judge: float = 0.5
    sigma0: float = 1.0
    sigma_u: float = 0.5
    lr: float = 0.03
    epochs: int = 1500
    mc_samples: int = 5
    posterior_samples: int = 200
    seed: int = 0


def estimate_votes_bayes_vi(
    J: torch.Tensor,
    active: torch.Tensor,
    elim_sets: Dict[int, List[int]],
    cfg: BayesVIConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], float]:
    torch.manual_seed(cfg.seed)
    n, T = J.shape

    mu = torch.zeros((n, T), requires_grad=True)
    rho = torch.full((n, T), -2.0, requires_grad=True)
    opt = torch.optim.Adam([mu, rho], lr=cfg.lr)

    def std_from_rho(r: torch.Tensor) -> torch.Tensor:
        return F.softplus(r) + 1e-4

    for _ in range(cfg.epochs):
        opt.zero_grad()
        std = std_from_rho(rho)
        neg_elbo = torch.tensor(0.0)

        for _k in range(cfg.mc_samples):
            U = mu + std * torch.randn_like(mu)

            loglik = torch.tensor(0.0)
            for t in range(1, T + 1):
                mask = active[:, t - 1]
                idx = mask.nonzero(as_tuple=True)[0]
                if idx.numel() < 3:
                    continue

                Jt = J[idx, t - 1]
                jp = Jt / (Jt.sum() + 1e-8)

                vp = torch.softmax(U[idx, t - 1], dim=0)
                C = cfg.weight_judge * jp + (1 - cfg.weight_judge) * vp

                Es = [e for e in elim_sets.get(t, []) if e in idx.tolist()]
                if len(Es) == 0:
                    continue

                Epos = torch.tensor([idx.tolist().index(e) for e in Es], dtype=torch.long)
                all_pos = torch.arange(idx.numel())
                keep = torch.ones(idx.numel(), dtype=torch.bool)
                keep[Epos] = False
                Kpos = all_pos[keep]

                Ce = C[Epos].unsqueeze(1)
                Ck = C[Kpos].unsqueeze(0)
                loglik = loglik + torch.log(torch.sigmoid(Ck - Ce) + 1e-8).mean()

            sigma0 = torch.tensor(cfg.sigma0)
            sigmau = torch.tensor(cfg.sigma_u)

            logprior = (-0.5 * (U[:, 0] / sigma0) ** 2 - torch.log(sigma0)).mean()
            d = U[:, 1:] - U[:, :-1]
            logprior = logprior + (-0.5 * (d / sigmau) ** 2 - torch.log(sigmau)).mean()

            logq = (-0.5 * (((U - mu) / std) ** 2 + 2 * torch.log(std) + math.log(2 * math.pi))).mean()

            neg_elbo = neg_elbo - (loglik + logprior - logq)

        neg_elbo = neg_elbo / cfg.mc_samples
        neg_elbo.backward()
        opt.step()

    # posterior sampling => mean/sd of vote share
    mu_d = mu.detach()
    std_d = std_from_rho(rho).detach()

    p_mean = np.full((n, T), np.nan, dtype=float)
    p_sd = np.full((n, T), np.nan, dtype=float)

    with torch.no_grad():
        for t in range(1, T + 1):
            mask = active[:, t - 1]
            idx = mask.nonzero(as_tuple=True)[0]
            if idx.numel() < 2:
                continue

            vps = []
            for _ in range(cfg.posterior_samples):
                U = mu_d + std_d * torch.randn_like(mu_d)
                vp = torch.softmax(U[idx, t - 1], dim=0).cpu().numpy()
                vps.append(vp)
            vps = np.stack(vps, axis=0)

            p_mean[idx.cpu().numpy(), t - 1] = vps.mean(axis=0)
            p_sd[idx.cpu().numpy(), t - 1] = vps.std(axis=0)

    pred_elim: Dict[int, int] = {}
    with torch.no_grad():
        for t in range(1, T + 1):
            mask = active[:, t - 1]
            idx = mask.nonzero(as_tuple=True)[0]
            if idx.numel() < 2:
                continue

            vp = p_mean[idx.cpu().numpy(), t - 1]
            if np.all(np.isnan(vp)):
                continue

            Jt = J[idx, t - 1].cpu().numpy()
            jp = Jt / (Jt.sum() + 1e-8)
            C = cfg.weight_judge * jp + (1 - cfg.weight_judge) * vp
            pred_elim[t] = int(idx.cpu().numpy()[np.argmin(C)])

    correct, total = 0, 0
    for t, Es in elim_sets.items():
        if len(Es) == 0 or t not in pred_elim:
            continue
        total += 1
        if pred_elim[t] in Es:
            correct += 1
    acc = correct / total if total else float("nan")

    return p_mean, p_sd, pred_elim, acc


# -----------------------------
# Save outputs
# -----------------------------

def wide_to_long_votes(names: List[str], season_id: int, p: np.ndarray, kind: str, p_sd: np.ndarray = None) -> pd.DataFrame:
    n, T = p.shape
    rows = []
    for i in range(n):
        for t in range(1, T + 1):
            if np.isnan(p[i, t - 1]):
                continue
            row = {
                "season": season_id,
                "week": t,
                "celebrity_name": names[i],
                f"{kind}_vote_share": float(p[i, t - 1]),
            }
            if p_sd is not None:
                row[f"{kind}_vote_share_sd"] = float(p_sd[i, t - 1]) if not np.isnan(p_sd[i, t - 1]) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def parse_seasons_arg(seasons_arg: str, all_seasons: List[int]) -> List[int]:
    s = seasons_arg.strip().lower()
    if s in ("all", "*"):
        return all_seasons
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
    return sorted(set(out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(DATA_DIR / "ProbC_Data.csv"))
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR_DEFAULT))

    parser.add_argument("--seasons", type=str, default="all")

    parser.add_argument("--epochs_nonbayes", type=int, default=1200)
    parser.add_argument("--lr_nonbayes", type=float, default=0.05)
    parser.add_argument("--smooth_lambda", type=float, default=5.0)

    parser.add_argument("--epochs_bayes", type=int, default=1500)
    parser.add_argument("--lr_bayes", type=float, default=0.03)
    parser.add_argument("--mc_samples", type=int, default=5)
    parser.add_argument("--post_samples", type=int, default=200)
    parser.add_argument("--sigma0", type=float, default=1.0)
    parser.add_argument("--sigma_u", type=float, default=0.5)

    parser.add_argument("--weight_judge", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = load_data(args.csv)
    score_cols = find_score_columns(df)

    all_seasons = sorted(df["season"].astype(int).unique().tolist())
    seasons = parse_seasons_arg(args.seasons, all_seasons)

    nonbayes_rows = []
    bayes_rows = []
    summary_rows = []

    for s in seasons:
        df_s = df[df["season"].astype(int) == int(s)].copy()
        if df_s.empty:
            continue

        season_id, jt, elim_week, names = season_tensor(df_s, score_cols, max_week=11)
        T = int(max(jt.columns)) if len(jt.columns) else 11

        J = torch.tensor(jt.values, dtype=torch.float32)
        active = ~torch.isnan(J)
        elim_sets = build_elim_sets(elim_week, T)

        nb_cfg = NonBayesConfig(
            weight_judge=args.weight_judge,
            smooth_lambda=args.smooth_lambda,
            lr=args.lr_nonbayes,
            epochs=args.epochs_nonbayes,
            seed=args.seed,
        )
        p_nb, pred_nb, acc_nb = estimate_votes_nonbayes(J, active, elim_sets, nb_cfg)
        nonbayes_rows.append(wide_to_long_votes(names, season_id, p_nb, kind="nonbayes"))

        bv_cfg = BayesVIConfig(
            weight_judge=args.weight_judge,
            sigma0=args.sigma0,
            sigma_u=args.sigma_u,
            lr=args.lr_bayes,
            epochs=args.epochs_bayes,
            mc_samples=args.mc_samples,
            posterior_samples=args.post_samples,
            seed=args.seed,
        )
        p_bm, p_bs, pred_b, acc_b = estimate_votes_bayes_vi(J, active, elim_sets, bv_cfg)
        bayes_rows.append(wide_to_long_votes(names, season_id, p_bm, kind="bayes", p_sd=p_bs))

        summary_rows.append({
            "season": season_id,
            "n_contestants": len(names),
            "nonbayes_elim_acc": acc_nb,
            "bayes_vi_elim_acc": acc_b,
        })

        print(f"[Season {season_id}] nonbayes_acc={acc_nb:.3f} | bayes_vi_acc={acc_b:.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if nonbayes_rows:
        pd.concat(nonbayes_rows, ignore_index=True).to_csv(out_dir / "nonbayes_vote_estimates.csv", index=False)
    if bayes_rows:
        pd.concat(bayes_rows, ignore_index=True).to_csv(out_dir / "bayes_vi_vote_estimates.csv", index=False)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / "model_summary.csv", index=False)

    print("Done. Wrote: nonbayes_vote_estimates.csv, bayes_vi_vote_estimates.csv, model_summary.csv")


if __name__ == "__main__":
    main()
