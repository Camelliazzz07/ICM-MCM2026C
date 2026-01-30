# Run from project root
# Example:
#   python src/infer_fan_votes.py --seasons all --sigma_u 1.2 --epochs_bayes 2500 --post_samples 400
#
# Input CSV:
#   - If you pass a cleaned long file: data/clean_long.csv
#     Must contain: season, week, celebrity_name, judge_total, active, active_next or elim_after_week
#   - Otherwise falls back to the original raw file: data/ProbC_Data.csv
#
# Output (in --out_dir):
#   nonbayes_vote_estimates.csv, bayes_vi_vote_estimates.csv, model_summary.csv

import argparse
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

# Project paths (robust no matter where you run)
SRC_DIR = Path(__file__).resolve().parent          # .../src
PROJ_DIR = SRC_DIR.parent                          # .../ICM-MCM2026C
DATA_DIR = PROJ_DIR / "data"
OUT_DIR_DEFAULT = PROJ_DIR / "output"

# -----------------------------
# Voting-rule utilities
# -----------------------------

def voting_method_for_season(season_id: int) -> str:
    """Return which combination rule applies.

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


def soft_rank(x: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    """Differentiable approximation of rank (1 = best), higher x means better.

    r_i = 1 + sum_j sigmoid((x_j - x_i)/tau)
    If many others are larger than x_i, rank increases (worse).
    """
    x_col = x.unsqueeze(0)              # [1, m]
    x_row = x.unsqueeze(1)              # [m, 1]
    # compare all pairs: j - i
    pair = (x_col - x_row) / tau        # [m, m]
    # sigmoid close to 1 when x_j > x_i
    return 1.0 + torch.sigmoid(pair).sum(dim=1) - 0.5  # subtract ~0.5 to reduce self-comparison bias


def compute_badness(
    season_id: int,
    Jt: torch.Tensor,
    vp: torch.Tensor,
    weight_judge: float,
    rank_tau: float,
) -> torch.Tensor:
    """Compute a per-contestant 'badness' score for elimination modeling.

    Convention: larger badness => more likely to be eliminated (worse).
    """
    method = voting_method_for_season(season_id)

    if method == "percent":
        jp = Jt / (Jt.sum() + 1e-8)          # higher is better
        C = weight_judge * jp + (1 - weight_judge) * vp
        return -C                             # lower C => worse, so badness = -C

    # rank-based
    rj = soft_rank(Jt, tau=rank_tau)         # 1 best, larger worse
    rv = soft_rank(vp, tau=rank_tau)
    return rj + rv                            # larger worse


def elim_loglik_standard(b: torch.Tensor, epos: torch.Tensor, tau_elim: float) -> torch.Tensor:
    """Log-likelihood that eliminated have higher badness than all non-eliminated."""
    m = b.numel()
    all_pos = torch.arange(m, device=b.device)
    keep = torch.ones(m, dtype=torch.bool, device=b.device)
    keep[epos] = False
    kpos = all_pos[keep]

    be = b[epos].unsqueeze(1)          # [E,1]
    bk = b[kpos].unsqueeze(0)          # [1,K]
    # Want be > bk. Use logistic: P = sigmoid((be - bk)/tau)
    return torch.log(torch.sigmoid((be - bk) / tau_elim) + 1e-8).mean()


def elim_penalty_bottom2(b: torch.Tensor, epos: torch.Tensor, tau_elim: float) -> torch.Tensor:
    """Penalty encouraging eliminated to be in bottom-2 (judges-save seasons).

    If eliminated is in bottom-2, at most one other contestant has badness > eliminated.
    We use a soft count of how many are worse than eliminated: sum sigmoid((b_k - b_e)/tau).
    Then penalize counts above 1.
    """
    m = b.numel()
    all_pos = torch.arange(m, device=b.device)
    keep = torch.ones(m, dtype=torch.bool, device=b.device)
    keep[epos] = False
    kpos = all_pos[keep]

    be = b[epos]                     # [E]
    bk = b[kpos]                     # [K]
    # For each eliminated (could be multiple), count how many have higher badness
    counts = torch.sigmoid((bk.unsqueeze(0) - be.unsqueeze(1)) / tau_elim).sum(dim=1)  # [E]
    # want counts <= 1
    return F.softplus(counts - 1.0).mean()

# -----------------------------
# Data utilities
# -----------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df


def find_score_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if re.match(r"week\d+_judge\d+_score", c)]


def build_judge_totals_from_raw(df_season: pd.DataFrame, score_cols: List[str], max_week: int = 11) -> pd.DataFrame:
    week_totals = {}
    for w in range(1, max_week + 1):
        cols = [c for c in score_cols if c.startswith(f"week{w}_")]
        if not cols:
            continue
        vals = df_season[cols].apply(pd.to_numeric, errors="coerce")
        week_totals[w] = vals.sum(axis=1, min_count=1)  # NaN if all NaN
    return pd.DataFrame(week_totals, index=df_season.index)


def parse_elimination_week_from_raw(df_season: pd.DataFrame) -> pd.Series:
    res = df_season["results"].astype(str)
    ew = res.str.extract(r"Eliminated Week (\d+)")[0].astype(float)
    return ew


def season_tensor_from_raw(df_season: pd.DataFrame, score_cols: List[str], max_week: int = 11):
    jt = build_judge_totals_from_raw(df_season, score_cols, max_week=max_week)
    jt = jt.replace(0.0, np.nan)  # dataset sometimes uses 0 after elimination; treat as missing
    elim_week = parse_elimination_week_from_raw(df_season)

    # mask out weeks after elimination
    for ridx, ew in elim_week.items():
        if not np.isnan(ew):
            ew_int = int(ew)
            for w in jt.columns:
                if w > ew_int:
                    jt.loc[ridx, w] = np.nan

    names = df_season["celebrity_name"].tolist()
    season_id = int(df_season["season"].iloc[0])
    T = int(max(jt.columns)) if len(jt.columns) else max_week
    J = torch.tensor(jt.values, dtype=torch.float32)
    active = ~torch.isnan(J)

    # build elim_sets from elim_week
    elim_sets = {t: [] for t in range(1, T + 1)}
    for i, ew in enumerate(elim_week.fillna(-1).astype(int).values):
        if 1 <= ew <= T:
            elim_sets[ew].append(i)

    return season_id, J, active, elim_sets, names, T


def season_tensor_from_clean_long(df_season_long: pd.DataFrame):
    """Build tensors from cleaned long format.

    Required columns:
      season, week, celebrity_name, judge_total, active, active_next or elim_after_week
    """
    season_id = int(df_season_long["season"].iloc[0])
    # ensure numeric week
    df_season_long = df_season_long.copy()
    df_season_long["week"] = pd.to_numeric(df_season_long["week"], errors="coerce").astype(int)

    # Determine max week observed in this season among active rows
    T = int(df_season_long["week"].max())

    # stable name order
    names = sorted(df_season_long["celebrity_name"].unique().tolist())
    name_to_i = {n: i for i, n in enumerate(names)}
    n = len(names)

    J = torch.full((n, T), float("nan"))
    active = torch.zeros((n, T), dtype=torch.bool)

    # Fill judge totals and active
    for _, r in df_season_long.iterrows():
        i = name_to_i[r["celebrity_name"]]
        t = int(r["week"])
        if t < 1 or t > T:
            continue
        if bool(r.get("active", True)):
            active[i, t - 1] = True
            jt = pd.to_numeric(r.get("judge_total", np.nan), errors="coerce")
            J[i, t - 1] = float(jt) if not pd.isna(jt) else float("nan")

    # Some pipelines set judge_total=0 after elimination; ignore those if inactive
    # We only treat 'active' rows as meaningful.

    # Build elimination sets:
    # eliminated after week t if active at t and not active at t+1, or elim_after_week flag
    elim_sets = {t: [] for t in range(1, T + 1)}
    if "elim_after_week" in df_season_long.columns:
        df_el = df_season_long[df_season_long["elim_after_week"] == True]
        for _, r in df_el.iterrows():
            i = name_to_i[r["celebrity_name"]]
            t = int(r["week"])
            if 1 <= t <= T:
                elim_sets[t].append(i)
    elif "active_next" in df_season_long.columns:
        df_el = df_season_long[(df_season_long["active"] == True) & (df_season_long["active_next"] == False)]
        for _, r in df_el.iterrows():
            i = name_to_i[r["celebrity_name"]]
            t = int(r["week"])
            if 1 <= t <= T:
                elim_sets[t].append(i)

    # Active mask should be based on explicit active flag (more reliable than ~isnan(J))
    return season_id, J, active, elim_sets, names, T


# -----------------------------
# Helpers
# -----------------------------

def safe_mean(x: torch.Tensor) -> torch.Tensor:
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
    rank_tau: float = 0.05
    elim_tau: float = 0.05
    bottom2_lambda: float = 2.0  # only used for judges-save seasons


def estimate_votes_nonbayes(
    season_id: int,
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
            # If judge totals are missing (NaN) even when active, skip those entries
            ok = ~torch.isnan(Jt)
            idx = idx[ok]
            Jt = Jt[ok]
            if idx.numel() < 3:
                continue

            ut = U[idx, t - 1]
            ut = ut - ut.mean()  # remove location invariance
            vp = torch.softmax(ut, dim=0)

            b = compute_badness(season_id, Jt, vp, cfg.weight_judge, cfg.rank_tau)

            Es = [e for e in elim_sets.get(t, []) if e in idx.tolist()]
            if len(Es) == 0:
                continue
            epos = torch.tensor([idx.tolist().index(e) for e in Es], dtype=torch.long)

            method = voting_method_for_season(season_id)
            if method == "rank_judges_save":
                # Encourage eliminated to be in bottom-2, plus (optionally) still worse than others
                loss = loss - elim_loglik_standard(b, epos, cfg.elim_tau)
                loss = loss + cfg.bottom2_lambda * elim_penalty_bottom2(b, epos, cfg.elim_tau)
            else:
                loss = loss - elim_loglik_standard(b, epos, cfg.elim_tau)

        # temporal smoothness regularization
        diff = U[:, 1:] - U[:, :-1]
        loss = loss + cfg.smooth_lambda * safe_mean((diff.reshape(-1)) ** 2)

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

            Jt = J[idx, t - 1]
            ok = ~torch.isnan(Jt)
            idx = idx[ok]
            Jt = Jt[ok]
            if idx.numel() < 2:
                continue

            ut = U[idx, t - 1]
            ut = ut - ut.mean()
            vp = torch.softmax(ut, dim=0).cpu().numpy()
            p_est[idx.cpu().numpy(), t - 1] = vp

            # predict elimination
            b = compute_badness(season_id, Jt, torch.tensor(vp), cfg.weight_judge, cfg.rank_tau).cpu().numpy()
            method = voting_method_for_season(season_id)
            if method == "rank_judges_save" and len(b) >= 2:
                bottom2 = np.argsort(-b)[:2]  # highest badness (worst two)
                # Judges-save: assume eliminated is the one with lower judge total among bottom2 (simple proxy)
                jvals = Jt.cpu().numpy()
                choice = bottom2[np.argmin(jvals[bottom2])]
                pred_elim[t] = int(idx.cpu().numpy()[choice])
            else:
                pred_elim[t] = int(idx.cpu().numpy()[np.argmax(b)])

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
    rank_tau: float = 0.05
    elim_tau: float = 0.05
    bottom2_lambda: float = 2.0


def estimate_votes_bayes_vi(
    season_id: int,
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
                ok = ~torch.isnan(Jt)
                idx = idx[ok]
                Jt = Jt[ok]
                if idx.numel() < 3:
                    continue

                ut = U[idx, t - 1]
                ut = ut - ut.mean()
                vp = torch.softmax(ut, dim=0)

                b = compute_badness(season_id, Jt, vp, cfg.weight_judge, cfg.rank_tau)

                Es = [e for e in elim_sets.get(t, []) if e in idx.tolist()]
                if len(Es) == 0:
                    continue
                epos = torch.tensor([idx.tolist().index(e) for e in Es], dtype=torch.long)

                method = voting_method_for_season(season_id)
                if method == "rank_judges_save":
                    loglik = loglik + elim_loglik_standard(b, epos, cfg.elim_tau)
                    loglik = loglik - cfg.bottom2_lambda * elim_penalty_bottom2(b, epos, cfg.elim_tau)
                else:
                    loglik = loglik + elim_loglik_standard(b, epos, cfg.elim_tau)

            sigma0 = torch.tensor(cfg.sigma0)
            sigmau = torch.tensor(cfg.sigma_u)

            # Prior over latent popularity (random walk)
            logprior = (-0.5 * (U[:, 0] / sigma0) ** 2 - torch.log(sigma0)).mean()
            d = U[:, 1:] - U[:, :-1]
            logprior = logprior + (-0.5 * (d / sigmau) ** 2 - torch.log(sigmau)).mean()

            # Variational density (mean-field Gaussian)
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

            Jt = J[idx, t - 1]
            ok = ~torch.isnan(Jt)
            idx = idx[ok]
            if idx.numel() < 2:
                continue

            vps = []
            for _ in range(cfg.posterior_samples):
                U_s = mu_d + std_d * torch.randn_like(mu_d)
                ut = U_s[idx, t - 1]
                ut = ut - ut.mean()
                vp = torch.softmax(ut, dim=0).cpu().numpy()
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

            Jt = J[idx, t - 1]
            ok = ~torch.isnan(Jt)
            idx = idx[ok]
            Jt = Jt[ok]
            if idx.numel() < 2:
                continue

            vp = p_mean[idx.cpu().numpy(), t - 1]
            if np.all(np.isnan(vp)):
                continue

            b = compute_badness(season_id, Jt, torch.tensor(vp), cfg.weight_judge, cfg.rank_tau).cpu().numpy()
            method = voting_method_for_season(season_id)
            if method == "rank_judges_save" and len(b) >= 2:
                bottom2 = np.argsort(-b)[:2]
                jvals = Jt.cpu().numpy()
                choice = bottom2[np.argmin(jvals[bottom2])]
                pred_elim[t] = int(idx.cpu().numpy()[choice])
            else:
                pred_elim[t] = int(idx.cpu().numpy()[np.argmax(b)])

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

def wide_to_long_votes(names: List[str], season_id: int, p: np.ndarray, kind: str, p_sd: Optional[np.ndarray] = None) -> pd.DataFrame:
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


def guess_default_input() -> str:
    """Prefer cleaned long file if present, otherwise raw."""
    clean = DATA_DIR / "clean_long.csv"
    if clean.exists():
        return str(clean)
    return str(DATA_DIR / "ProbC_Data.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=guess_default_input())
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
    parser.add_argument("--rank_tau", type=float, default=0.05, help="temperature for soft-rank (smaller = closer to hard rank)")
    parser.add_argument("--elim_tau", type=float, default=0.05, help="temperature for elimination likelihood")
    parser.add_argument("--bottom2_lambda", type=float, default=2.0, help="strength of bottom-2 constraint in judges-save seasons")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = load_data(args.csv)

    # Decide which format we are using
    is_clean_long = {"week", "judge_total", "active"}.issubset(set(df.columns))

    if is_clean_long:
        all_seasons = sorted(df["season"].astype(int).unique().tolist())
    else:
        score_cols = find_score_columns(df)
        all_seasons = sorted(df["season"].astype(int).unique().tolist())

    seasons = parse_seasons_arg(args.seasons, all_seasons)

    nonbayes_rows = []
    bayes_rows = []
    summary_rows = []

    for s in seasons:
        s = int(s)
        if is_clean_long:
            df_s = df[df["season"].astype(int) == s].copy()
            if df_s.empty:
                continue
            season_id, J, active, elim_sets, names, T = season_tensor_from_clean_long(df_s)
        else:
            df_s = df[df["season"].astype(int) == s].copy()
            if df_s.empty:
                continue
            season_id, J, active, elim_sets, names, T = season_tensor_from_raw(df_s, score_cols, max_week=11)

        nb_cfg = NonBayesConfig(
            weight_judge=args.weight_judge,
            smooth_lambda=args.smooth_lambda,
            lr=args.lr_nonbayes,
            epochs=args.epochs_nonbayes,
            seed=args.seed,
            rank_tau=args.rank_tau,
            elim_tau=args.elim_tau,
            bottom2_lambda=args.bottom2_lambda,
        )
        p_nb, pred_nb, acc_nb = estimate_votes_nonbayes(season_id, J, active, elim_sets, nb_cfg)
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
            rank_tau=args.rank_tau,
            elim_tau=args.elim_tau,
            bottom2_lambda=args.bottom2_lambda,
        )
        p_bm, p_bs, pred_b, acc_b = estimate_votes_bayes_vi(season_id, J, active, elim_sets, bv_cfg)
        bayes_rows.append(wide_to_long_votes(names, season_id, p_bm, kind="bayes", p_sd=p_bs))

        summary_rows.append({
            "season": season_id,
            "method": voting_method_for_season(season_id),
            "n_contestants": len(names),
            "nonbayes_elim_acc": acc_nb,
            "bayes_vi_elim_acc": acc_b,
            "T_weeks": T,
        })

        print(f"[Season {season_id} | {voting_method_for_season(season_id)}] nonbayes_acc={acc_nb:.3f} | bayes_vi_acc={acc_b:.3f}")

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
