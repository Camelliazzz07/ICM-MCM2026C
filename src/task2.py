# src/task2.py
import argparse
import os
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CLEAN = os.path.join(ROOT, "data", "clean_long.csv")
OUT_DIR = os.path.join(ROOT, "output")

VOTE_FILES = {
    "bayes": os.path.join(OUT_DIR, "bayes_vi_vote_estimates.csv"),
    "nonbayes": os.path.join(OUT_DIR, "nonbayes_vote_estimates.csv"),
}

# 题目点名的四个争议人物
CONTROVERSIES = [
    (2, "Jerry Rice"),
    (4, "Billy Ray Cyrus"),
    (11, "Bristol Palin"),
    (27, "Bobby Bones"),
]


# ---------- IO & Cleaning ----------

def _strip_df_strings(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()
    return df


def _to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    x = s.astype(str).str.strip().str.lower()
    return x.isin(["true", "1", "yes", "y", "t"])


def read_clean_long(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = _strip_df_strings(df)

    # 必要字段检查
    need_cols = {"season", "week", "celebrity_name", "judge_total", "active"}
    miss = need_cols - set(df.columns)
    if miss:
        raise ValueError(f"clean_long.csv 缺少列: {miss}")

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df["judge_total"] = pd.to_numeric(df["judge_total"], errors="coerce")
    df["celebrity_name"] = df["celebrity_name"].astype(str).str.strip()

    df["active"] = _to_bool_series(df["active"])

    # elim_after_week 若缺失则默认 False（但你文件里一般有）
    if "elim_after_week" in df.columns:
        df["elim_after_week"] = _to_bool_series(df["elim_after_week"])
    else:
        df["elim_after_week"] = False

    # results 可选（用于展示官方结果）
    if "results" in df.columns:
        df["results"] = df["results"].astype(str).str.strip()

    return df


def read_votes(vote_path: str, source: str) -> pd.DataFrame:
    v = pd.read_csv(vote_path)
    v.columns = [c.strip() for c in v.columns]
    v = _strip_df_strings(v)

    if source == "bayes":
        need = {"season", "week", "celebrity_name", "bayes_vote_share"}
        miss = need - set(v.columns)
        if miss:
            raise ValueError(f"{vote_path} 缺少列: {miss}")
        v = v[["season", "week", "celebrity_name", "bayes_vote_share"]].copy()
        v = v.rename(columns={"bayes_vote_share": "vote_share"})
    else:
        need = {"season", "week", "celebrity_name", "nonbayes_vote_share"}
        miss = need - set(v.columns)
        if miss:
            raise ValueError(f"{vote_path} 缺少列: {miss}")
        v = v[["season", "week", "celebrity_name", "nonbayes_vote_share"]].copy()
        v = v.rename(columns={"nonbayes_vote_share": "vote_share"})

    v["season"] = pd.to_numeric(v["season"], errors="coerce").astype("Int64")
    v["week"] = pd.to_numeric(v["week"], errors="coerce").astype("Int64")
    v["celebrity_name"] = v["celebrity_name"].astype(str).str.strip()
    v["vote_share"] = pd.to_numeric(v["vote_share"], errors="coerce")

    return v


# ---------- Helpers ----------

def compute_k_official(df_week_all: pd.DataFrame) -> int:
    # 官方当周淘汰人数（清洗后 elim_after_week=True 的个数）
    if "elim_after_week" not in df_week_all.columns:
        return 1
    return int(df_week_all["elim_after_week"].sum())


def stable_sort(df: pd.DataFrame, cols, ascending):
    # mergesort 稳定，便于可重复
    return df.sort_values(cols, ascending=ascending, kind="mergesort")


def split_names(s: str):
    if not isinstance(s, str) or not s.strip():
        return []
    return [t.strip() for t in s.split(";") if t.strip()]


def compute_combined_scores(
    wk: pd.DataFrame,
    method: str,
    rank_tie_method: str
) -> pd.DataFrame:
    """
    给当周 dataframe 计算 combined，并返回一个“最差在前”的排序结果（用于淘汰）。
    """
    wk = wk.copy()

    if method == "rank":
        # tie 处理：min / average / dense
        # judge_total/vote_share 从高到低是更好，所以 rank 用 ascending=False
        wk["judge_rank"] = wk["judge_total"].rank(ascending=False, method=rank_tie_method) # type: ignore
        wk["fan_rank"] = wk["vote_share"].rank(ascending=False, method=rank_tie_method) # type: ignore
        wk["combined"] = wk["judge_rank"] + wk["fan_rank"]
        # 最差：combined 大；tie-break：fan_rank 大（粉丝更差），再 judge_rank 大
        wk_sorted = stable_sort(wk, ["combined", "fan_rank", "judge_rank"], [False, False, False])
        return wk_sorted

    if method == "percent":
        wk["judge_pct"] = wk["judge_total"] / wk["judge_total"].sum()
        wk["fan_pct"] = wk["vote_share"]
        wk["combined"] = wk["judge_pct"] + wk["fan_pct"]
        # 最差：combined 小；tie-break：fan_pct 小，再 judge_total 小
        wk_sorted = stable_sort(wk, ["combined", "fan_pct", "judge_total"], [True, True, True])
        return wk_sorted

    raise ValueError("method 必须是 'rank' 或 'percent'")


def compute_final_ranking(
    df_season: pd.DataFrame,
    votes_season: pd.DataFrame,
    method: str,
    rank_tie_method: str,
    finalists: list[str],
    last_week: int
) -> pd.DataFrame:
    """
    对“最后一周仍在场”的 finalists 计算最终名次：
      - rank：combined 越小越好
      - percent：combined 越大越好
    返回：celebrity_name -> pred_final_place
    """
    wk = df_season[df_season["week"] == last_week].copy()
    wk = wk[wk["celebrity_name"].isin(finalists)].copy()
    wk = wk.merge(
        votes_season[votes_season["week"] == last_week],
        on=["season", "week", "celebrity_name"],
        how="left"
    )

    if wk["vote_share"].isna().any():
        wk["vote_share"] = wk["vote_share"].fillna(1.0 / max(len(wk), 1))

    wk_scored = compute_combined_scores(wk, method=method, rank_tie_method=rank_tie_method)

    # wk_scored 当前是“最差在前”的排序；决赛名次需要“最好在前”
    if method == "rank":
        # combined 越小越好
        best_first = stable_sort(wk_scored, ["combined", "fan_rank", "judge_rank"], [True, True, True])
    else:
        # percent：combined 越大越好
        best_first = stable_sort(wk_scored, ["combined", "fan_pct", "judge_total"], [False, False, False])

    best_first["pred_final_place"] = np.arange(1, len(best_first) + 1)
    return best_first[["celebrity_name", "pred_final_place", "combined"]].copy()


# ---------- Simulation Core ----------

def simulate_one_season(
    df_season: pd.DataFrame,
    votes_season: pd.DataFrame,
    method: str,
    bottom2_save: bool,
    rank_tie_method: str
):
    """
    返回三个表：
    1) weekly_detail：每周淘汰明细
    2) contestant_exit：每个选手预测退出周 / 是否决赛 / 决赛名次（若没被淘汰）
    3) final_ranking：决赛周排名明细（只含 finalists）
    """
    season_id = int(df_season["season"].dropna().iloc[0])
    weeks = sorted(df_season["week"].dropna().unique().tolist())
    last_week = int(weeks[-1])

    # 初始 active_set：用第一周 active==True
    w0 = int(weeks[0])
    init = df_season[(df_season["week"] == w0) & (df_season["active"] == True)]
    active_set = set(init["celebrity_name"].tolist())

    # 记录每个选手何时被淘汰
    elim_week = {}  # name -> week

    weekly_rows = []

    for w in weeks:
        w = int(w)
        df_w_all = df_season[df_season["week"] == w].copy()
        k = compute_k_official(df_w_all)

        wk = df_w_all[df_w_all["celebrity_name"].isin(active_set)].copy()
        n_active_before = int(len(wk))

        if wk.empty:
            weekly_rows.append({
                "season": season_id,
                "week": w,
                "method": method,
                "bottom2_save": bottom2_save,
                "rank_tie_method": rank_tie_method if method == "rank" else "",
                "k_official": int(k),
                "n_active_before": 0,
                "eliminated_pred": "",
            })
            continue

        wk = wk.merge(
            votes_season[votes_season["week"] == w],
            on=["season", "week", "celebrity_name"],
            how="left"
        )
        if wk["vote_share"].isna().any():
            wk["vote_share"] = wk["vote_share"].fillna(1.0 / len(wk))

        wk_sorted = compute_combined_scores(wk, method=method, rank_tie_method=rank_tie_method)

        eliminated = []
        if k > 0:
            if bottom2_save and len(wk_sorted) >= 2:
                tmp_active = set(wk_sorted["celebrity_name"].tolist())
                for _ in range(min(k, len(tmp_active))):
                    tmp = wk_sorted[wk_sorted["celebrity_name"].isin(tmp_active)].copy()
                    if len(tmp) == 1:
                        loser = tmp["celebrity_name"].iloc[0]
                    else:
                        bottom2 = tmp.head(2).copy()  # 最差两名
                        # judges save：淘汰 judge_total 更低者（技术更差）
                        bottom2 = stable_sort(bottom2, ["judge_total", "vote_share"], [True, True])
                        loser = bottom2["celebrity_name"].iloc[0]
                    eliminated.append(loser)
                    tmp_active.remove(loser)
            else:
                eliminated = wk_sorted.head(min(k, len(wk_sorted)))["celebrity_name"].tolist()

        for name in eliminated:
            active_set.discard(name)
            if name not in elim_week:
                elim_week[name] = w

        weekly_rows.append({
            "season": season_id,
            "week": w,
            "method": method,
            "bottom2_save": bottom2_save,
            "rank_tie_method": rank_tie_method if method == "rank" else "",
            "k_official": int(k),
            "n_active_before": n_active_before,
            "eliminated_pred": "; ".join(eliminated),
        })

    # 决赛周：对剩余 active_set 给出最终名次
    finalists = sorted(list(active_set))
    final_rank_df = pd.DataFrame(columns=["celebrity_name", "pred_final_place", "combined"])
    if len(finalists) > 0:
        final_rank_df = compute_final_ranking(
            df_season=df_season,
            votes_season=votes_season,
            method=method,
            rank_tie_method=rank_tie_method,
            finalists=finalists,
            last_week=last_week
        )

    # contestant_exit 表：全赛季初始参赛者（第一周 active）
    contestants = sorted(init["celebrity_name"].unique().tolist())
    final_place_map = {r["celebrity_name"]: int(r["pred_final_place"]) for _, r in final_rank_df.iterrows()}

    exit_rows = []
    for name in contestants:
        is_finalist = name in finalists
        # 若没被淘汰：退出周=last_week，并给 pred_final_place
        if is_finalist:
            exit_rows.append({
                "season": season_id,
                "method": method,
                "bottom2_save": bottom2_save,
                "rank_tie_method": rank_tie_method if method == "rank" else "",
                "celebrity_name": name,
                "pred_exit_week": last_week,
                "pred_elim_week": np.nan,  # 用 NaN 表示未被淘汰（但不会空着不管）
                "is_finalist": True,
                "pred_final_place": final_place_map.get(name, np.nan),
            })
        else:
            exit_rows.append({
                "season": season_id,
                "method": method,
                "bottom2_save": bottom2_save,
                "rank_tie_method": rank_tie_method if method == "rank" else "",
                "celebrity_name": name,
                "pred_exit_week": int(elim_week.get(name, np.nan)),
                "pred_elim_week": int(elim_week.get(name, np.nan)) if name in elim_week else np.nan,
                "is_finalist": False,
                "pred_final_place": np.nan,
            })

    weekly_detail = pd.DataFrame(weekly_rows)
    contestant_exit = pd.DataFrame(exit_rows)

    # 给 final_rank_df 加上规则标识
    if not final_rank_df.empty:
        final_rank_df = final_rank_df.copy()
        final_rank_df.insert(0, "season", season_id)
        final_rank_df.insert(1, "method", method)
        final_rank_df.insert(2, "bottom2_save", bottom2_save)
        final_rank_df.insert(3, "rank_tie_method", rank_tie_method if method == "rank" else "")
        final_rank_df.insert(4, "week", last_week)

    return weekly_detail, contestant_exit, final_rank_df


# ---------- Evaluation ----------

def build_official_elim_sets(df_clean: pd.DataFrame) -> pd.DataFrame:
    off = df_clean[df_clean["elim_after_week"] == True].copy()
    g = off.groupby(["season", "week"])["celebrity_name"].apply(lambda s: set(s.tolist())).reset_index()
    return g.rename(columns={"celebrity_name": "eliminated_official_set"})


def add_fan_bias_metrics(weekly_detail: pd.DataFrame, df_clean: pd.DataFrame, votes: pd.DataFrame) -> pd.DataFrame:
    """
    为每个 season-week-rule 增加：
      - elim_vote_share_mean：被淘汰者 vote_share 平均（越低越偏粉丝）
      - elim_vote_share_pctile：被淘汰者 vote_share 分位均值（0~1，越低越偏粉丝）
    """
    base = df_clean[["season", "week", "celebrity_name", "active"]].copy()
    base = base.merge(votes, on=["season", "week", "celebrity_name"], how="left")
    base["vote_share"] = pd.to_numeric(base["vote_share"], errors="coerce")

    base_active = base[base["active"] == True].copy()
    base_active["vote_pctile"] = base_active.groupby(["season", "week"])["vote_share"].rank(pct=True, ascending=True)

    rows = []
    for _, r in weekly_detail.iterrows():
        season, week = int(r["season"]), int(r["week"])
        elim_names = split_names(r["eliminated_pred"])
        if not elim_names:
            rows.append({**r.to_dict(), "elim_vote_share_mean": np.nan, "elim_vote_share_pctile": np.nan})
            continue

        sub = base_active[
            (base_active["season"] == season) &
            (base_active["week"] == week) &
            (base_active["celebrity_name"].isin(elim_names))
        ].copy()

        if sub.empty:
            rows.append({**r.to_dict(), "elim_vote_share_mean": np.nan, "elim_vote_share_pctile": np.nan})
        else:
            rows.append({
                **r.to_dict(),
                "elim_vote_share_mean": float(sub["vote_share"].mean()),
                "elim_vote_share_pctile": float(sub["vote_pctile"].mean()),
            })

    return pd.DataFrame(rows)


def evaluate_agreement(weekly_detail: pd.DataFrame, df_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    off = build_official_elim_sets(df_clean)

    pred = weekly_detail.copy()
    pred["eliminated_pred_set"] = pred["eliminated_pred"].apply(
        lambda x: set(split_names(x))
    )

    merged = pred.merge(off, on=["season", "week"], how="left")
    merged["eliminated_official_set"] = merged["eliminated_official_set"].apply(lambda x: x if isinstance(x, set) else set())
    merged["week_agree"] = merged.apply(lambda r: r["eliminated_pred_set"] == r["eliminated_official_set"], axis=1)

    season_summary = merged.groupby(["season", "method", "bottom2_save", "rank_tie_method"]).agg(
        n_weeks=("week", "count"),
        weekly_agreement_rate=("week_agree", "mean"),
    ).reset_index()

    return merged, season_summary


def controversy_summary(
    contestant_exit_all: pd.DataFrame,
    df_clean: pd.DataFrame
) -> pd.DataFrame:
    """
    输出：四个争议人物在不同规则下：
      - pred_elim_week（若被淘汰）
      - pred_final_place（若进决赛）
      - pred_exit_week（统一出口：被淘汰周 or last_week）
    """
    rows = []
    for season, celeb in CONTROVERSIES:
        df_off = df_clean[(df_clean["season"] == season) & (df_clean["celebrity_name"] == celeb)]
        off_result = df_off["results"].iloc[0] if ("results" in df_clean.columns and len(df_off)) else "N/A"

        sub_all = contestant_exit_all[
            (contestant_exit_all["season"] == season) &
            (contestant_exit_all["celebrity_name"] == celeb)
        ].copy()

        if sub_all.empty:
            # 找不到该选手记录（可能名字不一致/清洗差异）
            continue

        for _, r in sub_all.iterrows():
            rows.append({
                "season": season,
                "celebrity_name": celeb,
                "method": r["method"],
                "bottom2_save": bool(r["bottom2_save"]),
                "rank_tie_method": r["rank_tie_method"],
                "pred_elim_week": r["pred_elim_week"],
                "pred_exit_week": r["pred_exit_week"],
                "pred_final_place": r["pred_final_place"],
                "official_result": off_result,
                "is_finalist": bool(r["is_finalist"]),
            })

    out = pd.DataFrame(rows)
    # 排序方便看
    if not out.empty:
        out = out.sort_values(["season", "method", "bottom2_save", "rank_tie_method"])
    return out


# ---------- Main ----------

def parse_rank_tie_methods(s: str):
    items = [x.strip().lower() for x in s.split(",") if x.strip()]
    allowed = {"min", "average", "dense"}
    for it in items:
        if it not in allowed:
            raise ValueError(f"--rank_tie_methods 仅支持: min,average,dense；你给的是 {it}")
    # 去重且保序
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            out.append(it)
            seen.add(it)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vote_source", choices=["bayes", "nonbayes"], default="bayes",
                    help="用哪个任务一输出的粉丝票份额（默认 bayes）")
    ap.add_argument("--rank_tie_methods", default="min,average",
                    help="rank 法并列处理敏感性分析：逗号分隔，如 min,average 或 min,average,dense")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    df = read_clean_long(DATA_CLEAN)

    vote_path = VOTE_FILES[args.vote_source]
    if not os.path.exists(vote_path):
        raise FileNotFoundError(f"找不到投票估计文件：{vote_path}（确认 output/ 下有任务一输出）")

    votes = read_votes(vote_path, args.vote_source)

    rank_tie_methods = parse_rank_tie_methods(args.rank_tie_methods)

    weekly_all = []
    contestant_exit_all = []
    final_rank_all = []

    seasons = sorted(df["season"].dropna().unique().tolist())
    for season in seasons:
        df_s = df[df["season"] == season].copy()
        v_s = votes[votes["season"] == season].copy()
        if v_s.empty:
            continue

        # percent 规则不依赖 rank_tie_method，但为了统一输出，仍填空字符串
        for method in ["percent", "rank"]:
            if method == "rank":
                tie_list = rank_tie_methods
            else:
                tie_list = [""]  # 占位

            for tie in tie_list:
                for bottom2 in [False, True]:
                    wdetail, cexit, frank = simulate_one_season(
                        df_season=df_s,
                        votes_season=v_s,
                        method=method,
                        bottom2_save=bottom2,
                        rank_tie_method=tie if method == "rank" else "min"  # percent 不用，但传个合法值
                    )
                    weekly_all.append(wdetail)
                    contestant_exit_all.append(cexit)
                    if frank is not None and not frank.empty:
                        final_rank_all.append(frank)

    if not weekly_all:
        raise RuntimeError(
            "没有生成任何预测明细（weekly_detail为空）。\n"
            "检查：clean_long.csv 的 season/week/active；以及 vote_estimates 是否覆盖这些 season/week。"
        )

    weekly_detail = pd.concat(weekly_all, ignore_index=True)
    contestant_exit = pd.concat(contestant_exit_all, ignore_index=True) if contestant_exit_all else pd.DataFrame()
    final_ranking = pd.concat(final_rank_all, ignore_index=True) if final_rank_all else pd.DataFrame()

    # 统一：percent 的 rank_tie_method 输出为空字符串（更清晰）
    weekly_detail.loc[weekly_detail["method"] == "percent", "rank_tie_method"] = ""
    contestant_exit.loc[contestant_exit["method"] == "percent", "rank_tie_method"] = ""
    if not final_ranking.empty:
        final_ranking.loc[final_ranking["method"] == "percent", "rank_tie_method"] = ""

    # 输出：每周淘汰明细
    weekly_out = os.path.join(OUT_DIR, "task2_weekly_elims_detail.csv")
    weekly_detail.to_csv(weekly_out, index=False)

    # 2.1：一致率
    merged_weekly, season_summary = evaluate_agreement(weekly_detail, df)

    # 2.1：偏粉丝指标
    weekly_with_bias = add_fan_bias_metrics(weekly_detail, df, votes)
    bias_summary = weekly_with_bias.groupby(["season", "method", "bottom2_save", "rank_tie_method"]).agg(
        elim_vote_share_mean=("elim_vote_share_mean", "mean"),
        elim_vote_share_pctile=("elim_vote_share_pctile", "mean"),
    ).reset_index()

    season_summary = season_summary.merge(bias_summary, on=["season", "method", "bottom2_save", "rank_tie_method"], how="left")

    season_out = os.path.join(OUT_DIR, "task2_season_comparison.csv")
    season_summary.to_csv(season_out, index=False)

    # 决赛周排名输出（新增）
    final_out = os.path.join(OUT_DIR, "task2_final_ranking.csv")
    if not final_ranking.empty:
        final_ranking.to_csv(final_out, index=False)
    else:
        # 仍输出空表，避免脚本下游找不到文件
        pd.DataFrame(columns=["season", "method", "bottom2_save", "rank_tie_method", "week",
                              "celebrity_name", "pred_final_place", "combined"]).to_csv(final_out, index=False)

    # 2.2：争议人物复盘（新增：若没淘汰则给名次）
    cont = controversy_summary(contestant_exit, df)
    cont_out = os.path.join(OUT_DIR, "task2_controversy_summary.csv")
    cont.to_csv(cont_out, index=False)

    print("Done.")
    print(f"- weekly detail: {weekly_out}")
    print(f"- season comparison (2.1): {season_out}")
    print(f"- final ranking (NEW): {final_out}")
    print(f"- controversy summary (2.2): {cont_out}")
    print("Note: rank tie sensitivity is controlled by --rank_tie_methods (default min,average).")


if __name__ == "__main__":
    main()
