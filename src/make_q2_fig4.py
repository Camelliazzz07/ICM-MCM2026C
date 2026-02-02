# src/make_q2_figures.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Config (尽量少让你手动改)
# -----------------------------
INPUT_DIR = "output"
OUT_DIR_LATEX = "images"          # 论文里引用的是 images/xxx.png
OUT_DIR_BACKUP = os.path.join("output", "images")

WEEKLY_FILE = os.path.join(INPUT_DIR, "task2_weekly_elims_detail.csv")
SEASON_FILE = os.path.join(INPUT_DIR, "task2_season_comparison.csv")
FINAL_FILE  = os.path.join(INPUT_DIR, "task2_final_ranking.csv")  # 用于判断“没被淘汰的人最终名次”

FIG3_NAME = "Bobby_Bones_Survival_Fixed.png"
FIG4_NAME = "Rule_Fairness_Comparison_Fixed.png"


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """清理列名/字符串字段的前后空格，避免你们 CSV 里的 ' week' / ' method ' 这种坑。"""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c].isin(["", "nan", "None"]), c] = np.nan
    return df


def _ensure_dirs():
    os.makedirs(OUT_DIR_LATEX, exist_ok=True)
    os.makedirs(OUT_DIR_BACKUP, exist_ok=True)


def _rule_label(method: str, bottom2_save: bool) -> str:
    # 论文里用的 4 种规则组合
    m = str(method).strip().lower()
    save = bool(bottom2_save)
    if m.startswith("rank"):
        return "Rank (w/ Save)" if save else "Rank (No Save)"
    if m.startswith("percent"):
        return "Percent (w/ Save)" if save else "Percent (No Save)"
    return f"{method} | save={save}"

def make_figure4_rule_fairness_boxplot(season_cmp: pd.DataFrame):
    """
    Figure 4:
    用 task2_season_comparison.csv 的 elim_vote_share_mean (每 season 一个值)
    对四种规则做箱线图比较。
    """
    df = season_cmp.copy()

    # elim_vote_share_mean 在你们 CSV 里是带空格的字符串，强转 float
    df["elim_vote_share_mean"] = pd.to_numeric(df["elim_vote_share_mean"], errors="coerce")
    df = df.dropna(subset=["elim_vote_share_mean", "method", "bottom2_save"])

    # 规则标签
    df["bottom2_save_bool"] = df["bottom2_save"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    df["rule"] = df.apply(lambda r: _rule_label(r["method"], r["bottom2_save_bool"]), axis=1)

    order = ["Rank (No Save)", "Rank (w/ Save)", "Percent (No Save)", "Percent (w/ Save)"]
    data = [df.loc[df["rule"] == lab, "elim_vote_share_mean"].values for lab in order]

    # 画箱线图
    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=order, showfliers=False)
    plt.ylabel("Average Fan Vote Share of Eliminated Contestants (per season)")
    plt.title("Systemic Fairness Comparison Across Voting Rules")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    fig_path1 = os.path.join(OUT_DIR_LATEX, FIG4_NAME)
    fig_path2 = os.path.join(OUT_DIR_BACKUP, FIG4_NAME)
    plt.savefig(fig_path1, dpi=300)
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"[OK] Figure 4 saved: {fig_path1} and {fig_path2}")


def main():
    _ensure_dirs()

    # 读数据
    if not os.path.exists(WEEKLY_FILE):
        raise FileNotFoundError(f"Missing {WEEKLY_FILE}")
    if not os.path.exists(SEASON_FILE):
        raise FileNotFoundError(f"Missing {SEASON_FILE}")
    if not os.path.exists(FINAL_FILE):
        raise FileNotFoundError(f"Missing {FINAL_FILE}")

    weekly = _clean_df(pd.read_csv(WEEKLY_FILE))
    season_cmp = _clean_df(pd.read_csv(SEASON_FILE))
    final_rank = _clean_df(pd.read_csv(FINAL_FILE))

    # sanity print：避免你担心字段名被我搞错
    print("[INFO] weekly columns:", list(weekly.columns))
    print("[INFO] season columns:", list(season_cmp.columns))
    print("[INFO] final columns:", list(final_rank.columns))

    make_figure4_rule_fairness_boxplot(season_cmp)


if __name__ == "__main__":
    main()
