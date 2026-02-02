import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_DIR = "output"
OUT_DIR_LATEX = "images"
OUT_DIR_BACKUP = os.path.join("output", "images")

SEASON_FILE = os.path.join(INPUT_DIR, "task2_season_comparison.csv")
FIG4_NAME = "Rule_Fairness_Comparison_Fixed.png"

COLOR_MAP = {
    "Rank (No Save)":    "#F3CCDB",
    "Rank (w/ Save)":    "#E5E5F3",
    "Percent (No Save)": "#A8D1E1",
    "Percent (w/ Save)": "#62A9C8",
}

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
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
    m = str(method).strip().lower()
    save = bool(bottom2_save)
    if m.startswith("rank"):
        return "Rank (w/ Save)" if save else "Rank (No Save)"
    if m.startswith("percent"):
        return "Percent (w/ Save)" if save else "Percent (No Save)"
    return f"{method} | save={save}"

def make_figure4_rule_fairness_boxplot(season_cmp: pd.DataFrame):
    df = season_cmp.copy()

    df["elim_vote_share_mean"] = pd.to_numeric(df["elim_vote_share_mean"], errors="coerce")
    df = df.dropna(subset=["elim_vote_share_mean", "method", "bottom2_save"])

    df["bottom2_save_bool"] = df["bottom2_save"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    df["rule"] = df.apply(lambda r: _rule_label(r["method"], r["bottom2_save_bool"]), axis=1)

    order = ["Rank (No Save)", "Rank (w/ Save)", "Percent (No Save)", "Percent (w/ Save)"]
    data = [df.loc[df["rule"] == lab, "elim_vote_share_mean"].values for lab in order]

    plt.figure(figsize=(10, 5))

    bp = plt.boxplot(
        data,
        labels=order,
        showfliers=False,
        patch_artist=True,   # 关键：允许填充箱体颜色
        medianprops=dict(color="black", linewidth=1.4),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        boxprops=dict(edgecolor="black", linewidth=1.0),
    )

    # 给每个箱体上色
    for patch, lab in zip(bp["boxes"], order):
        patch.set_facecolor(COLOR_MAP.get(lab, "#DDDDDD"))
        patch.set_alpha(0.9)

    plt.ylabel("Average Fan Vote Share of Eliminated Contestants (per season)")
    plt.title("Systemic Fairness Comparison Across Voting Rules")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    fig_path1 = os.path.join(OUT_DIR_LATEX, FIG4_NAME)
    fig_path2 = os.path.join(OUT_DIR_BACKUP, FIG4_NAME)
    plt.savefig(fig_path1, dpi=300)
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"[OK] Figure 4 saved: {fig_path1} and {fig_path2}")

def main():
    _ensure_dirs()
    if not os.path.exists(SEASON_FILE):
        raise FileNotFoundError(f"Missing {SEASON_FILE}")

    season_cmp = _clean_df(pd.read_csv(SEASON_FILE))
    make_figure4_rule_fairness_boxplot(season_cmp)

if __name__ == "__main__":
    main()
