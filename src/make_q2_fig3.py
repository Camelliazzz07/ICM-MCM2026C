import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

WEEKLY = "output/task2_weekly_elims_detail.csv"
FINAL  = "output/task2_final_ranking.csv"

OUTDIR = "images"
FIGNAME = "Bobby_Bones_Survival_Fixed.png"

SEASON = 27
NAME = "Bobby Bones"
TOTAL_WEEKS = 9
FINAL_N = 4

# 你的色卡（按规则映射）
COLOR_MAP = {
    "Rank (No Save)":    "#F3CCDB",
    "Rank (w/ Save)":    "#E5E5F3",
    "Percent (No Save)": "#A8D1E1",
    "Percent (w/ Save)": "#62A9C8",
}

def clean(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c].isin(["nan", "None", ""]), c] = np.nan
    return df

def rule_label(method, save):
    if method == "rank":
        return "Rank (w/ Save)" if save else "Rank (No Save)"
    else:
        return "Percent (w/ Save)" if save else "Percent (No Save)"

def outcome_index_from_place(place: int) -> float:
    # 进入决赛：y > TOTAL_WEEKS，并且冠军最高
    return TOTAL_WEEKS + (FINAL_N - place + 1) / FINAL_N

def infer_elim_week_fallback(weekly_sub: pd.DataFrame) -> int | None:
    """
    兜底推断淘汰周：
    如果 weekly_sub 有一列表示“当周还在场的选手名单”，用它推断 Bobby 最后出现在哪周。
    """
    candidate_cols = [
        "remaining_contestants",
        "contestants_remaining",
        "alive_contestants",
        "cast_remaining",
        "participants_remaining",
    ]
    col = None
    for c in candidate_cols:
        if c in weekly_sub.columns:
            col = c
            break
    if col is None:
        return None

    last_seen = None
    for _, r in weekly_sub.sort_values("week").iterrows():
        s = r.get(col, None)
        if isinstance(s, str):
            tokens = [t.strip() for t in s.replace(";", ",").split(",")]
            if NAME in tokens:
                last_seen = int(r["week"])
    if last_seen is None:
        return None

    return min(last_seen + 1, TOTAL_WEEKS)

def get_bobby_outcome(weekly: pd.DataFrame, final: pd.DataFrame, method: str, save: bool):
    label = rule_label(method, save)

    ws = weekly[
        (weekly["method"] == method) &
        (weekly["bottom2_save"].astype(str).str.lower() == str(save).lower())
    ].copy()

    # 1) 直接用 eliminated_pred 找淘汰周
    elim_week = None
    if "eliminated_pred" in ws.columns:
        hits = ws[ws["eliminated_pred"] == NAME]
        if not hits.empty:
            elim_week = int(hits.sort_values("week")["week"].iloc[0])

    if elim_week is not None:
        return label, float(elim_week), f"Eliminated Week {elim_week}"

    # 2) 用 final_ranking 找名次
    fr = final[
        (final["method"] == method) &
        (final["bottom2_save"].astype(str).str.lower() == str(save).lower()) &
        (final["celebrity_name"] == NAME)
    ]
    if not fr.empty and "pred_final_place" in fr.columns and pd.notna(fr["pred_final_place"].iloc[0]):
        place = int(fr["pred_final_place"].iloc[0])
        return label, outcome_index_from_place(place), f"Final Place {place}"

    # 3) 兜底：名单消失推断淘汰周
    fallback_week = infer_elim_week_fallback(ws)
    if fallback_week is not None:
        return label, float(fallback_week), f"Inferred Elim Week {fallback_week}"

    # 4) 再兜底：活到最后一周但未知名次
    return label, float(TOTAL_WEEKS), "Reached Final Week (place unknown)"

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    weekly = clean(pd.read_csv(WEEKLY))
    final  = clean(pd.read_csv(FINAL))

    weekly = weekly[weekly["season"] == SEASON]
    final  = final[final["season"] == SEASON]

    rules = [
        ("rank", False),
        ("rank", True),
        ("percent", False),
        ("percent", True),
    ]

    labels, yvals, notes = [], [], []
    for method, save in rules:
        lab, y, note = get_bobby_outcome(weekly, final, method, save)
        labels.append(lab)
        yvals.append(y)
        notes.append(note)

    # --- Plot (更好看版) ---
    plt.figure(figsize=(10, 5))

    xs = np.arange(len(labels))

    # 1) 先画一条“中性连接线”（灰色），避免整体只有一种颜色太突兀
    plt.plot(xs, yvals, linewidth=2, color="gray", alpha=0.5, zorder=1)

    # 2) 再画 4 个彩色点（每条规则一个颜色）
    for i, lab in enumerate(labels):
        c = COLOR_MAP.get(lab, "#333333")
        plt.scatter(xs[i], yvals[i], s=110, color=c, edgecolor="black", linewidth=0.6, zorder=3)

    # Final week reference
    plt.axhline(9, linestyle="--", color="gray", alpha=0.35, zorder=0)

    plt.ylabel("Outcome Index (higher = better)")
    plt.title("Bobby Bones Outcome under Alternative Voting Rules (Season 27)")

    # y 轴：只看决赛相关区间
    plt.ylim(8.5, 10.3)
    plt.yticks(
        [9, 9.25, 9.5, 9.75, 10],
        ["Final Week",
         "Final 4 (4th)",
         "Final 4 (3rd)",
         "Final 4 (2nd)",
         "Champion"]
    )

    plt.xticks(xs, labels, rotation=0)

    plt.grid(axis="y", alpha=0.25)

    # 只给第四种规则加“Week 8”标注（你之前要求的）
    if "Percent (w/ Save)" in labels:
        idx = labels.index("Percent (w/ Save)")
        plt.annotate(
            "(Eliminated at Week 8)",
            (xs[idx], yvals[idx]),
            textcoords="offset points",
            xytext=(0, -18),
            ha="center",
            fontsize=9,
            color="gray",
            arrowprops=dict(arrowstyle="-", color="gray", alpha=0.6)
        )

    plt.tight_layout()
    outpath = os.path.join(OUTDIR, FIGNAME)
    plt.savefig(outpath, dpi=300)
    plt.close()

    print("[OK] Saved:", outpath)
    for lab, y, note in zip(labels, yvals, notes):
        print(f"{lab:18s} -> y={y:.2f} | {note}")

if __name__ == "__main__":
    main()
