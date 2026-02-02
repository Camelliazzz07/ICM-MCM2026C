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
    如果 weekly_sub 有一列能表示“当周还在场的选手名单”，用它推断 Bobby 最后出现在哪周。
    由于你们的 CSV 字段可能不统一，这里做“多列候选名”的尝试。
    """
    # 常见候选列名（你们如果实际列名不同，可以继续加）
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

    # 假设该列是用 ; 或 , 拼接的字符串
    last_seen = None
    for _, r in weekly_sub.sort_values("week").iterrows():
        s = r.get(col, None)
        if isinstance(s, str):
            tokens = [t.strip() for t in s.replace(";", ",").split(",")]
            if NAME in tokens:
                last_seen = int(r["week"])
    if last_seen is None:
        return None

    # 如果最后一次出现是第 k 周，说明第 k+1 周他不在了 -> 推断淘汰周 = k+1
    # 但不要超过总周数
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

    # 3) 兜底：用“是否仍在下一周名单中”推断淘汰周（如果你们 weekly 有对应列）
    fallback_week = infer_elim_week_fallback(ws)
    if fallback_week is not None:
        return label, float(fallback_week), f"Inferred Elim Week {fallback_week}"

    # 4) 再兜底：如果啥都没有，就放在 TOTAL_WEEKS（表示“至少活到最后一周但名次未知”）
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

    plt.figure(figsize=(9, 5))
    plt.plot(labels, yvals, marker="o", linewidth=2)

    # Final week reference
    plt.axhline(9, linestyle="--", color="gray", alpha=0.4)

    plt.ylabel("Outcome Index (higher = better)")
    plt.title("Bobby Bones Outcome under Alternative Voting Rules (Season 27)")

    # ✅ 核心修改在这里
    plt.ylim(8.5, 10.3)
    plt.yticks(
        [9, 9.25, 9.5, 9.75, 10],
        ["Final Week",
        "Final 4 (4th)",
        "Final 4 (3rd)",
        "Final 4 (2nd)",
        "Champion"]
    )

    plt.grid(axis="y", alpha=0.3)
    # 单独标注 Percent (w/ Save) 的淘汰周
    idx = labels.index("Percent (w/ Save)")
    x = idx
    y = yvals[idx]

    plt.annotate(
        "(Eliminated at Week 8)",
        (x, y),
        textcoords="offset points",
        xytext=(0, -18),
        ha="center",
        fontsize=9,
        color="gray",
        arrowprops=dict(arrowstyle="-", color="gray", alpha=0.6)
    )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, FIGNAME), dpi=300)
    plt.close()


    print("[OK] Saved")
    for lab, y, note in zip(labels, yvals, notes):
        print(f"{lab:18s} -> y={y:.2f} | {note}")

if __name__ == "__main__":
    main()
