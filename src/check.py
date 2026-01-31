import pandas as pd
from pathlib import Path

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 200)

# ----------------------------
# Robust paths (no dependence on working directory)
# ----------------------------
SRC_DIR = Path(__file__).resolve().parent          # .../ICM-MCM2026C/src
PROJ_DIR = SRC_DIR.parent                          # .../ICM-MCM2026C
DATA_DIR = PROJ_DIR / "data"

CLEAN_LONG = DATA_DIR / "clean_long.csv"

if not CLEAN_LONG.exists():
    print(f"[ERROR] Cannot find: {CLEAN_LONG}")
    print("Please confirm that you generated data/clean_long.csv and that it sits under the project root.")
    raise SystemExit(1)

df = pd.read_csv(CLEAN_LONG)

# ----------------------------
# Basic column checks
# ----------------------------
need_cols = ["season", "week", "celebrity_name", "judge_total", "active", "elim_after_week"]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    print("[ERROR] Missing required columns:", missing)
    print("Available columns:", list(df.columns))
    raise SystemExit(1)

# Ensure types
df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")

# bool-ish columns may come as True/False strings
for c in ["active", "elim_after_week"]:
    if df[c].dtype != bool:
        df[c] = df[c].astype(str).str.strip().str.lower().map({"true": True, "false": False})

# Optional
has_active_next = "active_next" in df.columns
if has_active_next and df["active_next"].dtype != bool:
    df["active_next"] = df["active_next"].astype(str).str.strip().str.lower().map({"true": True, "false": False})

print("======== LOADED ========")
print("File:", CLEAN_LONG)
print("Rows:", len(df))
print("Seasons:", df["season"].nunique())
print("Total elim_after_week==True:", int(df["elim_after_week"].sum()))
print()

# ----------------------------
# 1) Elimination count per (season, week)
# ----------------------------
elim_per_week = (
    df[df["elim_after_week"] == True]
    .groupby(["season", "week"])
    .size()
    .rename("n_elim")
    .sort_index()
)

print("======== 1) 淘汰人数分布（每个 season-week） ========")
dist = elim_per_week.value_counts(dropna=False).sort_index()
print(dist)
print()

print("======== 2) 异常 season-week（淘汰人数 != 1） ========")
abnormal = elim_per_week[elim_per_week != 1]
if len(abnormal) == 0:
    print("没有发现异常：所有发生淘汰的周都是 1 人淘汰。")
else:
    print(abnormal.head(50))
    print(f"... 共 {len(abnormal)} 个异常周（显示前 50）")
print()

# Note: also check weeks with zero elimination within a season-week where there are active contestants
# This is optional info; finals / special weeks may have 0.
active_groups = df.groupby(["season", "week"])["active"].sum().rename("n_active")
weeks_with_any_active = active_groups[active_groups > 0].index
weeks_with_elim = set(elim_per_week.index)
zero_elim_weeks = [idx for idx in weeks_with_any_active if idx not in weeks_with_elim]

print("======== 3) 有选手但没有淘汰的周（n_active>0 且无 elim_after_week） ========")
print("Count:", len(zero_elim_weeks))
if len(zero_elim_weeks) > 0:
    print("Examples:", zero_elim_weeks[:20])
print()

# ----------------------------
# 2) Check 'last week elimination' should typically be 0
# ----------------------------
max_week = df.groupby("season")["week"].transform("max")
bad_last_week = df[(df["elim_after_week"] == True) & (df["week"] == max_week)]

print("======== 4) 最后一周仍被标记淘汰（通常应为 0，除非你定义特殊） ========")
print("Count:", len(bad_last_week))
if len(bad_last_week) > 0:
    print(bad_last_week[["season", "week", "celebrity_name", "judge_total", "active"]].head(30))
print()

# ----------------------------
# 3) Consistency check for elim definition: active True and active_next False
# ----------------------------
if has_active_next:
    incons = df[(df["elim_after_week"] == True) & ~((df["active"] == True) & (df["active_next"] == False))]
    print("======== 5) elim_after_week 与 (active==True & active_next==False) 是否一致 ========")
    print("Inconsistent rows count:", len(incons))
    if len(incons) > 0:
        print(incons[["season", "week", "celebrity_name", "active", "active_next", "elim_after_week"]].head(30))
    print()
else:
    print("======== 5) active_next 列不存在：跳过一致性检查 ========")
    print()

# ----------------------------
# 4) Diagnostic for your preprocess risk:
#    next_week_exists should equal (week < season_max_week)
# ----------------------------
if "next_week_exists" in df.columns:
    nwe = df["next_week_exists"]
    if nwe.dtype != bool:
        nwe = nwe.astype(str).str.strip().str.lower().map({"true": True, "false": False})

    correct_nwe = (df["week"] < max_week)
    mismatch = (nwe != correct_nwe) & ~(nwe.isna()) & ~(correct_nwe.isna())

    print("======== 6) next_week_exists 是否等价于 (week < season_max_week) ========")
    print("Mismatch count:", int(mismatch.sum()))
    if int(mismatch.sum()) > 0:
        show = df.loc[mismatch, ["season", "week", "celebrity_name", "next_week_exists"]].head(30)
        show["expected"] = correct_nwe[mismatch].head(30).values
        print(show)
        print("说明：你 preprocess 里 next_week_exists 的计算可能受排序影响，建议改为 week < max_week 的版本。")
    print()
else:
    print("======== 6) next_week_exists 列不存在：跳过该诊断 ========")
    print()

# ----------------------------
# 5) Sample a few elimination rows for manual spot check
# ----------------------------
print("======== 7) 随机抽样 10 条淘汰记录（人工快速检查） ========")
elim_rows = df[df["elim_after_week"] == True]
if len(elim_rows) == 0:
    print("没有任何淘汰记录（这不正常，除非你的数据被过滤掉了）。")
else:
    sample = elim_rows.sample(min(10, len(elim_rows)), random_state=0)
    cols = ["season", "week", "celebrity_name", "judge_total", "active"]
    if has_active_next:
        cols += ["active_next"]
    if "next_week_exists" in df.columns:
        cols += ["next_week_exists"]
    cols += ["elim_after_week"]
    print(sample[cols].sort_values(["season", "week"]))
print()

print("Done.")
