import pandas as pd
from pathlib import Path

pd.set_option("future.no_silent_downcasting", True)

# =========================
# Paths
# =========================
RAW_PATH = Path("../data/ProbC_Data.csv")
OUT_PATH = Path("../data/clean_long.csv")

print("Reading:", RAW_PATH.resolve())
print("Will write:", OUT_PATH.resolve())

# =========================
# Load raw data
# =========================
df = pd.read_csv(RAW_PATH, dtype=str)
df.columns = (
    df.columns
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)

# =========================
# Convert numeric columns
# =========================
judge_cols = [
    c for c in df.columns
    if c.startswith("week") and c.endswith("_score")
]
for c in judge_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

for c in ["season", "placement", "celebrity_age_during_season"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# =========================
# Weekly judge totals
# =========================
for w in range(1, 12):
    cols = [c for c in judge_cols if c.startswith(f"week{w}_")]
    if cols:
        df[f"week{w}_total"] = df[cols].sum(axis=1)

# =========================
# Melt to long format
# =========================
id_cols = [
    "celebrity_name",
    "ballroom_partner",
    "celebrity_industry",
    "celebrity_homestate",
    "celebrity_homecountry/region",
    "celebrity_age_during_season",
    "season",
    "results",
    "placement",
]
id_cols = [c for c in id_cols if c in df.columns]

value_cols = [
    f"week{w}_total"
    for w in range(1, 12)
    if f"week{w}_total" in df.columns
]

long_df = df.melt(
    id_vars=id_cols,
    value_vars=value_cols,
    var_name="week",
    value_name="judge_total",
)

long_df["week"] = (
    long_df["week"]
    .str.extract(r"week(\d+)")
    .astype(int)
)

# =========================
# Active indicator
# =========================
long_df["judge_total"] = pd.to_numeric(long_df["judge_total"], errors="coerce")
long_df["active"] = long_df["judge_total"].fillna(0) > 0

# number of active contestants per season-week
long_df["n_active"] = (
    long_df
    .groupby(["season", "week"])["active"]
    .transform("sum")
)

# =========================
# OFFICIAL elimination logic
# =========================
# Parse elimination week from results text
long_df["elim_week_official"] = (
    long_df["results"]
    .astype(str)
    .str.extract(r"Eliminated Week (\d+)")
    .astype(float)
)

# Eliminated after week t iff:
#   this contestant's official elimination week == t
long_df["elim_after_week"] = (
    long_df["elim_week_official"].notna() &
    (long_df["week"] == long_df["elim_week_official"])
)

# =========================
# Final cleanup
# =========================
# Keep only weeks where the show is running
long_df = long_df[long_df["n_active"] > 0].copy()

# Sort for sanity
long_df = long_df.sort_values(
    ["season", "week", "celebrity_name"]
)

# =========================
# Save
# =========================
long_df.to_csv(OUT_PATH, index=False)

print("Wrote clean_long.csv")
print("Rows:", len(long_df))
print("Seasons:", long_df["season"].nunique())
print("Total elim_after_week == True:",
      int(long_df["elim_after_week"].sum()))

