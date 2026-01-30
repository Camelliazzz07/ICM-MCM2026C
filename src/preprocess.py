import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

df = pd.read_csv("../data/ProbC_Data.csv", dtype=str)
df.columns = df.columns.str.replace("\ufeff","",regex=False).str.strip()

judge_cols = [c for c in df.columns if c.startswith("week") and c.endswith("_score")]

for c in judge_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

for c in ["season","placement","celebrity_age_during_season"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# weekly totals
for w in range(1,12):
    cols = [c for c in judge_cols if c.startswith(f"week{w}_")]
    if cols:
        df[f"week{w}_total"] = df[cols].sum(axis=1)

id_cols = [
    "celebrity_name","ballroom_partner","celebrity_industry",
    "celebrity_homestate","celebrity_homecountry/region",
    "celebrity_age_during_season","season","results","placement"
]
id_cols = [c for c in id_cols if c in df.columns]

value_cols = [f"week{w}_total" for w in range(1,12) if f"week{w}_total" in df.columns]

long_df = df.melt(
    id_vars=id_cols,
    value_vars=value_cols,
    var_name="week",
    value_name="judge_total"
)

long_df["week"] = long_df["week"].str.extract(r"week(\d+)").astype(int)

long_df["active"] = long_df["judge_total"].fillna(0) > 0
long_df["n_active"] = long_df.groupby(["season","week"])["active"].transform("sum")

long_df = long_df.sort_values(["season","celebrity_name","week"])
long_df["season_week_active"] = long_df.groupby(["season","week"])["active"].transform("sum") > 0

long_df["next_week_exists"] = long_df.groupby("season")["week"].shift(-1)
long_df["next_week_exists"] = long_df["next_week_exists"].notna()

long_df["active_next"] = (
    long_df.groupby(["season","celebrity_name"])["active"]
    .shift(-1)
)

long_df["elim_after_week"] = (
    (long_df["active"]==True) &
    (long_df["active_next"]==False) &
    (long_df["next_week_exists"])
)

long_df = long_df[long_df["n_active"]>0]

long_df.to_csv("../data/clean_long.csv", index=False)

print("done", long_df.shape)
