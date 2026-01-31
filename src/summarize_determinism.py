import pandas as pd
import numpy as np
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
prob_path = PROJ / "output" / "mc_elimination_probs.csv"
week_path = PROJ / "output" / "mc_week_metrics.csv"

dfp = pd.read_csv(prob_path)
dfw = pd.read_csv(week_path)

# 1) 每周：p_max, top2 gap
g = dfp.groupby(["season","week"], as_index=False)
rows = []
for (s,w), sub in g:
    p = sub["p_eliminated_mc"].to_numpy()
    p_sorted = np.sort(p)[::-1]
    pmax = float(p_sorted[0]) if len(p_sorted) else np.nan
    gap = float(p_sorted[0]-p_sorted[1]) if len(p_sorted) >= 2 else np.nan
    n = int(sub.shape[0])
    rows.append({"season":s,"week":w,"n_active":n,"p_max":pmax,"gap_top1_top2":gap})
det = pd.DataFrame(rows)

# 2) 合并熵，算归一化确定性 D = 1 - H/log(n_active)
out = det.merge(dfw[["season","week","entropy_elim","method","p_true_elim_proxy"]], on=["season","week"], how="left")
out["H_norm"] = out["entropy_elim"] / np.log(out["n_active"].clip(lower=2))
out["D_entropy"] = 1 - out["H_norm"]

# 3) 给出分类（你可以调整阈值）
out["det_level_pmax"] = pd.cut(out["p_max"], bins=[-1,0.6,0.8,1.01], labels=["弱确定","中等确定","强确定"])
out["sens_level_gap"] = pd.cut(out["gap_top1_top2"], bins=[-1,0.05,0.2,1.01], labels=["高度敏感","临界","稳健"])

out.to_csv(PROJ / "output" / "determinism_summary_by_week.csv", index=False)

print("Wrote: output/determinism_summary_by_week.csv")
print("\n=== Overall stats ===")
print("weeks:", len(out))
print("p_max mean/median:", out["p_max"].mean(), out["p_max"].median())
print("D_entropy mean/median:", out["D_entropy"].mean(), out["D_entropy"].median())
print("strong-deterministic rate (p_max>=0.8):", (out["p_max"]>=0.8).mean())
print("highly-sensitive rate (gap<0.05):", (out["gap_top1_top2"]<0.05).mean())
print("low-prob true elimination rate (p_true<=0.3):", (out["p_true_elim_proxy"]<=0.3).mean())
