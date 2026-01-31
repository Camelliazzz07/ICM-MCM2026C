import pandas as pd
import numpy as np

df = pd.read_csv("output/vote_driver_softmax_predictions.csv")

# 1) week 内相关性：bayes_vote_share vs judge_total
cors = []
for (s,w), g in df.groupby(["season","week"]):
    if len(g) >= 3:
        cors.append(np.corrcoef(g["bayes_vote_share"], g["judge_total"])[0,1])
print("avg corr(vote, judge_total) within week:", np.nanmean(cors))

# 2) week 内相关性：bayes_vote_share vs vote_share_hat
cors2 = []
for (s,w), g in df.groupby(["season","week"]):
    if len(g) >= 3:
        cors2.append(np.corrcoef(g["bayes_vote_share"], g["vote_share_hat"])[0,1])
print("avg corr(vote, vote_hat) within week:", np.nanmean(cors2))
