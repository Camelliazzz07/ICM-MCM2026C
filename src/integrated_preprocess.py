import os
import pandas as pd
import numpy as np
import re

# # 获取当前脚本所在目录的上一级目录作为项目根目录 (ROOT)
# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# # 定义数据输入目录和结果输出目录
# DATA_DIR = os.path.join(ROOT, "data")
# OUT_DIR = os.path.join(ROOT, "output")

# # 确保输出目录存在，如果不存在则创建
# os.makedirs(OUT_DIR, exist_ok=True)

# # 定义具体的文件路径
# INPUT_FILE = os.path.join(DATA_DIR, "ProbC_Data.csv")  # 原始输入文件
# OUTPUT_FILE = os.path.join(OUT_DIR, "visualization_data_final.csv") # 结果输出文件

# print(f"项目根目录: {ROOT}")
# print(f"读取数据路径: {INPUT_FILE}")
# print(f"输出保存路径: {OUTPUT_FILE}")

# ==========================================
# 辅助函数定义
# ==========================================

def get_era(season):
    """
    Step 3: 动态规则映射 (Dynamic Rule Mapping)
    根据赛季定义规则时代
    """
    if season <= 2:
        return "Rank Era (S1-S2)"
    elif season <= 27:
        return "Percentage Era (S3-S27)"
    else:
        return "Judges' Save Era (S28+)"

def clean_industry(ind):
    """
    Step 5: 特征工程 - 职业归类
    将繁杂的职业标签归纳为 6 大类，便于绘图
    """
    if pd.isna(ind):
        return "Other"
    
    ind = str(ind).lower().strip()
    
    # 关键词匹配逻辑
    if any(x in ind for x in ['nfl', 'nba', 'olympic', 'athlete', 'player', 'boxer', 'ufc', 'wrestler', 'gymnast', 'skater', 'swimmer', 'snowboarder', 'jockey']):
        return "Athlete"
    elif any(x in ind for x in ['actor', 'actress', 'star', 'disney', 'marvel', 'sitcom', 'soap']):
        return "Actor/Actress"
    elif any(x in ind for x in ['singer', 'musician', 'rapper', 'pop', 'band', 'idol', 'country', 'song']):
        return "Musician"
    elif any(x in ind for x in ['reality', 'bachelor', 'housewife', 'survivor', 'shark', 'tv personality']):
        return "Reality Star"
    elif any(x in ind for x in ['host', 'anchor', 'meteorologist', 'journalist', 'correspondent']):
        return "Host/Journalist"
    else:
        return "Other" # 包括 Model, Politician, Influencer 等

# ==========================================
# 主处理流程
# ==========================================

def main():
    print(">>> 开始执行数据预处理流水线...")

    # -------------------------------------------------------
    # Step 1: 基础清洗与数字化 (Data Cleaning)
    # -------------------------------------------------------
    print("Step 1: 加载原始数据并清洗...")
    # 使用 utf-8-sig 处理 BOM (Byte Order Mark) 问题
    try:
       df = pd.read_csv("data/ProbC_Data.csv", dtype=str)
    except UnicodeDecodeError:
        df = pd.read_csv("data/ProbC_Data.csv", encoding='latin1')

    # 清洗列名
    df.columns = df.columns.str.strip().str.lower()
    
    # 转换数值列
    cols_to_numeric = ['season', 'celebrity_age_during_season']
    for c in cols_to_numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 识别评委打分列 (weekX_judgeY_score)
    judge_cols = [c for c in df.columns if re.match(r'week\d+_judge\d+_score', c)] # type: ignore
    for c in judge_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # 计算每周评委总分 (Weekly Totals)
    # 注意：自动处理 3 评委或 4 评委的情况
    for w in range(1, 13): # 假设最多 12 周
        week_cols = [c for c in judge_cols if c.startswith(f"week{w}_")]
        if week_cols:
            # 只对非空的行求和，避免把没参赛的人算成 0 分
            # min_count=1 确保全是 NaN 的时候结果是 NaN 而不是 0
            df[f"week{w}_total"] = df[week_cols].sum(axis=1, min_count=1)

    # -------------------------------------------------------
    # Step 2: 结构重组 - 宽表转长表 (Restructuring / Melt)
    # -------------------------------------------------------
    print("Step 2: 执行时空折叠 (Wide to Long)...")
    
    # 定义保留的静态身份列
    id_vars = [
        'celebrity_name', 'season', 'celebrity_industry', 
        'celebrity_homestate', 'celebrity_age_during_season', 
        'results', 'placement'
    ]
    # 确保这些列存在于数据中
    id_vars = [c for c in id_vars if c in df.columns]

    # 定义要折叠的值列 (即上面算出来的 weekX_total)
    value_vars = [c for c in df.columns if c.endswith('_total') and c.startswith('week')] # type: ignore

    # Melt 操作
    long_df = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='week_str',
        value_name='judge_score_raw'
    )

    # 从 'week1_total' 中提取数字 1
    long_df['week'] = long_df['week_str'].str.extract(r'(\d+)').astype(int)
    
    # 过滤无效数据：分数为空或为0通常表示该周未参赛
    long_df = long_df.dropna(subset=['judge_score_raw'])
    long_df = long_df[long_df['judge_score_raw'] > 0]
    
    # 排序方便后续计算
    long_df = long_df.sort_values(by=['season', 'week', 'judge_score_raw'], ascending=[True, True, False])

    # -------------------------------------------------------
    # Step 3: 动态规则映射 (Dynamic Rule Mapping)
    # -------------------------------------------------------
    print("Step 3: 注入赛季规则标签 (Eras)...")
    long_df['era_label'] = long_df['season'].apply(get_era)

    # -------------------------------------------------------
    # Step 4: 基准化处理 (Benchmarking)
    # -------------------------------------------------------
    print("Step 4: 计算标准化指标 (Shares & Ranks)...")
    
    # 计算 Judge Share (评委分占比)：该选手得分 / 当周所有人总分
    # 这是一个归一化的 0-1 值，消除了 30分制/40分制 的影响
    grouped = long_df.groupby(['season', 'week'])['judge_score_raw']
    long_df['judge_score_sum_weekly'] = grouped.transform('sum')
    long_df['judge_share'] = long_df['judge_score_raw'] / long_df['judge_score_sum_weekly']
    
    # 计算 Judge Rank (排名)：用于 S1-S2 分析
    # ascending=False 表示分数越高排名越前 (Rank 1)
    long_df['judge_rank'] = grouped.rank(ascending=False, method='min')

    # -------------------------------------------------------
    # Step 5: 特征工程 (Feature Engineering)
    # -------------------------------------------------------
    print("Step 5: 生成绘图特征 (Industry Clusters & Survival)...")
    
    # 职业聚类
    long_df['industry_cluster'] = long_df['celebrity_industry'].apply(clean_industry)
    
    # 标记生存状态
    # 逻辑：对于每一个选手，找到他参加的“最大周数”。
    # 如果这周不是决赛周（通常10-12周），那这一周就是他的“淘汰周”。
    max_weeks = long_df.groupby(['season', 'celebrity_name'])['week'].max().reset_index()
    max_weeks.rename(columns={'week': 'last_week_active'}, inplace=True)
    
    long_df = pd.merge(long_df, max_weeks, on=['season', 'celebrity_name'], how='left')
    
    # 标记每一行是否是该选手的“最后一舞”
    long_df['is_elimination_week'] = long_df['week'] == long_df['last_week_active']
    
    # 简单修正决赛选手的逻辑：如果 Placement 是 "Winner", "Runner-up", "3rd" 等，则最后一周不算被淘汰
    # 这里用简单的字符串包含判断
    def is_finalist_logic(row):
        res = str(row['results']).lower()
        if 'winner' in res or 'runner' in res or '3rd' in res or '4th' in res:
            return True
        return False

    # 这是一个近似判断，主要用于绘图区分颜色
    long_df['is_finalist'] = long_df.apply(is_finalist_logic, axis=1)
    
    # 如果是决赛选手，那他的最后一周状态不是“淘汰”，而是“完成比赛”
    long_df.loc[long_df['is_finalist'], 'is_elimination_week'] = False

    # -------------------------------------------------------
    # 导出与清理
    # -------------------------------------------------------
    # 选择最终需要的列，让CSV干净一点
    final_cols = [
        'season', 'era_label', 'week', 
        'celebrity_name', 'industry_cluster', 'celebrity_age_during_season',
        'judge_score_raw', 'judge_share', 'judge_rank',
        'is_elimination_week', 'is_finalist'
    ]
    
    output_df = long_df[final_cols]
    
    print(f"处理完成！生成数据形状: {output_df.shape}")
    print(f"正在保存至: output/visualization_data_final.csv")
    output_df.to_csv("output/visualization_data_final.csv", index=False, encoding='utf-8-sig')
    print(">>> 完成。你可以用这份 CSV 去画 Sankey 图、热力图或分布图了。")

if __name__ == "__main__":
    main()