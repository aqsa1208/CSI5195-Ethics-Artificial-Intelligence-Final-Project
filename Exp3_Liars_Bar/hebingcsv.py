import pandas as pd

# 读取两个 csv 文件
df1 = pd.read_csv("C:\\Users\\Gawai\\Desktop\\Liars_Bar\\all_plays_with_challenge_info_1.csv")        # 主文件
df2 = pd.read_csv("C:\\Users\\Gawai\\Desktop\\Liars_Bar\\all_plays_with_challenge_info.csv")        # 附加信息来源

# 只提取你要合并的那一列（加上对齐用的列）
df2_subset = df2[['behavior', 'opinion_label']]     # 只保留需要的列

# 合并：以 df1 为主表，按 'id' 合并
merged = pd.merge(df1, df2_subset, on='behavior', how='left')

# 保存结果
merged.to_csv('merged_output.csv', index=False)
