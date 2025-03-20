import pandas as pd
import os

# 定义桌面路径
desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')

# 加载数据
athletes_df = pd.read_csv('summerOly_athletes.csv', encoding='utf-8')
medal_counts_df = pd.read_csv('summerOly_medal_counts.csv', encoding='utf-8')

# 清洗 NOC 列，统一格式
medal_counts_df['NOC'] = medal_counts_df['NOC'].str.upper().str.strip()
athletes_df['NOC'] = athletes_df['NOC'].str.upper().str.strip()

# 获取出现在奖牌榜上的国家
appeared_nocs = medal_counts_df['NOC'].unique()

# 筛选 athletes_df 中的国家
# 出现在奖牌榜上的国家
appeared_athletes_df = athletes_df[athletes_df['NOC'].isin(appeared_nocs)]
# 未出现在奖牌榜上的国家
not_appeared_athletes_df = athletes_df[~athletes_df['NOC'].isin(appeared_nocs)]

# 筛选 medal_counts_df 中的国家
# 出现在奖牌榜上的国家（本身就是 medal_counts_df 中的国家）
appeared_medal_counts_df = medal_counts_df
# 未出现在奖牌榜上的国家（理论上这部分数据为空，因为 medal_counts_df 中的国家都在奖牌榜上）
not_appeared_medal_counts_df = pd.DataFrame(columns=medal_counts_df.columns)  # 创建一个空的 DataFrame

# 检查筛选结果
print("出现在奖牌榜上的国家数量:", len(appeared_nocs))
print("未出现在奖牌榜上的国家数量:", len(athletes_df['NOC'].unique()) - len(appeared_nocs))

# 打印一些示例来验证
print("\n部分出现在奖牌榜上的国家:")
print(appeared_athletes_df[['NOC']].drop_duplicates().head())

print("\n部分未出现在奖牌榜上的国家:")
print(not_appeared_athletes_df[['NOC']].drop_duplicates().head())

# 保存结果到桌面
appeared_athletes_df.to_csv(os.path.join(desktop_path, 'appeared_athletes.csv'), index=False)
not_appeared_athletes_df.to_csv(os.path.join(desktop_path, 'not_appeared_athletes.csv'), index=False)
appeared_medal_counts_df.to_csv(os.path.join(desktop_path, 'appeared_medal_counts.csv'), index=False)
not_appeared_medal_counts_df.to_csv(os.path.join(desktop_path, 'not_appeared_medal_counts.csv'), index=False)

print(f"\n出现在奖牌榜上的国家数据已保存到桌面：{os.path.join(desktop_path, 'appeared_athletes.csv')}")
print(f"未出现在奖牌榜上的国家数据已保存到桌面：{os.path.join(desktop_path, 'not_appeared_athletes.csv')}")
print(f"出现在奖牌榜上的国家奖牌数据已保存到桌面：{os.path.join(desktop_path, 'appeared_medal_counts.csv')}")
print(f"未出现在奖牌榜上的国家奖牌数据已保存到桌面：{os.path.join(desktop_path, 'not_appeared_medal_counts.csv')}")