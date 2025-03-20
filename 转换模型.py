import pandas as pd

# 读取medal_counts文件
medal_counts = pd.read_csv('summerOly_medal_counts.csv')

# 读取athlete文件
athlete = pd.read_csv('summerOly_athletes.csv')

# 创建一个从Team（全称）到NOC（简称）的映射字典
team_to_noc = athlete.set_index('Team')['NOC'].to_dict()

# 更新medal_counts中的NOC列
# 如果medal_counts中的NOC列的值在team_to_noc字典中有对应的键，则替换为对应的值
# 如果没有对应的键，则保留原值
medal_counts['NOC'] = medal_counts['NOC'].apply(lambda x: team_to_noc.get(x, x))

# 保存更新后的文件
medal_counts.to_csv('summerOly_medal_counts1.csv', index=False)

print("文件已成功更新并保存为原文件名称。")