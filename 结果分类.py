import pandas as pd
import os

# 获取桌面路径（Windows系统）
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# 读取两个文件
predictions = pd.read_excel('predictions_2028.xlsx')
summer_oly = pd.read_csv('summerOly_medal_counts.csv')

# 提取summerOly_medal_counts.csv中的所有NOC
summer_oly_noc = summer_oly['NOC'].unique()

# 筛选出出现在summerOly_medal_counts.csv中的国家
predictions_in_summer_oly = predictions[predictions['NOC'].isin(summer_oly_noc)]

# 筛选出未出现在summerOly_medal_counts.csv中的国家
predictions_not_in_summer_oly = predictions[~predictions['NOC'].isin(summer_oly_noc)]

# 输出结果
print("出现在summerOly_medal_counts.csv中的国家：")
print(predictions_in_summer_oly)

print("\n未出现在summerOly_medal_counts.csv中的国家：")
print(predictions_not_in_summer_oly)

# 保存到桌面
predictions_in_summer_oly.to_excel(os.path.join(desktop_path, 'predictions_in_summer_oly.xlsx'), index=False)
predictions_not_in_summer_oly.to_excel(os.path.join(desktop_path, 'predictions_not_in_summer_oly.xlsx'), index=False)

print(f"文件已保存到桌面：{desktop_path}")