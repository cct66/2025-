import matplotlib.pyplot as plt
import numpy as np

# 数据准备
years = [2016, 2020, 2024, 2028]
countries = ["USA", "CHN", "JPN", "GBR", "AUS", "NED", "FRA", "KOR", "ITA", "GER"]

# 奖牌数据（每个国家在每个年份的金牌数和总奖牌数）
medal_data = {
    "USA": [(46, 121), (39, 113), (40, 126), (35, 106)],
    "CHN": [(26, 70), (38, 89), (40, 91), (39, 91)],
    "JPN": [(12, 41), (27, 58), (20, 45), (25, 53)],
    "GBR": [(22, 67), (22, 64), (14, 65), (28, 68)],
    "AUS": [(17, 46), (17, 46), (18, 53), (16, 44)],
    "NED": [(8, 20), (10, 36), (15, 34), (11, 32)],
    "FRA": [(10, 42), (10, 33), (16, 64), (10, 56)],
    "KOR": [(9, 21), (6, 20), (13, 32), (8, 32)],
    "ITA": [(8, 28), (10, 40), (12, 40), (15, 45)],
    "GER": [(17, 42), (10, 37), (12, 33), (11, 37)],
}

# 设置柱状图的位置和宽度
bar_width = 0.2
index = np.arange(len(countries))

# 创建图形和轴
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

# 绘制金牌数柱状图
for i, year in enumerate(years):
    gold = [medal_data[country][i][0] for country in countries]
    ax1.bar(index + i * bar_width, gold, bar_width, label=f'{year} Gold', color=plt.cm.tab20(i * 3))

# 设置x轴刻度标签
ax1.set_xticks(index + bar_width * 1.5)
ax1.set_xticklabels(countries)

# 添加标题和标签
ax1.set_title('Olympic Gold Medals Comparison (2016-2028)', fontsize=14)
ax1.set_ylabel('Number of Gold Medals', fontsize=12)
ax1.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

# 绘制总奖牌数柱状图
for i, year in enumerate(years):
    total = [medal_data[country][i][1] for country in countries]
    ax2.bar(index + i * bar_width, total, bar_width, label=f'{year} Total', color=plt.cm.tab20(i * 3 + 1))

# 添加标题和标签
ax2.set_title('Olympic Total Medals Comparison (2016-2028)', fontsize=14)
ax2.set_xlabel('Countries', fontsize=12)
ax2.set_ylabel('Total Medals', fontsize=12)
ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

# 显示图形
plt.tight_layout()
plt.show()
