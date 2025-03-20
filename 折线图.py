import pandas as pd
import matplotlib.pyplot as plt

# 创建模拟数据（从1996年开始，仅金牌数）
data = {
    'Year': [1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024],
    'United States': [44, 37, 36, 36, 46, 46, 39, 40],  # 美国的金牌数
    'China': [16, 28, 32, 48, 39, 26, 38, 40],         # 中国的金牌数
    'Great Britain': [1, 11, 9, 19, 29, 27, 22, 14],   # 英国的金牌数
    'France': [15, 13, 11, 7, 11, 10, 10, 16],         # 法国的金牌数
    'Australia': [9, 16, 17, 14, 8, 8, 17, 18],        # 澳大利亚的金牌数
    'Japan': [3, 5, 16, 9, 7, 12, 27, 20],             # 日本的金牌数
    'Brazil': [3, 0, 5, 3, 3, 7, 7, 3]                # 巴西的金牌数
}

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 将Year列设置为索引
df.set_index('Year', inplace=True)

# 绘制折线图
plt.figure(figsize=(12, 8))

for country in df.columns:
    plt.plot(df.index, df[country], label=country, marker='o')  # 使用marker标记数据点

plt.xlabel('Year')
plt.ylabel('Gold Medals')
plt.title('Gold Medals Over Time (1996-2024)')
plt.legend()
plt.grid(True)
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# 创建模拟数据（从1996年开始）
data = {
    'Year': [1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024],
    'United States': [101, 93, 101, 112, 104, 121, 113, 126],
    'China': [50, 58, 63, 100, 92, 70, 89, 91],
    'Great Britain': [15, 28, 30, 51, 65, 67, 64, 65],
    'France': [37, 38, 33, 43, 35, 42, 33, 64],
    'Australia': [41, 58, 50, 46, 35, 29, 46, 53],
    'Japan': [14, 18, 37, 25, 38, 41, 58, 45],
    'Brazil': [15, 12, 10, 17, 17, 19, 21, 20]
}

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 将Year列设置为索引
df.set_index('Year', inplace=True)

# 绘制折线图
plt.figure(figsize=(12, 8))

for country in df.columns:
    plt.plot(df.index, df[country], label=country, marker='o')  # 使用marker标记数据点

plt.xlabel('Year')
plt.ylabel('Total Medals')
plt.title('Total Medals Over Time (1996-2024)')
plt.legend()
plt.grid(True)
plt.show()