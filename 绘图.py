import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
medals_df = pd.read_csv('summerOly_medal_counts.csv')  # 奖牌统计数据
athletes_df = pd.read_csv('summerOly_athletes.csv')  # 运动员数据

# 数据预处理
medals_df['Total'] = medals_df['Gold'] + medals_df['Silver'] + medals_df['Bronze']

# 检查列名
print(medals_df.columns)
print(athletes_df.columns)

# 如果 medals_df 中没有 'Sport' 列，从 athletes_df 中获取
if 'Sport' not in medals_df.columns:
    merged_df = pd.merge(medals_df, athletes_df[['NOC', 'Year', 'Sport']], on=['NOC', 'Year'], how='left')
else:
    merged_df = medals_df

# 按国家和体育项目分组并计算总奖牌数
country_sport_medals = merged_df.groupby(['NOC', 'Sport'])['Total'].sum().reset_index()
# 找出每个国家的前3个优势项目
top_sports_per_country = country_sport_medals.groupby('NOC').apply(lambda x: x.nlargest(3, 'Total')).reset_index(drop=True)

# 查看结果
print(top_sports_per_country.head(20))
# 选择前10个国家
top_countries = top_sports_per_country['NOC'].unique()[:10]
filtered_data = top_sports_per_country[top_sports_per_country['NOC'].isin(top_countries)]

# 绘制柱状图
plt.figure(figsize=(14, 10))
sns.barplot(x='Total', y='Sport', hue='NOC', data=filtered_data, orient='h', palette='viridis')
plt.title('Top 3 Sports by Medal Count for Top 10 Countries')
plt.xlabel('Total Medals')
plt.ylabel('Sport')
plt.legend(title='Country')
plt.show()
# 创建一个透视表，展示各国在不同体育项目中的奖牌数
country_sport_pivot = country_sport_medals.pivot_table(index='NOC', columns='Sport', values='Total', aggfunc='sum', fill_value=0)

# 选择前10个国家
country_sport_pivot = country_sport_pivot.loc[top_countries]

# 绘制热力图
plt.figure(figsize=(14, 10))
sns.heatmap(country_sport_pivot, annot=True, fmt=".0f", cmap='YlGnBu', cbar_kws={'label': 'Total Medals'})
plt.title('Medal Distribution by Country and Sport')
plt.xlabel('Sport')
plt.ylabel('Country')
plt.show()