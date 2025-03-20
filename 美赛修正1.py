import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据
athletes_df = pd.read_csv('summerOly_athletes.csv')  # 参赛运动员数据
medals_df = pd.read_csv('summerOly_medal_counts.csv')  # 奖牌统计数据
hosts_df = pd.read_csv('summerOly_hosts.csv')  # 主办国数据
programs_df = pd.read_csv('summerOly_programs.csv')  # 每届奥运会项目数量
# 将国家名转换为缩写（可以通过一个预定义的字典或者外部文件来实现）
country_abbreviations = {
    'United States': 'USA',
    'China': 'CHN',
    'Germany': 'GER',
    'Japan': 'JPN',
    'Russia': 'RUS',
    'France': 'FRA',
    'Great Britain': 'GBR',
    'Italy': 'ITA',
    'Australia': 'AUS',
    'South Korea': 'KOR',
    # 其他国家名转换...
}

# 将奖牌榜数据中的国家名转化为缩写
medals_df['NOC'] = medals_df['NOC'].map(country_abbreviations)
# 强势国家：奖牌数总数前10且金牌数≥8
strong_countries = medals_df[medals_df['Gold'] >= 8].nlargest(10, 'Total')

# 中等国家：出现在20届奥运会奖牌榜上
medium_countries = medals_df.groupby('NOC').filter(lambda x: len(x) > 20)

# 弱势国家：没有历史奖牌的国家
weak_countries = medals_df[~medals_df['NOC'].isin(strong_countries['NOC'])
                           & ~medals_df['NOC'].isin(medium_countries['NOC'])]
# 将数据从宽格式转换为长格式
programs_long_df = pd.melt(programs_df, id_vars=['Sport', 'Discipline', 'Code', 'Sports Governing Body'],
                           var_name='Year', value_name='Events')

# 显示转换后的长格式数据
print(programs_long_df.head())

# 假设 medals_df 是奖牌数据
# 假设 medals_df 已经是读取的奖牌数据并且包含 Year 和 NOC 列

# 合并每届奥运会的参赛项目数量
merged_data = pd.merge(medals_df, programs_df[['Year', 'Event']], on='Year', how='left')

# 合并主办国效应
merged_data = pd.merge(merged_data, hosts_df[['Year', 'HostCountry']], on='Year', how='left')
merged_data['HostEffect'] = merged_data['HostCountry'].apply(lambda x: 1 if x in strong_countries['NOC'].values else 0)

# 强势和中等国家的特征
X_strong_medium = merged_data[['Gold', 'Total', 'Events', 'HostEffect']]
y_gold_strong_medium = merged_data['Gold']
y_total_strong_medium = merged_data['Total']

# 弱势国家的特征
merged_data['HistoricalParticipants'] = merged_data['NOC'].map(athletes_df.groupby('NOC')['AthleteID'].nunique())

X_weak = merged_data[['HistoricalParticipants', 'Events']]
y_gold_weak = merged_data['Gold']
y_total_weak = merged_data['Total']
# 强势和中等国家线性回归模型
regressor_gold_strong_medium = LinearRegression()
regressor_gold_strong_medium.fit(X_strong_medium, y_gold_strong_medium)

regressor_total_strong_medium = LinearRegression()
regressor_total_strong_medium.fit(X_strong_medium, y_total_strong_medium)

# 弱势国家线性回归模型
regressor_gold_weak = LinearRegression()
regressor_gold_weak.fit(X_weak, y_gold_weak)

regressor_total_weak = LinearRegression()
regressor_total_weak.fit(X_weak, y_total_weak)
# 预测
gold_preds_strong_medium = regressor_gold_strong_medium.predict(X_strong_medium)
total_preds_strong_medium = regressor_total_strong_medium.predict(X_strong_medium)

gold_preds_weak = regressor_gold_weak.predict(X_weak)
total_preds_weak = regressor_total_weak.predict(X_weak)

# 评估模型
mse_gold_strong_medium = mean_squared_error(y_gold_strong_medium, gold_preds_strong_medium)
r2_gold_strong_medium = r2_score(y_gold_strong_medium, gold_preds_strong_medium)

mse_total_strong_medium = mean_squared_error(y_total_strong_medium, total_preds_strong_medium)
r2_total_strong_medium = r2_score(y_total_strong_medium, total_preds_strong_medium)

mse_gold_weak = mean_squared_error(y_gold_weak, gold_preds_weak)
r2_gold_weak = r2_score(y_gold_weak, gold_preds_weak)

mse_total_weak = mean_squared_error(y_total_weak, total_preds_weak)
r2_total_weak = r2_score(y_total_weak, total_preds_weak)

# 输出模型评估结果
print("Strong/Medium Country - Gold Medal: MSE =", mse_gold_strong_medium, "R² =", r2_gold_strong_medium)
print("Strong/Medium Country - Total Medal: MSE =", mse_total_strong_medium, "R² =", r2_total_strong_medium)
print("Weak Country - Gold Medal: MSE =", mse_gold_weak, "R² =", r2_gold_weak)
print("Weak Country - Total Medal: MSE =", mse_total_weak, "R² =", r2_total_weak)
