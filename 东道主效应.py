import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据
medal_counts = pd.read_csv('summerOly_medal_counts.csv')
hosts = pd.read_csv('summerOly_hosts.csv')

# 修复除以零错误
def calculate_host_boost(noc):
    host_years = hosts[hosts['Host'] == noc]['Year'].values
    boosts = []
    for year in host_years:
        prev1 = medal_counts[(medal_counts['NOC'] == noc) & (medal_counts['Year'] == year - 4)]
        prev2 = medal_counts[(medal_counts['NOC'] == noc) & (medal_counts['Year'] == year - 8)]
        current = medal_counts[(medal_counts['NOC'] == noc) & (medal_counts['Year'] == year)]
        if not prev1.empty and not prev2.empty and not current.empty:
            avg_prev_gold = (prev1['Gold'].values[0] + prev2['Gold'].values[0]) / 2
            if avg_prev_gold != 0:  # 检查是否为零
                boost_gold = current['Gold'].values[0] / avg_prev_gold
                boosts.append(boost_gold)
    return np.mean(boosts) if boosts else 1.0  # 默认增益为1（无增益）

# 计算所有国家的东道主增益
host_boost = {}
for noc in hosts['Host'].unique():
    boost = calculate_host_boost(noc)
    host_boost[noc] = {'Gold_Boost': boost}

# 添加历史特征
def create_historical_features(df, years_ago=4):
    df = df.copy()
    for year_ago in [years_ago]:  # 可以扩展为多个历史年份
        df_shifted = df[['Year', 'NOC', 'Gold', 'Total']].copy()
        df_shifted['Year'] = df_shifted['Year'] + year_ago
        df_shifted.rename(columns={
            'Gold': f'Gold_Past_{year_ago}Y',
            'Total': f'Total_Past_{year_ago}Y'
        }, inplace=True)
        df = pd.merge(df, df_shifted, on=['Year', 'NOC'], how='left')
    return df.fillna(0)  # 用0填充缺失值

medal_counts = create_historical_features(medal_counts)

# 合并东道主信息
medal_counts = pd.merge(medal_counts, hosts.rename(columns={'Host': 'NOC'}), on=['Year', 'NOC'], how='left')
medal_counts['IsHost'] = medal_counts['Year'].isin(hosts['Year']).astype(int)

# 添加增益系数
medal_counts['Gold_Boost'] = 1.0
for idx, row in medal_counts.iterrows():
    if row['IsHost'] == 1 and row['NOC'] in host_boost:
        medal_counts.at[idx, 'Gold_Boost'] = host_boost[row['NOC']]['Gold_Boost']

# 特征和目标变量
features = ['Year', 'Gold_Past_4Y', 'Total_Past_4Y', 'IsHost', 'Gold_Boost']
target_gold = 'Gold'

# 划分训练集和测试集
train_data = medal_counts[medal_counts['Year'] < 2020]
test_data = medal_counts[medal_counts['Year'] >= 2020]

X_train = train_data[features]
y_train = train_data[target_gold]
X_test = test_data[features]
y_test = test_data[target_gold]

# 训练模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测并评估
y_pred = model.predict(X_test)
print(f'MSE: {mean_squared_error(y_test, y_pred)}, R²: {r2_score(y_test, y_pred)}')

# 创建2028年数据
future_data = medal_counts[medal_counts['Year'] == 2024].copy()
future_data['Year'] = 2028
future_data['IsHost'] = 0
future_data.loc[future_data['NOC'] == 'USA', 'IsHost'] = 1
future_data.loc[future_data['NOC'] == 'USA', 'Gold_Boost'] = host_boost['USA']['Gold_Boost']

# 预测
future_pred_gold = model.predict(future_data[features])
future_data['Predicted_Gold_2028'] = future_pred_gold

# 导出预测结果
future_data[['NOC', 'Predicted_Gold_2028']].to_excel('predictions_2028_with_host_effect.xlsx', index=False)
print("预测结果已保存至 predictions_2028_with_host_effect.xlsx")
def bootstrap_confidence_interval(model, X_train, y_train, X_test, n_bootstrap=1000, alpha=0.05):
    """
    使用Bootstrap方法计算预测值的置信区间
    :param model: 基础模型（如RandomForestRegressor）
    :param X_train: 训练集特征
    :param y_train: 训练集目标
    :param X_test: 测试集特征
    :param n_bootstrap: Bootstrap样本数量
    :param alpha: 置信水平（默认95%）
    :return: 预测值的下界和上界
    """
    predictions = []
    for _ in range(n_bootstrap):
        # 重采样训练数据
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_sample = X_train.iloc[indices]
        y_sample = y_train.iloc[indices]

        # 训练模型并预测
        model_sample = RandomForestRegressor(n_estimators=100, random_state=42)
        model_sample.fit(X_sample, y_sample)
        y_pred = model_sample.predict(X_test)
        predictions.append(y_pred)

    # 计算置信区间
    lower = np.percentile(predictions, (alpha / 2) * 100, axis=0)
    upper = np.percentile(predictions, (1 - alpha / 2) * 100, axis=0)
    return lower, upper
# 使用Bootstrap计算置信区间
lower_gold, upper_gold = bootstrap_confidence_interval(model, X_train, y_train, future_data[features])

# 将置信区间添加到预测结果中
future_data['Gold_Lower_95CI'] = lower_gold
future_data['Gold_Upper_95CI'] = upper_gold

# 导出包含置信区间的预测结果
future_data[['NOC', 'Predicted_Gold_2028', 'Gold_Lower_95CI', 'Gold_Upper_95CI']].to_excel(
    'predictions_2028_with_confidence_intervals.xlsx', index=False
)
print("包含置信区间的预测结果已保存至 predictions_2028_with_confidence_intervals.xlsx")