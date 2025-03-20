import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# 加载数据
athletes_df = pd.read_csv('summerOly_athletes.csv')  # 运动员数据
medals_df = pd.read_csv('summerOly_medal_counts.csv')  # 奖牌统计数据
hosts_df = pd.read_csv('summerOly_hosts.csv')  # 主办国数据
programs_df = pd.read_csv('summerOly_programs.csv')  # 每届奥运会项目数量

# **减少数据列**：只选择需要的列来减少内存使用
athletes_df = athletes_df[['Name', 'Sex', 'NOC', 'Year', 'Medal']]
medals_df = medals_df[['NOC', 'Gold', 'Silver', 'Bronze', 'Total', 'Year']]

# **转换数据类型以节省内存**：转换为更小的数据类型
for col in ['Gold', 'Silver', 'Bronze', 'Total', 'Year']:
    if col in medals_df.columns:
        medals_df[col] = pd.to_numeric(medals_df[col], downcast='integer')

# **计算运动员的参赛经验（首次参赛年份到当前参赛年份的差值）**
athletes_df['FirstParticipation'] = athletes_df.groupby('Name')['Year'].transform('min')
athletes_df['YearsSinceFirstParticipation'] = athletes_df['Year'] - athletes_df['FirstParticipation']

# **处理性别：假设 'M' 表示男性，'F' 表示女性**
athletes_df['Sex'] = athletes_df['Sex'].apply(lambda x: 1 if x == 'M' else 0)

# **判断运动员是否在本国参赛：如果运动员的 NOC 与主办国一致，则为 1，否则为 0**
athletes_df['HomeCountryParticipation'] = athletes_df['NOC'].apply(lambda x: 1 if x in hosts_df['Host'].values else 0)

# **汇总每个运动员的奖牌数量**
athletes_df['TotalMedals'] = athletes_df.groupby('Name')['Medal'].transform('count')

# **提取运动项目数据，转换为长格式**
programs_long_df = pd.melt(programs_df, id_vars=['Sport', 'Discipline', 'Code', 'Sports Governing Body'],
                           var_name='Year', value_name='Events')

# 由于年份是字符串类型，需要转换为整数
programs_long_df['Year'] = programs_long_df['Year'].astype(int)

# **将每个运动员的参赛项目数量合并到运动员数据中**
athletes_df = pd.merge(athletes_df, programs_long_df[['Year', 'Events']], on='Year', how='left')

# **按年份分割数据**
train_data = athletes_df[(athletes_df['Year'] >= 2000) & ( athletes_df['Year'] <= 2020)]
test_data = athletes_df[(athletes_df['Year'] == 2024)]

# **合并训练数据和奖牌数据**
train_data = pd.merge(train_data, medals_df[(medals_df['Year'] <= 2000) & (medals_df['Year'] >= 2000)], on=['NOC', 'Year'], how='left')
test_data = pd.merge(test_data, medals_df[medals_df['Year'].isin([ 2024])], on=['NOC', 'Year'], how='left')

# **处理 NaN 值**
train_data.dropna(subset=['Gold', 'Total', 'TotalMedals', 'Events'], inplace=True)
test_data.dropna(subset=['Gold', 'Total', 'TotalMedals', 'Events'], inplace=True)

# 特征选择：距离第一次参赛时间、性别、是否在本国参赛、历史奖牌数、参赛项目数量
X_train = train_data[['YearsSinceFirstParticipation', 'Sex', 'HomeCountryParticipation', 'TotalMedals', 'Events']]
y_train_gold = train_data['Gold']
y_train_total = train_data['Total']

X_test = test_data[['YearsSinceFirstParticipation', 'Sex', 'HomeCountryParticipation', 'TotalMedals', 'Events']]
y_test_gold = test_data['Gold']
y_test_total = test_data['Total']

# **标准化特征**
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# **训练XGBoost模型**
xgb_model_gold = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model_gold.fit(X_train_scaled, y_train_gold)

xgb_model_total = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model_total.fit(X_train_scaled, y_train_total)

# **预测**
gold_preds_xgb = xgb_model_gold.predict(X_test_scaled)
total_preds_xgb = xgb_model_total.predict(X_test_scaled)

# **东道主效应模型（灰色预测）**
def gray_predict(data):
    """
    改进的灰色预测模型 GM(1,1)
    """
    if len(data) < 2:
        return np.mean(data)  # 少于两个数据点时返回均值预测

    # 将 Pandas Series 转换为 NumPy 数组
    data_cumsum = np.cumsum(data.values)  # 使用 .values 将 Series 转换为 NumPy 数组
    B = np.vstack([-0.5 * (data_cumsum[i] + data_cumsum[i+1]) for i in range(len(data_cumsum) - 1)])
    Y = data.values[1:]  # 使用 .values 提取 NumPy 数组

    # 使用最小二乘法求解参数
    a = np.linalg.lstsq(B, Y, rcond=None)[0]
    a = a[0]  # 提取系数

    # 构建预测模型
    def predict(n):
        x0 = data.values[0]  # 初始值
        return x0 * (1 - np.exp(a)) * np.exp(-a * n)

    # 生成预测序列
    predictions = [predict(i) for i in range(len(data) + 1)]
    return predictions[-1]  # 返回最后一个预测值

# 对主办国进行灰色预测
hosts_medals = medals_df[medals_df['NOC'].isin(hosts_df['Host'])]
gray_predictions_gold = hosts_medals.groupby('NOC')['Gold'].apply(gray_predict).values
gray_predictions_total = hosts_medals.groupby('NOC')['Total'].apply(gray_predict).values

# **确保 gray_predictions 的长度与测试集一致**
# 扩展 gray_predictions 的长度以匹配测试集
repeat_times_gold = len(y_test_gold) // len(gray_predictions_gold) + 1
repeat_times_total = len(y_test_total) // len(gray_predictions_total) + 1

gray_predictions_gold = np.tile(gray_predictions_gold, repeat_times_gold)[:len(y_test_gold)]
gray_predictions_total = np.tile(gray_predictions_total, repeat_times_total)[:len(y_test_total)]

# 堆叠数组
stacked_X = np.column_stack((gold_preds_xgb, total_preds_xgb, gray_predictions_gold, gray_predictions_total))

# **预测训练集**
gold_preds_xgb_train = xgb_model_gold.predict(X_train_scaled)
total_preds_xgb_train = xgb_model_total.predict(X_train_scaled)

# 扩展 gray_predictions 的长度以匹配训练集
repeat_times_gold_train = len(y_train_gold) // len(gray_predictions_gold) + 1
repeat_times_total_train = len(y_train_total) // len(gray_predictions_total) + 1

gray_predictions_gold_train = np.tile(gray_predictions_gold, repeat_times_gold_train)[:len(y_train_gold)]
gray_predictions_total_train = np.tile(gray_predictions_total, repeat_times_total_train)[:len(y_train_total)]

stacked_X_train = np.column_stack((gold_preds_xgb_train, total_preds_xgb_train, gray_predictions_gold_train, gray_predictions_total_train))

# **使用线性回归进行最终预测**
stacked_model_gold = LinearRegression()
stacked_model_gold.fit(stacked_X_train, y_train_gold)  # 训练金牌数预测

stacked_model_total = LinearRegression()
stacked_model_total.fit(stacked_X_train, y_train_total)  # 训练总奖牌数预测

# **评估模型**
final_gold_preds = stacked_model_gold.predict(stacked_X)
final_total_preds = stacked_model_total.predict(stacked_X)

# 输出评估结果
mse_gold = mean_squared_error(y_test_gold, final_gold_preds)
r2_gold = r2_score(y_test_gold, final_gold_preds)

mse_total = mean_squared_error(y_test_total, final_total_preds)
r2_total = r2_score(y_test_total, final_total_preds)

print("Stacked Model with Gray Prediction - Gold Medal - MSE =", mse_gold, "R² =", r2_gold)
print("Stacked Model with Gray Prediction - Total Medal - MSE =", mse_total, "R² =", r2_total)