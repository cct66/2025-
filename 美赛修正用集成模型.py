import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
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

# **按块处理数据**：避免一次性加载所有数据
chunk_size = 5000  # 每个分块的大小
chunks = [medals_df[i:i + chunk_size] for i in range(0, medals_df.shape[0], chunk_size)]

# 分块合并数据
merged_chunks = []
for chunk in chunks:
    # 仅保留所需的列，避免不必要的列占用内存
    chunk = chunk[['NOC', 'Gold', 'Silver', 'Bronze', 'Total', 'Year']]
    merged_chunk = pd.merge(chunk, athletes_df, on='NOC', how='left')
    merged_chunks.append(merged_chunk)

# 将所有分块合并成一个数据框
merged_data = pd.concat(merged_chunks, ignore_index=True)

# 特征选择：运动员的年龄、性别、是否在本国参赛、历史奖牌数、参赛项目数量
X = merged_data[['YearsSinceFirstParticipation', 'Sex', 'HomeCountryParticipation', 'TotalMedals', 'Events']]
y_gold = merged_data['Gold']
y_total = merged_data['Total']

# **划分训练集和测试集**
X_train, X_test, y_train_gold, y_test_gold = train_test_split(X, y_gold, test_size=0.2, random_state=42)
X_train, X_test, y_train_total, y_test_total = train_test_split(X, y_total, test_size=0.2, random_state=42)

# **子模型1：运动员预测模型（随机森林回归）**
rf_model_gold = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_gold.fit(X_train, y_train_gold)

rf_model_total = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_total.fit(X_train, y_train_total)

# 预测
gold_preds = rf_model_gold.predict(X_test)
total_preds = rf_model_total.predict(X_test)

# 评估模型
mse_gold = mean_squared_error(y_test_gold, gold_preds)
r2_gold = r2_score(y_test_gold, gold_preds)

mse_total = mean_squared_error(y_test_total, total_preds)
r2_total = r2_score(y_test_total, total_preds)

# 输出模型评估结果
print("Gold Medal - MSE =", mse_gold, "R² =", r2_gold)
print("Total Medal - MSE =", mse_total, "R² =", r2_total)

# **子模型2：东道主效应模型（灰色预测）**
# 合并主办国数据和奖牌数据
hosts_medals = medals_df[medals_df['NOC'].isin(hosts_df['Host'])]

# 灰色预测模型（简单示例）
def gray_predict(data):
    if data.empty:
        return 0  # 如果数据为空，返回 0
    return np.mean(data)  # 模拟预测，实际可以采用灰色预测模型

gray_predictions_gold = hosts_medals.groupby('NOC')['Gold'].apply(gray_predict).values
gray_predictions_total = hosts_medals.groupby('NOC')['Total'].apply(gray_predict).values

# 确保 gray_predictions_gold 和 gray_predictions_total 的长度与 X_train 一致
if len(gray_predictions_gold) != X_train.shape[0] or len(gray_predictions_total) != X_train.shape[0]:
    print("gray_predictions_gold 或 gray_predictions_total 的长度与 X_train 不一致，需要检查生成逻辑。")
else:
    # 堆叠法（Stacking）：使用子模型1和子模型2的预测结果作为新的输入特征
    stacked_X = np.column_stack((rf_model_gold.predict(X_train), rf_model_total.predict(X_train), gray_predictions_gold, gray_predictions_total))
    stacked_X_test = np.column_stack((rf_model_gold.predict(X_test), rf_model_total.predict(X_test), gray_predictions_gold, gray_predictions_total))

    # 使用线性回归进行最终预测
    stacked_model = LinearRegression()
    stacked_model.fit(stacked_X, y_train_gold)  # 训练金牌数预测
    stacked_model.fit(stacked_X, y_train_total)  # 训练总奖牌数预测

    # 评估模型
    final_gold_preds = stacked_model.predict(stacked_X_test)
    final_total_preds = stacked_model.predict(stacked_X_test)

    # 输出评估结果
    print("Final Gold Medal Prediction MSE:", mean_squared_error(y_test_gold, final_gold_preds))
    print("Final Total Medal Prediction MSE:", mean_squared_error(y_test_total, final_total_preds))




