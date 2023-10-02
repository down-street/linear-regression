import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,PowerTransformer, FunctionTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import least_square_diy as lsd
from sklearn.compose import ColumnTransformer

# 1. 读取数据集 https://www.kaggle.com/datasets/mirichoi0218/insurance
data = pd.read_csv('concrete_data.csv')

# 2. 数据预处理
# 2.1 检查表格是否有缺失值，如果有则用该列数据平均值代替
if data.isnull().any().any():
    print("数据中存在缺失值，将使用各列的平均值来填充缺失值。")
    data.fillna(data.mean(), inplace=True)

# # 2.2 独热编码分类特征
# data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)

# 2.3 数据规范化（标准化）
scaler = StandardScaler()
# data[['Cement','Blast Furnace Slag','Fly Ash','Water','Superplasticizer','Coarse Aggregate','Fine Aggregate','Age']] = scaler.fit_transform(data[['Cement','Blast Furnace Slag','Fly Ash','Water','Superplasticizer','Coarse Aggregate','Fine Aggregate','Age']])

# 2.4 检查并处理同名列
if data.columns.duplicated().any():
    print("数据中存在同名列，将它们重命名以便区分。")
    data.columns = [col if data.columns.get_loc(col) == idx else col + f'_{idx}' for idx, col in enumerate(data.columns)]




# 3. 准备数据集
X = data.drop('Strength', axis=1)  # 特征
y = data['Strength']  # 目标变量

# 4. 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
final_selected_features = ['Age', 'Superplasticizer', 'Water', 'Coarse Aggregate', 'Blast Furnace Slag', 'Cement']
transformer = ColumnTransformer(transformers=[
    ('power_transformer',PowerTransformer(),['Cement','Superplasticizer','Water','Coarse Aggregate']),
    ('log_transformer',FunctionTransformer(func=np.log1p,inverse_func=np.expm1),['Age'])
],remainder='passthrough')
X_train = X_train[final_selected_features]
X_test = X_test[final_selected_features]

X_train_tf = transformer.fit_transform(X_train)
X_train_tf = pd.DataFrame(X_train,columns=X_train.columns)
X_test_tf = transformer.transform(X_test)
X_test_tf = pd.DataFrame(X_test,columns=X_test.columns)
features = X_train.columns
final_X_train_tf = scaler.fit_transform(X_train_tf)
final_X_train_tf = pd.DataFrame(final_X_train_tf,columns=features)
final_X_test_tf = scaler.transform(X_test_tf)
final_X_test_tf = pd.DataFrame(final_X_test_tf,columns=features)
X_train = final_X_train_tf
X_test = final_X_test_tf
print(X_train)
# 5. 创建线性回归模型并训练
# model = lsd.MyModel()
model = LinearRegression()
model.fit(X_train, y_train)

# 6. 获取线性回归的预测值
y_pred = model.predict(X_test)

# 7. 打印线性回归的方程、截距和系数
intercept = model.intercept_
coefficients = model.coef_
equation = f"y = {intercept:.2f} + "
equation += " + ".join([f"({coeff:.2f})*{col}" for coeff, col in zip(coefficients, X.columns)])

print("线性回归方程：")
print(equation)
print("截距（intercept）:", intercept)
print("系数（coefficients）:", coefficients)

# 8. 可视化线性回归结果
plt.scatter(y_pred, y_test, color='blue', label='Linear Regression')
plt.plot([min(y_pred), max(y_pred)], [min(y_pred), max(y_pred)], color='red', linewidth=2, label='Linear regression Fitting Line')
plt.xlabel('Predictive Value')
plt.ylabel('Actual Value')
plt.legend()
plt.title('Result of Linear Regression')
plt.savefig('linear_regression_plot.png')
plt.show()

# # 9. 识别和移除异常值，即数据清洗
# # 使用箱线图来识别异常值
# plt.figure(figsize=(12, 6))
# sns.boxplot(data=data[['age', 'bmi', 'children', 'Strength']], orient="h")
# plt.title('Box Plot of Features')
# plt.savefig('box_plot.png')
# plt.show()

# 识别并移除 'Strength' 列中的异常值
# Q1 = data['Strength'].quantile(0.25)
# Q3 = data['Strength'].quantile(0.75)
# IQR = Q3 - Q1
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR
# data = data[(data['Strength'] >= lower_bound) & (data['Strength'] <= upper_bound)]

# # 重新准备数据集，移除异常值后的数据
# X = data.drop('Strength', axis=1)  # 特征
# y = data['Strength']  # 目标变量

# 重新拆分数据集为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 重新创建线性回归模型并训练
# model = LinearRegression()
# model.fit(X_train, y_train)

# # 重新获取线性回归的预测值
# y_pred = model.predict(X_test)

# # 打印重新训练后的线性回归方程、截距和系数
# intercept = model.intercept_
# coefficients = model.coef_
# equation = f"y = {intercept:.2f} + "
# equation += " + ".join([f"({coeff:.2f})*{col}" for coeff, col in zip(coefficients, X.columns)])

# print("重新训练后的线性回归方程：")
# print(equation)
# print("重新训练后的截距（intercept）:", intercept)
# print("重新训练后的系数（coefficients）:", coefficients)

# # 10. 可视化重新训练后的线性回归结果
# plt.scatter(y_pred, y_test, color='blue', label='Linear Regression (Re-trained)')
# plt.plot([min(y_pred), max(y_pred)], [min(y_pred), max(y_pred)], color='red', linewidth=2, label='Linear regression Fitting Line (Re-trained)')
# plt.xlabel('Predictive Value (Re-trained)')
# plt.ylabel('Actual Value')
# plt.legend()
# plt.title('Result of Linear Regression (Re-trained)')
# plt.savefig('linear_regression_plot_retrained.png')
# plt.show()

# 11. 计算皮尔逊系数并绘制热力图
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()

# 12. 计算平均绝对误差（MAE）和均方误差（MSE）
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("平均绝对误差（MAE）:", mae)
print("均方误差（MSE）:", mse)
r_squared = r2_score(y_test, y_pred)
print(f'R方值（决定系数）: {r_squared}')
