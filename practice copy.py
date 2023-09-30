import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import linear_regression_diy as lrd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据集 https://www.kaggle.com/datasets/mirichoi0218/insurance
data = pd.read_csv('insurance.csv')

# 2. 数据预处理
# 2.1 检查表格是否有缺失值，如果有则用该列数据平均值代替
if data.isnull().any().any():
    print("数据中存在缺失值，将使用各列的平均值来填充缺失值。")
    data.fillna(data.mean(), inplace=True)

# 2.2 独热编码分类特征
data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)

# 2.3 数据规范化（标准化）
scaler = StandardScaler()
data[['age', 'bmi', 'children', 'charges']] = scaler.fit_transform(data[['age', 'bmi', 'children', 'charges']])

# 2.4 检查并处理同名列
if data.columns.duplicated().any():
    print("数据中存在同名列，将它们重命名以便区分。")
    data.columns = [col if data.columns.get_loc(col) == idx else col + f'_{idx}' for idx, col in enumerate(data.columns)]

# 3. 准备数据集
X = data.drop('charges', axis=1)  # 特征
y = data['charges']  # 目标变量

# 4. 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 5. 创建线性回归模型并训练
model = lrd.LinearRegression(8, 100, 0.01)
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

# 9. 识别和移除异常值，即数据清洗
# 使用箱线图来识别异常值
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[['age', 'bmi', 'children', 'charges']], orient="h")
plt.title('Box Plot of Features')
plt.savefig('box_plot.png')
plt.show()

# 识别并移除 'charges' 列中的异常值
Q1 = data['charges'].quantile(0.25)
Q3 = data['charges'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['charges'] >= lower_bound) & (data['charges'] <= upper_bound)]

# 重新准备数据集，移除异常值后的数据
X = data.drop('charges', axis=1)  # 特征
y = data['charges']  # 目标变量

# 重新拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 重新创建线性回归模型并训练
model = lrd.LinearRegression(8, 100, 0.01)
model.fit(X_train, y_train)

# 重新获取线性回归的预测值
y_pred = model.predict(X_test)

# 打印重新训练后的线性回归方程、截距和系数
intercept = model.intercept_
coefficients = model.coef_
equation = f"y = {intercept:.2f} + "
equation += " + ".join([f"({coeff:.2f})*{col}" for coeff, col in zip(coefficients, X.columns)])

print("重新训练后的线性回归方程：")
print(equation)
print("重新训练后的截距（intercept）:", intercept)
print("重新训练后的系数（coefficients）:", coefficients)

# 10. 可视化重新训练后的线性回归结果
plt.scatter(y_pred, y_test, color='blue', label='Linear Regression (Re-trained)')
plt.plot([min(y_pred), max(y_pred)], [min(y_pred), max(y_pred)], color='red', linewidth=2, label='Linear regression Fitting Line (Re-trained)')
plt.xlabel('Predictive Value (Re-trained)')
plt.ylabel('Actual Value')
plt.legend()
plt.title('Result of Linear Regression (Re-trained)')
plt.savefig('linear_regression_plot_retrained.png')
plt.show()

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
