import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import linear_regression_diy as lrd

data = pd.read_csv('train.csv')
print(data.head())

# 删除包含缺失值的行
data.dropna(inplace=True)

#数据转换为numpy array
X = data['X'].values.reshape(-1,1)
y = data['y'].values.reshape(-1,1)
#print(type(X))
#print(X,y)
#print(X.shape,y.shape)
#print(X,y)
#print(y.size)
model = lrd.LinearRegression(1, y.size, 0.0003)
# 创建线性回归模型
for i in range(0, 5000):
    model.train_for_one_epoch(X,y)  
    #print(model.getTheta().data)
#print(X,y)


# 打印模型参数
print("斜率 (Theta1):", model.getTheta().data[0])
print("截距 (Theta0):", model.getTheta().data[1])

plt.figure(1)
# 绘制数据点
plt.scatter(X, y, s=30, label="Data Points")

# 绘制线性回归线
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Linear Regression")

# 添加标签和图例
plt.xlabel("X")
plt.ylabel("y")
plt.legend()

# 显示图形
plt.plot()

#测试集
data = pd.read_csv('test.csv')

#数据转换为二维数组
X = data['X']#.values.reshape(-1, 1)
y = data['y']#.values.reshape(-1, 1)

# 删除包含缺失值的行
data.dropna(inplace=True)

plt.figure(2)
# 绘制数据点
plt.scatter(X, y, s=30, label="Data Points")

# 绘制线性回归线
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Linear Regression")

# 添加标签和图例
plt.xlabel("X")
plt.ylabel("y")
plt.legend()

# 显示图形
plt.plot()

