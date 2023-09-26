import linear_regression_diy as lrd
import matplotlib.pyplot as plt
import numpy as np
import random

model = lrd.LinearRegression(4, 100, 0.3)

k = 32
k2 = 51
k3 = 12
k4 = 13
b = -65

x = []
y = []
for i in range(0, 100):
    cur = random.uniform(0, 1)
    cur2 = random.uniform(0, 2)
    cur3 = random.uniform(0, 1)
    cur4 = random.uniform(0, 3)
    x.append([cur, cur2, cur3, cur4])
    y.append([k * cur + k2 * cur2 + k3 * cur3 + k4 * cur4 +  b ])


# X = [xx[0] for xx in x]
# Y = [yy[0] for yy in y]
# fig, ax = plt.subplots()
# ax.scatter(X, Y)
#print(x,y)

for i in range(0, 5000):
    model.train_for_one_epoch(np.array(x), np.array(y))
    print(model.getTheta().data)