import numpy as np

class LinearRegression(object):
    """
    线性回归模型
    初始化时定义自变量个数、batch_size、学习率
    """
    def __init__(self, feature_nums: int, batch_size: int, learning_rate: float):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.feature_nums = feature_nums

        self.theta = Parameter(feature_nums + 1, 1)

    def train_for_one_epoch(self, x, y):
        assert isinstance(x, np.ndarray), "Input should be a numpy array, instead has type {}".format(type(x).__name__)
        assert isinstance(y, np.ndarray), "Label should be a numpy array, instead has type {}".format(type(y).__name__)

        assert x.shape == (self.batch_size, self.feature_nums), "Expected input shape ({}, {}), got {}".format(self.batch_size, self.feature_nums, x.shape)
        assert y.shape == (self.batch_size, 1), "Expected input shape({}, 1), got {}".format(self.batch_size, y.shape)

        # 为常数项增加一个恒为1的feature
        x = np.array([np.concatenate((row, [1])) for row in x])
        regression = np.dot(x, self.theta.data)

        # 求算loss的梯度并更新参数
        self.theta.update(self.learning_rate, np.dot(x.T, (regression - y)) / self.batch_size)

    def getTheta(self):
        return self.theta

class Parameter(object):
    """
    线性回归模型的参数
    """
    def __init__(self, *shape):
        assert len(shape) == 2, "Expected a two-dimension shape, got {}".format(len(shape))
        # 初始化参数
        limit = np.sqrt(3.0 / np.mean(shape))
        self.data = np.random.uniform(low=-limit, high=limit, size=shape)

    def update(self, alpha, gradient):
        assert gradient.shape == self.data.shape, "Expected gradient shape {}, got {}".format(self.data.shape, gradient.shape)
        self.data = self.data - alpha * gradient