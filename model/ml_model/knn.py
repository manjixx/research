import numpy as np
import math


# 曼哈顿距离
def manhattan(self, a, b):
    return np.sum(np.abs(self.w * (a - b)), axis=1)


# 曼哈顿距离反函数
def manhattan_inverse(self, a, b):
    dist = np.sum(np.abs(self.w * (a - b)), axis=1)
    return 1 / dist


# 曼哈顿高斯
def manhattan_gauss(self, a, b):
    h = 1
    o = 0
    wid = 0.3
    dist = np.sum(np.abs(self.w * (a - b)), axis=1)
    return h * math.e ** (-(dist - o) ** 2 / (2 * wid ** 2))


# 欧式距离
def euclid(self, a, b):
    return np.sqrt(np.sum((self.w * (a - b)) ** 2, axis=1).astype('float'))


# 欧式距离反函数
def euclid_inverse(self, a, b):
    dist = np.sqrt(np.sum((self.w * (a - b)) ** 2, axis=1).astype('float'))
    return 1 / dist


# 高斯距离加权
def euclid_gauss(self, a, b):
    h = 1
    o = 0
    wid = 0.3
    dist = np.sqrt(np.sum((self.w * (a - b)) ** 2, axis=1).astype('float'))
    return h * math.e ** (-(dist - o) ** 2 / (2 * wid ** 2))


class KNN(object):
    """
    分类器的实现
    """

    # 定义类的构造方法
    def __init__(self, x_train, y_train, neighbors=1, distance=euclid, w=1):
        self.neighbors = neighbors
        self.distance = distance
        self.x_train = x_train
        self.y_train = y_train
        self.w = w

    # 预测模型的方法
    def predict(self, x, w):
        # 初始化预测分类数组
        y_pred = np.zeros((x.shape[0], 1), dtype=self.y_train.dtype)
        # 遍历输入的x个测试点,取出每个数据点的序号i和数据x_test
        for i, x_test in enumerate(x):
            # 1.x_test要跟所有训练数据计算距离
            distance = self.distance(self.x_train, x_test, w)
            # 2.得到的距离按照由近到远排序,从近到远对应的索引
            nn_index = np.argsort(distance)
            # 3.选取距离最近的k个点,保存它们的对应类别
            nn_y = self.y_train[nn_index[:self.neighbors]].ravel().astype(int)
            # 4.统计类别中出现频率最高的，赋给y_pred
            y_pred[i] = np.argmax(np.bincount(nn_y))
        return y_pred

    def score(self, x_test, y_test, w):
        right_count = 0
        n = 10
        for X, y in zip(x_test, y_test):
            label = self.predict(X, w)
            if label == y:
                right_count += 1
        return right_count / len(x_test)
