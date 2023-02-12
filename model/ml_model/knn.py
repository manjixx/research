import numpy as np
import pandas as pd
import random
import math
from sklearn.metrics import *
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# 曼哈顿距离
def manhattan(a, b, w):
    return np.sum(np.abs(w * (a - b)), axis=1)


# 曼哈顿距离反函数
def manhattan_inverse(a, b, w):
    w = 1
    dist = np.sum(np.abs(w * (a - b)), axis=1)
    return 1 / dist


# 曼哈顿高斯
def manhattan_gaus(a, b, w):
    h = 1
    o = 0
    wid = 0.3
    dist = np.sum(np.abs(w * (a - b)), axis=1)
    return h * math.e ** (-(dist - o) ** 2 / (2 * wid ** 2))


# 欧式距离
def euclid(a, b, w):
    return np.sqrt(np.sum((w * (a - b)) ** 2, axis=1).astype('float'))


# 欧式距离反函数
def euclid_inverse(a, b, w):
    w = 1
    dist = np.sqrt(np.sum((w * (a - b)) ** 2, axis=1).astype('float'))
    return 1 / dist


# 高斯距离加权
def euclid_gaus(a, b, w):
    h = 1
    o = 0
    wid = 0.3
    dist = np.sqrt(np.sum((w * (a - b)) ** 2, axis=1).astype('float'))
    return h * math.e ** (-(dist - o) ** 2 / (2 * wid ** 2))


class KNN(object):
    """
    分类器的实现
    """

    # 定义类的构造方法
    def __init__(self, x_train, y_train, n_neighbors=1, dis_fun=euclid, w=1):
        self.n_neighbors = n_neighbors
        self.dis = dis_fun
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
            distance = self.dis(self.x_train, x_test, w)
            # 2.得到的距离按照由近到远排序,从近到远对应的索引
            nn_index = np.argsort(distance)
            # 3.选取距离最近的k个点,保存它们的对应类别
            nn_y = self.y_train[nn_index[:self.n_neighbors]].ravel().astype(int)
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


def knn(x_train, y_train, x_test_list, y_test_list, neighbors):
    distance = [manhattan,  euclid]
    name = ['low', 'mid', 'high']
    result = []

    for d in distance:
        knn = KNN(x_train, y_train)
        # 网格步长
        h = .02
        # 创建色彩图
        cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
        cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

        dis_name = str(d).split(' ')[1]

        print("距离函数为：" + dis_name)

        for i in range(0, len(x_test_list)):
            x_test = np.array(x_test_list[i][:, 0:2], dtype=np.float64)  # can use np.float32 as well
            w = x_test_list[i][0][-1]
            y_test = np.array(y_test_list[i])

            k_range = []  # 设置循环次数
            k_error = []

            # 考虑到不同的k值，尽量选奇数,所以步长为2
            for k in range(1, neighbors, 2):
                print("第{}个数据集".format(str(i)) + "近邻数为:" + str(k))
                knn.dis = d
                knn.n_neighbors = k
                y_pred = knn.predict(x_test, w)
                # 计算准确率
                accuracy = accuracy_score(y_test, y_pred)
                print("准确率为:" + str(accuracy))
                k_range.append(k)
                k_error.append(accuracy)

                result.append([name[i], k, dis_name, accuracy])

                # 绘制决策边界
                x_min, x_max = x_test[:, -2].min() - 1, x_test[:, -2].max() + 1
                y_min, y_max = x_test[:, -1].min() - 1, x_test[:, -1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                     np.arange(y_min, y_max, h))

                Z = knn.predict(np.c_[xx.ravel(), yy.ravel()], w).reshape(xx.shape)
                plt.figure()
                plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

                # 绘制训练点
                plt.scatter(x_test[:, -2], x_test[:, -1], c=y_test, cmap=cmap_bold, edgecolor='k', s=20)
                plt.xlim(xx.min(), xx.max())
                plt.ylim(yy.min(), yy.max())
                plt.show()
            # 画图，x轴为k值，y值为误差值
            plt.plot(k_range, k_error)
            plt.xlabel('Value of K in KNN')
            plt.ylabel('Accuracy')
            plt.show()
        df = pd.DataFrame(result, columns=['name', 'k', '距离函数', '准确率'])
    return df


def train_data_knn(x_train_pmv, y_train_pmv, x_train_list, y_train_list):
    x_train = []
    y_train = []

    for i in range(0, len(x_train_pmv)):
        x_train.append(x_train_pmv[i])
        y_train.append(y_train_pmv[i])

    for i in range(0, 3):
        x = x_train_list[i]
        y = y_train_list[i]
        for j in range(0, len(x)):
            x_train.append(x[i])
            y_train.append(y[i])

    x_train = np.array(x_train, dtype=object)
    y_train = np.array(y_train, dtype=object).astype(int)

    return x_train, y_train


def split_pmv_data(data, count):
    x_grade = []
    y_grade = []
    if count == 0:
        return x_grade, y_grade
    pmv = data[['ta', 'hr', 'pmv']]

    arr = random.sample(list(np.array(pmv)), count)
    arr = np.array(arr)
    x_train = arr[:, 0:2]
    x_grade = preprocessing.MaxAbsScaler().fit_transform(x_train)
    y_grade = arr[:, 2:3]
    return x_grade, y_grade


def split_filed_data(x_list, y_list, w, testsize):
    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []

    for i in range(0, len(x_list)):
        x = x_list[i]
        y = y_list[i]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testsize, random_state=35, stratify=y)
        x_test = preprocessing.MaxAbsScaler().fit_transform(x_test)
        x_train = preprocessing.MaxAbsScaler().fit_transform(x_train)
        x_test = np.insert(x_test, 2, values=w[i], axis=1)
        x_train_list.append(x_train)
        y_train_list.append(y_train)
        x_test_list.append(x_test)
        y_test_list.append(y_test)

    return x_train_list, y_train_list, x_test_list, y_test_list
