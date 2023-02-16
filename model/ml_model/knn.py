import random
from sklearn.model_selection import train_test_split
from model.utils import *


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
def euclid_gauss(a, b, w):
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
