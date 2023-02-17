import numpy as np
import pandas as pd
import copy
from scipy import special
from sklearn import preprocessing
import math
import matplotlib.pyplot as plt
from sklearn.metrics import *
from matplotlib.colors import ListedColormap


def read_data(file_path, data_type, season, algorithm):
    """
    按照季节读取数据
    :param features:
    :param algorithm:
    :param data_type:
    :param file_path:
    :param season: 读取季节
    :return:
    """
    data = pd.read_csv(file_path, encoding='gbk')
    data = data.dropna(axis=0, how='any', inplace=False)

    if season:
        data = data.loc[data['season'] == season]

    if data_type == 'pmv':
        index = 'pmv'
    else:
        index = 'thermal sensation'
    if algorithm == 'knn':
        data.loc[(data[index] > 0.5), index] = 2
        data.loc[(-0.5 <= data[index]) & (data[index] <= 0.5), index] = 1
        data.loc[(data[index] < -0.5), index] = 0
    elif algorithm == 'svm':
        data.loc[(data[index] > 0.5), index] = 1
        data.loc[((-0.5 <= data[index]) & (data[index] <= 0.5)), index] = 0
        data.loc[(data[index] < -0.5), index] = -1
        data = data[(data[index] != 0)]
    return data


def split(data, target, index):
    if index == 'bmi':
        low_x = data[(data[index] <= 18)]
        low_y = target[(target[index] <= 18)]
        mid_x = target[(target[index] < 24) & (data[index] > 18)]
        mid_y = data[(data[index] < 24) & (data[index] > 18)]
        high_x = data[(data[index] >= 24)]
        high_y = data[(data[index] >= 24)]
    elif index == 'griffith':
        low_x = data[(data[index] <= down)]
        low_y = data[(data[index] <= down)]
        mid_x = data[(data[index] < up) & (data[index] > down)]
        mid_y = data[(data[index] < up) & (data[index] > down)][[y_features]]
        high_x = data[(data[index] >= up)][x_features]
        high_y = data[(data[index] >= up)][[y_features]]
    elif index == 'preference':
        low_x = data[(data[index] == -1)][x_features]
        low_y = data[(data[index] == -1)][[y_features]]
        mid_x = data[(data[index] == 0)][x_features]
        mid_y = data[(data[index] == 0)][[y_features]]
        high_x = data[(data[index] == 1)][x_features]
        high_y = data[(data[index] == 1)][[y_features]]
    elif index == 'sensitivity':
        low_x = data[(data[index] == 0)][x_features]
        low_y = data[(data[index] == 0)][[y_features]]
        mid_x = data[(data[index] == 1)][x_features]
        mid_y = data[(data[index] == 1)][[y_features]]
        high_x = data[(data[index] == 2)][x_features]
        high_y = data[(data[index] == 2)][[y_features]]

    low_y = low_y.astype(int)
    mid_y = mid_y.astype(int)
    high_y = high_y.astype(int)

    x_list = [low_x, mid_x, high_x]
    y_list = [low_y, mid_y, high_y]

    for i in range(0, 3):
        x = x_list[i]
        y_list[i] = y_list[i].stack()
        y_list[i] = np.array(y_list[i])
        # 归一化
        x_list[i] = preprocessing.MaxAbsScaler().fit_transform(x)

    return x_list, y_list


def sign(x):
    """
    符号函数
    :param x:
    :return:
    """
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def sigmoid(x2):
    """
    sigmoid函数
    :param x2:
    :return:
    """
    x = copy.deepcopy(x2)
    if type(x) is int:
        x = 20.0 if x > 20.0 else x
        x = -100.0 if x < -100.0 else x
    else:
        # 避免下溢
        x[x > 20.0] = 20.0
        # 避免上溢
        x[x < -100.0] = -100.0
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """
    softmax函数
    :param x:
    :return:
    """
    if x.ndim == 1:
        return np.exp(x) / np.exp(x).sum()
    else:
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)


def entropy(x, sample_weight=None):
    """
    计算熵
    :param x:
    :param sample_weight:
    :return:
    """
    x = np.asarray(x)
    # x中元素个数
    x_num = len(x)
    # 如果sample_weight为None设均设置一样
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    x_counter = {}
    weight_counter = {}
    # 统计各x取值出现的次数以及其对应的sample_weight列表
    for index in range(0, x_num):
        x_value = x[index]
        if x_counter.get(x_value) is None:
            x_counter[x_value] = 0
            weight_counter[x_value] = []
        x_counter[x_value] += 1
        weight_counter[x_value].append(sample_weight[index])

    # 计算熵
    ent = .0
    for key, value in x_counter.items():
        p_i = 1.0 * value * np.mean(weight_counter.get(key)) / x_num
        ent += -p_i * math.log(p_i)
    return ent


def cond_entropy(x, y, sample_weight=None):
    """
    计算条件熵:H(y|x)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    # x中元素个数
    x_num = len(x)
    # 如果sample_weight为None设均设置一样
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    # 计算
    ent = .0
    for x_value in set(x):
        x_index = np.where(x == x_value)
        new_x = x[x_index]
        new_y = y[x_index]
        new_sample_weight = sample_weight[x_index]
        p_i = 1.0 * len(new_x) / x_num
        ent += p_i * entropy(new_y, new_sample_weight)
    return ent


def muti_info(x, y, sample_weight=None):
    """
    互信息/信息增益:H(y)-H(y|x)
    """
    x_num = len(x)
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    return entropy(y, sample_weight) - cond_entropy(x, y, sample_weight)


def info_gain_rate(x, y, sample_weight=None):
    """
    信息增益比
    """
    x_num = len(x)
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    return 1.0 * muti_info(x, y, sample_weight) / (1e-12 + entropy(x, sample_weight))


def gini(x, sample_weight=None):
    """
    计算基尼系数 Gini(D)
    :param x:
    :param sample_weight:
    :return:
    """
    x_num = len(x)
    # 如果sample_weight为None设均设置一样
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    x_counter = {}
    weight_counter = {}
    # 统计各x取值出现的次数以及其对应的sample_weight列表
    for index in range(0, x_num):
        x_value = x[index]
        if x_counter.get(x_value) is None:
            x_counter[x_value] = 0
            weight_counter[x_value] = []
        x_counter[x_value] += 1
        weight_counter[x_value].append(sample_weight[index])

    # 计算gini系数
    gini_value = 1.0
    for key, value in x_counter.items():
        p_i = 1.0 * value * np.mean(weight_counter.get(key)) / x_num
        gini_value -= p_i * p_i
    return gini_value


def cond_gini(x, y, sample_weight=None):
    """
    计算条件gini系数:Gini(y,x)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    # x中元素个数
    x_num = len(x)
    # 如果sample_weight为None设均设置一样
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    # 计算
    gini_value = .0
    for x_value in set(x):
        x_index = np.where(x == x_value)
        new_x = x[x_index]
        new_y = y[x_index]
        new_sample_weight = sample_weight[x_index]
        p_i = 1.0 * len(new_x) / x_num
        gini_value += p_i * gini(new_y, new_sample_weight)
    return gini_value


def gini_gain(x, y, sample_weight=None):
    """
    gini值的增益
    """
    x_num = len(x)
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    return gini(y, sample_weight) - cond_gini(x, y, sample_weight)


def square_error(x, sample_weight=None):
    """
    平方误差
    :param x:
    :param sample_weight:
    :return:
    """
    x = np.asarray(x)
    # x_mean = np.mean(x)
    x_num = len(x)
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    x_mean = np.dot(x, sample_weight / np.sum(sample_weight))
    error = 0.0
    for index in range(0, x_num):
        error += (x[index] - x_mean) * (x[index] - x_mean) * sample_weight[index]
    return error


def cond_square_error(x, y, sample_weight=None):
    """
    计算按x分组的y的误差值
    :param x:
    :param y:
    :param sample_weight:
    :return:
    """
    x = np.asarray(x)
    y = np.asarray(y)
    # x中元素个数
    x_num = len(x)
    # 如果sample_weight为None设均设置一样
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    # 计算
    error = .0
    for x_value in set(x):
        x_index = np.where(x == x_value)
        new_y = y[x_index]
        new_sample_weight = sample_weight[x_index]
        error += square_error(new_y, new_sample_weight)
    return error


def square_error_gain(x, y, sample_weight=None):
    """
    平方误差带来的增益值
    :param x:
    :param y:
    :param sample_weight:
    :return:
    """
    x_num = len(x)
    if sample_weight is None:
        sample_weight = np.asarray([1.0] * x_num)
    return square_error(y, sample_weight) - cond_square_error(x, y, sample_weight)


def gaussian_1d(x, u, sigma):
    """
    一维高斯概率分布函数
    :param x:
    :param u:
    :param sigma:
    :return:
    """
    return 1 / (np.sqrt(2 * np.pi) * sigma + 1e-12) * np.exp(-1 * np.power(x - u, 2) / (2 * sigma ** 2 + 1e-12))


def gaussian_nd(x, u, sigma):
    """
    高维高斯函数
    :param x:
    :param u:
    :param sigma:
    :return:
    """
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
    return 1.0 / (np.power(2 * np.pi, x.shape[1] / 2) * np.sqrt(np.linalg.det(sigma))) * np.exp(
        np.sum(-0.5 * (x - u).dot(np.linalg.inv(sigma)) * (x - u), axis=1))


def dirichlet(u, alpha):
    """
    狄利克雷分布
    :param u: 随机变量
    :param alpha: 超参数
    :return:
    """
    # 计算归一化因子
    beta = special.gamma(np.sum(alpha))
    for alp in alpha:
        beta /= special.gamma(np.sum(alp))
    rst = beta
    # 计算结果
    for idx in range(0, len(alpha)):
        rst *= np.power(u[idx], alpha[idx] - 1)
    return rst


def wishart(Lambda, W, v):
    """
    wishart分布
    :param Lambda:随机变量
    :param W:超参数
    :param v:超参数
    :return:
    """
    # 维度
    D = W.shape[0]
    # 先计算归一化因子
    B = np.power(np.linalg.det(W), -1 * v / 2)
    B_ = np.power(2.0, v * D / 2) * np.power(np.pi, D * (D - 1) / 4)
    for i in range(1, D + 1):
        B_ *= special.gamma((v + 1 - i) / 2)
    B = B / B_
    # 计算剩余部分
    rst = B * np.power(np.linalg.det(Lambda), (v - D - 1) / 2)
    rst *= np.exp(-0.5 * np.trace(np.linalg.inv(W) @ Lambda))
    return rst


def St(X, mu, Lambda, v):
    """
    学生t分布
    :param X: 随机变量
    :param mu: 超参数
    :param Lambda: 超参数
    :param v: 超参数
    :return:
    """
    n_sample, D = X.shape
    return special.gamma(D / 2 + v / 2) * np.power(np.linalg.det(Lambda), 0.5) * np.power(
        1 + np.sum((X - mu) @ Lambda * (X - mu), axis=1) / v, -1.0 * D / 2 - v / 2) / special.gamma(v / 2) / np.power(
        np.pi * v, D / 2)


def plot_decision_function(X, y, clf, support_vectors=None):
    """
    绘制决策边界
    """
    plot_step = 0.02
    # 创建色彩图
    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    # 绘制支持向量
    if support_vectors is not None:
        plt.scatter(X[support_vectors, 0], X[support_vectors, 1], s=80, c='none', alpha=0.7, edgecolor='red')

    plt.show()


def knn_plot(model, x_test, y_test):
    h = .02
    # 创建色彩图
    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])
    # 绘制决策边界
    x_min, x_max = x_test[:, -2].min() - 1, x_test[:, -2].max() + 1
    y_min, y_max = x_test[:, -1].min() - 1, x_test[:, -1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # 绘制训练点
    plt.scatter(x_test[:, -2], x_test[:, -1], c=y_test, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


def plot_contourf(data, func, lines=3):
    """
    绘制等高线
    """
    n = 256
    x = np.linspace(data[:, 0].min(), data[:, 0].max(), n)
    y = np.linspace(data[:, 1].min(), data[:, 1].max(), n)
    X, Y = np.meshgrid(x, y)
    C = plt.contour(X, Y, func(np.c_[X.reshape(-1), Y.reshape(-1)]).reshape(X.shape), lines, colors='g', linewidth=0.5)
    plt.clabel(C, inline=True, fontsize=10)
    plt.scatter(data[:, 0], data[:, 1])


def evaluating_indicator(y_pre, y_test):
    accuracy = {}
    precision = {}
    recall = {}
    f1 = {}
    # 准确率
    accuracy.update('训练集准确率：', accuracy_score(y_test, y_pre))

    # 精确率
    precision.update('精确率-macro：', precision_score(y_test, y_pre, average='macro'))
    precision.update('精确率-micro：', precision_score(y_test, y_pre, average='micro'))
    precision.update('精确率-weighted：', precision_score(y_test, y_pre, average='weighted'))
    precision.update('精确率-None：', precision_score(y_test, y_pre, average='None'))

    # 召回率
    recall.update('召回率-macro：', recall_score(y_test, y_pre, average='macro'))
    recall.update('召回率-micro：', recall_score(y_test, y_pre, average='micro'))
    recall.update('召回率-weighted：', recall_score(y_test, y_pre, average='weighted'))
    recall.update('召回率-None：', recall_score(y_test, y_pre, average='None'))

    # F1 score
    f1.update('F1 score-macro：', f1_score(y_test, y_pre, average='macro'))
    f1.update('F1 score-micro：', f1_score(y_test, y_pre, average='micro'))
    f1.update('F1 score-weighted：', f1_score(y_test, y_pre, average='weighted'))
    f1.update('F1 score-None：', f1_score(y_test, y_pre, average='None'))

    for kv in accuracy.items():
        print(kv)
    for kv in precision.items():
        print(kv)
    for kv in recall.items():
        print(kv)
    for kv in f1.items():
        print(kv)

    return accuracy, precision, recall, f1


def knn_plot(model, x_test, y_test):
    h = .02
    # 创建色彩图
    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])
    # 绘制决策边界
    x_min, x_max = x_test[:, -2].min() - 1, x_test[:, -2].max() + 1
    y_min, y_max = x_test[:, -1].min() - 1, x_test[:, -1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # 绘制训练点
    plt.scatter(x_test[:, -2], x_test[:, -1], c=y_test, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


