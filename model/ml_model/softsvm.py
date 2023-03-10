from sklearn.svm import SVC
from model.utils import *
import matplotlib.pyplot as plt
import numpy as np


def soft_svm(x_train, y_train, x_test, y_test, kernel, C):
    model = SVC(kernel=kernel, decision_function_shape='ovr', C=C).fit(x_train, y_train)
    y_pre = model.predict(x_test)
    accuracy, precision, recall, f1 = evaluating_indicator(y_pre, y_test)
    # plot_svm(model, x_train, y_train)
    return accuracy, precision, recall, f1


def plot_svm(model, x_train, y_train):
    hot_ta = []
    hot_hr = []
    cool_ta = []
    cool_hr = []

    for i in range(0, len(y_train)):
        if y_train[i] == 1:
            hot_ta.append(x_train[i][0])
            hot_hr.append(x_train[i][1])
        else:
            cool_ta.append(x_train[i][0])
            cool_hr.append(x_train[i][1])

    # 绘制数据
    plt.scatter(hot_ta, hot_hr, s=30, marker='x', c='b')
    plt.scatter(cool_ta, cool_hr, s=30, marker=None, c='r')
    # plt.xlim((0.5, 4))
    # plt.ylim((0.5, 4))

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 绘制最大间隔超平面和边界
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    XX, YY = np.meshgrid(xx, yy)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])

    # 绘制支持向量
    ax.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=10,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()

