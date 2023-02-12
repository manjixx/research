from sklearn.metrics import *
from utils import *
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np


def soft_svm(c, x_train, y_train, x_test, y_test):
    ker = ['linear', 'poly']
    for k in ker:
        print("核心函数为：" + k)
        model = SVC(kernel=k, decision_function_shape='ovr', C=c).fit(x_train, y_train)
        y_pred = model.predict(x_test)

        # 准确率
        print("准确率")
        print("训练集的准确率为：", model.score(x_train, y_train))  # 精度
        print("测试集的准确率为：", model.score(x_test, y_test))

        # 精确率
        print("精确率")
        print(precision_score(y_test, y_pred, average='macro'))
        print(precision_score(y_test, y_pred, average='micro'))
        print(precision_score(y_test, y_pred, average='weighted'))
        print(precision_score(y_test, y_pred, average=None))

        # 召回率
        print("回召率")
        print(recall_score(y_test, y_pred, average='macro'))  # 0.3333333333333333
        print(recall_score(y_test, y_pred, average='micro'))  # 0.3333333333333333
        print(recall_score(y_test, y_pred, average='weighted'))  # 0.3333333333333333
        print(recall_score(y_test, y_pred, average=None))  # [1. 0. 0.]

        # P-R曲线
        # F1 score
        print("F1 score")
        print(f1_score(y_test, y_pred, average='macro'))  # 0.26666666666666666
        print(f1_score(y_test, y_pred, average='micro'))  # 0.3333333333333333
        print(f1_score(y_test, y_pred, average='weighted'))  # 0.26666666666666666
        print(f1_score(y_test, y_pred, average=None))  # [0.8 0.  0. ]

        plot_svm(model, x_train, y_train)


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

