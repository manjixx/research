import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import *
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


def read_data(file_path, season, algorithm):
    data = pd.read_csv(file_path, encoding='gbk').dropna(axis=0, how='any', inplace=False)
    if season:
        data = data.loc[data['season'] == season]

    index = 'thermal sensation'

    if algorithm == 'svm':
        data.loc[(data[index] > 0.5), index] = 1
        data.loc[((-0.5 <= data[index]) & (data[index] <= 0.5)), index] = 0
        data.loc[(data[index] < -0.5), index] = -1
        data = data[(data[index] != 0)]
    elif algorithm == 'pmv':
        return data
    else:
        data.loc[(data[index] > 0.5), index] = 2
        data.loc[(-0.5 <= data[index]) & (data[index] <= 0.5), index] = 1
        data.loc[(data[index] < -0.5), index] = 0
    return data


def split_by_index(data, index, x_feature, y_feature):
    if index == 'bmi':
        low_x = data[(data[index] <= 18)][x_feature]
        low_y = data[(data[index] <= 18)][y_feature]
        mid_x = data[(data[index] < 24) & (data[index] > 18)][x_feature]
        mid_y = data[(data[index] < 24) & (data[index] > 18)][y_feature]
        high_x = data[(data[index] >= 24)][x_feature]
        high_y = data[(data[index] >= 24)][y_feature]
    elif index == 'griffith':
        low_x = data[(data[index] <= 0.8)][x_feature]
        low_y = data[(data[index] <= 0.8)][y_feature]
        mid_x = data[(data[index] < 1.5) & (data[index] > 0.8)][x_feature]
        mid_y = data[(data[index] < 1.5) & (data[index] > 0.8)][y_feature]
        high_x = data[(data[index] >= 1.5)][x_feature]
        high_y = data[(data[index] >= 1.5)][y_feature]
    elif index == 'preference':
        low_x = data[(data[index] == -1)][x_feature]
        low_y = data[(data[index] == -1)][y_feature]
        mid_x = data[(data[index] == 0)][x_feature]
        mid_y = data[(data[index] == 0)][y_feature]
        high_x = data[(data[index] == 1)][x_feature]
        high_y = data[(data[index] == 1)][y_feature]
    elif index == 'sensitivity':
        low_x = data[(data[index] == 0)][x_feature]
        low_y = data[(data[index] == 0)][y_feature]
        mid_x = data[(data[index] == 1)][x_feature]
        mid_y = data[(data[index] == 1)][y_feature]
        high_x = data[(data[index] == 2)][x_feature]
        high_y = data[(data[index] == 2)][y_feature]
    low_x = np.array(low_x)
    mid_x = np.array(mid_x)
    high_x = np.array(high_x)
    low_y = low_y.astype(int)
    mid_y = mid_y.astype(int)
    high_y = high_y.astype(int)
    x_list = [low_x, mid_x, high_x]
    y_list = [low_y, mid_y, high_y]
    return x_list, y_list


def split_svm_knn(data, algorithm, index, x_feature, y_feature, normalization):
    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []

    x_list, y_list = split_by_index(data, index, x_feature, y_feature)

    for i in range(0, 3):
        # ?????????
        if normalization:
            x = preprocessing.MaxAbsScaler().fit_transform(x_list[i])
            y = np.array(y_list[i].stack())
        else:
            x = x_list[i]
            y = np.array(y_list[i])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        x_train_list.append(x_train)
        y_train_list.append(y_train)
        x_test_list.append(x_test)
        y_test_list.append(y_test.astype(int))

    if algorithm == 'knn':
        x_list = []
        y_list = []
        for i in range(0, 3):
            for j in range(0, len(x_train_list[i])):
                x_list.append(x_train_list[i][j].tolist())
                y_list.append(y_train_list[i][j])
        x_train_list = x_list
        y_train_list = y_list

    return x_train_list, y_train_list, x_test_list, y_test_list


def split_ensemble_wcs(data, index, x_feature, y_feature, class_weight):

    x_train_list = []
    y_train_list = []
    sample_weight = []

    x_test_list = []
    y_test_list = []
    test_sample_weight = []

    x_list, y_list = split_by_index(data, index, x_feature, y_feature)

    for i in range(0, 3):
        # ?????????
        x = preprocessing.MaxAbsScaler().fit_transform(x_list[i])
        y = np.array(y_list[i].stack())
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        for j in range(0, len(x_train)):
            x_train_list.append(x_train[j])
            y_train_list.append(y_train[j])
            sample_weight.append(class_weight[i])

        for k in range(0, len(x_test)):
            x_test_list.append(x_test[k])
            y_test_list.append(y_test[k])
            test_sample_weight.append(class_weight[i])

    return x_train_list, x_test_list, y_train_list, y_test_list, sample_weight, test_sample_weight


def split_ensemble_wsw(data, x_feature, y_feature, sample_weight):

    data['sample_weight'] = sample_weight
    x_feature.append('sample_weight')
    x = data[x_feature]
    y = data[y_feature]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    x_feature.remove('sample_weight')

    x_train_list = preprocessing.MaxAbsScaler().fit_transform(x_train[x_feature])
    sample_weight = np.array(x_train[['sample_weight']]).flatten()
    x_test_list = preprocessing.MaxAbsScaler().fit_transform(x_test[x_feature])
    test_sample_weight = np.array(x_test[['sample_weight']]).flatten()

    return x_train_list, x_test_list, y_train, y_test, sample_weight, test_sample_weight


def plot_decision_function(X, y, clf, support_vectors=None):
    """
    ??????????????????
    """
    plot_step = 0.02
    # ???????????????
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
    # ??????????????????
    if support_vectors is not None:
        plt.scatter(X[support_vectors, 0], X[support_vectors, 1], s=80, c='none', alpha=0.7, edgecolor='red')

    plt.show()


def knn_plot(model, x_test, y_test):
    h = .02
    # ???????????????
    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])
    # ??????????????????
    x_min, x_max = x_test[:, -2].min() - 1, x_test[:, -2].max() + 1
    y_min, y_max = x_test[:, -1].min() - 1, x_test[:, -1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # ???????????????
    plt.scatter(x_test[:, -2], x_test[:, -1], c=y_test, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


def plot_contourf(data, func, lines=3):
    """
    ???????????????
    """
    n = 256
    x = np.linspace(data[:, 0].min(), data[:, 0].max(), n)
    y = np.linspace(data[:, 1].min(), data[:, 1].max(), n)
    X, Y = np.meshgrid(x, y)
    C = plt.contour(X, Y, func(np.c_[X.reshape(-1), Y.reshape(-1)]).reshape(X.shape), lines, colors='g', linewidth=0.5)
    plt.clabel(C, inline=True, fontsize=10)
    plt.scatter(data[:, 0], data[:, 1])


def avg_indicator(precision, recall, f1):

    p_macro = 0
    p_micro = 0
    p_weight = 0
    r_macro = 0
    r_micro = 0
    r_weight = 0
    f1_macro = 0
    f1_micro = 0
    f1_weight = 0
    for i in range(0, 3):
        p_macro += 1 / 3 * precision[i][0]
        p_micro += 1 / 3 * precision[i][1]
        p_weight += 1 / 3 * precision[i][2]
        r_macro += 1 / 3 * recall[i][0]
        r_micro += 1 / 3 * recall[i][1]
        r_weight += 1 / 3 * recall[i][2]
        f1_macro += 1 / 3 * f1[i][0]
        f1_micro += 1 / 3 * f1[i][1]
        f1_weight += 1 / 3 * f1[i][2]
    print("?????????-macro:" + str(p_macro))
    print("?????????-micro:" + str(p_micro))
    print("?????????-weight:" + str(p_weight))
    print("?????????-macro:" + str(r_macro))
    print("?????????-micro:" + str(r_micro))
    print("?????????-weight:" + str(r_weight))
    print("F1 score-macro:" + str(f1_macro))
    print("F1 score-micro:" + str(f1_micro))
    print("F1 score-weight:" + str(f1_weight))


def evaluating_indicator(y_pre, y_test):
    accuracy = {}
    precision = {}
    recall = {}
    f1 = {}
    # ?????????
    accuracy.update({'?????????????????????': accuracy_score(y_test, y_pre)})

    # ?????????
    precision.update({'?????????-macro???': precision_score(y_test, y_pre, average='macro')})
    precision.update({'?????????-micro???': precision_score(y_test, y_pre, average='micro')})
    precision.update({'?????????-weighted???': precision_score(y_test, y_pre, average='weighted')})
    # precision.update({'?????????-None???': precision_score(y_test, y_pre, average=None)})

    # ?????????
    recall.update({'?????????-macro???': recall_score(y_test, y_pre, average='macro')})
    recall.update({'?????????-micro???': recall_score(y_test, y_pre, average='micro')})
    recall.update({'?????????-weighted???': recall_score(y_test, y_pre, average='weighted')})
    # recall.update({'?????????-None???': recall_score(y_test, y_pre, average=None)})

    # F1 score
    f1.update({'F1 score-macro???': f1_score(y_test, y_pre, average='macro')})
    f1.update({'F1 score-micro???': f1_score(y_test, y_pre, average='micro')})
    f1.update({'F1 score-weighted???': f1_score(y_test, y_pre, average='weighted')})
    # f1.update({'F1 score-None???': f1_score(y_test, y_pre, average=None)})
    #
    # for kv in accuracy.items():
    #     print(kv)
    # for kv in precision.items():
    #     print(kv)
    # for kv in recall.items():
    #     print(kv)
    # for kv in f1.items():
    #     print(kv)

    return accuracy, precision, recall, f1


def knn_plot(model, x_test, y_test):
    h = .02
    # ???????????????
    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])
    # ??????????????????
    x_min, x_max = x_test[:, -2].min() - 1, x_test[:, -2].max() + 1
    y_min, y_max = x_test[:, -1].min() - 1, x_test[:, -1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # ???????????????
    plt.scatter(x_test[:, -2], x_test[:, -1], c=y_test, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


