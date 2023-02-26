import numpy as np

from model.models import *
from model.weight import *
from feature_select import *

if __name__ == '__main__':
    file_path = "../dataset/2021.csv"
    synthetic = False
    index = 'bmi'
    x_features = ['ta', 'hr']
    y_feature = ['thermal sensation']

    print("算法：pmv")
    data = read_data(file_path, synthetic=synthetic, season='summer', algorithm='pmv')
    pmv(np.array(data[x_features]), np.array(data[y_feature]))

    weight = proportion(data, index=index)
    s_weight = sample_weight(data)

    print("算法：svm, 分类指标： bmi")
    data = read_data(file_path, synthetic=synthetic, season='summer', algorithm='svm')
    x_train, y_train, x_test, y_test = split_svm_knn(
        data, algorithm='svm', index='bmi',
        x_feature=x_features, y_feature=y_feature,
        normalization=False
    )
    # c越大，对错误的容忍越小，所以对样本少的类别，应该C取较大值
    c = []
    for i in range(0, len(weight)):
        c.append(1 / weight[i])

    # kernel function
    kernel = ['linear', 'poly', 'rbf']

    svm(x_train, y_train, x_test, y_test, kernel, c)

    print("算法：knn, 分类指标： bmi")

    data = read_data(file_path, synthetic=synthetic, season='summer', algorithm='knn')
    x_train, y_train, x_test, y_test = split_svm_knn(
        data, algorithm='knn', index='bmi',
        x_feature=x_features, y_feature=y_feature,
        normalization=False
    )

    # w越大，距离函数越大，所以对样本少的类别，w应该取较大值，此时计算距离越大，即较少考虑该样本
    w = []
    for i in range(0, len(weight)):
        w.append(1 / weight[i])

    # distance function
    distance = ['manhattan','euclid']
    # ['manhattan_inverse', 'manhattan_gauss', , 'euclid_inverse', 'euclid_gauss']
    knn(x_train, y_train, x_test, y_test, w, distance)
    
    # x_train, y_train, x_test, y_test, sample_weights, sample_weights_test = split_ensemble(
    #     data, index='bmi', x_feature=x_features,
    #     y_feature=y_feature, weights=weight
    # )
    #
    # print("算法：adaboost，分类指标：bmi")
    # adaboost(x_train, y_train, x_test, y_test, sample_weights, sample_weights_test)
    #
    # print("算法：random forest，分类指标：bmi")
    #
    # random_forest(x_train, y_train, x_test, y_test, sample_weights, sample_weights_test)
    #
    # print("算法：xgboost，分类指标：bmi")
    # xgboost(x_train, y_train, x_test, y_test, sample_weights, sample_weights_test)
