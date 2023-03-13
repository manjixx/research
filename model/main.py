import numpy as np

from model.models import *
from model.weight import *

if __name__ == '__main__':
    file_path = "../dataset/2021.csv"
    index = 'preference'
    x_features = ['bmi', 'sensitivity', 'preference', 'griffith', 'ta', 'hr', 'thermal comfort', 'thermal preference']
    y_feature = ['thermal sensation']

    """
     PMV 模型
    """

    print("算法：pmv")
    data = read_data(file_path, season='summer', algorithm='pmv')
    pmv(np.array(data[['ta', 'hr']]), np.array(data[y_feature]))

    weight = proportion(data, index=index)
    s_weight = sample_weight(data)

    """
     SVM 模型
    """

    print("算法：svm, 分类指标： " + index)
    data = read_data(file_path, season='summer', algorithm='svm')
    x_train, y_train, x_test, y_test = split_svm_knn(
        data, algorithm='svm', index=index,
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

    """
     KNN 模型
    """

    print("算法：knn, 分类指标:" + index)

    data = read_data(file_path, season='summer', algorithm='knn')
    x_train, y_train, x_test, y_test = split_svm_knn(
        data, algorithm='knn', index=index,
        x_feature=x_features, y_feature=y_feature,
        normalization=False
    )
    # w越大，距离函数越大，所以对样本少的类别，w应该取较大值，此时计算距离越大，即较少考虑该样本
    w = []
    for i in range(0, len(weight)):
        w.append(1 / weight[i])
    # distance function
    # ['manhattan_inverse', 'manhattan_gauss', , 'euclid_inverse', 'euclid_gauss']
    distance = ['manhattan', 'euclid']
    knn(x_train, y_train, x_test, y_test, w, distance)

    """
     ensemble 算法数据生成
    """
    # 获取集成学习数据集
    data = read_data(file_path, season='summer', algorithm='ensemble')

    x_train_c, x_test_c, y_train_c, y_test_c, sample_weight_c, test_sample_weight_c = split_ensemble_wcs(
        data, index=index, x_feature=x_features,
        y_feature=y_feature, class_weight=weight
    )

    x_train_s, x_test_s, y_train_s, y_test_s, sample_weight_s, test_sample_weight_s = split_ensemble_wsw(
        data=data, x_feature=x_features, y_feature=y_feature, sample_weight=s_weight
    )

    """
    Adaboost
    """
    print("算法：adaboost，分类指标：" + index + "，权重：类别权重")
    adaboost(x_train_c, x_test_c, y_train_c, y_test_c, sample_weight_c, test_sample_weight_c)

    print("算法：adaboost，分类指标：" + index + "，权重：人员权重")
    adaboost(x_train_s, x_test_s, y_train_s, y_test_s, sample_weight_s, test_sample_weight_s)

    """
    Random Forest
    """
    print("算法：random forest, 分类指标：" + index + "权重：类别权重")
    # class_weight = {0: weight[0], 1: weight[1], 2: weight[2]}
    class_weight = {0: 1, 1: 1, 2: 1}
    random_forest(x_train_s, x_test_s, y_train_s, y_test_s, class_weight=class_weight,
                  sample_weight=None, test_sample_weight=None)

    print("算法：random forest，分类指标：" + index + "权重：人员权重")
    random_forest(x_train_s, x_test_s, y_train_s, y_test_s, class_weight=None,
                  sample_weight=sample_weight_s, test_sample_weight=test_sample_weight_s)

    """
        xgboost
    """
    print("算法：xgboost,分类指标：" + index + "权重：类别权重")
    # class_weight = {0: weight[0], 1: weight[1], 2: weight[2]}
    class_weight = {0: 1, 1: 1, 2: 1}

    xgboost(x_train_s, x_test_s, y_train_s, y_test_s,
            class_weight=class_weight,
            sample_weight=None, test_sample_weight=None)
    print("算法：xgboost,分类指标：" + index + " 权重：人员权重")

    xgboost(x_train_s, x_test_s, y_train_s, y_test_s,
            class_weight=None,
            sample_weight=sample_weight_s,
            test_sample_weight=test_sample_weight_s)

