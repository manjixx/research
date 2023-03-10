import numpy as np

from ml_model.pmv import *
from ml_model.softsvm import *
from ml_model.distance import *
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm.sklearn import LGBMClassifier
from utils import *


def pmv(data, target):
    """
    该模型为pmv预测热舒适投票值模型
    :param data: 训练数据特征集
    :param target: label
    :return:
    """
    # 初始化参数
    m = 1.1
    clo = 0.5
    vel = 0.06

    # 初始化存储列表
    pmv_pred = []
    pmv_pred_level = []

    for i in range(0, len(data)):
        ta = data[i][0]
        rh = data[i][1]
        pmv_result = pmv_model(M=m * 58.15, clo=clo, tr=ta, ta=ta, vel=vel, rh=rh)
        pmv_pred.append(round(pmv_result, 2))

        if pmv_result > 0.5:
            pmv_pred_level.append(2)
        elif pmv_result < -0.5:
            pmv_pred_level.append(0)
        else:
            pmv_pred_level.append(1)
    print("pmv模型预测精度为：", end='')
    count = 0
    for i in range(0, len(target)):
        if pmv_pred[i] == target[i]:
            count = count + 1
    print(count/len(target))

    for i in range(0, len(target)):
        if target[i] > 0.5:
            target[i] = 2
        elif target[i] < -0.5:
            target[i] = 0
        else:
            target[i] = 1
    print("pmv模型类别精度为：", end='')
    accuracy = accuracy_score(pmv_pred_level, target)
    print(accuracy)

    print("pmv模型类别precision为：", end='')
    precision = precision_score(pmv_pred_level, target, average='macro')
    print(precision)

    print("pmv模型类别recall为：", end='')
    recall = recall_score(pmv_pred_level, target, average='macro')
    print(recall)

    print("pmv模型类别f1为：", end='')
    f1 = f1_score(pmv_pred_level, target, average='macro')
    print(f1)


def svm(x_train, y_train, x_test, y_test, kernel, C):
    precision_all = []
    recall_all = []
    f1_all = []
    for k in kernel:
        print("核函数为：" + k)
        count = 0
        accuracy_count = 0
        for i in range(0, len(C)):
            # print(f'第{i + 1}类：', end='')
            accuracy, precision, recall, f1 = soft_svm(x_train[i], y_train[i], x_test[i], y_test[i], k, C[i])
            count += len(x_test[i])
            accuracy_count += accuracy.get('测试集准确率：') * len(x_test[i])
            precision_all.append(list(precision.values()))
            recall_all.append(list(recall.values()))
            f1_all.append(list(f1.values()))

        print("测试集精确率为：" + str(accuracy_count / count))
        avg_indicator(precision_all, recall_all, f1_all)


def knn(x_train, y_train, x_test, y_test, weights, distance):
    precision_all = []
    recall_all = []
    f1_all = []
    for d in distance:
        count = 0
        accuracy_count = 0
        print("距离函数为：" + d)
        for i in range(0, 3):
            # print("开始第"+str(i)+"轮训练")
            dis = Distance(weights[i])
            model = KNeighborsClassifier(
                n_neighbors=5,
                weights='uniform',
                algorithm='auto',
                leaf_size=30,
                # p=2,
                metric=getattr(dis, d),
                metric_params=None,
                n_jobs=None
            )
            model.fit(x_train, y_train)
            y_pre = model.predict(x_test[i])
            accuracy, precision, recall, f1 = evaluating_indicator(y_pre, y_test[i])
            # plot_decision_function(x_test, y_test, model)
            count += len(x_test[i])
            accuracy_count += accuracy.get('测试集准确率：') * len(x_test[i])
            precision_all.append(list(precision.values()))
            recall_all.append(list(recall.values()))
            f1_all.append(list(f1.values()))

        print("测试集精确率为：" + str(accuracy_count / count))
        avg_indicator(precision_all, recall_all, f1_all)


def adaboost(x_train, x_test, y_train, y_test, sample_weight, test_sample_weight):

    # 基分类器为：决策树
    print("AdaBoost: 基分类器为：决策树")
    classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=3)
    classifier.fit(x_train, y_train, sample_weight)
    y_pred = classifier.predict(x_test)
    print("不带权重的测试集准确率为：", end='')
    print(classifier.score(x_test, y_test))
    print("带权重的测试集准确率为：", end='')
    print(classifier.score(x_test, y_test, test_sample_weight))
    accuracy, precision, recall, f1 = evaluating_indicator(y_pred, y_test)
    # for kv in accuracy.items():
    #     print(kv)
    for kv in precision.items():
        print(kv)
    for kv in recall.items():
        print(kv)
    for kv in f1.items():
        print(kv)

    # # 基分类器为：SVM
    # print("AdaBoost: 基分类器为：SVM")
    # classifier = AdaBoostClassifier(SVC(probability=True, kernel='linear'), n_estimators=50)
    # classifier.fit(x_train, y_train, sample_weight)
    # y_pred = classifier.predict(x_test)
    # print("不带权重的测试集准确率为：", end='')
    # print(classifier.score(x_test, y_test))
    # print("带权重的测试集准确率为：", end='')
    # print(classifier.score(x_test, y_test, test_sample_weight))
    # accuracy, precision, recall, f1 = evaluating_indicator(y_pred, y_test)
    # # utils.plot_decision_function(x_train, y_train, classifier)
    # for kv in precision.items():
    #     print(kv)
    # for kv in recall.items():
    #     print(kv)
    # for kv in f1.items():
    #     print(kv)
    #
    # # 基分类器为：决策树, 权重衰减
    # print("AdaBoost: 基分类器为：决策树, 权重衰减")
    # classifier = AdaBoostClassifier(
    #     DecisionTreeClassifier(max_depth=3),
    #     n_estimators=2048,
    #     learning_rate=0.0075
    # )
    # classifier.fit(x_train, y_train, sample_weight)
    # y_pred = classifier.predict(x_test)
    # print("不带权重的测试集准确率为：", end='')
    # print(classifier.score(x_test, y_test))
    # print("带权重的测试集准确率为：", end='')
    # print(classifier.score(x_test, y_test, test_sample_weight))
    # accuracy, precision, recall, f1 = evaluating_indicator(y_pred, y_test)
    # # utils.plot_decision_function(x_train, y_train, classifier)
    # for kv in precision.items():
    #     print(kv)
    # for kv in recall.items():
    #     print(kv)
    # for kv in f1.items():
    #     print(kv)


def random_forest(x_train, x_test, y_train, y_test, class_weight, sample_weight, test_sample_weight):

    # 样本权重

    if class_weight:
        classifier = RandomForestClassifier(
            max_depth=2, random_state=0, class_weight=class_weight
        )
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        accuracy, precision, recall, f1 = evaluating_indicator(y_pred, y_test)
        # utils.plot_decision_function(x_train, y_train, classifier)
        for kv in accuracy.items():
            print(kv)
        for kv in precision.items():
            print(kv)
        for kv in recall.items():
            print(kv)
        for kv in f1.items():
            print(kv)

    # 人员权重

    if (sample_weight is not None) and (sample_weight.size != 0):
        classifier = RandomForestClassifier(
            max_depth=2, random_state=0
        )
        classifier.fit(x_train, y_train, sample_weight)
        y_pred = classifier.predict(x_test)
        print("不带权重的测试集准确率为：", end='')
        print(classifier.score(x_test, y_test))
        print("带权重的测试集准确率为：", end='')
        print(classifier.score(x_test, y_test, test_sample_weight))
        accuracy, precision, recall, f1 = evaluating_indicator(y_pred, y_test)
        # utils.plot_decision_function(x_train, y_train, classifier)
        for kv in precision.items():
            print(kv)
        for kv in recall.items():
            print(kv)
        for kv in f1.items():
            print(kv)


def xgboost(x_train, x_test, y_train, y_test, class_weight, sample_weight, test_sample_weight):

    # 样本权重

    if class_weight:

        classifier = LGBMClassifier(
            learning_rate=0.0075,
            max_depth=7,
            n_estimators=2048,
            num_leaves=63,
            random_state=2019,
            n_jobs=-1,
            reg_alpha=0.8,
            reg_lambda=0.8,
            subsample=0.2,
            colsample_bytree=0.5,
            class_weight=class_weight
        )
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        accuracy, precision, recall, f1 = evaluating_indicator(y_pred, y_test)
        # utils.plot_decision_function(x_train, y_train, classifier)
        for kv in accuracy.items():
            print(kv)
        for kv in precision.items():
            print(kv)
        for kv in recall.items():
            print(kv)
        for kv in f1.items():
            print(kv)

    if (sample_weight is not None) and (sample_weight.size != 0):
        classifier = LGBMClassifier(
            learning_rate=0.0075,
            max_depth=7,
            n_estimators=2048,
            num_leaves=63,
            random_state=2019,
            n_jobs=-1,
            reg_alpha=0.8,
            reg_lambda=0.8,
            subsample=0.2,
            colsample_bytree=0.5,
            # class_weight=class_weight
        )
        classifier.fit(x_train, y_train, sample_weight)
        y_pred = classifier.predict(x_test)
        print("不带权重的测试集准确率为：", end='')
        print(classifier.score(x_test, y_test))
        print("带权重的测试集准确率为：", end='')
        print(classifier.score(x_test, y_test, test_sample_weight))
        accuracy, precision, recall, f1 = evaluating_indicator(y_pred, y_test)
        # utils.plot_decision_function(x_train, y_train, classifier)
        for kv in precision.items():
            print(kv)
        for kv in recall.items():
            print(kv)
        for kv in f1.items():
            print(kv)
