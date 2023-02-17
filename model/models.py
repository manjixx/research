from ml_model.pmv import *
from ml_model.softsvm import *
from ml_model.distance import *
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
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

    for i in range(0, len(data)):
        ta = data[i][0]
        rh = data[i][1]
        pmv_result = pmv_model(M=m * 58.15, clo=clo, tr=ta, ta=ta, vel=vel, rh=rh)
        if pmv_result > 0.5:
            pmv_pred.append(2)
        elif pmv_result < -0.5:
            pmv_pred.append(0)
        else:
            pmv_pred.append(1)

    accuracy = accuracy_score(pmv_pred, target)

    print("pmv模型预测精度为：" + str(accuracy))


def svm(x_train, y_train, x_test, y_test, kernel, C):
    for i in range(0, len(C)):
        print('权重为：' + str(C[i]))
        for k in kernel:
            print("核函数为：" + kernel)
            soft_svm(x_train[i], y_train[i], x_test[i], y_test[i], k, C[i])


def knn(x_train, y_train, x_test, y_test, weights, distance):
    for i in range(0, len(x_test)):
        print("开始第"+str(i)+"轮训练")
        for d in distance:
            distance = Distance(weights[i])
            model = KNeighborsClassifier(
                n_neighbors=5,
                weights='uniform',
                algorithm='',
                leaf_size='30',
                p=2,
                metric=getattr(distance, d),
                metric_params=None,
                n_jobs=None
            )
            model.fit(x_train, y_train)
            y_pre = model.predict(x_test[i])
            accuracy, precision, recall, f1 = evaluating_indicator(y_pre, y_test[i])
            plot_decision_function(x_test[i], y_test[i], model)


# def knn(x_train, y_train, x_test, y_test, weights, distance, neighbor):
#     model = KNN(x_train, y_train)
#     model.neighbors = neighbor
#     for i in range(0, len(x_test)):
#         print("开始第"+str(i)+"轮训练")
#         for d in distance:
#             print("距离函数为" + d)
#             model.distance = d
#             y_pre = model.predict(x_test[i], weights[i])
#             accuracy, precision, recall, f1 = evaluating_indicator(y_pre, y_test)
#             knn_plot(model, x_test[i], y_test[i], weights[i])


def adaboost(x_train, y_train, x_test, y_test, sample_weights, sample_weights_test):
    # 同质
    print("AdaBoost: 同质")
    classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=10)
    classifier.fit(x_train, y_train, sample_weights)
    y_pred = classifier.predict(x_test)
    print("不带权重的测试集准确率为：" + classifier.score(x_test, y_test))
    print("带权重的测试集准确率为：" + classifier.score(x_test, y_test, sample_weights_test))
    evaluating_indicator(y_pred, y_test)
    # utils.plot_decision_function(x_train, y_train, classifier)

    # 异质
    print("AdaBoost: 异质")
    classifier = AdaBoostClassifier(
        base_estimator=[
            DecisionTreeClassifier(max_depth=2),
            LogisticRegression(),
            SVC(C=5.0, kernel='rbf', probability=True)],
        n_estimators=10,
        learning_rate=1
    )
    classifier.fit(x_train, y_train, sample_weights)
    y_pred = classifier.predict(x_test)
    print("不带权重的测试集准确率为：" + classifier.score(x_test, y_test))
    print("带权重的测试集准确率为：" + classifier.score(x_test, y_test, sample_weights_test))
    evaluating_indicator(y_pred, y_test)

    # utils.plot_decision_function(x_train, y_train, classifier)

    # 权重衰减
    print("AdaBoost: 权重衰减")
    classifier = AdaBoostClassifier(
        base_estimator=[
            DecisionTreeClassifier(max_depth=2),
            LogisticRegression(),
            SVC(C=5.0, kernel='rbf', probability=True)],
        n_estimators=10,
        learning_rate=0.5
    )
    classifier.fit(x_train, y_train, sample_weights, sample_weights_test)
    y_pred = classifier.predict(x_test)
    print("不带权重的测试集准确率为：" + classifier.score(x_test, y_test))
    print("带权重的测试集准确率为：" + classifier.score(x_test, y_test, sample_weights_test))
    evaluating_indicator(y_pred, y_test)
    # utils.plot_decision_function(x_train, y_train, classifier)


def random_forest(x_train, y_train, x_test, y_test, sample_weights, sample_weights_test):
    # 同质
    classifier = RandomForestClassifier(
        base_estimator_=DecisionTreeClassifier(max_depth=2)
    )
    classifier.fit(x_train, y_train, sample_weights)
    y_pred = classifier.predict(x_test)
    print("不带权重的测试集准确率为：" + classifier.score(x_test, y_test))
    print("带权重的测试集准确率为：" + classifier.score(x_test, y_test, sample_weights_test))
    evaluating_indicator(y_pred, y_test)
    # utils.plot_decision_function(x_train, y_train, classifier)

    # 异质
    classifier = RandomForestClassifier(
        base_estimator=[
            DecisionTreeClassifier(max_depth=2),
            LogisticRegression(),
            SVC(C=5.0, kernel='rbf', probability=True)],
        class_weight={0: 1, 1: 1.2, 2: 1}
    )
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print("不带权重的测试集准确率为：" + classifier.score(x_test, y_test))
    print("带权重的测试集准确率为：" + classifier.score(x_test, y_test, sample_weights_test))
    evaluating_indicator(y_pred, y_test)
    # utils.plot_decision_function(x_train, y_train, classifier)


def xgboost(x_train, y_train, x_test, y_test, sample_weights, sample_weights_test):
    category = [23, 25, 27, 28]  ##30_改
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
        class_weight={0: 1, 1: 1.2, 2: 1}
    )
    # classifier.fit(train_x, train_y, categorical_feature=category)
    classifier.fit(x_test, y_test, sample_weights)
    y_pred = classifier.predict(x_train)
    print("不带权重的测试集准确率为：" + classifier.score(x_test, y_test))
    print("带权重的测试集准确率为：" + classifier.score(x_test, y_test, sample_weights_test))
    evaluating_indicator(y_pred, y_test)
    # utils.plot_decision_function(x_train, y_train, classifier)
