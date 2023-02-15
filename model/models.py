from ml_model.pmv import *
from ml_model.softsvm import *
from ml_model.knn import *
from ml_model.adaboost_classifier import *
from ml_model.random_forest_classifier import *
from ml_model.xgboost_classifier import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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

    accuracy = utils.cal_accuracy(pmv_pred, target)
    print("pmv模型预测精度为：" + str(accuracy))


def svm(data, target, C):
    for i in range(0, len(data)):
        x = data[i]
        y = target[i]
        c = C[i]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=35, stratify=y)
        print('权重为：' + str(c))
        soft_svm(c, x_train, y_train, x_test, y_test)






def knn(data, target, weights):



def knn_model(file_path, x_features, w, neighbours):
    data = read_data(file_path=file_path, data_type='filed', season="summer", algorithm='knn')
    x_list, y_list = split_dataset(data, 'griffith', down=0.8, up=1.2, x_features=x_features,
                                   y_features='thermal sensation')

    x_train_list, y_train_list, x_test_list, y_test_list = split_filed_data(x_list, y_list, w, testsize=0.2)

    size = len(x_train_list[0]) + len(x_train_list[1]) + len(x_train_list[2])

    print("原始数据集中有" + str(size) + "条数据")
    pmv_data = read_data('../dataset/synthetic.csv', data_type='pmv', season="summer", algorithm='knn')
    print("生成数据集中共有" + str(pmv_data.shape[0]) + "条数据！")
    x_train_pmv, y_train_pmv = split_pmv_data(pmv_data, 2000 - size)

    x_train, y_train = train_data_knn(x_train_pmv, y_train_pmv, x_train_list, y_train_list)

    result = knn(x_train, y_train, x_test_list, y_test_list, neighbours)
    print(result)


def cart(data, target):
    # 造伪数据
    from sklearn.datasets import make_classification
    data, target = make_classification(n_samples=100, n_features=2, n_classes=2, n_informative=1, n_redundant=0,
                                       n_repeated=0, n_clusters_per_class=1, class_sep=.5, random_state=21)
    # 训练并查看效果
    tree = CARTClassifier()
    tree.fit(data, target)
    # utils.plot_decision_function(data, target, tree)

    # 一样的，如果不加以限制，同样会存在过拟合现象，所以可以剪枝...

    # 剪枝
    tree.prune(5)
    utils.plot_decision_function(data, target, tree)


def adaboost(x_train, y_train, x_test, y_test):
    # 同质
    print("AdaBoost: 同质")
    classifier = AdaBoostClassifier(base_estimator=CARTClassifier(max_depth=2), n_estimators=10)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    count = 0
    for i in range(0, len(y_pred)):
        if y_pred[i] == y_test[i]:
            count = count + 1
    # 准确率
    print("准确率")
    print("测试集的准确率为：", count / len(y_pred))
    # plot_decision_function(x_train, y_train, classifier)

    # 异质
    print("AdaBoost: 异质")
    classifier = AdaBoostClassifier(base_estimator=[LogisticRegression(), SVC(C=5.0, kernel='rbf', probability=True), CARTClassifier()])
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    count = 0
    for i in range(0, len(y_pred)):
        if y_pred[i] == y_test[i]:
            count = count + 1
    # 准确率
    print("准确率")
    print("测试集的准确率为：", count / len(y_pred))
    # utils.plot_decision_function(x_train, y_train, classifier)

    # 权重衰减
    print("AdaBoost: 权重衰减")
    classifier = AdaBoostClassifier(base_estimator=[LogisticRegression(), SVC(C=5.0, kernel='rbf', probability=True), CARTClassifier()], learning_rate=0.5)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    count = 0
    for i in range(0, len(y_pred)):
        if y_pred[i] == y_test[i]:
            count = count + 1
    # 准确率
    print("准确率")
    print("测试集的准确率为：", count / len(y_pred))
    # utils.plot_decision_function(x_train, y_train, classifier)


def random_forest(x_train, y_train, x_test, y_test):
    # 同质
    classifier = RandomForestClassifier(feature_sample=0.6)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    count = 0
    for i in range(0, len(y_pred)):
        if y_pred[i] == y_test[i]:
            count = count + 1
    # 准确率
    print("准确率")
    print("测试集的准确率为：", count / len(y_pred))
    # utils.plot_decision_function(x_train, y_train, classifier)

    # 异质
    classifier = RandomForestClassifier(
        base_estimator=[LogisticRegression(), SVC(C=5.0, kernel='rbf', probability=True), CARTClassifier(max_depth=2)],
        feature_sample=0.6)
    classifier.fit(x_train, y_train)
    # utils.plot_decision_function(x_train, y_train, classifier)
    y_pred = classifier.predict(x_test)
    count = 0
    for i in range(0, len(y_pred)):
        if y_pred[i] == y_test[i]:
            count = count + 1
    # 准确率
    print("准确率")
    print("测试集的准确率为：", count / len(y_pred))


def xgboost(x_train, y_train, x_test, y_test):
    classifier = XGBoostClassifier()
    classifier.fit(x_train, y_train)
    # utils.plot_decision_function(x_train, y_train, classifier)
    y_pred = classifier.predict(x_test)
    count = 0
    for i in range(0, len(y_pred)):
        if y_pred[i] == y_test[i]:
            count = count + 1
    # 准确率
    print("准确率")
    print("测试集的准确率为：", count / len(y_pred))
