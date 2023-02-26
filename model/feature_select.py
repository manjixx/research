# -*- coding: utf-8 -*-
"""
@Project ：data- mechanism 
@File ：feature_select.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/2/16 21:00 
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def feature_importance(x_train, y_train):
    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                 max_depth=None, max_features='auto', max_leaf_nodes=None,
                                 min_impurity_decrease=0.0, min_impurity_split=None,
                                 min_samples_leaf=1, min_samples_split=2,
                                 min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                                 oob_score=False, random_state=None, verbose=0,
                                 warm_start=False)
    # clf = RandomForestClassifier(n_estimators=100)
    clf.fit(x_train, y_train)
    index = ['gender', 'age', 'height', 'weight', 'bmi', 'preference',
                   'sensitivity', 'environment', 'thermal comfort ',
                   'thermal preference', 'ta', 'hr', 'griffith']
    feature_imp = pd.Series(clf.feature_importances_, index=index).sort_values(ascending=False)
    print(feature_imp)
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.legend()
    plt.show()

    # data = read_data(file_path, synthetic=synthetic, season='summer',
    #                  algorithm='other')
    #
    # x = data[['gender', 'age', 'height', 'weight', 'bmi', 'preference',
    #                'sensitivity', 'environment', 'thermal comfort ',
    #                'thermal preference', 'ta', 'hr', 'griffith']]
    # y = data[["thermal sensation"]]
    #
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # feature_importance(x_train, y_train)


def cross_val(x_train, y_train, neighbour):
    scores = []
    ks = []
    for k in range(3, neighbour, 2):
        model = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(model, x_train, y_train, cv=6).mean()
        scores.append(score)
        ks.append(k)
    # 画图，x轴为k值，y值为误差值
    plt.plot(ks, scores)
    plt.xlabel('Value of K in KNN')
    plt.ylabel('scores')
    plt.show()
