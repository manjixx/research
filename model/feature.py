# -*- coding: utf-8 -*-
"""
@Project ：data- mechanism 
@File ：feature.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/2/16 21:00 
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    feature_imp = pd.Series(clf.feature_importances_, index=x_train.feature_names).sort_values(ascending=False)
    print(feature_imp)
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.legend()
    plt.show()


