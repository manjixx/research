# -*- coding: utf-8 -*-
"""
@Project ：data- mechanism 
@File ：xgboost.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/3/6 20:33 
"""

import numpy as np
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

env = np.load('dataset/env.npy')
body = np.load('dataset/body.npy')
label = np.load('dataset/label.npy')
feature = np.concatenate((env, body), axis=1)
train_feature, test_feature, train_label, test_label = train_test_split(feature, label, test_size=0.2)

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
)
classifier.fit(train_feature, train_label)
y_pred = classifier.predict(test_feature)
print(accuracy_score(y_pred, test_label))