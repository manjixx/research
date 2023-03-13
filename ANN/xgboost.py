# -*- coding: utf-8 -*-
"""
@Project ：data- mechanism 
@File ：xgboost.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/3/6 20:33 
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

file_path = './dataset/experimental_v1/'
env = np.load(file_path + 'env.npy')
body = np.load(file_path+'body.npy')
label = np.load(file_path+'label.npy')
feature = np.concatenate((env, body), axis=1)
train_feature, test_feature, train_label, test_label = train_test_split(feature, label, test_size=0.2)

print("soft-svm")

model = SVC(kernel='linear', decision_function_shape='ovr', C=0.1).fit(train_feature, train_label)
model.fit(train_feature, train_label)
y_pred = model.predict(test_feature)
print('准确率：' + str(accuracy_score(y_pred, test_label)))
print('精确率：' + str(accuracy_score(y_pred, test_label)))
print('recall：' + str(accuracy_score(y_pred, test_label)))
print('f1：' + str(accuracy_score(y_pred, test_label)))


print("knn,距离函数 manhattan")
model = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    algorithm='auto',
    leaf_size=30,
    p=1,
    n_jobs=None
)
model.fit(train_feature, train_label)
y_pred = model.predict(test_feature)
print('准确率：' + str(accuracy_score(y_pred, test_label)))
print('精确率：' + str(accuracy_score(y_pred, test_label)))
print('recall：' + str(accuracy_score(y_pred, test_label)))
print('f1：' + str(accuracy_score(y_pred, test_label)))


print("Random-Forest")

classifier = RandomForestClassifier(
    max_depth=2,
    random_state=0
)

classifier.fit(train_feature, train_label)
y_pred = classifier.predict(test_feature)
print('准确率：' + str(accuracy_score(y_pred, test_label)))
print('精确率：' + str(accuracy_score(y_pred, test_label)))
print('recall：' + str(accuracy_score(y_pred, test_label)))
print('f1：' + str(accuracy_score(y_pred, test_label)))

print("Xgboost")

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
print('准确率：' + str(accuracy_score(y_pred, test_label)))
print('精确率：' + str(accuracy_score(y_pred, test_label)))
print('recall：' + str(accuracy_score(y_pred, test_label)))
print('f1：' + str(accuracy_score(y_pred, test_label)))