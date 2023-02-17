# -*- coding: utf-8 -*-
"""
@Project ：data- mechanism 
@File ：split_dataset.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/2/17 15:16 
"""

import random
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def train_data_knn(x_train_pmv, y_train_pmv, x_train_list, y_train_list):
    x_train = []
    y_train = []

    for i in range(0, len(x_train_pmv)):
        x_train.append(x_train_pmv[i])
        y_train.append(y_train_pmv[i])

    for i in range(0, 3):
        x = x_train_list[i]
        y = y_train_list[i]
        for j in range(0, len(x)):
            x_train.append(x[i])
            y_train.append(y[i])

    x_train = np.array(x_train, dtype=object)
    y_train = np.array(y_train, dtype=object).astype(int)

    return x_train, y_train


def split_pmv_data(data, count):
    x_grade = []
    y_grade = []
    if count == 0:
        return x_grade, y_grade
    pmv = data[['ta', 'hr', 'pmv']]

    arr = random.sample(list(np.array(pmv)), count)
    arr = np.array(arr)
    x_train = arr[:, 0:2]
    x_grade = preprocessing.MaxAbsScaler().fit_transform(x_train)
    y_grade = arr[:, 2:3]
    return x_grade, y_grade


def split_filed_data(x_list, y_list, w, testsize):
    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []

    for i in range(0, len(x_list)):
        x = x_list[i]
        y = y_list[i]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testsize, random_state=35, stratify=y)
        x_test = preprocessing.MaxAbsScaler().fit_transform(x_test)
        x_train = preprocessing.MaxAbsScaler().fit_transform(x_train)
        x_test = np.insert(x_test, 2, values=w[i], axis=1)
        x_train_list.append(x_train)
        y_train_list.append(y_train)
        x_test_list.append(x_test)
        y_test_list.append(y_test)

    return x_train_list, y_train_list, x_test_list, y_test_list
