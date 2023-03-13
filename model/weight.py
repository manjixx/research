# -*- coding: utf-8 -*-
import math
import numpy as np

"""
@Project ：data- mechanism 
@File ：weight.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/2/24 20:05 
"""


def proportion(data, index):
    weight = []

    count = data.shape[0];

    if index == "bmi":
        low_count = data[(data[index] <= 18)].shape[0]
        mid_count = data[(data[index] < 24) & (data[index] > 18)].shape[0]
        high_count = data[(data[index] >= 24)].shape[0]
    elif index == 'griffith':
        low_count = data[(data[index] <= 0.8)].shape[0]
        mid_count = data[(data[index] < 1.5) & (data[index] > 0.8)].shape[0]
        high_count = data[(data[index] >= 1.5)].shape[0]
    elif index == 'preference':
        low_count = data[(data[index] == -1)].shape[0]
        mid_count = data[(data[index] == 0)].shape[0]
        high_count = data[(data[index] == 1)].shape[0]
    elif index == 'sensitivity':
        low_count = data[(data[index] == 0)].shape[0]
        mid_count = data[(data[index] == 1)].shape[0]
        high_count = data[(data[index] == 2)].shape[0]

    weight.append(round(low_count / count, 2))
    weight.append(round(mid_count / count, 2))
    weight.append(round(high_count / count, 2))

    w_max = max(weight)

    for i in range(0, len(weight)):
        weight[i] = weight[i] / w_max;

    return weight


def sample_weight(data):

    weight = []

    for i in range(0, data.shape[0]):
        sensitivity = np.array(data.iloc[i: i + 1, 7:8]).flatten()[0]  # 选取df的第i行和第9列

        g = np.array(data.iloc[i: i + 1, 9:10]).flatten()[0]  # 选取df的第i行和第9列
        if g <= 0.8:
            griffith = 0
        elif 0.8 < g < 1.2:
            griffith = 1
        elif g >= 1.2:
            griffith = 2

        p = np.array(data.iloc[i:i + 1, 6:7]).flatten()[0]
        if p == -1:
            preference = 0
        elif p == 0:
            preference = 1
        elif p == 1:
            preference = 2

        b = np.array(data.iloc[i:i + 1, 5:6]).flatten()[0]
        if b <= 18:
            bmi = 0
        elif 18 < b < 24:
            bmi = 1
        elif b >= 24:
            bmi = 2

        w = sensitivity * 4 + griffith * 3 + preference * 2 + bmi

        weight.append(w)

    w_max = min(weight)

    for i in range(0, len(weight)):
        weight[i] = round(weight[i] / w_max, 2)

    return weight
