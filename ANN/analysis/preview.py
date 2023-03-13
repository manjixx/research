# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
@Project ：data- mechanism
@File ：csv2npy_Ver1.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/3/5 20:18
"""

import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # 先用pandas读入csv
    df = pd.read_csv('../../dataset/2021.csv', encoding='gbk').dropna(axis=0, how='any', inplace=False)
    df = df.loc[df['season'] == 'summer']

    no = np.array(df['no'].unique())

    y_feature = 'thermal sensation'
    df.loc[(df[y_feature] > 0.5), y_feature] = 2
    df.loc[((-0.5 <= df[y_feature]) & (df[y_feature] <= 0.5)), y_feature] = 1
    df.loc[(df[y_feature] < -0.5), y_feature] = 0

    hot_ta = df[(df['thermal sensation'] == 2)][['ta']]
    hot_hr = df[(df['thermal sensation'] == 2)][['hr']]
    cool_ta = df[(df['thermal sensation'] == 0)][['ta']]
    cool_hr = df[(df['thermal sensation'] == 0)][['hr']]
    com_ta = df[(df['thermal sensation'] == 1)][['ta']]
    com_hr = df[(df['thermal sensation'] == 1)][['hr']]

    # hot_ta.hist(color='r')
    # com_ta.hist(color='g')
    # cool_ta.hist(color='b')
    #
    # plt.show()
    #
    # hot_hr.hist(color='r')
    # com_hr.hist(color='g')
    # cool_hr.hist(color='b')
    # plt.show()

    # 绘制分布图
    plt.figure(figsize=(8, 5), dpi=80)
    axes = plt.subplot(111)
    plt.title('all')
    plt.xlabel("temp(℃)")
    plt.ylabel("humid(%)")
    plt.plot(com_ta, com_hr, "o", marker='+', c="green", )
    # plt.show()
    len = {}
    for n in no:
        # if n < 56:
        #     continue
        data = df.loc[df['no'] == n]
        l = data.shape[0]
        len.update({n: l})
        hot_ta = data[(data['thermal sensation'] == 2)][['ta']]
        hot_hr = data[(data['thermal sensation'] == 2)][['hr']]
        cool_ta = data[(data['thermal sensation'] == 0)][['ta']]
        cool_hr = data[(data['thermal sensation'] == 0)][['hr']]
        com_ta = data[(data['thermal sensation'] == 1)][['ta']]
        com_hr = data[(data['thermal sensation'] == 1)][['hr']]

        # hot_ta.hist(color='r')
        # com_ta.hist(color='g')

        # cool_ta.hist(color='b')

        # 绘制分布图
        plt.figure(figsize=(8, 5), dpi=80)
        axes = plt.subplot(111)
        label1 = axes.scatter(hot_ta, hot_hr, s=50, marker=None, c="red")
        label2 = axes.scatter(cool_ta, cool_hr, s=50, marker='x', c="blue")
        label3 = axes.scatter(com_ta, com_hr, s=50, marker='+', c="green")
        plt.title(n)
        plt.xlabel("temp(℃)")
        plt.ylabel("humid(%)")
        axes.legend((label1, label2, label3), ("hot", "cool", "comfort"), loc=3)
        plt.show()

    for key, value in len.items():
        print('%s:%s' % (key, value))
