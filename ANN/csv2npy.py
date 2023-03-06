# -*- coding: utf-8 -*-
"""
@Project ：data- mechanism 
@File ：csv2npy.py
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
    df = pd.read_csv('../dataset/2021.csv', encoding='gbk').dropna(axis=0, how='any', inplace=False)
    data = df.loc[df['season'] == 'summer']

    # '''plot'''
    # hot_ta = data[(data['thermal sensation'] > 0.5)][['ta']]
    # hot_hr = data[(data['thermal sensation'] > 0.5)][['hr']]
    # cool_ta = data[(data['thermal sensation'] < -0.5)][['ta']]
    # cool_hr = data[(data['thermal sensation'] < -0.5)][['hr']]
    # com_ta = data[(data['thermal sensation'] <= 0.5) & (data['thermal sensation'] >= -0.5)][['ta']]
    # com_hr = data[(data['thermal sensation'] <= 0.5) & (data['thermal sensation'] >= -0.5)][['hr']]
    #
    # # 绘制分布图
    # plt.figure(figsize=(8, 5), dpi=80)
    # axes = plt.subplot(111)
    # label1 = axes.scatter(hot_ta, hot_hr, s=50, marker=None, c="red")
    # label2 = axes.scatter(cool_ta, cool_hr, s=50, marker='x', c="blue")
    # label3 = axes.scatter(com_ta, com_hr, s=50, marker='+', c="green")
    # plt.xlabel("temp(℃)")
    # plt.ylabel("humid(%)")
    # axes.legend((label1, label2, label3), ("hot", "cool", "comfort"), loc=3)
    # # plt.savefig('./result/pic/feedback distribution plot in ' + season + 'dataset.png')
    # plt.show()

    '''environment data'''
    va = []
    for i in range(0, data.shape[0]):
        va.append(0.7 * round(random.random(), 1))
    va = pd.DataFrame({'va': va})

    env_feature = ['ta', 'hr']
    env = pd.DataFrame(MinMaxScaler().fit_transform(data[env_feature]))
    env_data = pd.concat([env, va], axis=1)

    '''body data'''
    gender = pd.DataFrame(data['gender'].tolist())
    body_feature = ['age', 'height', 'weight', 'bmi']
    body_data = pd.DataFrame(MinMaxScaler().fit_transform(data[body_feature]))
    body = pd.concat([gender, body_data], axis=1)

    '''label data'''

    y_feature = 'thermal sensation'
    data.loc[(data[y_feature] > 0.5), y_feature] = 2
    data.loc[((-0.5 <= data[y_feature]) & (data[y_feature] <= 0.5)), y_feature] = 1
    data.loc[(data[y_feature] < -0.5), y_feature] = 0

    y = data[y_feature].values.astype(int)
    # y = data[y_feature]

    '''save data'''
    np.save('dataset/env.npy', env_data)
    np.save('dataset/body.npy', body_data)
    np.save('dataset/label.npy', y)


