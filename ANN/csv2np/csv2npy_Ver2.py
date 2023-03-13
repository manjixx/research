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
    data = df.loc[df['season'] == 'summer']

    '''environment data'''
    va = []
    for i in range(0, data.shape[0]):
        va.append(0.7 * round(random.random(), 1))
    va = pd.DataFrame({'va': va})

    env_feature = ['ta', 'hr']
    env = pd.DataFrame(np.array(data[env_feature]))
    env_data = pd.concat([env, va], axis=1)
    print(env_data)

    '''body data'''
    gender = data['gender']
    body_feature = ['age', 'height', 'weight', 'bmi']
    body = data[body_feature]

    '''label data'''

    y_feature = 'thermal sensation'
    data.loc[(data[y_feature] > 0.5), y_feature] = 2
    data.loc[((-0.5 <= data[y_feature]) & (data[y_feature] <= 0.5)), y_feature] = 1
    data.loc[(data[y_feature] < -0.5), y_feature] = 0

    y = data[y_feature].values.astype(int)
    # y = data[y_feature]

    '''save data'''
    np.save('../dataset/experimental_v2/env.npy', env_data)
    np.save('../dataset/experimental_v2/gender.npy', gender)
    np.save('../dataset/experimental_v2/body.npy', body)
    np.save('../dataset/experimental_v2/label.npy', y)


