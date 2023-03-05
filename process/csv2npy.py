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
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # 先用pandas读入csv
    data = pd.read_csv('../dataset/2021.csv', encoding='gbk').dropna(axis=0, how='any', inplace=False)

    va = []
    for i in range(0, data.shape[0]):
        va.append(0.7 * round(random.random(), 1))
    va = pd.DataFrame({'va': va})   #'A'是columns，对应的是list

    env_feature = ['ta', 'hr']
    env_data = data[env_feature]
    env_data = pd.concat([env_data, va], axis=1)

    body_feature = ['gender', 'age', 'height', 'weight', 'bmi']

    body_data = data[body_feature]

    y_feature = 'thermal sensation'

    data.loc[(data[y_feature] > 0.5), y_feature] = 1
    data.loc[((-0.5 <= data[y_feature]) & (data[y_feature] <= 0.5)), y_feature] = 0
    data.loc[(data[y_feature] < -0.5), y_feature] = -1

    y = data[y_feature].values.astype(int)
    print(env_data)
    print(body_data)
    print(y)
    # 再使用numpy保存为npy
    np.save('../dataset/npy/env.npy', env_data)
    np.save('../dataset/npy/body.npy', body_data)
    np.save('../dataset/npy/label.npy', y)


