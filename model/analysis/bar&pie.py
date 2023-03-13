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
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    # 先用pandas读入csv
    df = pd.read_csv('../../dataset/2021.csv', encoding='gbk').dropna(axis=0, how='any', inplace=False)
    # df = df.loc[df['season'] == 'summer']
    no = np.array(df['no'].unique())

    '''查看BMI分布'''
    bmi = df[['no', 'bmi']]
    bmi.drop_duplicates('no', keep='first', inplace=True)
    # 直方图
    plt.hist(bmi['bmi'], edgecolor='white')
    plt.title("实验人员BMI直方图")
    plt.show()
    # 饼图
    thin = bmi[(bmi['bmi'] < 18)][['bmi']].shape[0]
    weight = bmi[(bmi['bmi'] > 24)][['bmi']].shape[0]
    norm = bmi['bmi'].shape[0] - thin - weight
    x = [thin, weight, norm]
    labels = ['thin', 'weight', 'norm']
    plt.title("实验人员BMI饼图")
    plt.pie(x, labels=labels, autopct='%.2f%%')
    plt.show()

    '''查看主观热敏感度分布'''
    sensitivity = df[['no', 'sensitivity']]
    sensitivity.drop_duplicates('no', keep='first', inplace=True)
    normal = sensitivity[(sensitivity['sensitivity'] == 0)][['sensitivity']].shape[0]
    slightly = sensitivity[(sensitivity['sensitivity'] == 1)][['sensitivity']].shape[0]
    very = sensitivity[(sensitivity['sensitivity'] == 2)][['sensitivity']].shape[0]
    x = [0, 1, 2]
    y = [normal, slightly, very]
    # 直方图
    x = range(0, 3, 1)
    plt.bar(x=x, height=y, edgecolor='white', align='center', width=0.6)
    plt.xticks(x)
    plt.title("实验人员主观热敏感度直方图")
    plt.show()
    # 饼图

    labels = ['normal', 'slightly', 'very']
    plt.title("实验人员主观热敏感度饼图")
    plt.pie(y, labels=labels, autopct='%.2f%%')
    plt.show()

    '''查看主观热偏好分布'''
    preference = df[['no', 'preference']]
    preference.drop_duplicates('no', keep='first', inplace=True)
    cold = preference[(preference['preference'] == -1)][['preference']].shape[0]
    normal = preference[(preference['preference'] == 0)][['preference']].shape[0]
    warm = preference[(preference['preference'] == 1)][['preference']].shape[0]
    x = [-1, 0, 1]
    y = [cold, normal, warm]
    # 直方图
    # 设置figsize的大小
    plt.figure(figsize=(4, 5), dpi=80)
    plt.bar(x=x, height=y, edgecolor='white', align='center', width=0.6)
    plt.xticks(x)
    plt.title("实验人员主观热偏好直方图")
    plt.show()
    # 饼图
    labels = ['cold', 'normal', 'warm']
    plt.title("实验人员主观热偏好饼图")
    plt.pie(y, labels=labels, autopct='%.2f%%')
    plt.show()

    '''查看格里菲斯常数字分布'''
    griffith = df[['no', 'griffith']]
    griffith.drop_duplicates('no', keep='first', inplace=True)

    # 直方图
    plt.hist(griffith['griffith'], edgecolor='white')
    plt.title("实验人员格里菲斯常数直方图")
    plt.show()

    # 饼图
    part1 = griffith[(griffith['griffith'] < 0.8)][['griffith']].shape[0]
    part3 = griffith[(griffith['griffith'] > 1.5)][['griffith']].shape[0]
    part2 = griffith.shape[0] - cold - warm
    y = [part1, part2, part3]
    labels = ['g < 0.8', '0.8 <= g <= 1.5 ', 'g > 1.5']
    plt.title("实验人员格里菲斯常数饼图")
    plt.pie(y, labels=labels, autopct='%.2f%%')
    plt.show()


    # y_feature = 'thermal sensation'
    # df.loc[(df[y_feature] > 0.5), y_feature] = 2
    # df.loc[((-0.5 <= df[y_feature]) & (df[y_feature] <= 0.5)), y_feature] = 1
    # df.loc[(df[y_feature] < -0.5), y_feature] = 0
    #
    # hot_ta = df[(df['thermal sensation'] == 2)][['ta']]
    # hot_hr = df[(df['thermal sensation'] == 2)][['hr']]
    # cool_ta = df[(df['thermal sensation'] == 0)][['ta']]
    # cool_hr = df[(df['thermal sensation'] == 0)][['hr']]
    # com_ta = df[(df['thermal sensation'] == 1)][['ta']]
    # com_hr = df[(df['thermal sensation'] == 1)][['hr']]
    #
    # print(type(hot_ta))
    #
    # # hot_ta.hist(color='r')
    # # com_ta.hist(color='g')
    # # cool_ta.hist(color='b')
    # #
    # # plt.show()
    # #
    # # hot_hr.hist(color='r')
    # # com_hr.hist(color='g')
    # # cool_hr.hist(color='b')
    # # plt.show()
    #
    # # 绘制分布图
    # plt.figure(figsize=(8, 5), dpi=80)
    # axes = plt.subplot(111)
    # plt.title('all')
    # plt.xlabel("temp(℃)")
    # plt.ylabel("humid(%)")
    # plt.plot(com_ta, com_hr, "o", marker='+', c="green", )
    # # plt.show()
    # len = {}
    # for n in no:
    #     # if n < 56:
    #     #     continue
    #     data = df.loc[df['no'] == n]
    #     l = data.shape[0]
    #     len.update({n: l})
    #     hot_ta = data[(data['thermal sensation'] == 2)][['ta']]
    #     hot_hr = data[(data['thermal sensation'] == 2)][['hr']]
    #     cool_ta = data[(data['thermal sensation'] == 0)][['ta']]
    #     cool_hr = data[(data['thermal sensation'] == 0)][['hr']]
    #     com_ta = data[(data['thermal sensation'] == 1)][['ta']]
    #     com_hr = data[(data['thermal sensation'] == 1)][['hr']]
    #
    #     # hot_ta.hist(color='r')
    #     # com_ta.hist(color='g')
    #
    #     # cool_ta.hist(color='b')
    #
    #     # 绘制分布图
    #     plt.figure(figsize=(8, 5), dpi=80)
    #     axes = plt.subplot(111)
    #     label1 = axes.scatter(hot_ta, hot_hr, s=50, marker=None, c="red")
    #     label2 = axes.scatter(cool_ta, cool_hr, s=50, marker='x', c="blue")
    #     label3 = axes.scatter(com_ta, com_hr, s=50, marker='+', c="green")
    #     plt.title(n)
    #     plt.xlabel("temp(℃)")
    #     plt.ylabel("humid(%)")
    #     axes.legend((label1, label2, label3), ("hot", "cool", "comfort"), loc=3)
    #     plt.show()
    #
    # for key, value in len.items():
    #     print('%s:%s' % (key, value))
