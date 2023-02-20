# -*- coding: utf-8 -*-
"""
@Project ：data- mechanism 
@File ：distance.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/2/17 15:03 
"""
import math
import numpy as np


class Distance(object):

    def __init__(self, w):
        self.w = w

    # 曼哈顿距离
    def manhattan(self, a, b):
        return np.sum(np.abs(self.w * (a - b)), axis=0)

    # 曼哈顿距离反函数
    def manhattan_inverse(self, a, b):
        dist = np.sum(np.abs(self.w * (a - b)), axis=0)
        return 1 / dist

    # 曼哈顿高斯
    def manhattan_gauss(self, a, b):
        h = 1
        o = 0
        wid = 0.3
        dist = np.sum(np.abs(self.w * (a - b)), axis=0)
        return h * math.e ** (-(dist - o) ** 2 / (2 * wid ** 2))

    # 欧式距离
    def euclid(self, a, b):
        return np.sqrt(np.sum((self.w * (a - b)) ** 2, axis=0).astype('float'))

    # 欧式距离反函数
    def euclid_inverse(self, a, b):
        dist = np.sqrt(np.sum((self.w * (a - b)) ** 2, axis=0).astype('float'))
        return 1 / dist

    # 高斯距离加权
    def euclid_gauss(self, a, b):
        h = 1
        o = 0
        wid = 0.3
        dist = np.sqrt(np.sum((self.w * (a - b)) ** 2, axis=0).astype('float'))
        return h * math.e ** (-(dist - o) ** 2 / (2 * wid ** 2))

