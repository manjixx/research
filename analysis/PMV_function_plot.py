# -*- coding: utf-8 -*-
"""
@Project ：data- mechanism 
@File ：PMV_function_plot.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/3/9 10:13 
"""

# mpl_toolkits是matplotlib官方的工具包 mplot3d是用来画三维图像的工具包
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
import numpy as np

# 绘制z=x^2+y^2的3D图# 创建一个图像窗口
fig = plt.figure()
# 在图像窗口添加3d坐标轴
ax = Axes3D(fig)
# 使用np.arange定义 x:范围(-10,10);间距为0.1
ta = np.arange(20, 30, 1)
pa = 10 ** (8.07131 - 1730.63 / (233.426 + ta))
psk = 10 ** (8.07131 - 1730.63 / (233.426 + 33.7))
hc = 12.1 * math.sqrt(0.8)
hr = 6
fcl = 1 / (1 + (hr + hc) * 0.8)
fpcl = 1 / (1 + 0.923 * hc * 0.8)
M = 1.2
# 使用np.arange定义 y:范围(-10,10);间距为0.1
phia = np.arange(60, 80, 5)
# 创建x-y平面网络
ta, phia = np.meshgrid(ta, phia)
# 定义函数z=x^2+y^2

S = M * (1 - 0.0023 * (44 - phia * pa)) \
    - 2.2 * hc * (0.06 + 0.94 * 0.5) * (psk - phia * pa) * fpcl \
    - (hr + hc) * (33.7 - ta) * fcl

pmv = (0.303 * math.exp(-0.036 * M) + 0.028) * S

# 将函数显示为3d  rstride 和 cstride 代表 row(行)和column(列)的跨度 cmap为色图分类
ax.plot_surface(ta, phia, pmv, rstride=1, cstride=1, cmap='rainbow')
plt.show()
