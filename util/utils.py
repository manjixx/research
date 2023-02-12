import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def write_header(filepath, fieldnames):
    with open(filepath, "a", encoding='utf-8', newline='') as fs:
        fieldnames = fieldnames
        writer = csv.DictWriter(fs, fieldnames=fieldnames)
        writer.writeheader()


def read(filepath,season):
    """
    读取数据
    :param filepath: 文件路径
    :param season: 季节
    :return:
    """
    df = pd.read_csv(filepath, encoding="gbk")

    df.dropna(axis=0, how='any', inplace=True)

    if season:
        return df.loc[df['season'] == season]
    else:
        return df


def corr(df, name, index, count):

    """
    相关性分析
    :param df: 原始数据集
    :param name：数据集名称
    :param index:相关性分析主题
    :param count: 相关性最高的个数
    :return:
    """

    result = df.corr()
    print(name + "数据集中各参数与"+ index + "之间的相关性分析如下：")
    result1 = df.corr()[index].sort_values()
    print(result1)

    # 绘制热力图
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
    plt.rcParams['axes.unicode_minus'] = False
    plt.subplots(figsize=(14, 14), dpi=100)  # 设置画面大小
    sns.heatmap(result, annot=True, vmax=1, square=True, cmap="Blues")
    plt.savefig('./' + name + '数据集中参数相关系数热力图.png')
    plt.show()

    # 选取相关性最强的6个
    most_correlated = df.corr().abs()[index].sort_values(ascending=False)

    if count:
        most_correlated = most_correlated[count]
        print(name + "数据集中各参数与" + index + "相关性最强的" + str(count) + "个参数依次为：")
        print(most_correlated)
    else:
        print(name + "数据集中各参数与 "+  index + "相关性排序如下:")
        print(most_correlated)

def scatter(data,index):
