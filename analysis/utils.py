import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def read(file_path, season):
    """
    按照季节读取数据
    :param file_path:
    :param season: 读取季节
    :return:
    """
    data = pd.read_csv(file_path, encoding='gbk')

    data.dropna(axis=0, how='any', inplace=True)

    if season == 'all' or season == 'synthetic':
        return data
    else:
        return data.loc[data['season'] == season]


def corr(df, index, count, season):
    """
    相关性分析
    :param df: 原始数据集
    :param index:
    :param count: 相关性最高的个数
    :param season: 季节
    :return:
    """

    if season == 'summer':
        name = '夏季'
    elif season == "winter":
        name = '冬季'
    elif season == "synthetic":
        name = "合成"
    elif season == "all":
        name = "全年"

    result = df.corr()
    print(name + "数据中各参数与" + index + "热投票值之间的相关性分析如下：")
    print(result[index].sort_values())

    # 绘制热力图
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
    plt.rcParams['axes.unicode_minus'] = False
    plt.subplots(figsize=(14, 14), dpi=100)  # 设置画面大小
    plt.title("Heatmap of coefficient correlations in " + season + " dataset")
    sns.heatmap(result, annot=True, vmax=1, square=True, cmap="Blues")
    # plt.savefig('./result/cor/Heatmap of coefficient correlations in ' + season + ' dataset.png')
    plt.show()

    # 选取相关性最强的6个
    most_correlated = df.corr().abs()[index].sort_values(ascending=False)

    # if count:
    #     most_correlated = most_correlated[count]
    #     print(name + "数据集中各参数与" + index + "相关性最强的" + str(count) + "个参数为：")
    #     print(most_correlated)
    # else:
    print(name + "数据集中各参数与" + index + "相关性排序如下:")
    print(most_correlated)


def distribution(df, index, season):
    """
    绘制分布图
    :param df: 原始数据集
    :param index: 查看的指标
    :param season: 季节
    :return:
    """

    index_distribution = sns.kdeplot(df[index], shade=True)
    index_distribution.axes.set_title("distribution plot of " + index + " in " + season + " dataset", fontsize=10)
    index_distribution.set_xlabel(index, fontsize=10)
    index_distribution.set_ylabel('density', fontsize=10)
    plt.grid(color='k', linestyle='--', linewidth=0.5)
    # plt.savefig('./result/distribution/distribution plot of' + index + "in " + season + " dataset", dpi=200, bbox_inches='tight')
    plt.show()


def distribution_person(df, season):
    """
    查看人员热舒适分布图
    :param df:
    :param season:
    :return:
    """

    count = len(df['no'].unique())

    for i in range(0, count):
        data = df.loc[df['no'] == i + 1]

        if len(data) != 0:
            pmv_distribution = sns.kdeplot(data['thermal sensation'], shade=True)
            pmv_distribution.axes.set_title(('distribution plot of pmv for the {} \'th person in the ' + season + ' dataset').format(str(i)),
                                            fontsize=10)
            pmv_distribution.set_xlabel('pmv', fontsize=10)
            pmv_distribution.set_ylabel('count', fontsize=10)
            # plt.savefig(('./result/person/distribution plot of pmv for the {} \'th person in the ' + season + ' dataset').format(str(i)),
            #     dpi=200, bbox_inches='tight')
            plt.show()


def gauss(df, index):
    """
    判断某个指标的分布是否符合高斯分布
    :param df:
    :param index:
    :return:
    """
    u = df[index].mean()  # 计算均值
    print(index + '均值为：' + str(u))
    std = df[index].std()  # 计算标准差
    result = stats.kstest(df[index], 'norm', (u, std))
    if result.pvalue > 0.05:
        print("符合正态分布")
    else:
        print("不符合正态分布")


def plot_all(data, season):
    """
    绘制所有数据冷热分布图
    :param season: 季节
    :param data: 原始数据
    :return:
    """

    hot = []
    cool = []
    normal = []

    if season == 'synthetic':
        hot = data[(data['pmv'] > 0.5)]
        cool = data[(data['pmv'] < -0.5)]
        normal = data[(data['pmv'] <= 0.5) & (data['pmv'] >= -0.5)]
    else:
        hot = data[(data['pmv'] > 0.5)]
        cool = data[(data['pmv'] < -0.5)]
        normal = data[(data['pmv'] <= 0.5) & (data['pmv'] >= -0.5)]

    print("There are " + str(hot.shape[0]) + " pieces of hot complain in " + season + " dataset")
    print("There are " + str(cool.shape[0]) + " pieces of cool complain in " + season + " dataset")
    print("There are " + str(normal.shape[0]) + " pieces of comfort feedback in " + season + " dataset")

    hot_ta = hot[['ta']]
    hot_hr = hot[['hr']]
    cool_ta = cool[['ta']]
    cool_hr = cool[['hr']]
    com_ta = normal[['ta']]
    com_hr = normal[['hr']]

    # 绘制冷热不适分布图
    plt.figure(figsize=(8, 5), dpi=80)
    axes = plt.subplot(111)
    label1 = axes.scatter(hot_ta, hot_hr, s=50, marker=None, c="red")
    label2 = axes.scatter(cool_ta, cool_hr, s=50, marker='x', c="blue")
    plt.title('feedback distribution plot in ' + season + 'dataset.png')
    plt.xlabel("temp(℃)")
    plt.ylabel("humid(%)")
    axes.legend((label1, label2), ("hot", "cool"), loc=2)
    # plt.savefig('./result/pic/feedback distribution plot in ' + season + 'dataset.png')
    plt.show()

    # 绘制分布图
    plt.figure(figsize=(8, 5), dpi=80)
    axes = plt.subplot(111)
    label1 = axes.scatter(hot_ta, hot_hr, s=50, marker=None, c="red")
    label2 = axes.scatter(cool_ta, cool_hr, s=50, marker='x', c="blue")
    label3 = axes.scatter(com_ta, com_hr, s=50, marker='+', c="green")
    plt.title('feedback distribution plot in ' + season + 'dataset.png')
    plt.xlabel("temp(℃)")
    plt.ylabel("humid(%)")
    axes.legend((label1, label2, label3), ("hot", "cool", "comfort"), loc=3)
    # plt.savefig('./result/pic/feedback distribution plot in ' + season + 'dataset.png')
    plt.show()


def plot_bg(data, index, down, up, season):
    """
    根据bmi、griffiths分类，绘制每个群体成员冷热分布图
    :param season: 季节
    :param data: 原始数据
    :param down:分类下界
    :param up:分类上界
    :param index: 分类指标

    :return:
    """

    low = data[(data[index] <= down)]
    mid = data[(data[index] < up) & (data[index] > down)]
    high = data[(data[index] >= up)]

    print(season + "数据集中" + index + " <= " + str(down) + "的数据共计有" + str(low.shape[0]) + "条！")
    print(season + "数据集中" + str(down) + " < " + index + " < " + str(up) + "的数据共计" + str(mid.shape[0]) + "条！")
    print(season + "数据集中" + index + " >= " + str(up) + "的数据共计有" + str(high.shape[0]) + "条！")

    if season == 'synthetic':
        low = low[['pmv', 'ta', 'hr']]
        mid = mid[['pmv', 'ta', 'hr']]
        high = high[['pmv', 'ta', 'hr']]
    else:    # 初始化数组
        low = low[['thermal sensation', 'ta', 'hr']]
        mid = mid[['thermal sensation', 'ta', 'hr']]
        high = high[['thermal sensation', 'ta', 'hr']]

    arr = {'low': low,
           'mid': mid,
           'high': high}

    plot(arr, index)


def plot_sp(data, index, l1, l2, l3, season):
    """
    根据preference、sensitivity分类绘制不同群体的冷热分布图
    :param season: 季节
    :param data: 原始数据
    :param index: 分类指标
    :param l1: 层级1
    :param l2: 层级2
    :param l3: 层级3
    :return:
    """

    level1 = data[(data[index] == l1)][['thermal sensation', 'ta', 'hr']]
    level2 = data[(data[index] == l2)][['thermal sensation', 'ta', 'hr']]
    level3 = data[(data[index] == l3)][['thermal sensation', 'ta', 'hr']]

    print(season + "数据集中" + index + "等级为" + str(l1) + "的数据共计有" + str(level1.shape[0]) + "条！")
    print(season + "数据集中" + index + "等级为" + str(l2) + "的数据共计有" + str(level2.shape[0]) + "条！")
    print(season + "数据集中" + index + "等级为" + str(l3) + "的数据共计有" + str(level3.shape[0]) + "条！")
    print("***************************************************")

    arr = {'low': level1,
           'mid': level2,
           'high': level3
           }

    plot(arr, index)


def plot(arr, index):
    """
    根据传入参数绘制图形
    :param index: 绘制指标
    :param arr: 原始数据
    :return:
    """

    key_list = list(arr.keys())
    val_list = list(arr.values())
    names = {}

    if index == 'bmi':
        names = {
            "low": "bmi <= 18 ",
            "mid": "18 < bmi < 24 ",
            "high": "bmi >= 24 ",
        }
    elif index == 'sensitivity':
        names = {
            "low": "sensitive = 0 ",
            "mid": "sensitive = 0 ",
            "high": "sensitive = 0 ",
        }
    elif index == 'preference':
        names = {
            "low": "preference is cold ",
            "mid": "preference is normal ",
            "high": "preference is hot ",
        }
    elif index == 'griffith':
        names = {
            "low": "griffith <= 0.8 ",
            "mid": "0.8 < griffith < 1.2 ",
            "high": "griffith >= 1.2 ",
        }
    for i in range(0, len(key_list)):
        a = val_list[i]
        name = names.get(key_list[i])
        hot_ta = a[(a['thermal sensation'] > 0.5)][['ta']]
        hot_hr = a[(a['thermal sensation'] > 0.5)][['hr']]
        hot = hot_ta.shape[0]
        cool_ta = a[(a['thermal sensation'] < -0.5)][['ta']]
        cool_hr = a[(a['thermal sensation'] < -0.5)][['hr']]
        cool = cool_ta.shape[0]
        com_ta = a[(a['thermal sensation'] <= 0.5) & (a['thermal sensation'] >= -0.5)][['ta']]
        com_hr = a[(a['thermal sensation'] <= 0.5) & (a['thermal sensation'] >= -0.5)][['hr']]
        com = com_ta.shape[0]

        # 绘制冷热不适分布图
        plt.figure(figsize=(8, 5), dpi=80)
        axes = plt.subplot(111)
        label1 = axes.scatter(hot_ta, hot_hr, s=50, marker=None, c="red")
        label2 = axes.scatter(cool_ta, cool_hr, s=50, marker='x', c="blue")
        plt.xlabel("temp(℃)")
        plt.ylabel("humid(%)")
        plt.title(name + "hot and cold distribution map")
        axes.legend((label1, label2), ("hot", "cool"), loc=2)
        # plt.savefig('./result/BMIgt24.jpg')
        plt.show()

        # 绘制分布图
        plt.figure(figsize=(8, 5), dpi=80)
        axes = plt.subplot(111)
        label1 = axes.scatter(hot_ta, hot_hr, s=50, marker=None, c="red")
        label2 = axes.scatter(cool_ta, cool_hr, s=50, marker='x', c="blue")
        label3 = axes.scatter(com_ta, com_hr, s=50, marker='+', c="green")
        plt.xlabel("temp(℃)")
        plt.ylabel("humid(%)")
        plt.title(name + "distribution map")
        axes.legend((label1, label2, label3), ("hot", "cool", "comfort"), loc=3)
        # plt.savefig('./result/BMIgt24.jpg')
        plt.show()



