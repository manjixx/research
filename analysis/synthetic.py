from analysis.utils import *

if __name__ == "__main__":

    seasons = ['summer']
    filepath = '../dataset/synthetic.csv'

    for season in seasons:
        # 读取数据
        df = read(filepath, season)

        print(df)

        # 查看各个参数与pmv投票间的相关性
        corr(df, "thermal sensation", 5, season)

        # 查看全年pmv分布图并判断是否符合高斯分布
        distribution(df, 'thermal sensation', season)
        gauss(df, "thermal sensation")

        # 查看温度分布图并判断是否符合高斯分布
        distribution(df, "ta", season)
        gauss(df, "hr")

        # 查看湿度分布图并判断是否符合高斯分布
        distribution(df, "hr", season)
        gauss(df, "hr")

        # 绘制整体冷热不适分布图
        plot_all(df, season)
