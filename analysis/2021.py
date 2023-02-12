from analysis.utils import *

if __name__=="__main__":

    seasons = ['all']

    for season in seasons:
        # 读取数据
        filepath = '../dataset/2021.csv'
        df = read(filepath, season)

        # 查看各个参数与pmv投票间的相关性
        corr(df, "thermal sensation", 5, season)

        # 查看全年pmv分布图并判断是否符合高斯分布
        distribution(df, "thermal sensation", season)
        gauss(df, "thermal sensation")

        # 查看温度分布图并判断是否符合高斯分布
        distribution(df, "ta", season)
        gauss(df, "hr")

        # 查看湿度分布图并判断是否符合高斯分布
        distribution(df, "hr", season)
        gauss(df, "hr")

        # 绘制整体冷热不适分布图
        plot_all(df, "summer")

        # 绘制按照bmi和格里菲斯常数分类的冷热不适分布图
        plot_bg(df, 'bmi', 18, 24, season)
        plot_bg(df, 'griffith', 0.8, 1.2, season)

        # 绘制按照sensitive与preference分类的冷热不适应分布图
        plot_sp(df, 'sensitivity', 0, 1, 2, season)
        plot_sp(df, 'preference', -1, 0, 1, season)

        # 查看2021年全年每个人的pmv分布图
        distribution_person(df, season)




