from utils import *
from scipy.stats import linregress



def cal_griffith(data):
    griffith = []
    for i in range(50):

        data_person = data.loc[data["no"] == i + 1]

        temp = data_person['ta']

        pmv = data_person['pmv']

        scatter = sns.scatterplot(x=pmv, y=temp)

        scatter.set_xlabel('pmv')

        scatter.set_ylabel('temp')

        statistics = linregress(pmv, temp)

        griffith.append(statistics.slope)

        print(str(i + 1) + '号温度与PMV的斜率为' + str(statistics.slope))

    return griffith


def save(filepath, data, griffith):

    with open(filepath, "a", encoding='utf-8', newline='') as fs:
        for i in range(len(data)):
                no = data.iloc[i, 0]
                gender = data.iloc[i, 1]
                age = data.iloc[i, 2]
                height = data.iloc[i, 3]
                weight = data.iloc[i, 4]
                bmi = round(data.iloc[i, 5], 2)
                season = data.iloc[i, 6]
                ta = data.iloc[i, 7]
                hr = data.iloc[i, 8]
                g = round(griffith[no - 1], 2)
                pmv = data.iloc[i, 9]
                datalist = [no, gender, age, height, weight, bmi, g, pmv, season, ta, hr]
                print(datalist)
                csv_write = csv.writer(fs)
                csv_write.writerow(datalist)


if __name__ == "__main__":

    # 读取原始数据
    filepath = '../dataset/synthetic_dataset.csv'
    df = read(filepath)

    # 计算格里菲斯常数
    griffiths = cal_griffith(df)

    # 写入新文件
    fieldnames = ['no', 'gender', 'age', 'height', 'weight', 'bmi', 'griffith',
                  'thermal sensation', 'season', 'ta', 'hr']

    write_header('../dataset/synthetic.csv', fieldnames)

    save('../dataset/synthetic.csv', df, griffiths)

