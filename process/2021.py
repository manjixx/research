from utils import *
from scipy.stats import linregress


def cal_griffith(data):

    data.dropna(axis=0, how='any', inplace=True)

    griffith = []
    for i in range(57):
        data_person = data.loc[data["no"] == i + 1]

        data_person = data_person[['thermal sensation', 'temp']]

        temp = data_person['temp']

        pmv = data_person['thermal sensation']

        scatter = sns.scatterplot(x=pmv, y=temp)

        scatter.set_xlabel('pmv')

        scatter.set_ylabel('temp')

        statistics = linregress(pmv, temp)

        griffith.append(statistics.slope)

        print(str(i + 1) + '号温度与PMV的斜率为' + str(statistics.slope))
    return griffith


def save(filepath, data, griffith):
    for i in range(len(data)):
        with open(filepath, "a", encoding='utf-8', newline='') as fs:
            no = data.iloc[i, 0]
            gender = data.iloc[i, 1]
            age = data.iloc[i, 2]
            height = data.iloc[i, 3]
            weight = data.iloc[i, 4]
            bmi = round(data.iloc[i, 5], 2)
            season = data.iloc[i, 6]
            preference = data.iloc[i, 7]
            sensitivity = data.iloc[i, 8]
            environment = data.iloc[i, 9]
            date = data.iloc[i, 10]
            time = data.iloc[i, 11]
            room = data.iloc[i, 12]
            thermal_sensation = data.iloc[i, 13]
            thermal_comfort = data.iloc[i, 14]
            thermal_preference = data.iloc[i, 15]
            temp = data.iloc[i, 16]
            humid = data.iloc[i, 17]
            grif = round(griffith[no - 1], 2)
            datalist = [no, gender, age, height, weight, bmi, preference, sensitivity, environment, grif,
                        thermal_sensation, thermal_comfort, thermal_preference,
                        season, date, time, room, temp, humid]
            print(datalist)
            csv_write = csv.writer(fs)
            csv_write.writerow(datalist)


if __name__ == "__main__":
    # 读取原始数据集
    filepath = '../dataset/2021_dateset.csv'
    df = read(filepath)

    # 计算griffiths常数
    griffiths = cal_griffith(df)

    '''
        写入头文件
        gender:0 female,1: male
        height:cm
        weight:kg
        preference: -1:cool 0:normal 1:warm
        sensitivity:0:insensitivity 1:slight 2:very
        environment: -1:cool 0:normal 1:warm
        thermal comfort: 1:comfort,0:uncomfort
        thermal preference: -1:cooler,0:not change,1:warmer
        temp: 摄氏度
        humid:%
        griffith:
    '''

    fieldnames = ['no', 'gender', 'age', 'height', 'weight', 'bmi',
                  'preference', 'sensitivity', 'environment', 'griffith',
                  'thermal sensation', 'thermal comfort', 'thermal preference',
                  'season', 'date', 'time', 'room', 'ta', 'hr']

    write_header('../dataset/2021.csv', fieldnames)

    save('../dataset/2021.csv', df, griffiths)
