import numpy as np

from utils import *
from scipy.stats import linregress
from pmv import pmv_model


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


def cal_m(data):
    result = []
    data = np.array(data[['gender(female:0,male:1)', 'age', 'height(cm)', 'weight(Kg)']]).tolist()
    for i in range(len(data)):
        if data[i][0] == 0:
            m = 661+9.6 * data[i][3]+1.72 * data[i][2]-4.7 * data[i][1]
        else:
            m = 66.5+13.73 * data[i][3]+5 * data[i][2]-6.9 * data[i][1]
        result.append(m)
    return result


def cal_pmv_res(data):
    result = []
    res = []
    m = 1.1
    clo_s = 0.50
    clo_w = 0.818
    vel = 0.1
    vote = data['thermal sensation']
    data = np.array(data[['season', 'temp', 'humid']]).tolist()

    for i in range(0, len(data)):
        ta = data[i][1]
        rh = data[i][2]
        if data[i][0] == 'summer':
            clo = clo_s
        else:
            clo = clo_w
        pmv = pmv_model(M=m * 58.15, clo=clo, tr=ta, ta=ta, vel=vel, rh=rh)
        r = vote[i] - pmv
        res.append(round(r, 2))
        result.append(round(pmv,2))
    return res, result


def save(filepath, data, griffith, pmv, res):
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
            p = round(pmv[i], 2)
            r = res[i]
            datalist = [no, gender, age, height, weight, bmi, preference, sensitivity, environment, grif,
                        thermal_sensation, thermal_comfort, thermal_preference,
                        season, date, time, room, temp, humid, p, r]
            print(datalist)
            csv_write = csv.writer(fs)
            csv_write.writerow(datalist)



if __name__ == "__main__":
    # 读取原始数据集
    filepath = '../dataset/2021_dateset.csv'
    df = read(filepath)

    # 计算griffiths常数
    griffiths = cal_griffith(df)
    res, pmv = cal_pmv_res(df)
    m = cal_m(df)
    print(res)
    print(pmv)
    print(m)

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
                  'season', 'date', 'time', 'room', 'ta', 'hr', 'pmv', 'res']

    # write_header('../dataset/2021_pmv_res.csv', fieldnames)
    #
    # save('../dataset/2021_pmv_res.csv', df, griffiths, pmv, res)
