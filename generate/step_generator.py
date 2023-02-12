import numpy as np
import random
from utils import *
from pmv import pmv_model
from constant import *


def step_generator(filepath):
    """
    随机生成器
    :param filepath: 文件保存路径
    :param count: 人数
    :return:
    """
    no = 0
    with open(filepath, "a", encoding='utf-8', newline='') as fs:
        csv_write = csv.writer(fs)
        for g in gender:
            for a in age:
                for bmi in BMI:
                    if g == 0:
                        for h in female_height:
                            no += 1
                            w = round(bmi * (h / 100) ** 2, 1)
                            for s in season:
                                if s == 'summer':
                                    clo = summer_clo
                                    for vel in air_vel:
                                        for ta in summer_ta:
                                            for rh in summer_rh:
                                                pmv = pmv_model(M=met * 58.15, clo=clo, tr=ta, ta=ta, vel=vel, rh=rh)
                                                data = [no, g, a, h, w, bmi, s, ta, rh, round(pmv, 2)]
                                                print(data)
                                                csv_write.writerow(data)
                                if s == 'winter':
                                    clo = winter_clo
                                    for vel in air_vel:
                                        for ta in winter_ta:
                                            for rh in winter_rh:
                                                pmv = pmv_model(M=met * 58.15, clo=clo, tr=ta, ta=ta, vel=vel, rh=rh)
                                                data = [no, g, a, h, w, bmi, s, ta, rh, round(pmv, 2)]
                                                print(data)
                                                csv_write.writerow(data)
                    if g == 1:
                        for h in male_height:
                            no += 1
                            w = round(bmi * (h / 100) ** 2, 1)
                            for s in season:
                                if s == 'summer':
                                    clo = summer_clo
                                    for vel in air_vel:
                                        for ta in summer_ta:
                                            for rh in summer_rh:
                                                pmv = pmv_model(M=met * 58.15, clo=clo, tr=ta, ta=ta, vel=vel, rh=rh)
                                                data = [no, g, a, h, w, bmi, s, ta, rh, round(pmv, 2)]
                                                print(data)
                                                csv_write.writerow(data)
                                if s == 'winter':
                                    clo = winter_clo
                                    for vel in air_vel:
                                        for ta in winter_ta:
                                            for rh in winter_rh:
                                                pmv = pmv_model(M=met * 58.15, clo=clo, tr=ta, ta=ta, vel=vel, rh=rh)
                                                data = [no, g, a, h, w, bmi, s, ta, rh, round(pmv, 2)]
                                                print(data)
                                                csv_write.writerow(data)


