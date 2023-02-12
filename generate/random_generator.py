import numpy as np
import random
from utils import *
from pmv import pmv_model


def random_generator(filepath, count):
    """
    随机生成器
    :param filepath: 文件保存路径
    :param count: 人数
    :return:
    """
    # 代谢率，衣服隔热，风速
    m = 1.1
    clo_s = 0.50
    clo_w = 0.818
    vel = 0.1

    # 生成夏季数据并保存
    with open(filepath, "a", encoding='utf-8', newline='') as fs:
        for i in range(0, count):
            age = np.random.randint(20, 30)
            h = 0.0
            w = 0.0
            bmi = round(np.random.uniform(17, 26), 2)
            season = ""
            ta = 0.0
            rh = 0.0
            gender = 0  # 0:female,1:male
            pmv = 0
            if i % 2 == 0:
                gender = 0
                h = round(random.gauss(160, 8), 1)
                w = round(bmi * (h / 100) ** 2, 1)
            else:
                gender = 1
                h = round(random.gauss(175, 8), 2)
                w = round(bmi * (h / 100) ** 2, 2)

            for r in range(0, 50):
                for j in range(0, 2):
                    if j == 0:
                        season = "summer"
                        ta = round(np.random.uniform(18, 30), 1)
                        rh = round(np.random.uniform(60, 81), 1)
                        # rh = round(rh_s[r], 2)
                        pmv = pmv_model(M=m * 58.15, clo=clo_s, tr=ta, ta=ta, vel=vel, rh=rh)
                    else:
                        season = "winter"
                        ta = round(np.random.uniform(18, 30), 1)
                        # rh = round(rh_w[r], 2)
                        rh = round(np.random.uniform(10, 31), 1)
                        pmv = pmv_model(M=m * 58.15, clo=clo_w, tr=ta, ta=ta, vel=vel, rh=rh)
                    # pmv = Decimal(pmv).quantize(Decimal('0.00'), rounding=ROUND_HALF_UP)
                    data = [i + 1, gender, age, h, w, bmi, season, ta, rh, round(pmv, 2)]
                    print(data)
                    csv_write = csv.writer(fs)
                    csv_write.writerow(data)

