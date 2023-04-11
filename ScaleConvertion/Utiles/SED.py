# -*- coding: utf-8 -*-
# @Time    : 2022/7/11 15:40
# @Author  : qxcnwu
# @FileName: SED.py
# @Software: PyCharm
import math
import os
import tkinter
from datetime import datetime
from tkinter import filedialog
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from tqdm import tqdm
import colour


class SED:
    def __init__(self, path: str = None, data_line: int = 33):
        """
        解析sed文件
        Args:
            path:
            data_line:
        """
        self.path = path
        self.data_line = data_line
        self.version = None
        self.File_Name = None
        self.USER_FIELD1 = None
        self.USER_FIELD2 = None
        self.USER_FIELD3 = None
        self.USER_FIELD4 = None
        self.Instrument = None
        self.Detectors = None
        self.Measurement = None
        self.Date = None
        self.Time = None
        self.Temperature_C = None
        self.Battery_Voltage = None
        self.Averages = None
        self.Integration = None
        self.Dark_Mode = None
        self.Foreoptic: None
        self.Radiometric_Calibration = None
        self.Units = None
        self.Wavelength_Range = None
        self.Latitude = None
        self.Longitude = None
        self.Altitude = None
        self.GPS_Time = None
        self.Satellites = None
        self.Calibrated_Reference_Correction = None
        self.File = None
        self.Channels = None
        self.ref = None
        if path:
            self.__parsePath()

    def __parsePath(self):
        """
        解析sed文件
        Returns:
        """
        assert os.path.exists(self.path), "No such file "+self.path
        fd = open(self.path, 'r')
        lines = fd.readlines()
        for i in range(0, self.data_line):
            lines[i] = lines[i].replace("\n", "")
            # 解析元数据
            kv = lines[i].split(":")
            if (len(kv) >= 2):
                key = kv[0].replace(" ", "_").replace("(", "").replace(")", "")
                value = []
                for v in kv[1:]:
                    for vs in v.split(","):
                        value.append(vs)
                if len(value) == 1:
                    value = value[0]
                self.__setattr__(key, value)
        temp = ""
        for i in self.GPS_Time:
            temp += i+":"
        self.GPS_Time = temp.strip(":")
        # 解析反射率测量数据
        self.ref = np.array(pd.read_csv(self.path, delimiter="\t",
                            skiprows=self.data_line, header=None, index_col=False))
        return

    def save_json(self, save_path: str):
        """
        保存sed为json文件
        Args:
            save_path:

        Returns:

        """
        dic = {}
        for field in self.__dict__.keys():
            value = self.__getattribute__(field)
            if isinstance(value, (str, int, float, list, dict, tuple, set)):
                dic.update({field: value})
            else:
                dic.update({field: np.array(value).tolist()})
        with open(save_path, "w") as fd:
            json.dump(dic, fd)
        fd.close()
        return

    @staticmethod
    def load_json(json_path: str):
        """
        加载json文件生成SED对象
        Args:
            json_path:
        Returns:
        """
        sed = SED()
        with open(json_path, "r") as fd:
            dic = json.load(fd)
        for k, v in dic.items():
            sed.__setattr__(k, v)
        sed.ref = np.array(sed.ref)
        return sed

    def draw_line(self, draw_index: List[int], show=False, save_path=None):
        """
        绘制波普曲线
        Args:
            draw_index: 绘制线条索引[1,2,3]
            show: 是否显示
            save_path: 保存路径
        Returns:
        """
        plt.title(str(self.Longitude)+" "+str(self.Latitude) +
                  " "+str(self.Date[0])+" "+self.GPS_Time)
        labels = ["wvl", "ref(rad)", "target(rad)", "reflect"]
        plt.xlabel("wvl")
        for i in draw_index:
            plt.plot(self.ref[:, 0], self.ref[:, i], label=labels[i])
        plt.legend()
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)
        plt.close()
        return


def compute_angle(time: str, lon: float, lat: float, TimeZone=8):
    """
    计算太阳高度角
    计算太阳方位角
    Args:
        time: 日期
        lon: 经度
        lat: 纬度
    Returns:
    """
    FORMAT = "%Y-%m-%d %H:%M:%S"
    t = datetime.strptime(time, FORMAT)

    year = t.year
    hour = t.hour
    month = t.month
    day = t.day
    sec = t.second
    min = t.minute

    # 儒略日 Julian day(由通用时转换到儒略日)
    JD0 = int(365.25 * (year - 1)) + int(30.6001 *
                                         (1 + 13)) + 1 + hour / 24 + 1720981.5
    if month <= 2:
        JD2 = int(365.25 * (year - 1)) + int(30.6001 *
                                             (month + 13)) + day + hour / 24 + 1720981.5
    else:
        JD2 = int(365.25 * year) + int(30.6001 * (month + 1)) + \
            day + hour / 24 + 1720981.5
    # 年积日 Day of year
    DOY = JD2 - JD0 + 1

    N0 = 79.6764 + 0.2422 * (year - 1985) - int((year - 1985) / 4.0)
    sitar = 2 * math.pi * (DOY - N0) / 365.2422
    ED1 = 0.3723 + 23.2567 * math.sin(sitar) + 0.1149 * math.sin(2 * sitar) - 0.1712 * math.sin(
        3 * sitar) - 0.758 * math.cos(sitar) + 0.3656 * math.cos(2 * sitar) + 0.0201 * math.cos(3 * sitar)
    ED = ED1 * math.pi / 180  # ED本身有符号

    if lon >= 0:
        if TimeZone == -13:
            dLon = lon - (math.floor((lon * 10 - 75) / 150) + 1) * 15.0
        else:
            dLon = lon - TimeZone * 15.0  # 地球上某一点与其所在时区中心的经度差
    else:
        if TimeZone == -13:
            dLon = (math.floor((lon * 10 - 75) / 150) + 1) * 15.0 - lon
        else:
            dLon = TimeZone * 15.0 - lon
    # 时差
    Et = 0.0028 - 1.9857 * math.sin(sitar) + 9.9059 * math.sin(2 * sitar) - 7.0924 * math.cos(
        sitar) - 0.6882 * math.cos(2 * sitar)
    gtdt1 = hour + min / 60.0 + sec / 3600.0 + dLon / 15  # 地方时
    gtdt = gtdt1 + Et / 60.0
    dTimeAngle1 = 15.0 * (gtdt - 12)
    dTimeAngle = dTimeAngle1 * math.pi / 180
    latitudeArc = lat * math.pi / 180
    HeightAngleArc = math.asin(
        math.sin(latitudeArc) * math.sin(ED) + math.cos(latitudeArc) * math.cos(ED) * math.cos(dTimeAngle))
    CosAzimuthAngle = (math.sin(HeightAngleArc) * math.sin(latitudeArc) - math.sin(ED)) / math.cos(
        HeightAngleArc) / math.cos(latitudeArc)
    AzimuthAngleArc = math.acos(CosAzimuthAngle)
    HeightAngle = HeightAngleArc * 180 / math.pi
    # 天顶角
    ZenithAngle = 90 - HeightAngle
    AzimuthAngle1 = AzimuthAngleArc * 180 / math.pi

    if dTimeAngle < 0:
        AzimuthAngle = 180 - AzimuthAngle1
    else:
        AzimuthAngle = 180 + AzimuthAngle1

    return round(HeightAngle, 4), round(AzimuthAngle, 4)


def read_all_sed(base_path: str) -> List[SED]:
    """
    读取文件夹下面的所有sed文件
    :param base_path:
    :return:
    """
    ans = []
    for file in os.listdir(base_path):
        if not file.endswith("sed"):
            continue
        ans.append(SED(os.path.join(base_path, file)))
    return ans

def getSeds():
    root = tkinter.Tk()
    root.withdraw()
    fnstr = filedialog.askopenfilenames(filetypes=[("sed file", "*.sed"), ("all", "*.*")])
    seds = []
    alts = []
    for name in fnstr:
        sed = SED(name)
        seds.append(sed.ref[:, -1])
        try:
            alts.append(float(sed.USER_FIELD1[1].replace('m', '')))
        except:
            alts.append(1.2)
    root.destroy()
    return seds, alts

if __name__ == '__main__':
    pass
