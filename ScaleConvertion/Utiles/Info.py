# -*- coding: utf-8 -*-
# @Time    : 2023/4/9 14:47
# @Author  : qxcnwu
# @FileName: Info.py
# @Software: PyCharm
import requests
from tqdm import tqdm


class Sensors_IFOV:
    """
    IFOV of Sensors
    """
    SEI_RS800 = 25


class UAV_IFOV:
    """
    IFOV of UAV
    """
    DJI = 84


def download_file(fname: str):
    try:
        resp = requests.get(
            "http://s9zdkfev2.hb-bkt.clouddn.com/save4_0.003732832917032445mt.pth", stream=True)
        total = int(resp.headers.get('content-length', 0))
        with open(fname, 'wb') as file, tqdm(
                desc=fname,
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    except:
        resp = requests.get(
            "http://81.70.190.126/model/save4_0.003732832917032445mt.pth", stream=True)
        total = int(resp.headers.get('content-length', 0))
        with open(fname, 'wb') as file, tqdm(
                desc=fname,
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
