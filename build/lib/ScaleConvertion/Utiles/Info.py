# -*- coding: utf-8 -*-
# @Time    : 2023/4/9 14:47
# @Author  : qxcnwu
# @FileName: Info.py
# @Software: PyCharm
import os
from tqdm import tqdm
import requests


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
    resp = requests.get(
        "https://models-1256388644.cos.ap-beijing.myqcloud.com/save4_0.003732832917032445mt.pth", stream=True)
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
