# -*- coding: utf-8 -*-
# @Time    : 2023/4/9 14:38
# @Author  : qxcnwu
# @FileName: PictureInfo.py
# @Software: PyCharm
import math
import os
import shutil
from typing import List

import exifread
import numpy as np
from PIL import Image
from osgeo import gdal


def read_pic(pic_path):
    """
    get picture's altitude,width,heigh,channel
    :param pic_path:
    :return: altitude,w,h,c
    """
    # 图像原位信息
    high = None
    f = open(pic_path, 'rb')
    contents = exifread.process_file(f)
    for tag, value in contents.items():
        if tag == 'GPS GPSAltitude':
            high = round(float(eval(str(value))), 2)
            break
    f.close()

    if high == None:
        raise AssertionError(pic_path, " has no altitude information please check.")

    img = np.array(Image.open(pic_path)).astype(np.int)
    img_h, img_w, img_c = img.shape
    return high, img_h, img_w, img_c


def copy_image(path: str, dst: str) -> str:
    """
    copy img to tmp.jpg
    :param path:
    :return:
    """
    if not os.path.exists(dst):
        os.mkdir(dst)
    shutil.copy(path, os.path.join(dst, "tmp.jpg"))
    return os.path.join(dst, "tmp.jpg")


def read_tiff(tiff_path: str):
    """
    read tiff file
    Args:
        tiff_path:

    Returns:
    """
    data = gdal.Open(tiff_path)
    img = data.ReadAsArray()
    img = (img.transpose(2, 1, 0) / 65536 * 256).astype(np.int_)
    return img


def get_ref(img: np.array, points: List[List[int]], rads: List[int]):
    """
    get simple point's ref
    Args:
        img:
        points:
        rads:
    Returns:
    """
    ans = []
    for point, rad in zip(points, rads):
        ans.append(
            np.mean(img[point[1] - rad:point[1] + rad, point[0] - rad:point[0] + rad], axis=(0, 1)))
    return ans


def get_true(img: np.array, point: List[int], rads: List[int]):
    """
    get true ref
    Args:
        img:
        point:
        rads:

    Returns:

    """
    ans = []
    for rad in rads:
        ans.append(np.mean(img[point[1] - rad:point[1] + rad,
                           point[0] - rad:point[0] + rad], axis=(0, 1)))
    return ans


def make_pic(tiff_path: str, f_num: List[List[int]], save_path: str):
    """

    Args:
        tiff_path:
        f_num:
        save_path:

    Returns:

    """
    img = read_tiff(tiff_path)
    b, g, r = np.mean(img[:, :, f_num[0][0]:f_num[0][1]], axis=-1), np.mean(
        img[:, :, f_num[1][0]:f_num[1][1]], axis=-1), np.mean(img[:, :, f_num[2][0]:f_num[2][1]], axis=-1)
    b = np.expand_dims(b, axis=-1).astype(np.int_)
    g = np.expand_dims(g, axis=-1).astype(np.int_)
    r = np.expand_dims(r, axis=-1).astype(np.int_)
    k = np.concatenate([r, g, b], axis=-1)
    Image.fromarray(k.astype(np.uint8)).save(save_path)
    return
