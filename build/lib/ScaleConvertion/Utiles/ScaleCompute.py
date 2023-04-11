# -*- coding: utf-8 -*-
# @Time    : 2023/4/9 14:50
# @Author  : qxcnwu
# @FileName: ScaleCompute.py
# @Software: PyCharm

from math import tan, pi


def compute_sensors_rad(img_h: int, img_w: int, altitude_uav: float, altitude_sensors: float, sensors_IFOV: float,
                        Uav_IFOV: float) -> int:
    """
    compute sensors bound's pixel of uav
    :param img_h:
    :param img_w:
    :param altitude_uav:
    :param altitude_sensors:
    :param sensors_IFOV:
    :param Uav_IFOV:
    :return:
    """
    rad = tan(sensors_IFOV/2 / 180 * pi) * altitude_sensors
    resolution = 6 * altitude_uav * tan(Uav_IFOV/2 / 180 * pi) / 5 / min(img_h, img_w)
    k = int(rad / resolution)
    if k == 0:
        raise UnboundLocalError("rad euqals ", k, " warning!")
    return k


def compute_pixel_rad(img_h: int, img_w: int, altitude_uav: float, Uav_IFOV: float, pixel_: float) -> int:
    """
    compute pixel rad
    :param img_h:
    :param img_w:
    :param altitude_uav:
    :param Uav_IFOV:
    :param pixel_:
    :return:
    """
    resolution = 6 * altitude_uav * tan(Uav_IFOV/2 / 180 * pi) / 5 / min(img_h, img_w)
    k = int(pixel_ / 2 / resolution)
    if k == 0:
        raise UnboundLocalError("rad euqals ", k, " warning!")
    return k
