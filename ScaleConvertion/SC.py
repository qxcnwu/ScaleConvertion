# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:45
# @Author  : qxcnwu
# @FileName: SC.py
# @Software: PyCharm
import tkinter
from tkinter import filedialog
from typing import List
import os

from .PicProcess.Draw_Window import read_, read_tif
from .Utiles.Info import Sensors_IFOV, UAV_IFOV
from .Utiles.PictureInfo import make_pic
from .Utiles.SED import getSeds, SED
from .Utiles.SaveFile import chose


def read_png(img_path: str = None, save_dir: str = "", seds: List[SED] = None,
             sensors_IFOV: float = Sensors_IFOV.SEI_RS800,
             uav_IFOV: float = UAV_IFOV.DJI,
             sensors_altitude: List[float] = None, pixel=None):
    """
    main function
    Args:
        img_path:
        save_dir:
        pixel:
    Returns:
    """
    print("initialize enviromental")
    if pixel is None:
        pixel = [2, 4, 8, 10, 20, 30]
    if seds == None or sensors_altitude == None:
        seds, sensors_altitude = getSeds()
    if save_dir == None:
        raise BaseException("should be create a savedir!!")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if img_path == None:
        img_path = chose("chose png file")
    return read_(img_path=img_path, save_dir=save_dir, seds=seds, sensors_IFOV=sensors_IFOV, uav_IFOV=uav_IFOV,
                 sensors_altitude=sensors_altitude, pixel=pixel)


def read_tiff(img_path: str = None, feature_bands: List[List[int]] = None, tiff_path: str = None, save_dir: str = "",
             sensors_IFOV: float = Sensors_IFOV.SEI_RS800,
             scale_tiff: float = 0.065,
             sen_alt=None, pixel: List[float] = None):
    print("initialize enviromental")
    if sen_alt is None:
        sen_alt = [1.2 for i in range(9)]
    if pixel is None:
        pixel = [2, 4, 8, 10, 20, 30]
    if save_dir == None:
        raise BaseException("should be create a savedir!!")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if img_path == None and feature_bands == None:
        img_path = chose("chose png file")
    if tiff_path == None:
        tiff_path = chose("chose tiff file")
    if img_path == None and feature_bands != None:
        img_path = os.path.join(save_dir, "feature.png")
        make_pic(tiff_path, feature_bands, img_path)
    return read_tif(img_path=img_path, tiff_path=tiff_path, save_dir=save_dir, sensors_IFOV=sensors_IFOV,
                     scale_tiff=scale_tiff, sen_alt=sen_alt, pixel=pixel)
