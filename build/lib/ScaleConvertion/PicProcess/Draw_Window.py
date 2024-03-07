# -*- coding: utf-8 -*-
# @Time    : 2023/4/9 15:07
# @Author  : qxcnwu
# @FileName: Draw_Window.py
# @Software: PyCharm
import os.path
import tkinter as tk
from math import tan, pi
from tkinter import filedialog
from typing import Any, List

import cv2
import numpy as np

from .DataPredict import predict
from .ConcatAnswer import concate
from .MakeData import DataMaker
from ..Utiles.Info import Sensors_IFOV, UAV_IFOV
from ..Utiles.PictureInfo import read_pic, copy_image, get_ref, get_true, read_tiff
from ..Utiles.SED import SED
from ..Utiles.SaveFile import save_csv_png, save_csv_tif
from ..Utiles.ScaleCompute import compute_sensors_rad, compute_pixel_rad


class struct_getPoint:
    def __init__(self, image: Any, name: str, sensors_rad: List[int], pixel_rad: List[int], img_h: int, img_w: int):
        """
        get point and pixel rad
        :param image: 
        :param name: 
        :param sensors_rad: 
        :param pixel_rad: 
        """
        self.im_h, self.im_w = img_h, img_w
        self.sensors_rad = sensors_rad
        self.pixel_rad = pixel_rad
        self.g_window_wh = [800, 600]
        self.location_click = [0, 0]
        self.location_release = [0, 0]
        self.image_original = image.copy()
        self.image_show = self.image_original[0: self.g_window_wh[1], 0:self.g_window_wh[0]]
        self.image_show = self.image_original
        self.location_win = [0, 0]
        self.location_win_click = [0, 0]
        self.image_zoom = self.image_original.copy()
        self.zoom = 1
        self.step = 0.1
        self.window_name = name
        self.point = []
        self.true_pixel = []

    # OpenCV mouth
    def getPoint(self):
        def mouse_callback(event, x, y, flags, param):
            global save_end, circle_end

            def check_location(img_wh, win_wh, win_xy):
                for i in range(2):
                    if win_xy[i] < 0:
                        win_xy[i] = 0
                    elif win_xy[i] + win_wh[i] > img_wh[i] and img_wh[i] > win_wh[i]:
                        win_xy[i] = img_wh[i] - win_wh[i]
                    elif win_xy[i] + win_wh[i] > img_wh[i] and img_wh[i] < win_wh[i]:
                        win_xy[i] = 0

            def count_zoom(flag, step, zoom, zoom_max):
                if flag > 0:
                    zoom += step
                    if zoom > 1 + step * 20:
                        zoom = 1 + step * 20
                else:
                    zoom -= step
                    if zoom < zoom_max:
                        zoom = zoom_max
                zoom = round(zoom, 2)
                return zoom

            if event or flags:
                w2, h2 = self.g_window_wh
                h1, w1 = param.image_zoom.shape[0:2]
                if event == cv2.EVENT_LBUTTONDOWN:
                    param.location_click = [x, y]
                    param.location_win_click = [param.location_win[0],
                                                param.location_win[1]]
                elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
                    param.location_release = [x, y]
                    if w1 <= w2 and h1 <= h2:
                        param.location_win = [0, 0]
                    elif w1 >= w2 and h1 < h2:
                        param.location_win[0] = param.location_win_click[0] + param.location_click[0] - \
                                                param.location_release[0]
                    elif w1 < w2 and h1 >= h2:
                        param.location_win[1] = param.location_win_click[1] + param.location_click[1] - \
                                                param.location_release[1]
                    else:
                        param.location_win[0] = param.location_win_click[0] + param.location_click[0] - \
                                                param.location_release[0]
                        param.location_win[1] = param.location_win_click[1] + param.location_click[1] - \
                                                param.location_release[1]
                    check_location([w1, h1], [w2, h2], param.location_win)
                elif event == cv2.EVENT_MOUSEWHEEL:
                    z = param.zoom
                    zoom_max = self.g_window_wh[0] / param.image_original.shape[1]
                    param.zoom = count_zoom(flags, param.step, param.zoom, zoom_max)
                    w1, h1 = [int(param.image_original.shape[1] * param.zoom),
                              int(param.image_original.shape[0] * param.zoom)]
                    param.image_zoom = cv2.resize(param.image_original, (w1, h1), interpolation=cv2.INTER_AREA)
                    param.location_win = [int((param.location_win[0] + x) * param.zoom / z - x),
                                          int((param.location_win[1] + y) * param.zoom / z - y)]
                    check_location([w1, h1], [w2, h2], param.location_win)
                elif event == cv2.EVENT_RBUTTONDOWN:
                    point_num = len(param.point)
                    if point_num > len(self.sensors_rad):
                        param.point.pop()
                        point_num = len(param.point)
                        param.image_original = circle_end.copy()
                    # draw circle
                    if point_num != len(self.sensors_rad):
                        [x_ori, y_ori] = [int((param.location_win[0] + x) / param.zoom),
                                          int((param.location_win[1] + y) / param.zoom)]
                        param.point.append([x_ori, y_ori])
                        cv2.circle(param.image_original, (x_ori, y_ori), self.sensors_rad[point_num], (255, 0, 0),
                                   thickness=-1)
                        cv2.putText(param.image_original, str(point_num + 1), (x_ori, y_ori), cv2.FONT_HERSHEY_PLAIN,
                                    1.0, (0, 255, 0), thickness=1)
                        save_end = point_num == len(self.sensors_rad) - 1
                    else:
                        # draw square
                        [x_ori, y_ori] = [int((param.location_win[0] + x) / param.zoom),
                                          int((param.location_win[1] + y) / param.zoom)]
                        param.point.append([x_ori, y_ori])
                        self.true_pixel = []
                        for pr in self.pixel_rad:
                            if x_ori - pr >= 0 and y_ori - pr >= 0 and x_ori + pr < self.im_w and y_ori + pr < self.im_h:
                                cv2.rectangle(param.image_original, (x_ori - pr, y_ori - pr), (x_ori + pr, y_ori + pr),
                                              color=(0, 0, 0), thickness=10)
                                self.true_pixel.append(pr)

                param.image_zoom = cv2.resize(param.image_original, (w1, h1), interpolation=cv2.INTER_AREA)  # 图片缩放
                param.image_show = param.image_zoom[param.location_win[1]:param.location_win[1] + h2,
                                   param.location_win[0]:param.location_win[0] + w2]
                if save_end:
                    circle_end = param.image_original.copy()
                    save_end = False
                cv2.imshow(param.window_name, param.image_show)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.g_window_wh[0], self.g_window_wh[1])
        cv2.imshow(self.window_name, self.image_show)
        cv2.setMouseCallback(self.window_name, mouse_callback, self)
        return


def read_(img_path: str = None, save_dir: str = "", seds: List[SED] = None,
          sensors_IFOV: float = Sensors_IFOV.SEI_RS800,
          uav_IFOV: float = UAV_IFOV.DJI,
          sensors_altitude: List[float] = None, pixel: List[float] = None):
    """
    read img and draw img
    :param seds:
    :param pixel:
    :param sensors_altitude:    :param img_path:
    :param sensors_IFOV:
    :param uav_IFOV:
    :return:
    """
    # read dataset init
    print("start get picture process step 1/6")
    img_path = copy_image(img_path, os.path.join(os.path.dirname(__file__), "tmp"))
    altitude, img_h, img_w, img_c = read_pic(img_path)
    sen_alt = [compute_sensors_rad(img_h, img_w, altitude, i, sensors_IFOV, uav_IFOV) for i in sensors_altitude]
    pixel_rad = [compute_pixel_rad(img_h, img_w, altitude, uav_IFOV, i) for i in pixel]
    print("start get points step 2/6")
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    tmp = struct_getPoint(img, os.path.basename(img_path), sen_alt, pixel_rad, img_h, img_w)
    tmp.getPoint()
    while True:
        # wait to exit
        flag = cv2.waitKey(1)
        if flag == ord('s'):
            cv2.destroyAllWindows()
            cv2.imwrite(os.path.join(save_dir,"con.jpg"), tmp.image_show)
            break
    # data process
    print("start make dataset step 3/6")
    dm = DataMaker(img_path, tmp.point[:-1], sen_alt, tmp.point[-1], tmp.true_pixel)
    # data pridict
    print("start predict dataset step 4/6")
    ans,err = predict(dm.small_path, dm.big_path)
    # save answer
    print("start concate answer step 5/6")
    out = concate(seds, ans)
    # end
    print("start concate save answer step 6/6")
    save_csv_png(out, pixel[0:len(tmp.true_pixel)], save_dir, seds,err)
    return out, ans, dm


def read_tif(img_path: str = None, tiff_path: str = None, save_dir: str = "",
             sensors_IFOV: float = Sensors_IFOV.SEI_RS800,
             scale_tiff: float = 0.065,
             sen_alt: List[float] = None, pixel: List[float] = None):
    """
    read img and draw img
    :param seds:
    :param pixel:
    :param sensors_altitude:    :param img_path:
    :param sensors_IFOV:
    :param uav_IFOV:
    :return:
    """
    # read dataset init
    print("start get picture process step 1/7")
    img_path = copy_image(img_path, os.path.join(os.path.dirname(__file__), "tmp"))
    imgs = read_tiff(tiff_path)
    img_h, img_w, _ = imgs.shape

    sen_alt = [int(tan(sensors_IFOV / 2 / 180 * pi) *
                   altitude_sensors / 0.065) for altitude_sensors in sen_alt]
    pixel_rad = [int(i / scale_tiff) for i in pixel]

    print("start get points step 2/7")
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    tmp = struct_getPoint(img, os.path.basename(img_path), sen_alt, pixel_rad, img_h, img_w)
    tmp.getPoint()
    while True:
        # wait to exit
        flag = cv2.waitKey(1)
        if flag == ord('s'):
            cv2.destroyAllWindows()
            cv2.imwrite(os.path.join(save_dir, "con.jpg"), tmp.image_show)
            break

    # get ref
    print("start get refs step 3/7")
    seds = get_ref(imgs, tmp.point[:-1], sen_alt)
    seds_true = get_true(imgs, tmp.point[-1], tmp.true_pixel)

    # data process
    print("start make dataset step 4/7")
    dm = DataMaker(img_path, tmp.point[:-1], sen_alt, tmp.point[-1], tmp.true_pixel)
    # data pridict
    print("start predict dataset step 5/7")
    ans,err = predict(dm.small_path, dm.big_path)
    # save answer
    print("start concate answer step 6/7")
    out = concate(seds, ans)
    # end
    print("start concate save answer step 7/7")
    save_csv_tif(out, pixel[0:len(tmp.true_pixel)], save_dir, seds_true[0:len(tmp.true_pixel)], seds,err)
    return out, ans, dm
