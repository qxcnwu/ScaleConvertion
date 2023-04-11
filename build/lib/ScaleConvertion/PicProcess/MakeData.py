# -*- coding: utf-8 -*-
# @Time    : 2023/4/9 17:49
# @Author  : qxcnwu
# @FileName: MakeData.py
# @Software: PyCharm
import os
from typing import Any, List

import numpy as np
from PIL import Image
from tqdm import tqdm


class DataMaker:
    def __init__(self, img_path: str, points_index: List[List[int]], points_rad: List[int], points_pixel: List[int],
                 points_rad_pixel: List[int]):
        """
        create dataset with different rad
        :param img_path:
        :param points_index:
        :param points_rad:
        :param points_pixel:
        :param points_rad_pixel:
        """
        self.img = np.array(Image.open(img_path))
        self.points_index = points_index
        self.points_rad = points_rad
        self.points_pixel = points_pixel
        self.points_rad_pixel = points_rad_pixel
        self.big_path = []
        self.dir = os.path.join(os.path.dirname(__file__))
        self.small_path = os.path.join(self.dir, "tmp", "small.jpg")

        self.make_big()
        self.make_small()

    def make_big(self):
        """
        make square to 224*224*3
        :return:
        """
        for idx, n in enumerate(tqdm(self.points_rad_pixel)):
            tmp = os.path.join(self.dir, "tmp", str(idx) + ".jpg")
            Image.fromarray(
                np.array(self.img[self.points_pixel[1] - n:self.points_pixel[1] + n, self.points_pixel[0] - n:
                                                                                     self.points_pixel[0] + n]).astype(
                    np.uint8)).resize((224, 224)).save(tmp)
            self.big_path.append(tmp)
        return

    def make_small(self):
        tmp = []
        for points, rad in zip(self.points_index, self.points_rad):
            y, x = points
            tmp.append(np.array(Image.fromarray(self.img[x - rad:x + rad, y - rad:y + rad]).resize((56, 56))))
        arr = np.zeros((224, 224, 3))
        for i in range(4):
            for j in range(4):
                arr[56 * i:56 * i + 56, j * 56:j * 56 + 56, :] = tmp[(i * 4 + j) % len(self.points_rad)]
        Image.fromarray(arr.astype(np.uint8)).save(self.small_path)
        return
