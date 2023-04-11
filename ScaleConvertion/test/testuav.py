# -*- coding: utf-8 -*-
# @Time    : 2023/4/9 21:43
# @Author  : qxcnwu
# @FileName: test.py
# @Software: PyCharm
from ScaleConvertion.SC import read_tiff, read_png

if __name__ == '__main__':
    tiff_path = r"D:\Transformer\test\dat1.tif"
    img_path = r"G:\UAVPICTURRE\13裸土-上东下西-H975m-6m-116.JPG"
    save_dir = "tmp"
    read_png(img_path=img_path, save_dir=save_dir, pixel=[2, 4, 8, 10, 20, 30])
