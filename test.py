# -*- coding: utf-8 -*-
# @Time    : 2023/4/9 21:43
# @Author  : qxcnwu
# @FileName: test.py
# @Software: PyCharm
from ScaleConvertion.SC import read_tiff

if __name__ == '__main__':
    tiff_path=r"D:\Transformer\test\dat1.tif"
    read_tiff(feature_bands=[[0,8],[8,16],[16,24]],save_dir=r"D:\Transformer\test",tiff_path=tiff_path,sen_alt=[9,9,9,9,9,9])
