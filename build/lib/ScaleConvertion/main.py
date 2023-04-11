# -*- coding: utf-8 -*-
# @Time    : 2023/4/9 14:31
# @Author  : qxcnwu
# @FileName: main.py
# @Software: PyCharm
import tkinter
from tkinter import filedialog
from typing import List

from .PicProcess.Draw_Window import read_
from .Utiles.SED import SED


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


def read(img_path: str, save_dir: str, pixel: List[int]):
    """
    main function
    Args:
        img_path:
        save_dir:
        pixel:
    Returns:
    """
    a, b = getSeds()
    read_(img_path, save_dir, a, sensors_altitude=b, pixel=pixel)
    return
