# -*- coding: utf-8 -*-
# @Time    : 2023/4/9 19:18
# @Author  : qxcnwu
# @FileName: SaveFile.py
# @Software: PyCharm
import os
import tkinter
from tkinter import filedialog
from typing import List

import numpy as np
import pandas as pd
import qt5_applications
from colour import Color
from matplotlib import pyplot as plt

dirname = os.path.dirname(qt5_applications.__file__)
plugin_path = os.path.join(dirname, 'Qt', 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path


def save_csv_png(refs: List[np.array], altitudes: List[float], save_dir: str, seds: List[np.array], errs: List[float]):
    """
    save ref to csv
    :param refs:
    :param altitudes:
    :param save_dir:
    :return:
    """
    # draw simple average
    plt.plot([i for i in range(350, 2501)], np.mean(
        np.array(seds), axis=0), label="simple average", linestyle='--')
    # draw predict
    for ref, alt, err in zip(refs, altitudes, errs):
        pd.DataFrame(np.mean(ref, axis=1)).to_csv(
            os.path.join(save_dir, str(alt)+"_"+str(err) + ".csv"))
        plt.plot([i for i in range(350, 2501)], np.mean(
            ref, axis=1), label=str(alt) + "m")
    plt.legend()
    plt.ylabel("ref")
    plt.xlabel("wavelength/nm")
    plt.savefig(os.path.join(save_dir, "draw.png"))
    return


def save_csv_tif(refs: List[np.array], altitudes: List[float], save_dir: str, trues: List[np.array],
                 seds: List[np.array], errs: List[float]):
    """
    save ref to csv
    :param refs:
    :param altitudes:
    :param save_dir:
    :return:
    """
    red = Color("red")
    colors = list(red.range_to(Color("blue"), len(altitudes) + 1))
    colors = [c.get_rgb() for c in colors]
    # draw simple average
    plt.plot([i for i in range(0, len(seds[0]))], np.mean(
        np.array(seds), axis=0), label="simple average", c=colors[0])
    i = 1
    # draw predict
    for ref, alt, err in zip(refs, altitudes, errs):
        pd.DataFrame(np.mean(ref, axis=1)).to_csv(
            os.path.join(save_dir, str(alt)+"_"+str(err) + ".csv"))
        plt.plot([i for i in range(0, len(seds[0]))], np.mean(
            ref, axis=1), label=str(alt) + "m pred", c=colors[i])
        plt.plot([i for i in range(0, len(seds[0]))], ref_true,
                 label=str(alt) + "m true", c=colors[i], linestyle='--')
        i += 1
    plt.legend()
    plt.ylabel("ref")
    plt.xlabel("wavelength/nm")
    plt.savefig(os.path.join(save_dir, "draw.png"))
    return


def chose(name):
    """
    chose file
    Args:
        name:

    Returns:

    """
    root = tkinter.Tk()
    root.withdraw()
    img_path = filedialog.askopenfilename(title=name)
    root.destroy()
    return img_path
