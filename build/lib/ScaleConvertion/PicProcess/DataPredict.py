# -*- coding: utf-8 -*-
# @Time    : 2023/4/9 18:24
# @Author  : qxcnwu
# @FileName: DataPredict.py
# @Software: PyCharm
import os
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable

from ..Model.restnet_transformer import resnet18
from ..Utiles.Info import download_file


def getOneValidationData(small_path: str, big_path: List[str]):
    """
    data process norm
    :param small_path:
    :param big_path:
    :return:
    """
    big_Pic = [np.array(Image.open(file)).astype(np.float32)
               for file in big_path]
    big_Pic_gray = [np.mean(np.array(Image.open(file).convert('L')).astype(np.float32))
                    for file in big_path]
    smallPic = np.array(Image.open(small_path)).astype(np.float32)
    smallPicGray = np.array(Image.open(
        small_path).convert('L')).astype(np.float32)
    arr = np.array([np.mean(smallPicGray[0:56, 0:56]),
                    np.mean(smallPicGray[0:56, 56:112]),
                    np.mean(smallPicGray[0:56, 112:168]),
                    np.mean(smallPicGray[0:56, 168:224]),
                    np.mean(smallPicGray[56:112, 0:56]),
                    np.mean(smallPicGray[56:112, 56:112]),
                    np.mean(smallPicGray[56:112, 112:168]),
                    np.mean(smallPicGray[56:112, 168:224]),
                    np.mean(smallPicGray[112:168, 0:56]),
                    np.mean(smallPicGray[112:168, 56:112]),
                    np.mean(smallPicGray[112:168, 112:168]),
                    np.mean(smallPicGray[112:168, 168:224]),
                    np.mean(smallPicGray[168:224, 0:56]),
                    np.mean(smallPicGray[168:224, 56:112]),
                    np.mean(smallPicGray[168:224, 112:168]),
                    np.mean(smallPicGray[168:224, 168:224])])

    smallPic = np.concatenate(
        [smallPic[0:56, 0:56],
         smallPic[0:56, 56:112],
         smallPic[0:56, 112:168],
         smallPic[0:56, 168:224],
         smallPic[56:112, 0:56],
         smallPic[56:112, 56:112],
         smallPic[56:112, 112:168],
         smallPic[56:112, 168:224],
         smallPic[112:168, 0:56],
         smallPic[112:168, 56:112],
         smallPic[112:168, 112:168],
         smallPic[112:168, 168:224],
         smallPic[168:224, 0:56],
         smallPic[168:224, 56:112],
         smallPic[168:224, 112:168],
         smallPic[168:224, 168:224]], axis=2)
    bigPic = [np.transpose(tmp, [2, 0, 1]).astype(
        np.float32) / 255 for tmp in big_Pic]
    smallPic = np.transpose(smallPic, [2, 0, 1]) / 255
    return bigPic, smallPic.astype(np.float32), big_Pic_gray, arr


def predict(small_path: str, big_path: List[str], device: str = "cuda"):
    """
    predict answer
    :param device:
    :param small_path:
    :param big_path:
    :return:
    """
    bigPics, small, big_Pic_gray, arr = getOneValidationData(
        small_path, big_path)
    model_path = os.path.join(os.path.dirname(
        __file__), "save4_0.003732832917032445mt.pth")
    if torch.cuda.is_available() and device == "cuda":
        device = torch.device("cuda")
        dev = "cuda"
    else:
        device = torch.device("cpu")
        dev = "cpu"
    model = resnet18(device=dev)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device.type))
    else:
        try:
            download_file(model_path)
            model.load_state_dict(torch.load(
                model_path, map_location=device.type))
        except:
            raise ModuleNotFoundError("model.pth not found!!")
    model = model.to(device)
    model.eval()
    ans = []
    err = []
    with torch.no_grad():
        for bigPic, bt in zip(bigPics, big_Pic_gray):
            x_batch1 = Variable(torch.from_numpy(
                np.expand_dims(bigPic, axis=0))).to(device=device)
            x_batch2 = Variable(torch.from_numpy(
                np.expand_dims(small, axis=0))).to(device=device)
            output = model(x_batch1, x_batch2)
            output = output.detach().cpu().numpy()
            ans.append(output[0])
            err.append(abs(np.mean(output[0] * arr)-bt)/bt*100)
            print("true:", bt, "pred:", np.mean(
                output[0] * arr), "simple average:", np.mean(arr))
    return ans, err
