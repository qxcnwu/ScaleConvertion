# -*- coding: utf-8 -*-
# @Time    : 2023/4/9 19:08
# @Author  : qxcnwu
# @FileName: ConcatAnswer.py
# @Software: PyCharm
from typing import List

import numpy as np
from tqdm import tqdm


def concate(ref: List[np.array], ans_index: List[np.array]):
    """
    concate
    :param ref:
    :param ans_index:
    :return:
    """
    k = []
    for i in range(16):
        k.append(ref[i % len(ref)])
    k = np.array(k)
    ans = []
    for rq in tqdm(ans_index):
        ans.append((k * np.expand_dims(rq, 1)).T)
    return ans
