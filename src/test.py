# -*- coding: utf-8 -*-
# @Time    : 2022/1/27 14:34
# @Author  : nieyuzhou
# @File    : test.py
# @Software: PyCharm
import numpy as np
import pandas as pd


def has_both_modality(df):
    if len(set(df["Modality"])) > 1:
        return 1
    else:
        return 0


data = pd.read_csv("../data/idaSearch_1_26_2022.csv")
length = len(data["Subject ID"].value_counts())
# 1449个
subject_ids = data.groupby(by = "Subject ID").apply(has_both_modality)
subject_ids = subject_ids[subject_ids == 1].index.to_list()
# 848、37927
data = data[data["Subject ID"].isin(subject_ids)]

label = data.groupby(by = "Subject ID").max()["Research Group"]
# CN     352
# MCI    256
# AD     240
# 选择有用的图像
