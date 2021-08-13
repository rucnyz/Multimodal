# -*- coding: utf-8 -*-
# @Time    : 2021/7/27 15:07
# @Author  : nieyuzhou
# @File    : test_mvlearn.py
# @Software: PyCharm

from mvlearn.datasets import load_UCImultifeature
from mvlearn.plotting import quick_visualize
import pandas as pd

full_data, full_labels = load_UCImultifeature()
# 76 Fourier coefficients of the character shapes
fou_data = pd.DataFrame(full_data[0])
# 216 profile correlations
fac_data = pd.DataFrame(full_data[1])
# 64 Karhunen-Love coefficients
kar_data = pd.DataFrame(full_data[2])
# 240 pixel averages in 2 x 3 windows
pix_data = pd.DataFrame(full_data[3])
# 47 Zernike moments
zer_data = pd.DataFrame(full_data[4])
# 6 morphological features
mor_data = pd.DataFrame(full_data[5])

# 把这六个模态的所有特征展平，降到2维
quick_visualize(full_data, labels = full_labels, title = "10-class data")