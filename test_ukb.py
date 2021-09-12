# -*- coding: utf-8 -*-
# @Time    : 2021/9/7 15:58
# @Author  : HCY
# @File    : test_ukb.py
# @Software: PyCharm

import pandas as pd
import os
import numpy as np

dataroot = os.path.join(os.getcwd() + '\\data' + '\\ukb_data')

data1 = pd.read_stata(dataroot + '\\【控制变量】depression_covariant.dta')
data1 = data1[data1['n_eid'] > 0]
data2 = pd.read_stata(dataroot + '\\mdd_grs.dta')
data2 = data2[data2['iid'] > 0]
data2.rename(columns = {'iid': 'n_eid'}, inplace = True)
data3 = pd.read_stata(dataroot + '\\【结局变量】depression_outcome.dta')
data_all = pd.merge(data1, data2, on = 'n_eid', how = 'inner')
data_all = pd.merge(data_all, data3[['dep_inc', 'n_eid']], on = 'n_eid', how = 'inner')
data_all.drop(
    ["n_eid", "pat_x", "pat_y", "mat_x", "mat_y", "sex_x", "gender", "fid_x", "fid_y", "phenotype_x", "phenotype_y"],
    axis = 1, inplace = True)
data_all.rename(columns = {'sex_y': 'sex'}, inplace = True)

population = ["age", "sex", "screening", "family_history"]
economy = ["lowincome", "workstatus", "highschool", "isolation2", "deprivation", "housing_tenure"]
lifestyle = ["healthy_PA", "healthy_diet", "healthy_smoking", "healthy_alcohol", "healthy_obesity", "sleep_score",
             "healthy_score"]
blood = ["n_30010_0_0", "n_30000_0_0", "n_30100_0_0", "n_30750_0_0", "n_30040_0_0", "n_30050_0_0", "n_30060_0_0",
         "n_30070_0_0", "n_30080_0_0", "n_30150_0_0", "n_30210_0_0", "n_30140_0_0", "n_30200_0_0", "n_30130_0_0",
         "n_30190_0_0", "n_30160_0_0", "n_30220_0_0", "n_30120_0_0", "n_30180_0_0"]
metabolism = ["n_30740_0_0", "n_30690_0_0", "n_30870_0_0", "n_30780_0_0", "n_30760_0_0", "n_30640_0_0", "n_30630_0_0"]
urine = ["n_30510_0_0", "n_30500_0_0", "n_30520_0_0", "n_30530_0_0"]
gene = data_all.columns[285:323].tolist()
others = ["n_20002_0_0", "n_20003_0_0"]
ill = ["dep_inc"]

data_final = data_all[population + economy + lifestyle + blood + metabolism + urine + others + gene + ill]
data_final = data_final.replace('NA', np.nan)
data_final = data_final.dropna(axis = 0, how = 'any')  # drop all rows that have any NaN values

full_data = {0: data_final[population], 1: data_final[economy], 2: data_final[lifestyle], 3: data_final[blood],
             4: data_final[metabolism], 5: data_final[urine], 6: data_final[gene], 7: data_final[others]}

full_labels = data_final[ill].values

print(data_final, full_data, full_labels)
