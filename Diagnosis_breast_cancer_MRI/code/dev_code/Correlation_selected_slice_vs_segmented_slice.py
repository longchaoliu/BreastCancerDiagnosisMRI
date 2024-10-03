#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 15:54:50 2022

@author: deeperthought
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/test_result.csv'

result = pd.read_csv(PATH)

result.columns

malignants_total = len(result.loc[result['y_true'] == 1])

result_M = result.loc[result['GT_slice'] > 0]

from scipy.stats import spearmanr, pearsonr
corr, p = spearmanr(result_M['GT_slice'],result_M['max_slice'])
corr, p = pearsonr(result_M['GT_slice'],result_M['max_slice'])

plt.figure(figsize=(5,5))
plt.plot([5,51],[5,51],'k--', alpha=0.5)

# plt.title('{} segmented scans (from a total of {})'.format(len(result_M),malignants_total ))
plt.scatter(result_M['GT_slice'],result_M['max_slice'], alpha=0.5)#, c=result_M['y_pred'], cmap='inferno'); plt.colorbar()
plt.text(30,10,'r={}'.format(round(corr,2)), fontsize=15)
plt.xlabel('segmented slice', fontsize=15)
plt.ylabel('slice with maximum risk', fontsize=15)
plt.grid()



plt.figure(figsize=(5,5))
# plt.title('{} segmented scans (from a total of {})'.format(len(result_M),malignants_total ))
plt.hist(0.3*(result_M['GT_slice'] - result_M['max_slice']), bins=40)#, c=result_M['y_pred'], cmap='inferno'); plt.colorbar()
plt.xlabel('Distance from index slice (cm)', fontsize=15)


# plt.savefig(PATH.replace('.csv','_slice_correlation.png'), dpi=400)

#%%
deltas = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/test_result_with_deltas.csv')

hits = deltas.loc[deltas['min_distance'] == 0]
miss = deltas.loc[deltas['min_distance'] > 0]


result_M_hits = result_M.loc[result_M['scan'].isin(hits['scan'])]
result_M_miss = result_M.loc[~result_M['scan'].isin(hits['scan'])]


plt.figure(figsize=(4,4))
plt.title('{} segmented scans (from a total of {})'.format(len(result_M),malignants_total ))
plt.scatter(result_M_hits['GT_slice'],result_M_hits['max_slice'], alpha=0.5, color='r', label='Hit')#, c=result_M['y_pred'], cmap='inferno'); plt.colorbar()
plt.scatter(result_M_miss['GT_slice'],result_M_miss['max_slice'], alpha=0.5, color='b', label='Miss')#, c=result_M['y_pred'], cmap='inferno'); plt.colorbar()

#plt.text(30,10,'r={}'.format(round(corr,2)), fontsize=15)
plt.xlabel('segmented slice', fontsize=15)
plt.ylabel('slice with maximum risk', fontsize=15)
plt.grid()
plt.legend()

plt.savefig(PATH.replace('.csv','_slice_correlation_hits.png'), dpi=400)