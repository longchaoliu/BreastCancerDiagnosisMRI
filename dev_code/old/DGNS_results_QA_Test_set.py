#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 16:07:16 2022

@author: deeperthought
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt



from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


df = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/SelectedSlices_and_Random_DGNS_train162742_val3150_DataAug_Clinical_depth6_filters42_L21e-05/Test_Set_result.csv'

GLOBAL_RESULTS_TEST = pd.read_csv(df)


#GLOBAL_RESULTS_TEST = GLOBAL_RESULTS_TEST.loc[GLOBAL_RESULTS_TEST['scan'].str[-10:-2] < '20050000' ]

bad = ['MSKCC_16-328_1_06701_20100224_l',  # WRONG SIDE! Benign scan!
       'MSKCC_16-328_1_09331_20060504_r',  # Strong motion Artifact
       'MSKCC_16-328_1_00992_20060511_l'] #Lesion out of frame?


GLOBAL_RESULTS_TEST = GLOBAL_RESULTS_TEST.loc[~ GLOBAL_RESULTS_TEST['scan'].isin(bad)]

GLOBAL_RESULTS_TEST.to_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/SelectedSlices_and_Random_DGNS_train162742_val3150_DataAug_Clinical_depth6_filters42_L21e-05/Test_Set_result_QA.csv', index=False)

name = 'test'

roc_auc_test_final = roc_auc_score( [int(x) for x in GLOBAL_RESULTS_TEST['y_true'].values],GLOBAL_RESULTS_TEST['y_pred'].values)
fpr_test, tpr_test, thresholds = roc_curve([int(x) for x in GLOBAL_RESULTS_TEST['y_true'].values],GLOBAL_RESULTS_TEST['y_pred'].values)
malignants_test = GLOBAL_RESULTS_TEST.loc[GLOBAL_RESULTS_TEST['y_true'] == 1.0, 'y_pred']
benigns_test = GLOBAL_RESULTS_TEST.loc[GLOBAL_RESULTS_TEST['y_true'] == 0.0, 'y_pred']         

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(fpr_test, tpr_test, label='ROC curve (area = %0.3f)' % roc_auc_test_final)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")   

plt.subplot(1,3,2)                   
plt.hist(malignants_test.values, color='r', alpha=1, bins=80)
plt.hist(benigns_test.values, color='g', alpha=0.5, bins=100)
plt.legend(['Malignants (N={})'.format(len(malignants_test)), 'Benigns  (N={})'.format(len(benigns_test))])
plt.title('{} N = {}'.format(name, len(GLOBAL_RESULTS_TEST)))

plt.subplot(1,3,3)                   
plt.hist(malignants_test.values, color='r', alpha=1, bins=80)
plt.hist(benigns_test.values, color='g', alpha=0.5, bins=100)
plt.yscale('log')
plt.legend(['Malignants (N={})'.format(len(malignants_test)), 'Benigns  (N={})'.format(len(benigns_test))])
plt.title('{} N = {}'.format(name, len(GLOBAL_RESULTS_TEST)))

plt.savefig(df.replace('.csv', '_QA.png'), dpi=200)

GLOBAL_RESULTS_TEST.to_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/U-Net_classifier_train28154_val1620_DataAug_Clinical_depth6_filters42_L20/TEST_result_initialQA.csv', index=False)


#%%
