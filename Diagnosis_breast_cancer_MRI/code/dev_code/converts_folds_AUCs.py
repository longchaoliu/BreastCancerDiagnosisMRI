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

PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Convert_finetuning/Sessions/Seeded_50epochs_Segmented_and_NotSegmented_10Folds_LastTwoLayersFineTune_TrainPrevalence/'


SAVE_RESULTS = True
OUTPUT_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Convert_finetuning/Sessions/Seeded_50epochs_Segmented_and_NotSegmented_10Folds_LastTwoLayersFineTune_TrainPrevalence/'
NAME = 'AUCs'


main_result = pd.read_csv('/home/deeperthought/Documents/Papers_and_grants/Risk_paper/data/RISK/RISK_PaperResults_TEST.csv')

#%%

folds = [x for x in os.listdir(PATH) if x.startswith('FOLD')]


aucs = {}
i = 0
for fold in folds:
    
    result = pd.read_csv(PATH + fold + '/test_result.csv')

    auc = roc_auc_score(result['y_true'], result['y_pred'])
    fpr, tpr, thr = roc_curve(result['y_true'], result['y_pred'])
    aucs[fold] = [auc,fpr, tpr]
    


plt.figure(1,figsize=(5,5))

aucs_v = []

i = 1
for fold in aucs.keys():
    
    auc, fpr, tpr = aucs[fold]

    plt.plot(1-fpr, tpr, label=f'fold {i}', alpha=0.75)
    
    aucs_v.append(aucs[fold][0])
    
    i += 1

# plt.legend()

    
fpr, tpr, thr = roc_curve(main_result['y_true'], main_result['y_pred'])
plt.plot(1-fpr, tpr, label=f'fold {i}', alpha=1, color='k')

    
plt.plot([0,1],[1,0],'--')
plt.xlabel('Specificity', fontsize=14)
plt.ylabel('Sensitivity', fontsize=14)
# plt.text(x=0.2,y=0.1,s=f'AUC-ROC = {round(auc,2)}', fontsize=14)
# plt.title(f'Out-of-site axial exams (n={N})', fontsize=14)
plt.xlim([0,1])
plt.ylim([0,1])

plt.savefig('/home/deeperthought/Documents/Papers_and_grants/Risk_paper/Academic Radiology/Figures/FigS11.png', dpi=300)