#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:34:51 2022

@author: deeperthought
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


#RESULT_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/SelectedSlices_and_Random_DGNS_train162742_val3150_DataAug_Clinical_depth6_filters42_L21e-05/VAL_Set_result.csv'
#RESULT_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/SelectedSlices_and_Random_DGNS_train27702_val1548_DataAug_Clinical_depth6_filters42_L20/VAL_Set_result.csv'

RESULT_PATH = '/home/deeperthought/Projects/DGNS/Saccades_BBOX/Results/DataGeneratorSessions/DGNS_CNN-GRU_Training_session_CNN32_GRUPATCH32_DENSE24_LR5e-05_ClinicalInfo_AddContralateral_AddPreviousExam_5ROIL20_dropout0_classWeight1.0/global_results_VAL.csv'

result = pd.read_csv(RESULT_PATH)

name = 'VAL'

df = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Data_Partitions.csv')
result.rename(columns={'scan':'Scan_ID'}, inplace=True)
result.rename(columns={'scanID':'Scan_ID'}, inplace=True)

result = pd.merge(result, df[['Scan_ID','BIRADS','Image_QA','Notes']], on='Scan_ID')

OUT = '/'.join(RESULT_PATH.split('/')[:-1])


result['BIRADS'].value_counts()

result = result.loc[result['BIRADS'] != 'NR']
result = result.loc[result['BIRADS'] != '0']


df2 = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/SelectedSlices_and_Random_DGNS_train162742_val3150_DataAug_Clinical_depth6_filters42_L21e-05/VAL_Set_result.csv')

result = result.loc[result['Scan_ID'].isin(df2['scan'])]

result = df2.loc[df2['scan'].isin(result['Scan_ID'])]
result.columns = [u'Scan_ID', u'y_pred', u'y_true', u'max_slice', u'GT_slice']
#%%


BINS = 100

roc_auc_test_final = roc_auc_score( [int(x) for x in result['y_true'].values],result['y_pred'].values)
fpr_test, tpr_test, thresholds = roc_curve([int(x) for x in result['y_true'].values],result['y_pred'].values)
malignants_test = result.loc[result['y_true'] == 1, 'y_pred']
benigns_test = result.loc[result['y_true'] == 0, 'y_pred']         

print('{} : AUC-ROC = {}'.format(name, roc_auc_test_final))

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
plt.hist(malignants_test.values, color='r', alpha=1, bins=BINS)
plt.hist(benigns_test.values, color='g', alpha=0.5, bins=BINS)
plt.legend(['Malignants (N={})'.format(len(malignants_test)), 'Benigns  (N={})'.format(len(benigns_test))])
plt.title('{} N = {}'.format(name, len(result)))

plt.subplot(1,3,3)                   
plt.hist(malignants_test.values, color='r', alpha=1, bins=BINS)
plt.hist(benigns_test.values, color='g', alpha=0.5, bins=BINS)
plt.yscale('log')
plt.legend(['Malignants (N={})'.format(len(malignants_test)), 'Benigns  (N={})'.format(len(benigns_test))])
plt.title('{} N = {}'.format(name, len(result)))
plt.tight_layout()
#plt.savefig(OUT +  '/{}_result_ROC_bins200.png'.format(name), dpi=200)   

#plt.close()
#%%

#%% QA

QA = ['MSKCC_16-328_1_09903_20060910_l', # no visible lesion...
'MSKCC_16-328_1_09071_20071108_r', # super noisy
'MSKCC_16-328_1_05846_20030226_r', # noisy, no visible lesion
'MSKCC_16-328_1_12064_20080513_r', # noisy. Should be able to compute SNR 
'MSKCC_16-328_1_03612_20030804_r'] # bad quality image.. 2003

result = result.loc[~ result['Scan_ID'].isin(QA)]

#%% QA

result['Image_QA'].value_counts()
result['BIRADS'].value_counts()

result = result.loc[result['Image_QA'].isnull()]

result = result.loc[result['BIRADS'] != 'NR']
result = result.loc[result['BIRADS'] != '99']
name = name + ' QA'

roc_auc_test_final = roc_auc_score( [int(x) for x in result['y_true'].values],result['y_pred'].values)
fpr_test, tpr_test, thresholds = roc_curve([int(x) for x in result['y_true'].values],result['y_pred'].values)
malignants_test = result.loc[result['y_true'] == 1, 'y_pred']
benigns_test = result.loc[result['y_true'] == 0, 'y_pred']         

print('{} : AUC-ROC = {}'.format(name, roc_auc_test_final))

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
plt.hist(malignants_test.values, color='r', alpha=1, bins=100)
plt.hist(benigns_test.values, color='g', alpha=0.5, bins=200)
plt.yscale('log')
plt.legend(['Malignants (N={})'.format(len(malignants_test)), 'Benigns  (N={})'.format(len(benigns_test))])
plt.title('{} N = {}'.format(name, len(result)))
plt.tight_layout()
plt.savefig(OUT +  '/{}_result_ROC_BIRADS_QA.png'.format(name), dpi=200)   

#%%
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(result['y_true'], result['y_pred'])
AUC_PR = auc(recall, precision)
average_precision = average_precision_score(result['y_true'], result['y_pred'])


#create precision recall curve
fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')

#add axis labels to plot
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')

no_skill = np.sum(result['y_true']==1) / float(len(result))
ax.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

plt.legend(['Average Precision: ' + str(np.round(average_precision,2))])
plt.text(0.1,0.2, 'AUC-PR: {}'.format(np.round(AUC_PR,2)))
#display plot
plt.savefig(OUT +  '/{}_result_PR_Curve_QA.png'.format(name), dpi=200)   

