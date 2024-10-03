#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 10:47:17 2023

@author: deeperthought
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

RESULTS_TABLE_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/test_result.csv'
# resM = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/validation_result_with_deltas.csv')
result = pd.read_csv(RESULTS_TABLE_PATH)

#%%

def SENS(TP,FN):
    return TP / (TP + FN)

def SPEC(TN, FP):
    'TNR'
    return TN / (TN + FP)

def PPV(TP,FP):
    if (TP + FP) == 0:
        return 1
    else:
        return TP / (TP + FP)


def NPV(TN,FN):
    if (TN + FN) == 0:
        return 1
    else:
        return TN / (TN + FN) 
    
def get_triage_curves(results, path='', steps=1000):
    df = pd.DataFrame(columns=['threshold','N_below_threshold','N_above_threshold','TP','TN','FP','FN','PPV','NPV','SENS','SPEC','TOTAL'])

    for risk_threshold in np.arange(0,1+1./steps,1./steps):
        vals_above = results.loc[results['y_pred'] > risk_threshold, 'y_true'].value_counts().astype(float)    
        vals_below = results.loc[results['y_pred'] <= risk_threshold, 'y_true'].value_counts().astype(float)    
        
        try:
            TN=vals_below[0]
        except:
            TN=0
        try:
            FP=vals_above[0]    
        except:
            FP=0
        
        try:
            TP=vals_above[1]
        except:
            TP=0
        try:
            FN=vals_below[1]
        except:
            FN=0

        
        df = df.append({'threshold':risk_threshold,
                        'N_below_threshold':TN + FN,
                        'N_above_threshold':TP + FP,
                        'TP':TP,'TN':TN,'FP':FP,'FN':FN,
                        'PPV':PPV(TP,FP),'NPV':NPV(TN,FN),
                        'SENS':SENS(TP,FN),'SPEC':SPEC(TN, FP),
                        'TOTAL':TP+FP+TN+FN}, ignore_index=True)    
    
    return df[:-1]


#%%

auc = roc_auc_score(result['y_true'],result['y_pred'])

triage_curve = get_triage_curves(result, path='', steps=1000)

triage_curve.keys()

plt.figure(figsize=(4,4))
plt.plot(triage_curve['SPEC'],triage_curve['SENS'])
plt.text(x=0.4,y=0.4,s=auc)

#%%

MASTER = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Data_Partitions.csv')


if 'BIRADS' not in result.columns:
    
    if 'scanID' not in result.columns:
        result.columns = [u'scanID', u'y_pred', u'y_true', u'max_slice', u'GT_slice','slice_preds']

    result = pd.merge(result, MASTER[['Scan_ID','BIRADS','Image_QA']], left_on='scanID', right_on='Scan_ID')
    
    for i, row in result.loc[result['BIRADS'].str.contains(',')].iterrows():
        x = row['BIRADS']
        scanid = row['scanID']
        BIRADS = np.array(x.split(','), dtype='int').max()
        result.loc[result['scanID'] == scanid, 'BIRADS'] = BIRADS
        


result.loc[result['BIRADS'] == 1, 'BIRADS'] = '1'
result.loc[result['BIRADS'] == 2, 'BIRADS'] = '2'
result.loc[result['BIRADS'] == 3, 'BIRADS'] = '3'
result.loc[result['BIRADS'] == 4, 'BIRADS'] = '4'
result.loc[result['BIRADS'] == 5, 'BIRADS'] = '5'
result.loc[result['BIRADS'] == 6, 'BIRADS'] = '6'

df = pd.DataFrame(columns=['BIRADS','Benign','Cancer'])

for B in result['BIRADS'].unique():
    print(B)
    nb=0
    nm=0
    nb = len(result.loc[(result['BIRADS'] == B)*(result['y_true']==0)])
    nm = len(result.loc[(result['BIRADS'] == B)*(result['y_true']==1)])

    df=df.append({'BIRADS':B, 'Benign':nb, 'Cancer':nm}, ignore_index=True)

df.to_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/test_result_BIRADS_table.csv')

result['y_true'].value_counts()

result['BIRADS'].value_counts()
        

result = result.loc[result['BIRADS'].isin(['4','5'])]
result.reset_index(drop=True, inplace=True)


result['y_true'].value_counts()

N_benigns = len(result.loc[result['y_true'] == 0])
N_cancer = len(result.loc[result['y_true'] == 1])

auc = roc_auc_score(result['y_true'],result['y_pred'])

triage_curve = get_triage_curves(result, path='', steps=1000)


plt.figure(figsize=(8,4))
plt.suptitle('BI-RADS 4 & 5')
plt.subplot(121)
plt.hist(result.loc[result['y_true'] == 0, 'y_pred'], color='g', alpha=0.7, bins=50, label=f'Benign (n={N_benigns})')
plt.hist(result.loc[result['y_true'] == 1, 'y_pred'], color='r', alpha=0.7, bins=50, label=f'Cancer (n={N_cancer})')
plt.xlabel('Network probability')
plt.legend()

plt.subplot(122)
plt.plot(triage_curve['SPEC'],triage_curve['SENS'])
plt.plot([1,0],[0,1],'--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.text(x=0.2,y=0.2,s=f'AUC = {round(auc,2)}')



#%%

result = result.loc[result['BIRADS'].isin(['1','2','3','4','5','6'])]

auc = roc_auc_score(result['y_true'],result['y_pred'])

triage_curve = get_triage_curves(result, path='', steps=1000)

triage_curve.keys()
N_benigns = len(result.loc[result['y_true'] == 0])
N_cancer = len(result.loc[result['y_true'] == 1])

result['y_true'].value_counts()

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.hist(result.loc[result['y_true'] == 0, 'y_pred'], color='g', alpha=0.7, bins=50, label=f'Benign (n={N_benigns})')
plt.hist(result.loc[result['y_true'] == 1, 'y_pred'], color='r', alpha=0.7, bins=50, label=f'Cancer (n={N_cancer})')
plt.xlabel('Network probability')
plt.yscale('log')
plt.legend()

plt.subplot(122)
plt.plot(triage_curve['SPEC'],triage_curve['SENS'])
plt.plot([1,0],[0,1],'--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.text(x=0.2,y=0.2,s=f'AUC = {round(auc,2)}')

plt.savefig('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/test_result_cleanBIRADS.png', dpi=300)