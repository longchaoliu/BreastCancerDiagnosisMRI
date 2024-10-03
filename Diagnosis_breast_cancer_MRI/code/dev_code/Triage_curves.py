#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:26:36 2023

@author: deeperthought
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.patches as patches

RESULTS_TABLE_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/test_result.csv'

MASTER = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Data_Partitions.csv')

USE_ONLY_BIRADS45 = 0
USE_INDIVIDUAL_BREASTS = 0
save_fig = 0

RESULT_OUTPUT = '/'.join(RESULTS_TABLE_PATH.split('/')[:-1])

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
    
        
    plt.figure(1, figsize=(10,10))
    plt.subplot(2,2,1)
    plt.title('ROC')
    plt.plot(1-df['SPEC'],df['SENS'])
    plt.xlabel('1-SPEC')
    plt.ylabel('SENS')
    plt.plot([0,1],[0,1])
    
    
    plt.subplot(2,2,2)
    plt.title('Triage curve')
    #plt.plot(((df['TN']+df['FN'])/df['TOTAL']), 1-(df['FN']/(df['FN']+df['TP'])))
    plt.plot(((df['TN']+df['FN'])/df['TOTAL']), df['SENS'])
    plt.ylabel('Fraction cancers detected')
    plt.xlabel('Fraction omitted readings')
    plt.grid()
    
    
    plt.subplot(2,2,3)
    plt.title('PR curve')
    plt.plot(df['SENS'],df['PPV'])
    plt.xlabel('SENS')
    plt.ylabel('PPV')
    plt.grid()
    
    plt.subplot(2,2,4)
    plt.title('PR curve for benigns')
    plt.plot(df['SPEC'],df['NPV'])
    plt.xlabel('SPEC')
    plt.ylabel('NPV')
    plt.grid()
    
    plt.tight_layout()
    
    if len(path) > 0:
        plt.savefig(path, dpi=400)
        df.to_csv(path.replace('.png','.csv'))
        
    return df


#%%
        

RESULT_OUTPUT = '/'.join(RESULTS_TABLE_PATH.split('/')[:-1])
RESULT_NAME = RESULTS_TABLE_PATH.split('/')[-1].split('.csv')[0] 

results = pd.read_csv(RESULTS_TABLE_PATH)
if 'scanID' not in results.columns:
    results.columns = [u'scanID', u'y_pred', u'y_true', u'max_slice', u'GT_slice','slice_preds']

results = results.sort_values('y_pred')

if USE_INDIVIDUAL_BREASTS:
    results['breast_ID'] = results['scanID'].str[:20] + '_' + results['scanID'].str[-1]
    unique_breasts = results['breast_ID'].unique()
    
    converts = results.loc[results['y_true'] == 1]
    healthy = results.loc[results['y_true'] == 0]
    
    healthy = healthy.sample(frac=1).drop_duplicates('breast_ID')
    converts = converts.sample(frac=1).drop_duplicates('breast_ID')
    
    results = pd.concat([converts, healthy])
    
    RESULT_NAME = RESULT_NAME + '_IndividualBreasts'

results.reset_index(drop=True, inplace=True)


if 'BIRADS' not in results.columns:
    
    if 'scanID' not in results.columns:
        results.columns = [u'scanID', u'y_pred', u'y_true', u'max_slice', u'GT_slice','slice_preds']

    results = pd.merge(results, MASTER[['Scan_ID','BIRADS','Image_QA']], left_on='scanID', right_on='Scan_ID')
    
    for i, row in results.loc[results['BIRADS'].str.contains(',')].iterrows():
        x = row['BIRADS']
        scanid = row['scanID']
        BIRADS = np.array(x.split(','), dtype='int').max()
        results.loc[results['scanID'] == scanid, 'BIRADS'] = BIRADS
        
        
results['BIRADS'].value_counts()
        


results.loc[results['BIRADS'] == '6', 'y_true'].value_counts()
results.loc[results['BIRADS'].isin(['4','5']), 'y_true'].value_counts()




results.loc[results['BIRADS'] == 1, 'BIRADS'] = '1'
results.loc[results['BIRADS'] == 2, 'BIRADS'] = '2'
results.loc[results['BIRADS'] == 3, 'BIRADS'] = '3'
results.loc[results['BIRADS'] == 4, 'BIRADS'] = '4'
results.loc[results['BIRADS'] == 5, 'BIRADS'] = '5'
results.loc[results['BIRADS'] == 6, 'BIRADS'] = '6'



if USE_ONLY_BIRADS45:
    results = results.loc[results['BIRADS'].isin(['4','5'])]
    RESULT_NAME = RESULT_NAME + '_BIRADS45'
    results.reset_index(drop=True, inplace=True)


results['BIRADS'].value_counts()
results['y_true'].value_counts()


#%%


thresholds = get_triage_curves(results, path=RESULT_OUTPUT + '/triage_curves.png', steps=1000)



thresholds_biopsy = get_triage_curves(results_biopsy, path=RESULT_OUTPUT + '/triage_curves_biopsy.png', steps=1000)

plt.figure(figsize=(7,7))
plt.title('Triage curve')
plt.plot(((thresholds['TN']+thresholds['FN'])/thresholds['TOTAL']), thresholds['SENS'])
plt.ylabel('Fraction cancers detected')
plt.xlabel('Fraction omitted readings')
plt.grid()
plt.ylim([0.92,1])
plt.xticks(np.arange(0,1,0.1), rotation=45)
plt.hlines(y=0.995, xmin=0, xmax=1, linestyle='--', color='r', alpha=0.5)
plt.tight_layout()
#plt.savefig(RESULT_OUTPUT + '/triage_curve.png', dpi=400)

plt.figure(figsize=(7,7))
plt.title('Triage curve')
plt.plot(((thresholds_biopsy['TN']+thresholds_biopsy['FN'])/thresholds_biopsy['TOTAL']), thresholds_biopsy['SENS'])
plt.ylabel('Fraction cancers detected')
plt.xlabel('Fraction omitted readings')
plt.grid()
plt.ylim([0.92,1])
plt.xlim([0,0.4])
plt.xticks(np.arange(0,1,0.1), rotation=45)
plt.hlines(y=0.995, xmin=0, xmax=1, linestyle='--', color='r', alpha=0.5)
plt.tight_layout()
#plt.savefig(RESULT_OUTPUT + '/triage_curve_biopsy_ZOOM.png', dpi=400)

