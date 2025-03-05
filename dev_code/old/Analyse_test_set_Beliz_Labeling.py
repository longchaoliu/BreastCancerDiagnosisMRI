#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 13:54:05 2023

@author: deeperthought
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


QA = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/DATA/sagittal_test_set_Beliz_QA/Labeling_results.csv')

QA['scan'] = QA['Exam'].str[:31]


QA = QA.drop_duplicates('Exam')

result = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/test_result.csv')

# result = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/PAPER_SESSIONS/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/result_best_model/test_result.csv')

result['y_true'].value_counts()

result = pd.merge(result, QA, on='scan')


result['image_problems'] = result['image_artifact_in_breast_detected'] + result['Bad_Image'] + result['fat_sat_artifact'] + result['image_artifact_outside_breast_detected']

result.loc[result['image_problems'] > 0, 'image_problems'] = 1

    
result['any_problem'] = result['image_problems'] +  result['biopsy_clip_detected']+  result['post_surgery_artifact']+  result['Implant']

result.loc[result['any_problem'] > 0, 'any_problem'] = 1


def bootstrap_auc_differences(result, COL, N=1000):
    
    auc1 = roc_auc_score(result.loc[result[COL] == 0, 'y_true'], result.loc[result[COL] == 0, 'y_pred'])
    auc2 = roc_auc_score(result.loc[result[COL] == 1, 'y_true'], result.loc[result[COL] == 1, 'y_pred'])
    
    delta_auc = auc1-auc2

    b1 = len(result.loc[(result[COL] == 0)*(result['y_true'] == 0)])
    m1 = len(result.loc[(result[COL] == 0)*(result['y_true'] == 1)])
     
    b2 = len(result.loc[(result[COL] == 1)*(result['y_true'] == 0)])
    m2 = len(result.loc[(result[COL] == 1)*(result['y_true'] == 1)])
    
    print(f'/nFor {COL}, we have: 0 [{b1}/{m1}] vs 1 [{b2}/{m2}]')
    print(f'AUC0 = {auc1}, AUC1 = {auc2}, Delta={delta_auc}')
    print('Generating samples from null distribution..')
    
    roc_deltas = []
    TMP = result.copy()

    for _ in range(N):

        
        
        SAMPLE1 = pd.concat([TMP.loc[TMP['y_true'] == 0].sample(replace=True, n=b1), TMP.loc[TMP['y_true'] == 1].sample(replace=True, n=m1)])
        SAMPLE2 = pd.concat([TMP.loc[TMP['y_true'] == 0].sample(replace=True, n=b2), TMP.loc[TMP['y_true'] == 1].sample(replace=True, n=m2)])

        
        roc_auc1 = roc_auc_score(SAMPLE1['y_true'], SAMPLE1['y_pred'])
        roc_auc2 = roc_auc_score(SAMPLE2['y_true'], SAMPLE2['y_pred'])

        roc_deltas.append(roc_auc1-roc_auc2)
            
    return delta_auc, roc_deltas




def bootstrap_auc_differences_v2(result, COL, N=1000):
    

    df1 = result.loc[result[COL] == 0]
    df2 = result.loc[result[COL] == 1]
    
    delta_auc = 0

    b1 = len(df1.loc[df1['y_true'] == 0])
    m1 = len(df1.loc[df1['y_true'] == 1])
     
    b2 = len(df2.loc[df2['y_true'] == 0])
    m2 = len(df2.loc[df2['y_true'] == 1])
    
    roc_deltas = []

    for _ in range(N):

        SAMPLE1 = pd.concat([df1.loc[df1['y_true'] == 0].sample(replace=True, n=b1), df1.loc[df1['y_true'] == 1].sample(replace=True, n=m1)])
        SAMPLE2 = pd.concat([df2.loc[df2['y_true'] == 0].sample(replace=True, n=b2), df2.loc[df2['y_true'] == 1].sample(replace=True, n=m2)])
    
        roc_auc1 = roc_auc_score(SAMPLE1['y_true'], SAMPLE1['y_pred'])
        roc_auc2 = roc_auc_score(SAMPLE2['y_true'], SAMPLE2['y_pred'])

        roc_deltas.append(roc_auc1-roc_auc2)
            
    return delta_auc, roc_deltas


'Implant',

u'fat_sat_artifact',
u'image_artifact_in_breast_detected',
u'image_artifact_in_breast_detected',
u'nipple_enhancing_detected',
u'Bad_Image' 

u'post_surgery_artifact', u'image_artifact_in_breast_detected',
u'image_artifact_outside_breast_detected', 
u'biopsy_clip_detected'







#%%


'''
For a condition , get AUC1 and AUC2, and store the difference. This is our target value to check.

Now we generate a Null Distribution:
    
This comes from the assumption "data from condition 1 and condition 2 are from the same distribution or population"

so we merge these two (its already merged)

we now generate fake samples1 and samples2 from our null:
    
this means we draw randomly with replacement from our null, generating population 1 (so keeping same numbers of classes in population with condition 1)
and the same for condition 2.

For examples: Biopsy_clips, No: 100 benigns, 10 malignants. Yes: 500 benigns, 50 malignants.
Sample1: draw random 100 benigns, and 10 malignants. Sample2L draw random 500 benigns, 50 malignants.

Get the AUCs and store the difference.

Do this 1000 times.

Now we have a distribution of AUC differences following the Null Distribution. This should be centered at 0 !!

Now we check how far from the center our statistic of interest is!

The area to the side is our p. In whiche percentile it lands.

'''

i = 1
plt.figure(figsize=(8,8))
for COL in ['biopsy_clip_detected','post_surgery_artifact', 'Implant', 'image_problems','any_problem']:#, 'image_artifact_in_breast_detected','image_artifact_outside_breast_detected' ,'Bad_Image','nipple_enhancing_detected','fat_sat_artifact']:
    print('\n')
    print(COL)

    delta_auc, auc_deltas = bootstrap_auc_differences(result, COL, N=1000)
    
    pval = sum(auc_deltas > delta_auc) / float(len(auc_deltas))

    plt.subplot(2,3,i)
    plt.hist(auc_deltas, label='H0', bins=80);
    plt.vlines(x=delta_auc,ymin=0,ymax=50, color='r', linestyle='--')
    plt.vlines(x=np.mean(auc_deltas),ymin=0,ymax=50, color='k')

    plt.legend()
    plt.title(f'{COL}: pval={pval}')

    # v = np.percentile(auc_deltas, 50)
    # sum(np.abs(auc_deltas) > np.abs(v)) / float(len(auc_deltas))
  
    i+=1
    
plt.tight_layout()






#%%



i = 1
plt.figure(figsize=(8,8))
for COL in ['biopsy_clip_detected','post_surgery_artifact', 'Implant', 'image_problems']:#, 'image_artifact_in_breast_detected','image_artifact_outside_breast_detected' ,'Bad_Image','nipple_enhancing_detected','fat_sat_artifact']:
    print('\n')
    print(COL)

    delta_auc, auc_deltas = bootstrap_auc_differences_v2(result, COL, N=1000)
    
    pval = sum(np.array(auc_deltas) > delta_auc) / float(len(auc_deltas))

    plt.subplot(2,2,i)
    plt.hist(auc_deltas, label='H0', bins=80);
    plt.vlines(x=delta_auc,ymin=0,ymax=50, color='r', linestyle='--')
    plt.vlines(x=np.mean(auc_deltas),ymin=0,ymax=50, color='k')

    plt.legend()
    plt.title(f'{COL}: pval={pval}')
    # v = np.percentile(auc_deltas, 50)
    # sum(np.abs(auc_deltas) > np.abs(v)) / float(len(auc_deltas))
  
    i+=1
    
plt.tight_layout()