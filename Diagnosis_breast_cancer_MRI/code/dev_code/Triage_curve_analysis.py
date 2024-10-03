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

RESULTS_TABLE_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/validation_result.csv'
resM = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/validation_result_with_deltas.csv')
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



# sens = tpr = tp / (tp + fp)

sens100_threshold = resM.loc[resM['min_distance'] > 0, 'y_pred'].min()



triage_curve = get_triage_curves(result, path='', steps=1000)
triage_curve.loc[triage_curve['SENS'] >= 0.995]
sensitivity_995_threshold = triage_curve.iloc[311]['threshold']



res_miss = resM.loc[resM['min_distance'] > 0]
resB = result.loc[result['y_true'] == 0]

result_misses = pd.concat([res_miss, resB])

triage_curve_misses = get_triage_curves(result_misses, path='', steps=1000)
triage_curve_misses.loc[triage_curve_misses['SENS'] >= 0.995]
sensitivity_995_threshold = triage_curve_misses.iloc[267]['threshold']


'''
the threshold for SENS=100 and SENS>99.5 is the same, at least defined over missed cancers 
'''

plt.figure(figsize=(8,14))

plt.subplot(3,1,1)

plt.figure(figsize=(4,4))
plt.title('All predictions, hits: threshold ==0 slices away')


plt.hist(result.loc[result['y_true'] == 0, 'y_pred'], color='g', bins=100, alpha=0.5, label='benigns')
plt.hist(resM.loc[resM['min_distance'] > 0, 'y_pred'], color='blue', bins=100, alpha=1, label='cancer, miss')
plt.hist(resM.loc[resM['min_distance'] == 0, 'y_pred'], color='r', bins=100, alpha=0.5, label='cancer, hit')
plt.yscale('log')
#plt.vlines(sensitivity_995_threshold, 0, 1000, linestyle='--')
plt.legend()
plt.xlabel('network prediction')
#plt.savefig('/home/deeperthought/Documents/THESIS/third draft/Triage_fig2.png', dpi=400)

plt.subplot(3,1,2)
plt.title('Only cancers, hits: threshold ==0 slices away')

plt.hist(resM.loc[resM['min_distance'] > 0, 'y_pred'], color='blue', bins=100, alpha=1, label='cancer, miss')
plt.hist(resM.loc[resM['min_distance'] == 0, 'y_pred'], color='r', bins=100, alpha=0.5, label='cancer, hit')
plt.legend()
plt.yscale('log')
plt.xlabel('network prediction')

plt.subplot(3,1,3)
plt.title('Only cancers, hits: threshold <=2 slices away')
plt.hist(resM.loc[resM['min_distance'] > 2, 'y_pred'], color='blue', bins=100, alpha=1, label='cancer, miss')
plt.hist(resM.loc[resM['min_distance'] <= 2, 'y_pred'], color='r', bins=100, alpha=0.5, label='cancer, hit')
plt.legend()
plt.yscale('log')
plt.xlabel('network prediction')

plt.savefig('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/validation_results_hits_and_misses.png', dpi=400)





'''
fix threshold (val set?)

50% of green histogram is constant 

meaningful error bars based on cancer distribution

sensitivity - effort curve  (triage)

vertical axis, effort, error bards

new triage curve for each shuffle.
error bars on the triage curve

with abbreviated reading on X% of data, we still put eyes on target on 99% +- CI

1- complete triage: threshold set for SENS 100% ~30% can be ommitted
2- abbreviated reading curve: 70% abbreviated reading but still putting all eyes on cancer. Here we can put error bars.


operating points: 100% SENS, or 

2 triage curves on test set:
first curve is including everything.  
second curve is only include misses (blue) and green - abbreviated reading curve.




'''


