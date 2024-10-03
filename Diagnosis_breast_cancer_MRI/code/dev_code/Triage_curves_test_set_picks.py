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



# TEST SET
RESULTS_TABLE_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/test_result.csv'
resM = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/test_result_with_deltas.csv')

axial_results = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/axial_results/results.csv')
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
    
    return df[:-1]



#%%
        

RESULT_OUTPUT = '/'.join(RESULTS_TABLE_PATH.split('/')[:-1])
RESULT_NAME = RESULTS_TABLE_PATH.split('/')[-1].split('.csv')[0] 

results = pd.read_csv(RESULTS_TABLE_PATH)


plt.hist(results.loc[results['y_true']==1,'y_pred'], color='r', bins=100)

results = pd.merge(results, resM[['scan','min_distance']], on='scan', how='left')

plt.hist(results.loc[results['y_true']==1,'y_pred'], color='r', bins=100)
plt.hist(results.loc[results['y_true']==0,'y_pred'], color='g', bins=100, alpha=0.5)



results = results.sort_values('y_pred')

def bootstrap_triage_curves(results, n_draws, n_steps):
    
    DRAWS = n_draws
    STEPS = n_steps
    triage_curves = np.empty(shape=(DRAWS,STEPS,2))
    
    for i in range(DRAWS):
        TMP = results.copy()
        TMP = TMP.sample(replace=True, n=len(TMP))
        tr_df = get_triage_curves(TMP, steps=STEPS)
    
        effort = (tr_df['TN']+tr_df['FN'])/tr_df['TOTAL']
        sensitivity = tr_df['SENS']
        
        triage_curves[i,:,0] = effort
        triage_curves[i,:,1] = sensitivity
        print('finished draw {}/{}'.format(i, DRAWS))
    
    
    return triage_curves


def plot_triage_curve(triage_curves, xlabel='', ylim=[0,1], xlim=[0,1], path='', **kwargs):
    
    median_effort = np.median(triage_curves[:,:,0],axis=0) # median over draws, not steps
    median_sensitivity = np.median(triage_curves[:,:,1],axis=0) # median over draws, not steps
    
    min_effort = np.min(triage_curves[:,:,0],axis=0) # median over draws, not steps
    min_sensitivity = np.min(triage_curves[:,:,1],axis=0) # median over draws, not steps
    
    max_effort = np.max(triage_curves[:,:,0],axis=0) # median over draws, not steps
    max_sensitivity = np.max(triage_curves[:,:,1],axis=0) # median over draws, not steps
    
    fig, ax = plt.subplots()#(figsize=kwargs.get('figsize',(5,5)))
    ax.plot(median_effort, median_sensitivity, color='k')
    ax.fill_between(x=median_effort, y1=min_sensitivity, y2=max_sensitivity, alpha=0.4)
    ax.set_ylabel('Sensitivity')
    ax.set_xlabel(xlabel)
    
    ax.grid()    
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    
    if len(path) >0:
        plt.savefig(path, dpi=400)

    return median_effort, median_sensitivity, min_sensitivity, max_sensitivity

#
#triage_curves_alldata = bootstrap_triage_curves(results, n_draws=10, n_steps=1000) #500,1000
triage_curves_alldata = np.load('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/triage_curves_alldata.npy', allow_pickle=True)



resB = results.loc[results['y_true'] == 0]
misses = resM.loc[resM['min_distance'] > 0]
result_misses = pd.concat([resB, misses])



triage_curves_misses = bootstrap_triage_curves(result_misses, n_draws=500, n_steps=1000)

triage_curves_misses = np.load('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/triage_curves_misses.npy', allow_pickle=True)

PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/'

triage_omitted_readings_plot = plot_triage_curve(triage_curves_alldata, xlabel='Effort (ommited readings)', ylim=[0.95,1.005], xlim=[0,0.6], path='', figsize=(4,4))#PATH+'Effort_ommited_readings.png')

plot_triage_curve(triage_curves_misses, xlabel='Effort (abbreviated readings)', ylim=[0.0,1.05], path='', figsize=(4,4))#path=PATH+'Effort_abbreviated_readings.png')


#triage_curves_alldata = bootstrap_triage_curves(results, n_draws=10, n_steps=1000) #500,1000

triage_abbreviated_readings_plot = plot_triage_curve(triage_curves_misses, xlabel='Effort (abbreviated readings)', ylim=[0.8,1.005], xlim=[0.5,0.8], path='', figsize=(4,4))#path=PATH+'Effort_abbreviated_readings.png')



np.save('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/triage_curves_alldata.npy', triage_curves_alldata)
np.save('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/triage_curves_misses.npy', triage_curves_misses)




scanIDs_biopsied = MASTER.loc[MASTER['BIRADS_clean'].isin([4,5]), 'Scan_ID'].values
results_biopsied = results.loc[results['scan'].isin(scanIDs_biopsied)]

triage_curves_biopsied = bootstrap_triage_curves(results_biopsied, n_draws=10, n_steps=1000) #500,1000


triage_curves_biopsied_plot = plot_triage_curve(triage_curves_biopsied, xlabel='Effort (abbreviated readings)', ylim=[0,1], xlim=[0,1], path='', figsize=(4,4))#path=PATH+'Effort_abbreviated_readings.png')



#%% Paper figure 3

N,M = 2,3

plt.figure(figsize=(12,8))

plt.subplot(N,M,1); plt.title('Sagittal exams')
plt.hist(result_misses.loc[result_misses['y_true']==0,'y_pred'], color='g', bins=100, alpha=0.5, label='benign (n=6353)')
plt.hist(results.loc[results['y_true']==1,'y_pred'], color='r', bins=100, alpha=1, label='cancer, hit (n=186)')
plt.hist(result_misses.loc[result_misses['y_true']==1,'y_pred'], color='b', bins=100, alpha=1, label='cancer, miss (n=76)')
plt.yscale('log')
plt.xlabel('Predicted risk')
plt.legend()

plt.subplot(N,M,2); plt.title('Omitted readings')
plt.plot(triage_omitted_readings_plot[0], triage_omitted_readings_plot[1], color='k')
plt.fill_between(x=triage_omitted_readings_plot[0], y1=triage_omitted_readings_plot[2], y2=triage_omitted_readings_plot[3], alpha=0.4)
plt.xlabel('Effort (omitted readings)')
plt.ylabel('Sensitivity')
plt.ylim([0.95,1.005])
plt.xlim([0,0.6])
plt.grid()

plt.subplot(N,M,3); plt.title('Abbreviated readings')
plt.plot(triage_abbreviated_readings_plot[0], triage_abbreviated_readings_plot[1], color='k')
plt.fill_between(x=triage_abbreviated_readings_plot[0], y1=triage_abbreviated_readings_plot[2], y2=triage_abbreviated_readings_plot[3], alpha=0.4)
plt.xlabel('Effort (abbreviated readings)')
plt.ylabel('Sensitivity')
plt.ylim([0.85,1.01])
plt.xlim([0.5,0.8])
plt.grid()

plt.subplot(N,M,4); plt.title('BI-RADS 4,5')
plt.hist(results_biopsied.loc[results_biopsied['y_true']==0,'y_pred'], color='g', bins=100, alpha=0.5, label='benign (n=484)')
plt.hist(results_biopsied.loc[results_biopsied['y_true']==1,'y_pred'], color='r', bins=100, alpha=1, label='cancer (n=94)')
plt.yscale('log')
plt.legend()
plt.xlabel('Predicted risk')



plt.subplot(N,M,5); plt.title('Omitted biopsies')
plt.plot(triage_curves_biopsied_plot[0], triage_curves_biopsied_plot[1], color='k')
plt.fill_between(x=triage_curves_biopsied_plot[0], y1=triage_curves_biopsied_plot[2], y2=triage_curves_biopsied_plot[3], alpha=0.4)
plt.xlabel('Effort (omitted biopsies)')
plt.ylabel('Sensitivity')
plt.ylim([0.85,1.01])
plt.xlim([0,0.4])
plt.grid()

plt.tight_layout()

plt.savefig('/home/deeperthought/Documents/Papers_and_grants/Diagnosis_paper/Fig3_clinical_applications.png', dpi=400)


#%%

plt.figure(figsize=(8,4))
sens100_threshold = 0.267

effort_range = triage_curves_alldata[:,int(sens100_threshold/(1./1000)),0] # range of effort values for threshold
sensitivity_range = triage_curves_alldata[:,int(sens100_threshold/(1./1000)),1] # range of SENS values for threshold

plt.subplot(1,2,1)
plt.title('Bootstrap N=500')
plt.scatter(effort_range, sensitivity_range, alpha=0.5, s=15)
plt.xlabel('Effort (% omitted readings)')
plt.ylabel('Sensitivity')
plt.grid()


sensitivity_995_threshold = 0.311 # including all cancers in val.


effort_range_995 = triage_curves_misses[:,int(sensitivity_995_threshold/(1./1000)),0] # range of effort values for threshold
sensitivity_range_995 = triage_curves_misses[:,int(sensitivity_995_threshold/(1./1000)),1] # range of SENS values for threshold

plt.subplot(1,2,2)
plt.title('Bootstrap N=500')
plt.scatter(effort_range_995, sensitivity_range_995, alpha=0.5, s=15)
plt.xlabel('Effort (% abbreviated readings)')
plt.ylabel('Sensitivity')
plt.grid()


np.percentile(effort_range, 2.5), np.median(effort_range), np.percentile(effort_range, 97.5)
np.percentile(sensitivity_range, 2.5), np.median(sensitivity_range), np.percentile(sensitivity_range, 97.5)


np.percentile(effort_range_995, 2.5), np.median(effort_range_995), np.percentile(effort_range_995, 97.5)
np.percentile(sensitivity_range_995, 2.5), np.median(sensitivity_range_995), np.percentile(sensitivity_range_995, 97.5)



'''
shuffle stats on test set
just pick points in the triage curve with error bars, done.

each bootstrap pick SENS=100
just draw errors on effort.

'''


# SENS = 100% 
median_SENS = np.median(triage_curves_alldata[:,:,1], axis=0)
threshold_sens100 = np.argwhere(median_SENS == 1)[-1][0]

effort_range_sens1 = triage_curves_alldata[:,threshold_sens100,0] # range of effort at sens=1
np.percentile(effort_range_sens1, 2.5), np.median(effort_range_sens1), np.percentile(effort_range_sens1, 97.5)

# Pick here effort= 50% or something.
triage_curves_misses
median_effort = np.median(triage_curves_misses[:,:,0], axis=0)
threshold_effort50 = np.argwhere(median_effort <= 0.5)[-1][0]
SENS_range_effort50 = triage_curves_misses[:,threshold_effort50,1] # range of effort at sens=1
np.percentile(SENS_range_effort50, 2.5), np.median(SENS_range_effort50), np.percentile(SENS_range_effort50, 97.5)

SENS_alldata_range_effort50 = triage_curves_alldata[:,threshold_effort50,1] # range of effort at sens=1
np.percentile(SENS_alldata_range_effort50, 2.5), np.median(SENS_alldata_range_effort50), np.percentile(SENS_alldata_range_effort50, 97.5)


'''
something wrong
if I get SENS=0.985 on the data with only misses, it means the threshold is above some missed cancers.
but the threshold is at 0.331, 
'''


#%% Figure ROC results:

axial_results = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/test_result_QA.csv')
# add birads. 
MASTER['BIRADS'].value_counts()

MASTER['BIRADS_clean'] = 999

MASTER['BIRADS_clean'].value_counts()

for row in MASTER.iterrows():
    birads = row[1]['BIRADS']
    scanID = row[1]['Scan_ID']
    # print(f'From: {birads}')
    if ',' in birads:
        vals = birads.split(',')
        birads = max(vals)
    if '.' in birads:
        vals = birads.split('.')
        birads = max(vals)        
    if (birads == 'MISSING') or (birads == 'NR'):
        continue

    birads = int(birads)
    # print(f'To: {birads}')

    MASTER.loc[MASTER['Scan_ID'] == scanID, 'BIRADS_clean'] = birads

scanIDs_biopsied = MASTER.loc[MASTER['BIRADS_clean'].isin([4,5]), 'Scan_ID'].values
results_biopsied = results.loc[results['scan'].isin(scanIDs_biopsied)]


triage_results_BIRADS45_df = get_triage_curves(results_biopsied)
triage_results_df = get_triage_curves(results)
triage_results_axial_df = get_triage_curves(axial_results)

roc_auc_score(results['y_true'], results['y_pred'])
roc_auc_score(results_biopsied['y_true'], results_biopsied['y_pred'])
roc_auc_score(axial_results['y_true'], axial_results['y_pred'])

plt.figure(figsize=(12,4))
n,m=1,3
# figure
plt.subplot(n,m,1); plt.title('Sagittal MRI (n=6,615)')
plt.plot(1-np.append(triage_results_df['SPEC'].values, 1), np.append(triage_results_df['SENS'].values, 0))
plt.plot([0,1],[0,1], linestyle='--', color='gray')
plt.xlabel('1-specificity')
plt.ylabel('sensitivity')

plt.subplot(n,m,2); plt.title('Sagittal MRI - BI-RADS 4,5 (n=578)')
plt.plot(1-np.append(triage_results_BIRADS45_df['SPEC'].values, 1), np.append(triage_results_BIRADS45_df['SENS'].values, 0))
plt.plot([0,1],[0,1], linestyle='--', color='gray')
plt.xlabel('1-specificity')
plt.ylabel('sensitivity')

plt.subplot(n,m,3); plt.title('Axial MRI (n=1,496)')
plt.plot(1-np.append(triage_results_axial_df['SPEC'].values, 1), np.append(triage_results_axial_df['SENS'].values, 0))
plt.plot([0,1],[0,1], linestyle='--', color='gray')
plt.xlabel('1-specificity')
plt.ylabel('sensitivity')


plt.figure(figsize=(4,4))
# figure
plt.plot(1-np.append(triage_results_df['SPEC'].values, 1), np.append(triage_results_df['SENS'].values, 0))
plt.plot(1-np.append(triage_results_BIRADS45_df['SPEC'].values, 1), np.append(triage_results_BIRADS45_df['SENS'].values, 0))
plt.plot(1-np.append(triage_results_axial_df['SPEC'].values, 1), np.append(triage_results_axial_df['SENS'].values, 0))
plt.plot([0,1],[0,1], linestyle='--', color='gray')



#%%

triage_curves = bootstrap_triage_curves(results, n_draws=10, n_steps=1000) #500,1000
plot_triage_curve(triage_curves, xlabel='Effort (omitted readings)', ylim=[0,1], xlim=[0,1], path='', figsize=(4,4))#path=PATH+'Effort_abbreviated_readings.png')


triage_curves_misses = bootstrap_triage_curves(result_misses, n_draws=10, n_steps=1000) #500,1000
plot_triage_curve(triage_curves_misses, xlabel='Effort (abbreviated readings)', ylim=[0,1], xlim=[0,1], path='', figsize=(4,4))#path=PATH+'Effort_abbreviated_readings.png')


triage_curves_biopsied = bootstrap_triage_curves(results_biopsied, n_draws=10, n_steps=1000) #500,1000
plot_triage_curve(triage_curves_biopsied, xlabel='Effort (abbreviated readings)', ylim=[0,1], xlim=[0,1], path='', figsize=(4,4))#path=PATH+'Effort_abbreviated_readings.png')




triage_curves_axial = bootstrap_triage_curves(axial_results, n_draws=10, n_steps=1000) #500,1000
plot_triage_curve(triage_curves_axial, xlabel='Effort (abbreviated readings)', ylim=[0,1], xlim=[0,1], path='', figsize=(4,4))#path=PATH+'Effort_abbreviated_readings.png')


plot_triage_curve(triage_curves, xlabel='Effort (omitted sagittal readings)', ylim=[0.9,1.005], xlim=[0,1], path='', figsize=(4,4))#path=PATH+'Effort_abbreviated_readings.png')
plot_triage_curve(triage_curves_misses, xlabel='Effort (abbreviated sagittal readings)', ylim=[0.9,1.005], xlim=[0,1], path='', figsize=(4,4))#path=PATH+'Effort_abbreviated_readings.png')
plot_triage_curve(triage_curves_biopsied, xlabel='Effort (omitted biopsies)', ylim=[0.9,1.005], xlim=[0,0.6], path='', figsize=(4,4))#path=PATH+'Effort_abbreviated_readings.png')
plot_triage_curve(triage_curves_axial, xlabel='Effort (omitted axial readings)', ylim=[0.9,1.005], xlim=[0,0.6], path='', figsize=(4,4))#path=PATH+'Effort_abbreviated_readings.png')

