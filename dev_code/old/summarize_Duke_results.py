#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:44:19 2024

@author: deeperthought
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.patches as patches
import math
import os




SAVE_RESULTS = False

OUTPUT_PATH = '/home/deeperthought/Documents/Papers_and_grants/Diagnosis_paper/data/'

NAME = 'Duke_test_result'



# res_duke = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Duke_predictions/predictions/results_duke_smartCrop.csv')

# No demographics
res_duke = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Duke_predictions/predictions/results_duke_smartCropChest.csv')

# With demographics
res_duke = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Duke_predictions/predictions/results_duke_smartCropChest_demographics.csv')

duke_clinical = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Duke_predictions/predictions/Duke_clinical_features_relevant.csv')





andy = pd.read_csv('/home/deeperthought/kirby/MSKdata/dukePublicData/metadata_andyDescriptor.csv')

andy.columns

andy.loc[~ andy['SubjectID'].isin(res_duke['Patient ID']), 'SubjectID']

#%% Some clean up

res_duke = res_duke.sort_values('Patient ID')
# res_duke = res_duke[:197]

res_duke = pd.merge(res_duke, duke_clinical, on='Patient ID')

# Check tumor location and contralateral breast
res_duke.columns

location_feats = ['Tumor Location', 'Bilateral Information','Multicentric/Multifocal','Contralateral Breast Involvement','For Other Side If Bilateral']

side_mismatch = []

for row in res_duke.iterrows():
    
    patient = row[1]['Patient ID']
    
    left_pathology = row[1]['left_breast_pathology']
    right_pathology = row[1]['right_breast_pathology']

    location = row[1]['Tumor Location']
    bilateral_info = row[1]['Bilateral Information']
    multicentric = row[1]['Multicentric/Multifocal']
    contralat_involvement = row[1]['Contralateral Breast Involvement']
    other_side = row[1]['For Other Side If Bilateral']
    
    if bilateral_info == '1':
        res_duke.loc[res_duke['Patient ID'] == patient, 'left_breast_pathology'] = 1
        res_duke.loc[res_duke['Patient ID'] == patient, 'right_breast_pathology'] = 1
        print('Corrected')
                        
        
    # if left_pathology and location == 'R' and bilateral_info == '0':
    #     side_mismatch.append(row[1]['Patient ID'])

    #     # print(f'\n\nInferred left breast pathology = {left_pathology}, right breast pathology = {right_pathology}')
    #     # print(f'Sheet:\n location = {location}\n bilateral_info = {bilateral_info}\n contralat_involvement={contralat_involvement}')


    # if right_pathology and location == 'L' and bilateral_info == '0':
    #     side_mismatch.append(row[1]['Patient ID'])
    
        # print(f'\n\nInferred left breast pathology = {left_pathology}, right breast pathology = {right_pathology}')
        # print(f'Sheet:\n location = {location}\n bilateral_info = {bilateral_info}\n contralat_involvement={contralat_involvement}')

    X1 = row[1]['X1']
    X2 = row[1]['X2']
    maxslice = row[1]['max_slice']
    
    if maxslice >= X1 and maxslice <= X2:
        res_duke.loc[res_duke['Patient ID'] == patient, 'Hit'] = 1

# res_duke.to_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Duke_predictions/predictions/results_duke_with_clinical.csv')





#%%

from scipy import stats

def z_test_proportion(sample_size, successes, hypothesized_proportion, confidence_level=0.95):
  """
  This function calculates the z-score and p-value for a z-test of proportions.

  Args:
      sample_size (int): Number of observations in the sample.
      successes (int): Number of successes in the sample.
      hypothesized_proportion (float): The expected proportion in the population.
      confidence_level (float, optional): Confidence level for the test (default is 0.95).

  Returns:
      tuple: A tuple containing the z-score and p-value.
  """

  # Calculate observed proportion
  observed_proportion = successes / sample_size

  # Calculate standard error
  standard_error = np.sqrt(hypothesized_proportion * (1 - hypothesized_proportion) / sample_size)

  # Calculate z-score
  z_score = (observed_proportion - hypothesized_proportion) / standard_error

  # Calculate p-value (two-tailed test)
  p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

  return z_score, p_value

#%%

# Witowski excluded

# witowski_excluded = ['Breast_MRI_065', 'Breast_MRI_120', 'Breast_MRI_127', 'Breast_MRI_134', 'Breast_MRI_232', 'Breast_MRI_279', 'Breast_MRI_465',
#  'Breast_MRI_514', 'Breast_MRI_574', 'Breast_MRI_596', 'Breast_MRI_700' , 'Breast_MRI_767']

# res_duke = res_duke.loc[ ~ res_duke['Patient ID'].isin(witowski_excluded)]

# I dont trust this one, segmentations and bounding box are different..
# res_duke = res_duke.loc[res_duke['Patient ID'] != 'Breast_MRI_005']

res_duke.columns

y_true = pd.concat([res_duke['left_breast_pathology'] , res_duke['right_breast_pathology']], ignore_index=True)
y_pred = pd.concat([res_duke['left_breast_global_pred'] , res_duke['right_breast_global_pred']], ignore_index=True)


auc = roc_auc_score(y_true, y_pred)
fpr, tpr, thr = roc_curve(y_true, y_pred)

res_duke['Hit'].value_counts()

n_hits = len(res_duke.loc[res_duke['Hit'] == 1])
n_miss = len(res_duke.loc[res_duke['Hit'] == 0])
N = len(res_duke)

sample_size = N
successes = n_hits
hypothesized_proportion = 0.5

z_score, p_value = z_test_proportion(sample_size, successes, hypothesized_proportion)


res = pd.DataFrame(columns=['y_true', 'y_pred'])
res['y_true'] = y_true
res['y_pred'] = y_pred



plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.hist(res.loc[res['y_true'] == 0, 'y_pred'], color='g', bins=50, label=f'benign breasts (N={len(res.loc[res["y_true"] == 0])})')
plt.hist(res.loc[res['y_true'] == 1, 'y_pred'], color='r', alpha=0.65, bins=50, label=f'malignant breasts (N={len(res.loc[res["y_true"] == 1])})')
plt.xlabel('Probability', fontsize=14)
plt.title(f'Out-of-site axial exams (n={N})', fontsize=14)
plt.legend()
# plt.yscale('log')

plt.subplot(1,2,2)
plt.plot(1-fpr, tpr)
plt.plot([0,1],[1,0],'--')
plt.xlabel('Specificity', fontsize=14)
plt.ylabel('Sensitivity', fontsize=14)
plt.text(x=0.2,y=0.1,s=f'AUC-ROC = {round(auc,2)}', fontsize=14)
plt.title(f'Out-of-site axial exams (n={N})', fontsize=14)
plt.xlim([0,1])
plt.ylim([0,1])

# plt.subplot(1,3,3)
# # plt.bar(0,N)
# plt.bar(0,n_miss, color='dodgerblue')
# plt.bar(1,n_hits, color='orange')
# plt.xticks([0,1],['Miss','Hit'], fontsize=14)
# plt.title('Localization', fontsize=14)
# # plt.text(x=-0.25,y=n_hits-2,s=f'z={round(z_score,2)}, p_val={round(p_value,3)}')
# plt.text(0-0.15,n_miss//2, s=f'{round(n_miss*100./N,1)} %', fontsize=14, weight='bold')
# plt.text(1-0.15,n_hits//2, s=f'{round(n_hits*100./N,1)} %', fontsize=14, weight='bold')
# plt.grid()
# plt.ylim([0,n_hits+5])
# plt.xlabel(f'z={round(z_score,2)}, p_val={round(p_value,2)}')

# if p_value < 0.05:
#     plt.text(x=0,y=n_hits+n_hits*0.05,s='_________________________', weight='bold')
#     plt.text(x=0.5,y=n_hits+n_hits*0.03,s='*', weight='bold', fontsize=15)
#     plt.ylim([0,n_hits+n_hits*0.1])

if SAVE_RESULTS:
    plt.savefig(OUTPUT_PATH + f'{NAME}.png', dpi=200)

    res.to_csv(OUTPUT_PATH + f'{NAME}.csv')



plt.figure(2, figsize=(4,5))

plt.bar(0,n_miss, color='dodgerblue')
plt.bar(1,n_hits, color='orange')
plt.xticks([0,1],['Miss','Hit'], fontsize=14)
plt.title('Localization', fontsize=14)
# plt.text(x=-0.25,y=n_hits-2,s=f'z={round(z_score,2)}, p_val={round(p_value,3)}')
plt.text(0-0.15,n_miss//2, s=f'{round(n_miss*100./N,1)} %', fontsize=14, weight='bold')
plt.text(1-0.15,n_hits//2, s=f'{round(n_hits*100./N,1)} %', fontsize=14, weight='bold')
plt.grid()
plt.ylim([0,n_hits+5])
plt.xlabel(f'z={round(z_score,2)}, p_val={round(p_value,2)}')

if p_value < 0.05:
    plt.text(x=0,y=n_hits+n_hits*0.05,s='_________________________', weight='bold')
    plt.text(x=0.5,y=n_hits+n_hits*0.03,s='*', weight='bold', fontsize=15)
    plt.ylim([0,n_hits+n_hits*0.1])

if SAVE_RESULTS:
    plt.savefig(OUTPUT_PATH + f'{NAME}_localization.png', dpi=200)

    res.to_csv(OUTPUT_PATH + f'{NAME}_localization.csv')



#%%

# df = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/axial_results/results.csv')


df1 = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Duke_predictions/predictions/results_duke_smartCrop.csv')
df2 = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Duke_predictions/predictions/results_duke_reshape512.csv')

df1 = df1.loc[df1['Patient ID'].isin(df2['Patient ID'])]

df = pd.merge(df1,df2, on='Patient ID')

df['delta'] = df['max_pred_x'] - df['max_pred_y']

df = df.sort_values('delta')

df.head(20)


# auc = roc_auc_score(df['y_true'], df['y_pred'])
