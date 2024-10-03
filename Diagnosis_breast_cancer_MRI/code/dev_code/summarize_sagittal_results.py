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


SAVE_RESULTS = True
OUTPUT_PATH = '/home/deeperthought/Documents/Papers_and_grants/Diagnosis_paper/data/'
NAME = 'sagittal_test_results'


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





result = pd.read_csv(PATH)

result.columns

malignants_total = len(result.loc[result['y_true'] == 1])

result_M = result.loc[result['GT_slice'] > 0]

from scipy.stats import spearmanr, pearsonr
corr, p = spearmanr(result_M['GT_slice'],result_M['max_slice'])
corr, p = pearsonr(result_M['GT_slice'],result_M['max_slice'])

plt.figure(figsize=(5,5))


plt.plot([5+2,51+2],[5,51],'r--', alpha=0.5)
plt.plot([5-2,51-2],[5,51],'r--', alpha=0.5)



# plt.title('{} segmented scans (from a total of {})'.format(len(result_M),malignants_total ))
plt.scatter(result_M['GT_slice'],result_M['max_slice'], alpha=0.5)#, c=result_M['y_pred'], cmap='inferno'); plt.colorbar()
plt.text(30,10,'r={}'.format(round(corr,2)), fontsize=15)
plt.xlabel('Index slice', fontsize=15)
plt.ylabel('Predicted slice', fontsize=15)
# plt.grid()

plt.tight_layout()

if SAVE_RESULTS:
    plt.savefig(OUTPUT_PATH + f'{NAME}_slice_correlation.png', dpi=200)





# plt.figure(figsize=(5,5))
# # plt.title('{} segmented scans (from a total of {})'.format(len(result_M),malignants_total ))
# plt.hist(0.3*(result_M['GT_slice'] - result_M['max_slice']), bins=40)#, c=result_M['y_pred'], cmap='inferno'); plt.colorbar()
# plt.xlabel('Distance from index slice (cm)', fontsize=15)


# plt.savefig(PATH.replace('.csv','_slice_correlation.png'), dpi=400)



#%%


auc = roc_auc_score(result['y_true'], result['y_pred'])
fpr, tpr, thr = roc_curve(result['y_true'], result['y_pred'])



result_M['Hit'] = -1

result_M['delta'] = result_M['max_slice'] - result_M['GT_slice']

result_M.loc[abs(result_M['delta']) <= 2, 'Hit'] = 1
result_M.loc[abs(result_M['delta']) > 2, 'Hit'] = 0

result_M['Hit'].value_counts()

n_hits = len(result_M.loc[result_M['Hit'] == 1])
n_miss = len(result_M.loc[result_M['Hit'] == 0])
N = len(result_M)

sample_size = N
successes = n_hits
hypothesized_proportion = 0.5

z_score, p_value = z_test_proportion(sample_size, successes, hypothesized_proportion)




plt.figure(1,figsize=(15,5))

plt.subplot(1,3,1)
plt.hist(result.loc[result['y_true'] == 0, 'y_pred'], color='g', bins=50, label=f'benign breasts (N={len(result.loc[result["y_true"] == 0])})')
plt.hist(result.loc[result['y_true'] == 1, 'y_pred'], color='r', alpha=0.65, bins=50, label=f'malignant breasts (N={len(result.loc[result["y_true"] == 1])})')
plt.xlabel('Probability', fontsize=14)
# plt.title(f'Out-of-site axial exams (n={N})', fontsize=14)
plt.legend()
plt.yscale('log')

plt.subplot(1,3,2)
plt.plot(1-fpr, tpr)
plt.plot([0,1],[1,0],'--')
plt.xlabel('Specificity', fontsize=14)
plt.ylabel('Sensitivity', fontsize=14)
plt.text(x=0.2,y=0.1,s=f'AUC-ROC = {round(auc,2)}', fontsize=14)
# plt.title(f'Out-of-site axial exams (n={N})', fontsize=14)
plt.xlim([0,1])
plt.ylim([0,1])

plt.subplot(1,3,3)
# plt.bar(0,N)
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
    plt.savefig(OUTPUT_PATH + f'{NAME}.png')

    result.to_csv(OUTPUT_PATH + f'{NAME}.csv')



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





#%%
# deltas = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/test_result_with_deltas.csv')

# hits = deltas.loc[deltas['min_distance'] == 0]
# miss = deltas.loc[deltas['min_distance'] > 0]


# result_M_hits = result_M.loc[result_M['scan'].isin(hits['scan'])]
# result_M_miss = result_M.loc[~result_M['scan'].isin(hits['scan'])]


# plt.figure(figsize=(4,4))
# plt.title('{} segmented scans (from a total of {})'.format(len(result_M),malignants_total ))
# plt.scatter(result_M_hits['GT_slice'],result_M_hits['max_slice'], alpha=0.5, color='r', label='Hit')#, c=result_M['y_pred'], cmap='inferno'); plt.colorbar()
# plt.scatter(result_M_miss['GT_slice'],result_M_miss['max_slice'], alpha=0.5, color='b', label='Miss')#, c=result_M['y_pred'], cmap='inferno'); plt.colorbar()

# #plt.text(30,10,'r={}'.format(round(corr,2)), fontsize=15)
# plt.xlabel('segmented slice', fontsize=15)
# plt.ylabel('slice with maximum risk', fontsize=15)
# plt.grid()
# plt.legend()

# plt.savefig(PATH.replace('.csv','_slice_correlation_hits.png'), dpi=400)