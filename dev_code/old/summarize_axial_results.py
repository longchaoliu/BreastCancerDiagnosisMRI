#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:11:32 2024

@author: deeperthought
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score



EXAM_LEVEL_RESULT = False
SAVE_RESULTS = False
OUTPUT_PATH = '/home/deeperthought/Documents/Papers_and_grants/Diagnosis_paper/data/'

NAME = 'Axial_test_result'


if EXAM_LEVEL_RESULT:
    NAME += '_exams'


PATH = "/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/axial_results/"

# PATH = "Z:\\Projects\\DGNS\\2D_Diagnosis_model\\Sessions\\FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8\\axial_results\\"

res_axial = pd.read_csv(PATH + 'results_noFineTune_res04_smartCropChest.csv')
res_axial2 = pd.read_csv(PATH + 'results_noFineTune_res04_smartCropChest_part2.csv')
res_axial3 = pd.read_csv(PATH + 'results_noFineTune_res04_smartCropChest_segmented.csv')

res_axial = pd.concat([res_axial, res_axial2, res_axial3])


# qa = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/DATA/axial_test_QA/QA_results.csv')

# res_axial = res_axial.loc[ ~ res_axial['Exam'].isin(qa['Scan'])]

# test_res = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/test_result_QA.csv')
test_res = pd.read_csv(PATH + 'results_noFineTune_res04_smartCrop_final.csv')

res_axial['left_breast_exam'] = res_axial['Exam'] + '_l'
res_axial['right_breast_exam'] = res_axial['Exam'] + '_r'


res_left = res_axial[['left_breast_exam','left_breast_pathology','left_breast_global_pred', 'slice_preds']]
res_right = res_axial[['right_breast_exam','right_breast_pathology','right_breast_global_pred', 'slice_preds']]


res_left['exam_max_slice'] = -1

for row in res_left.iterrows():
    p = row[1]['slice_preds']
    scanID = row[1]['left_breast_exam']
    p = p.replace('[','').replace(']','').replace('\n','').split(' ')
    
    p = [float(x) for x in p if len(x)>0]

    # LEN = len(p)
    # p_breast = p[:LEN//2]   
       
    res_left.loc[res_left['left_breast_exam'] == scanID,'exam_max_slice'] = np.argmax(p)
    


res_right['breast_max_pred'] = -1
res_right['breast_max_slice'] = -1

for row in res_right.iterrows():
    p = row[1]['slice_preds']
    scanID = row[1]['right_breast_exam']
    p = p.replace('[','').replace(']','').replace('\n','').split(' ')
    
    p = [float(x) for x in p if len(x)>0]

    # LEN = len(p)
    # p_breast = p[LEN//2:]       
    
    res_right.loc[res_right['right_breast_exam'] == scanID,'exam_max_slice'] = np.argmax(p)    
    
    
res_left = res_left[['left_breast_exam', 'left_breast_pathology', 'left_breast_global_pred', 'exam_max_slice']]
res_right = res_right[['right_breast_exam', 'right_breast_pathology', 'right_breast_global_pred', 'exam_max_slice']]

res_left.columns = ['Exam', 'y_true', 'y_pred','exam_max_slice']
res_right.columns = ['Exam', 'y_true', 'y_pred','exam_max_slice']



result = pd.concat([res_right,res_left])
result = result.drop_duplicates('Exam')


# result = result.loc[result['Exam'].isin(test_res['Exam'])]
test_res = test_res.loc[test_res['Exam'].isin(result['Exam'])]


result = result.dropna()
test_res = test_res.dropna()



#%% Remove overlap with sagittal set?

sagittal_data = np.load('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/DATA/Data.npy', allow_pickle=True).item()

# sagittal_data = np.load('Z:\\Projects\\DGNS\\2D_Diagnosis_model\\DATA\\Data.npy', allow_pickle=True).item()

sagittal_subjects = list(set([x[:20] for x in sagittal_data['train']]))

axial_subjects = list(set(result['Exam'].str[:20]))

len(sagittal_subjects), len(axial_subjects)
       
n1 = len(axial_subjects)

n2 = len(set(axial_subjects) - set(sagittal_subjects))

print(f'Removed {n1-n2} subjects that were part of the sagittal training cohort.')

result['subj'] = result['Exam'].str[:20]

result = result.loc[ ~ result['subj'].isin(sagittal_subjects)]


#%% Remove BIRADS 6 that had pathology more than 3 months before image was taken (possible tumor removal)

df = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/REDCAP/2023/Axial_cancers_date_of_pathology_vs_date_of_exam.csv')

# df = pd.read_csv('Z:\\Projects\\MSKCC_Data_Organization\\data\\REDCAP\\2023\\Axial_cancers_date_of_pathology_vs_date_of_exam.csv')

resultB = result.loc[result['y_true'] == 0]
resultM = result.loc[result['y_true'] == 1]

resultM = pd.merge(resultM, df, left_on='Exam', right_on='scanID')


# plt.scatter(resultM['delta_days'], resultM['y_pred'])


resultM['delta_days'].min()
resultM['delta_days'].max()

resultM = resultM.loc[resultM['delta_days'] > -90 ]

result = pd.concat([resultM, resultB])

# result.loc[result['Exam'] == 'RIA_19-093_000_02400_20190125_l']


# result.to_csv(PATH + 'results_noFineTune_res04_smartCropChest_final.csv')


# res_axial2 = res_axial2[['Exam', 'y_true', 'y_pred', 'exam_max_slice', 'subj', 'scanID', 'date_pathology', 'BIRADS', 'delta_days','pathology']]

result = result.drop_duplicates('Exam')

df_redcap = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/REDCAP/2023/REDCAP_EZ.csv')

df_redcap['Exam']

result_birads = result.copy()

result_birads['Exam'] = result_birads['Exam'].str[:-2]

df_redcap.loc[df_redcap['Exam'].isin(result_birads['Exam']), 'bi_rads_assessment_for_stu'].value_counts()

result.BIRADS


#%%  Localization



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



axial_annotations = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/DATA/Axial_segmented_annotations_lesions.csv')


result_segmented = result.loc[result['Exam'].isin(axial_annotations['Exam'])]

result_segmented.columns

localization_results = pd.merge(axial_annotations, result_segmented, left_on='Exam', right_on='Exam')

localization_results[['exam_max_slice','z1','z2','shape0']]

localization_results['Hit'] = 0

localization_results['Hit'] = (localization_results['exam_max_slice'] >= localization_results['z1']) * (localization_results['exam_max_slice'] <= localization_results['z2'])


localization_results['Hit']  = localization_results['Hit'].astype(int) 

N = len(localization_results)
n_hits = len(localization_results.loc[localization_results['Hit'] == 1])
n_miss = len(localization_results.loc[localization_results['Hit'] == 0])

print('\n #################################')
print('\n Localization Results')
print('\n')
print(f'{n_hits}/{N}')

sample_size = N
successes = n_hits
hypothesized_proportion = 0.5

z_score, p_value = z_test_proportion(sample_size, successes, hypothesized_proportion)

plt.figure(1, figsize=(4,5))
plt.bar(0,n_miss, color='dodgerblue')
plt.bar(1,n_hits, color='orange')
plt.xticks([0,1],['Miss','Hit'], fontsize=14)
plt.title(f'Localization (N={N})', fontsize=14)
# plt.text(x=-0.25,y=n_hits-2,s=f'z={round(z_score,2)}, p_val={round(p_value,3)}')
plt.text(0-0.15,n_miss//2, s=f'{round(n_miss*100./N,1)} %', fontsize=14, weight='bold')
plt.text(1-0.15,n_hits//2, s=f'{round(n_hits*100./N,1)} %', fontsize=14, weight='bold')
plt.grid()
plt.ylim([0,n_hits+5])
plt.xlabel(f'z={round(z_score,2)}, p_val={round(p_value,2)}')

if p_value < 0.05:
    plt.text(x=0,y=n_hits+n_hits*0.05,s='____________________________', weight='bold')
    plt.text(x=0.5,y=n_hits+n_hits*0.03,s='*', weight='bold', fontsize=15)
    plt.ylim([0,n_hits+n_hits*0.1])


if SAVE_RESULTS:
    plt.savefig(OUTPUT_PATH + f'{NAME}_localization.png', dpi=200)

#%% Method 1

if EXAM_LEVEL_RESULT:
    
    result.columns
    
    result = result[['Exam', 'y_pred', 'exam_max_slice','y_true']]
    
    result['side'] = result['Exam'].str[-1:]
    
    result['Exam'] = result['Exam'].str[:-2]
    
    
    result['scanID'] = result['Exam'] + '_' + result['side']
    
    
    exam_g = result.groupby('Exam')
    
    for g in exam_g:
        maxpred = np.max(g[1]['y_pred'])
        maxlabel = np.max(g[1]['y_true'])
        result.loc[result['Exam'] == g[0], 'y_pred'] = maxpred
        result.loc[result['Exam'] == g[0], 'y_true'] = maxlabel
        
    
    result = result.drop_duplicates('Exam')
        
#%% Method 2


# result = result.sort_values('y_pred', ascending=False)

# result['Exam']

# result_unique = result.drop_duplicates('Exam', keep='first')

# result_unique['y_true'].value_counts()


# result = result_unique
#%%


auc = roc_auc_score(result['y_true'], result['y_pred'])
fpr, tpr, thr = roc_curve(result['y_true'], result['y_pred'])

                                      
N = len(result)

BINS = 110

plt.figure(2,figsize=(10,5))

plt.subplot(1,2,1)
plt.hist(result.loc[result['y_true'] == 0, 'y_pred'], bins=BINS, color='g', label=f'benign (N={len(result.loc[result["y_true"] == 0])})')
plt.hist(result.loc[result['y_true'] == 1, 'y_pred'],  bins=BINS,  color='r', alpha=0.65, label=f'malignant (N={len(result.loc[result["y_true"] == 1])})')
plt.xlabel('Probability', fontsize=14)
plt.title(f'Axial Data (N = {N})', fontsize=14)
plt.legend()
plt.xlim([0.1,1.05])
# plt.yscale('symlog')
# plt.yscale('log')

plt.subplot(1,2,2)
plt.plot(1-fpr, tpr, label='All')
# plt.legend()

plt.plot([0,1],[1,0],'--')
plt.xlabel('Specificity', fontsize=14)
plt.ylabel('Sensitivity', fontsize=14)
plt.text(x=0.2,y=0.1,s=f'AUC-ROC = {round(auc,5)}', fontsize=14)
plt.title(f'Axial Data (N = {N})', fontsize=14)
plt.xlim([0,1])  
plt.ylim([0,1])


if SAVE_RESULTS:
    plt.savefig(OUTPUT_PATH + f'{NAME}.png', dpi=200)
    
    
    result.to_csv(OUTPUT_PATH + f'{NAME}.csv')



# # test_res = test_res.loc[test_res['scan'].isin(result['Exam'])]
# auc = roc_auc_score(test_res['y_true'], test_res['y_pred'])
# fpr2, tpr2, thr = roc_curve(test_res['y_true'], test_res['y_pred'])

# N = len(test_res)


# plt.subplot(2,2,3)
# plt.hist(test_res.loc[test_res['y_true'] == 0, 'y_pred'], bins=BINS,  color='g', label=f'benign (N={len(test_res.loc[test_res["y_true"] == 0])})')
# plt.hist(test_res.loc[test_res['y_true'] == 1, 'y_pred'], bins=BINS,   color='r', alpha=0.65, label=f'malignant (N={len(test_res.loc[test_res["y_true"] == 1])})')
# plt.xlabel('Probability', fontsize=14)
# plt.title(f'Axial Data (N = {N})', fontsize=14)
# plt.legend()


# plt.subplot(2,2,4)
# plt.plot(1-fpr, tpr, label='No finetune')
# plt.plot(1-fpr2, tpr2, label='Finetuned')
# # plt.plot(1-fpr_diagnostic, tpr_diagnostic, label='Diagnostic')
# # plt.plot(1-fpr_screen, tpr_screen, label='Screen')
# plt.legend(loc='center left')

# plt.plot([0,1],[1,0],'--')
# plt.xlabel('Specificity', fontsize=14)
# plt.ylabel('Sensitivity', fontsize=14)
# plt.text(x=0.2,y=0.1,s=f'AUC-ROC = {round(auc,2)}', fontsize=14)
# plt.title(f'Axial Data (N = {N})', fontsize=14)
# plt.xlim([0,1])
# plt.ylim([0,1])

# plt.tight_layout()


#%%


subj = list(set([x[:20] for x in result['Exam'].values]))
exams = list(set([x[:28] for x in result['Exam'].values]))

len(subj), len(exams)


