#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:16:42 2023

@author: deeperthought
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

result = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/axial_results/results.csv')

REDCAP = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/REDCAP/2023/REDCAP_EZ.csv') 

REMOVE_EXAMS_WITH_MALIGNANT_DIAGNOSIS_BEFORE = 0

NO_BIRADS6 = 0

BINS = 100

if REMOVE_EXAMS_WITH_MALIGNANT_DIAGNOSIS_BEFORE:
    
    TITLE = 'Exams with diagnosis after exam'
    
    result = pd.merge(result, REDCAP[['mrn_id','Exam','bi_rads_assessment_for_stu']], on='Exam')
    
    pathology_sheet = REDCAP.loc[REDCAP['redcap_repeat_instrument'] == 'malignant_pathology_mod']
    
    pathology_sheet = pathology_sheet[['mrn_id','date_cancer_diagnos_mod']]
    
    result_B = result.loc[result['y_true'] == 0]
    
    result_M = pd.merge(result.loc[result['y_true'] == 1], pathology_sheet, on='mrn_id')
    
    result_M['Date_Exam'] = result_M['Exam'].str[-8:]
    result_M['Date_Diag'] = result_M['date_cancer_diagnos_mod'].apply(lambda x : x.split('/')[-1] + x.split('/')[-3].zfill(2) + x.split('/')[-2].zfill(2))
    
    result_M['delta'] = result_M['Date_Exam'].astype(int) - result_M['Date_Diag'].astype(int)
     
    result_M[['Date_Exam','Date_Diag','delta']]
    
    
#    result_M = result_M.loc[np.abs(result_M['delta']) < 300]
#    plt.scatter(result_M['y_pred'],result_M['delta'])
#    plt.hist(result_M['delta'])
#
#    result_M = result_M.loc[result_M['delta'] < 0]
#    plt.scatter(result_M['y_pred'],result_M['delta'])
#    plt.hist(result_M['delta'])

    
    print('Removing all exams with a malignant diagnosis preceding exam. Tumor might have been already removed.')
    result_M = result_M.loc[result_M['delta'] < 0]#30]
    #result_M = result_M.loc[result_M['delta'] < 0]
    
    result_M = result_M[['Exam', u'y_pred', u'y_true', u'max_slice', u'mrn_id','bi_rads_assessment_for_stu']]
    result_M = result_M.drop_duplicates()

    result = pd.concat([result_M, result_B])

elif NO_BIRADS6:
    TITLE = 'No BIRADS 6'
    result = pd.merge(result, REDCAP[['mrn_id','Exam','bi_rads_assessment_for_stu']], on='Exam')    
    result = result.loc[result['bi_rads_assessment_for_stu'] != '6']
else:
    TITLE = 'All Exams'



#%% Split results by breast
    
# Only problem. If malignant max slice is on wrong breast, I still cant tell what prediction is assigned to other breast. Maybe both 
# had values >0.5 ??

    
#%%



name = 'Axial'
BINS = 20

roc_auc_test_final = roc_auc_score( [int(x) for x in result['y_true'].values],result['y_pred'].values)
fpr_test, tpr_test, thresholds = roc_curve([int(x) for x in result['y_true'].values],result['y_pred'].values)
malignants_test = result.loc[result['y_true'] == 1, 'y_pred']
benigns_test = result.loc[result['y_true'] == 0, 'y_pred']         

print('{} : AUC-ROC = {}'.format(name, roc_auc_test_final))

plt.figure(figsize=(8,4))
plt.suptitle(TITLE)
plt.subplot(1,2,1)
plt.plot(fpr_test, tpr_test, label='ROC curve (area = %0.3f)' % roc_auc_test_final)
plt.plot([0, 1], [0, 1], 'k--') 
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")   

plt.subplot(1,2,2)                   
plt.hist(malignants_test.values, color='r', alpha=1, bins=BINS)
plt.hist(benigns_test.values, color='g', alpha=0.5, bins=BINS)
plt.legend(['Malignants (N={})'.format(len(malignants_test)), 'Benigns  (N={})'.format(len(benigns_test))])
plt.title('{} N = {}'.format(name, len(result)))
plt.yscale('log')




#plt.savefig(OUT +  NAME + '/{}_result_ROC.png'.format(name), dpi=200)          
#plt.close()