#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 11:46:32 2023

@author: deeperthought
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from DeLong_AUC_test import get_DeLong_pValue

COMPARE_CONTRALATERAL_POPULATION = True

NAME = 'Models_Comparison'

CNN_SMALL_DATA_01_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/SmallerData0.1_classifier_train9833_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/VAL_result.csv'
CNN_SMALL_DATA_05_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/PAPER_SESSIONS/Half_of_data_for_training_classifier_train39064_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/VAL_result.csv'
CNN_FULL_DATA_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/validation_result.csv'
CNN_FULL_DATA_NoAUG_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/NODataAug_classifier_train52598_val5892_Clinical_depth6_filters42_L21e-05_batchsize8/validation_result.csv'
CNN_SMALL_DATA_01_NoAUG_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/SmallerData0.1_NoDataAug_classifier_train9833_val5892_Clinical_depth6_filters42_L21e-05_batchsize8/VAL_result.csv'
CNN_CONTRALATERAL_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/Contra_RandomSlices_DataAug_classifier_train36032_val4038_DataAug_Clinical_Contra_depth6_filters42_L21e-05_batchsize8/VAL_result.csv'

CNN_CONTRALAT_HALF1_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/Contra_LateInt_FreezeHalf1_classifier_train36031_val4038_DataAug_Clinical_Contra_depth6_filters42_L20_batchsize8/VAL_result.csv'
CNN_CONTRALAT_HALF2_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/Contra_LateInt_FreezeBothHalfs_classifier_train36031_val4038_DataAug_Clinical_Contra_depth6_filters42_L20_batchsize8/VAL_result.csv'

RESNET_IMAGENET = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/PAPER_SESSIONS/ResNet50_ImageNet_classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/validation_result.csv'
RESNET = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/PAPER_SESSIONS/ResNet50_RegularizedMore_classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/VAL_result.csv'

DEMOGRAPHICS_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/PAPER_SESSIONS/Demographics_Classifier/Demographics_classifier_validation_set_scanIDs.csv'

CNN_FULL_DATA = pd.read_csv(CNN_FULL_DATA_PATH)
CNN_SMALL_DATA_01 = pd.read_csv(CNN_SMALL_DATA_01_PATH)
CNN_SMALL_DATA_05 = pd.read_csv(CNN_SMALL_DATA_05_PATH)
CNN_FULL_DATA_NoAUG = pd.read_csv(CNN_FULL_DATA_NoAUG_PATH)
CNN_SMALL_DATA_01_NoAUG = pd.read_csv(CNN_SMALL_DATA_01_NoAUG_PATH)
CNN_CONTRALATERAL = pd.read_csv(CNN_CONTRALATERAL_PATH)
CNN_CONTRALAT_HALF1 = pd.read_csv(CNN_CONTRALAT_HALF1_PATH)
CNN_CONTRALAT_HALF2 = pd.read_csv(CNN_CONTRALAT_HALF2_PATH)
RESNET = pd.read_csv(RESNET)
RESNET_IMAGENET = pd.read_csv(RESNET_IMAGENET)

DEMOGRAPHICS = pd.read_csv(DEMOGRAPHICS_PATH)



def bootstrap_auc(DATAFRAME, N=1000):
    DATAFRAME = DATAFRAME.drop_duplicates()
    main_roc = roc_auc_score(DATAFRAME['y_true'], DATAFRAME['y_pred'])
    roc_list = []
    for _ in range(N):
        TMP = DATAFRAME.copy()
        TMP = TMP.sample(replace=True, n=len(TMP))
        
        TMP.sample(replace=True, n=1)
        
        roc_auc = roc_auc_score(TMP['y_true'], TMP['y_pred'])
        roc_list.append(roc_auc)
            
    return main_roc, main_roc - np.percentile(roc_list,2.5), np.percentile(roc_list,97.5) - main_roc



#%%  Make summary table:

from sklearn.metrics import f1_score, confusion_matrix, average_precision_score

models_list = [CNN_SMALL_DATA_01, CNN_SMALL_DATA_01_NoAUG,CNN_SMALL_DATA_05, CNN_FULL_DATA_NoAUG, 
               CNN_FULL_DATA, CNN_CONTRALAT_HALF1, CNN_CONTRALAT_HALF2,CNN_CONTRALATERAL,
               RESNET, RESNET_IMAGENET, DEMOGRAPHICS]

models_names = ['CNN_SMALL_DATA_01','CNN_SMALL_DATA_01_NoAUG', 'CNN_SMALL_DATA_05',  'CNN_FULL_DATA_NoAUG', 
               'CNN_FULL_DATA', 'CNN_CONTRALAT_HALF1', 'CNN_CONTRALAT_HALF2','CNN_CONTRALATERAL',
               'RESNET', 'RESNET_IMAGENET', 'DEMOGRAPHICS']


for row in zip(models_names, models_list):
    #print(row[0])
    auc = roc_auc_score(row[1]['y_true'], row[1]['y_pred'])

    f1 = f1_score(row[1]['y_true'], row[1]['y_pred'] > 0.5)
    
    avg_precision = average_precision_score(row[1]['y_true'], row[1]['y_pred'])

    #print('AUC: {}, F1: {}, Avg_precision: {}'.format(auc,f1,avg_precision))

    print(',',row[0], np.round(auc,3), np.round(f1,3), np.round(avg_precision,3),',')


f1_score(CNN_SMALL_DATA_01['y_true'], CNN_SMALL_DATA_01['y_pred'] > 0.5)
f1_score(CNN_SMALL_DATA_01_NoAUG['y_true'], CNN_SMALL_DATA_01_NoAUG['y_pred'] > 0.5)
f1_score(CNN_SMALL_DATA_05['y_true'], CNN_SMALL_DATA_05['y_pred'] > 0.5)
f1_score(CNN_FULL_DATA_NoAUG['y_true'], CNN_FULL_DATA_NoAUG['y_pred'] > 0.5)

f1_score(CNN_CONTRALATERAL['y_true'], CNN_CONTRALATERAL['y_pred'] > 0.5)
f1_score(CNN_CONTRALAT_HALF1['y_true'], CNN_CONTRALAT_HALF1['y_pred'] > 0.5)
f1_score(CNN_CONTRALAT_HALF2['y_true'], CNN_CONTRALAT_HALF2['y_pred'] > 0.5)

f1_score(RESNET['y_true'], RESNET['y_pred'] > 0.5)
f1_score(RESNET_IMAGENET['y_true'], RESNET_IMAGENET['y_pred'] > 0.5)
f1_score(DEMOGRAPHICS['y_true'], DEMOGRAPHICS['y_pred'] > 0.5)
f1_score(CNN_FULL_DATA['y_true'], CNN_FULL_DATA['y_pred'] > 0.5)

average_precision_score(CNN_SMALL_DATA_01['y_true'], CNN_SMALL_DATA_01['y_pred'])
average_precision_score(CNN_SMALL_DATA_01_NoAUG['y_true'], CNN_SMALL_DATA_01_NoAUG['y_pred'] )
average_precision_score(CNN_SMALL_DATA_05['y_true'], CNN_SMALL_DATA_05['y_pred'] )
average_precision_score(CNN_FULL_DATA_NoAUG['y_true'], CNN_FULL_DATA_NoAUG['y_pred'] )
average_precision_score(CNN_CONTRALATERAL['y_true'], CNN_CONTRALATERAL['y_pred'] )
average_precision_score(CNN_CONTRALAT_HALF1['y_true'], CNN_CONTRALAT_HALF1['y_pred'] )
average_precision_score(CNN_CONTRALAT_HALF2['y_true'], CNN_CONTRALAT_HALF2['y_pred'] )
average_precision_score(RESNET['y_true'], RESNET['y_pred'] )
average_precision_score(RESNET_IMAGENET['y_true'], RESNET_IMAGENET['y_pred'])
average_precision_score(DEMOGRAPHICS['y_true'], DEMOGRAPHICS['y_pred'])
average_precision_score(CNN_FULL_DATA['y_true'], CNN_FULL_DATA['y_pred'])


confusion_matrix(CNN_FULL_DATA['y_true'], CNN_FULL_DATA['y_pred'] > 0.5)
confusion_matrix(CNN_CONTRALAT_HALF2['y_true'], CNN_CONTRALAT_HALF2['y_pred'] > 0.5)

#%% Data sizes and augmentation

ROUND = 3

NAME = 'Data_sizes'


models = [ 'CNN_FULL_DATA',
'CNN_SMALL_DATA_05',
 'CNN_SMALL_DATA_01']


roc_auc_2, c1_2, c2_2 = bootstrap_auc(CNN_SMALL_DATA_01, N=100)
roc_auc_3, c1_3, c2_3 = bootstrap_auc(CNN_SMALL_DATA_05, N=100)
roc_auc_4, c1_4, c2_4 = bootstrap_auc(CNN_FULL_DATA, N=100)

plt.title('Data Sizes')
plt.bar([0,1,2],[roc_auc_2, roc_auc_3, roc_auc_4], yerr=[[c1_2, c1_3, c1_4],[c2_2,c2_3,c2_4]], width=0.5, capsize=7)
plt.xticks([0,1,2],['10% data', '50% data', '100% data'], rotation=25)
plt.ylabel('AUC-ROC val set (N=5,892)')
plt.ylim([0.5,1])
plt.grid( axis='y', alpha=0.75)
plt.tight_layout()

plt.savefig('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/PAPER_SESSIONS/DataSizes.png', dpi=400)

#%% Augmentation:

models = [ 'CNN_SMALL_DATA_01',
 'CNN_SMALL_DATA_01_NoAUG',
 'CNN_FULL_DATA',
 'CNN_FULL_DATA_NoAUG']



roc_auc, c1, c2 = bootstrap_auc(CNN_SMALL_DATA_01_NoAUG, N=100)
roc_auc_2, c1_2, c2_2 = bootstrap_auc(CNN_SMALL_DATA_01, N=100)
roc_auc_3, c1_3, c2_3 = bootstrap_auc(CNN_FULL_DATA_NoAUG, N=100)
roc_auc_4, c1_4, c2_4 = bootstrap_auc(CNN_FULL_DATA, N=100)

plt.title('Data Augmentation')
plt.bar([0,1,2,3],[roc_auc, roc_auc_2, roc_auc_3, roc_auc_4], yerr=[[c1, c1_2, c1_3, c1_4],[c2,c2_2,c2_3,c2_4]], width=0.5, capsize=7)
plt.xticks([0,1,2,3],['10% data\n- Aug', '10% data\n+ Aug', '100% data\n- Aug', '100% data\n+ Aug'], rotation=25)
plt.ylabel('AUC-ROC val set (N=5,892)')
plt.ylim([0.5,1])
plt.grid( axis='y', alpha=0.75)
plt.tight_layout()


plt.savefig('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/PAPER_SESSIONS/Augmentation.png', dpi=400)




#%% architectures:

ROUND = 3
ALPHA = 0.75


models = [ 'CNN_FULL_DATA',
 'CNN_CONTRALAT_HALF1',
 'RESNET',
 'DEMOGRAPHICS']




roc_auc, c1, c2 = bootstrap_auc(RESNET, N=100)
roc_auc_2, c1_2, c2_2 = bootstrap_auc(RESNET_IMAGENET, N=100)
roc_auc_3, c1_3, c2_3 = bootstrap_auc(CNN_FULL_DATA, N=100)
roc_auc_4, c1_4, c2_4 = bootstrap_auc(DEMOGRAPHICS, N=100)

plt.title('Model architectures')
plt.bar([0,1,2,3],[roc_auc, roc_auc_2, roc_auc_3, roc_auc_4], yerr=[[c1, c1_2, c1_3, c1_4],[c2,c2_2,c2_3,c2_4]], width=0.5, capsize=7)
plt.xticks([0,1,2,3],['ResNet50', 'ResNet50\nImageNet', 'CNN', 'Demographics'], rotation=25)
plt.ylabel('AUC-ROC val set (N=5,892)')
plt.ylim([0.5,1])
plt.grid( axis='y', alpha=0.75)
plt.tight_layout()


plt.savefig('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/PAPER_SESSIONS/Model_architectures.png', dpi=400)




#%% pre-training:

ROUND = 3
ALPHA = 0.75


models = [  'RESNET_IMAGENET',
 'RESNET']


roc_auc_2, c1_2, c2_2 = bootstrap_auc(RESNET_IMAGENET, N=100)
roc_auc, c1, c2 = bootstrap_auc(RESNET, N=100)

plt.figure(figsize=(4,4))
plt.title('Pretraining')
plt.bar([0,1],[roc_auc, roc_auc_2], yerr=[[c1, c1_2],[c2,c2_2]], width=0.5, capsize=7)
plt.xticks([0,1],['ResNet50', 'ResNet50\nImageNet'], rotation=25)
plt.ylabel('AUC-ROC val set (N=5,892)')
plt.ylim([0.5,1])
plt.grid( axis='y', alpha=0.75)
plt.tight_layout()


plt.savefig('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/PAPER_SESSIONS/Resnet_Pretraining.png', dpi=400)


#%% Contralateral:

ROUND = 3
ALPHA = 0.75


models = [  'CNN_CONTRALAT_HALF1',
 'CNN_FULL_DATA']

CNN_FULL_DATA = CNN_FULL_DATA.loc[CNN_FULL_DATA['scan'].isin(CNN_CONTRALAT_HALF1['scan'].values)]


roc_auc_contra, c1_contra, c2_contra = bootstrap_auc(CNN_CONTRALAT_HALF1, N=100)
roc_auc, c1, c2 = bootstrap_auc(CNN_FULL_DATA, N=100)


plt.figure(figsize=(4,4))
plt.title('Adding contralateral')
plt.bar([0,1],[roc_auc, roc_auc_contra], yerr=[[c1, c1_contra],[c2,c2_contra]], width=0.5, capsize=7)
plt.xticks([0,1],['CNN', 'CNN\nContralateral'], rotation=25)
plt.ylabel('AUC-ROC val set (N=4,038)')
plt.ylim([0.5,1])
plt.grid( axis='y', alpha=0.75)
plt.tight_layout()


plt.savefig('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/PAPER_SESSIONS/Contralateral.png', dpi=400)
