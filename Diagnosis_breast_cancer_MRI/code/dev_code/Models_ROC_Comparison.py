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

COMPARE_CONTRALATERAL_POPULATION = True

NAME = 'Models_Comparison'

PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/PAPER_SESSIONS/'

CNN_SMALL_DATA_01_PATH = PATH + 'SmallerData0.1_classifier_train9833_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/VAL_result.csv'
CNN_SMALL_DATA_05_PATH = PATH + 'Half_of_data_for_training_classifier_train39064_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/VAL_result.csv'
CNN_FULL_DATA_PATH = PATH + 'FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/validation_result.csv'
CNN_FULL_DATA_NoAUG_PATH = PATH + 'NODataAug_classifier_train52598_val5892_Clinical_depth6_filters42_L21e-05_batchsize8/validation_result.csv'
CNN_SMALL_DATA_01_NoAUG_PATH = PATH + 'SmallerData0.1_NoDataAug_classifier_train9833_val5892_Clinical_depth6_filters42_L21e-05_batchsize8/VAL_result.csv'
CNN_CONTRALATERAL_PATH = PATH + 'Contra_RandomSlices_DataAug_classifier_train36032_val4038_DataAug_Clinical_Contra_depth6_filters42_L21e-05_batchsize8/VAL_result.csv'

CNN_CONTRALAT_HALF1_PATH = PATH + 'Contra_LateInt_FreezeHalf1_classifier_train36031_val4038_DataAug_Clinical_Contra_depth6_filters42_L20_batchsize8/VAL_result.csv'
CNN_CONTRALAT_HALF2_PATH = PATH + 'Contra_LateInt_FreezeBothHalfs_classifier_train36031_val4038_DataAug_Clinical_Contra_depth6_filters42_L20_batchsize8/VAL_result.csv'

RESNET_IMAGENET = PATH + 'ResNet50_ImageNet_classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/validation_result.csv'
RESNET = PATH + 'ResNet50_RegularizedMore_classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/VAL_result.csv'

DEMOGRAPHICS_PATH = PATH + 'Demographics_Classifier/Demographics_classifier_validation_set_scanIDs.csv'

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

NAME = 'Data_augmentations_and_sizes'

FIGSIZE = (8,8)

plt.figure(figsize=(FIGSIZE))
plt.plot([0, 1], [0, 1], '--', color='gray')  # random predictions curve
#plt.xlim([-0.1, 1.1])
#plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)


fpr_test, tpr_test, _ = roc_curve(CNN_FULL_DATA['y_true'], CNN_FULL_DATA['y_pred'])
roc_auc = roc_auc_score(CNN_FULL_DATA['y_true'], CNN_FULL_DATA['y_pred'])
plt.plot(fpr_test, tpr_test, color='k', alpha=1, linewidth=2, label='CNN : AUC={}'.format(np.round(roc_auc,ROUND)))


fpr_test, tpr_test, _ = roc_curve(CNN_SMALL_DATA_05['y_true'], CNN_SMALL_DATA_05['y_pred'])
roc_auc = roc_auc_score(CNN_SMALL_DATA_05['y_true'], CNN_SMALL_DATA_05['y_pred'])
plt.plot(fpr_test, tpr_test, color='green',  alpha=1, linewidth=2, label='CNN 50% data : AUC={}'.format(np.round(roc_auc,ROUND)))


fpr_test, tpr_test, _ = roc_curve(CNN_FULL_DATA_NoAUG['y_true'], CNN_FULL_DATA_NoAUG['y_pred'])
roc_auc = roc_auc_score(CNN_FULL_DATA_NoAUG['y_true'], CNN_FULL_DATA_NoAUG['y_pred'])
plt.plot(fpr_test, tpr_test, color='k', linestyle='--', alpha=1, linewidth=2, label='CNN NoAug : AUC={}'.format(np.round(roc_auc,ROUND)))

fpr_test, tpr_test, _ = roc_curve(CNN_SMALL_DATA_01['y_true'], CNN_SMALL_DATA_01['y_pred'])
roc_auc = roc_auc_score(CNN_SMALL_DATA_01['y_true'], CNN_SMALL_DATA_01['y_pred'])
plt.plot(fpr_test, tpr_test, color='dodgerblue', alpha=1, linewidth=2, label='CNN 10% data : AUC={}'.format(np.round(roc_auc,ROUND)))

fpr_test, tpr_test, _ = roc_curve(CNN_SMALL_DATA_01_NoAUG['y_true'], CNN_SMALL_DATA_01_NoAUG['y_pred'])
roc_auc = roc_auc_score(CNN_SMALL_DATA_01_NoAUG['y_true'], CNN_SMALL_DATA_01_NoAUG['y_pred'])
plt.plot(fpr_test, tpr_test, color='dodgerblue',linestyle='--',  alpha=1, linewidth=2, label='CNN 10% data NoAug : AUC={}'.format(np.round(roc_auc,ROUND)))


fpr_test, tpr_test, _ = roc_curve(DEMOGRAPHICS['y_true'], DEMOGRAPHICS['y_pred'])
roc_auc = roc_auc_score(DEMOGRAPHICS['y_true'], DEMOGRAPHICS['y_pred'])
plt.plot(fpr_test, tpr_test, color='gray',  alpha=1, linewidth=2, label='Demographics : AUC={}'.format(np.round(roc_auc,ROUND)))

plt.legend(loc='lower right')

plt.savefig('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/PAPER_SESSIONS/{}.png'.format(NAME), dpi=400)

#%% All models:

ROUND = 3

FIGSIZE = (8,8)

if COMPARE_CONTRALATERAL_POPULATION:
    
    CNN_FULL_DATA = CNN_FULL_DATA.loc[CNN_FULL_DATA['scan'].isin(CNN_CONTRALAT_HALF2['scan'])]
    DEMOGRAPHICS = DEMOGRAPHICS.loc[DEMOGRAPHICS['scan'].isin(CNN_CONTRALAT_HALF2['scan'])]
    RESNET = RESNET.loc[RESNET['scan'].isin(CNN_CONTRALAT_HALF2['scan'])]
    RESNET_IMAGENET = RESNET_IMAGENET.loc[RESNET_IMAGENET['scan'].isin(CNN_CONTRALAT_HALF2['scan'])]
    CNN_FULL_DATA_NoAUG = CNN_FULL_DATA_NoAUG.loc[CNN_FULL_DATA_NoAUG['scan'].isin(CNN_CONTRALAT_HALF2['scan'])]
    


plt.figure(figsize=(FIGSIZE))
plt.plot([0, 1], [0, 1], '--', color='gray')  # random predictions curve
#plt.xlim([-0.1, 1.1])
#plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)


fpr_test, tpr_test, _ = roc_curve(CNN_CONTRALAT_HALF2['y_true'], CNN_CONTRALAT_HALF2['y_pred'])
roc_auc = roc_auc_score(CNN_CONTRALAT_HALF2['y_true'], CNN_CONTRALAT_HALF2['y_pred'])
plt.plot(fpr_test, tpr_test, color='dodgerblue', alpha=1, linewidth=2, label='CNN Contra Half2 : AUC={}'.format(np.round(roc_auc,ROUND)))


fpr_test, tpr_test, _ = roc_curve(CNN_CONTRALAT_HALF1['y_true'], CNN_CONTRALAT_HALF1['y_pred'])
roc_auc = roc_auc_score(CNN_CONTRALAT_HALF1['y_true'], CNN_CONTRALAT_HALF1['y_pred'])
plt.plot(fpr_test, tpr_test, color='blue', alpha=1, linewidth=2, label='CNN Contra Half1 : AUC={}'.format(np.round(roc_auc,ROUND)))


fpr_test, tpr_test, _ = roc_curve(CNN_FULL_DATA['y_true'], CNN_FULL_DATA['y_pred'])
roc_auc = roc_auc_score(CNN_FULL_DATA['y_true'], CNN_FULL_DATA['y_pred'])
plt.plot(fpr_test, tpr_test, color='k', alpha=1, linewidth=2, label='CNN : AUC={}'.format(np.round(roc_auc,ROUND)))



fpr_test, tpr_test, _ = roc_curve(RESNET_IMAGENET['y_true'], RESNET_IMAGENET['y_pred'])
roc_auc = roc_auc_score(RESNET_IMAGENET['y_true'], RESNET_IMAGENET['y_pred'])
plt.plot(fpr_test, tpr_test, color='Red', alpha=1, linewidth=2, label='ResNet50 Imagenet : AUC={}'.format(np.round(roc_auc,ROUND)))


fpr_test, tpr_test, _ = roc_curve(CNN_CONTRALATERAL['y_true'], CNN_CONTRALATERAL['y_pred'])
roc_auc = roc_auc_score(CNN_CONTRALATERAL['y_true'], CNN_CONTRALATERAL['y_pred'])
plt.plot(fpr_test, tpr_test, color='blue',linestyle='--',  alpha=1, linewidth=2, label='CNN + Contralat : AUC={}'.format(np.round(roc_auc,ROUND)))


fpr_test, tpr_test, _ = roc_curve(RESNET['y_true'], RESNET['y_pred'])
roc_auc = roc_auc_score(RESNET['y_true'], RESNET['y_pred'])
plt.plot(fpr_test, tpr_test, color='Darkorange', alpha=1, linewidth=2, label='ResNet50 : AUC={}'.format(np.round(roc_auc,ROUND)))

fpr_test, tpr_test, _ = roc_curve(CNN_FULL_DATA_NoAUG['y_true'], CNN_FULL_DATA_NoAUG['y_pred'])
roc_auc = roc_auc_score(CNN_FULL_DATA_NoAUG['y_true'], CNN_FULL_DATA_NoAUG['y_pred'])
plt.plot(fpr_test, tpr_test, color='k', linestyle='--', alpha=1, linewidth=2, label='CNN NoAug : AUC={}'.format(np.round(roc_auc,ROUND)))

fpr_test, tpr_test, _ = roc_curve(CNN_SMALL_DATA_01['y_true'], CNN_SMALL_DATA_01['y_pred'])
roc_auc = roc_auc_score(CNN_SMALL_DATA_01['y_true'], CNN_SMALL_DATA_01['y_pred'])
plt.plot(fpr_test, tpr_test, color='dodgerblue', alpha=1, linewidth=2, label='CNN 10% data : AUC={}'.format(np.round(roc_auc,ROUND)))

fpr_test, tpr_test, _ = roc_curve(CNN_SMALL_DATA_01_NoAUG['y_true'], CNN_SMALL_DATA_01_NoAUG['y_pred'])
roc_auc = roc_auc_score(CNN_SMALL_DATA_01_NoAUG['y_true'], CNN_SMALL_DATA_01_NoAUG['y_pred'])
plt.plot(fpr_test, tpr_test, color='dodgerblue',linestyle='--',  alpha=1, linewidth=2, label='CNN 10% data NoAug : AUC={}'.format(np.round(roc_auc,ROUND)))


fpr_test, tpr_test, _ = roc_curve(DEMOGRAPHICS['y_true'], DEMOGRAPHICS['y_pred'])
roc_auc = roc_auc_score(DEMOGRAPHICS['y_true'], DEMOGRAPHICS['y_pred'])
plt.plot(fpr_test, tpr_test, color='gray',  alpha=1, linewidth=2, label='Demographics : AUC={}'.format(np.round(roc_auc,ROUND)))

plt.legend(loc='lower right')

plt.savefig('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/PAPER_SESSIONS/All_models_{}.png'.format(NAME), dpi=400)



#%% Contralateral subset:

ROUND = 3
ALPHA = 0.75

if COMPARE_CONTRALATERAL_POPULATION:
    
    CNN_FULL_DATA = CNN_FULL_DATA.loc[CNN_FULL_DATA['scan'].isin(CNN_CONTRALAT_HALF2['scan'])]
    DEMOGRAPHICS = DEMOGRAPHICS.loc[DEMOGRAPHICS['scan'].isin(CNN_CONTRALAT_HALF2['scan'])]
    RESNET = RESNET.loc[RESNET['scan'].isin(CNN_CONTRALAT_HALF2['scan'])]
    RESNET_IMAGENET = RESNET_IMAGENET.loc[RESNET_IMAGENET['scan'].isin(CNN_CONTRALAT_HALF2['scan'])]
    CNN_FULL_DATA_NoAUG = CNN_FULL_DATA_NoAUG.loc[CNN_FULL_DATA_NoAUG['scan'].isin(CNN_CONTRALAT_HALF2['scan'])]
    

FIGSIZE = (8,8)

plt.figure(figsize=(FIGSIZE))
plt.plot([0, 1], [0, 1], '--', color='gray')  # random predictions curve
#plt.xlim([-0.1, 1.1])
#plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)



fpr_test, tpr_test, _ = roc_curve(CNN_CONTRALAT_HALF2['y_true'], CNN_CONTRALAT_HALF2['y_pred'])
roc_auc = roc_auc_score(CNN_CONTRALAT_HALF2['y_true'], CNN_CONTRALAT_HALF2['y_pred'])
plt.plot(fpr_test, tpr_test, color='blue', alpha=ALPHA, linewidth=2, label='CNN + Contra Finetuning Half2 : AUC={}'.format(np.round(roc_auc,ROUND)))



fpr_test, tpr_test, _ = roc_curve(CNN_CONTRALAT_HALF1['y_true'], CNN_CONTRALAT_HALF1['y_pred'])
roc_auc = roc_auc_score(CNN_CONTRALAT_HALF1['y_true'], CNN_CONTRALAT_HALF1['y_pred'])
plt.plot(fpr_test, tpr_test, color='blue', alpha=ALPHA, linewidth=2, label='CNN + Contra Finetuning Half2 : AUC={}'.format(np.round(roc_auc,ROUND)))


fpr_test, tpr_test, _ = roc_curve(CNN_FULL_DATA['y_true'], CNN_FULL_DATA['y_pred'])
roc_auc = roc_auc_score(CNN_FULL_DATA['y_true'], CNN_FULL_DATA['y_pred'])
plt.plot(fpr_test, tpr_test, color='k', alpha=ALPHA, linewidth=2, label='CNN : AUC={}'.format(np.round(roc_auc,ROUND)))


fpr_test, tpr_test, _ = roc_curve(CNN_CONTRALATERAL['y_true'], CNN_CONTRALATERAL['y_pred'])
roc_auc = roc_auc_score(CNN_CONTRALATERAL['y_true'], CNN_CONTRALATERAL['y_pred'])
plt.plot(fpr_test, tpr_test, color='green',  alpha=ALPHA, linewidth=2, label='CNN + Contralat channel : AUC={}'.format(np.round(roc_auc,ROUND)))


fpr_test, tpr_test, _ = roc_curve(RESNET_IMAGENET['y_true'], RESNET_IMAGENET['y_pred'])
roc_auc = roc_auc_score(RESNET_IMAGENET['y_true'], RESNET_IMAGENET['y_pred'])
plt.plot(fpr_test, tpr_test, color='Red', alpha=ALPHA, linewidth=2, label='ResNet50 Imagenet : AUC={}'.format(np.round(roc_auc,ROUND)))

fpr_test, tpr_test, _ = roc_curve(RESNET['y_true'], RESNET['y_pred'])
roc_auc = roc_auc_score(RESNET['y_true'], RESNET['y_pred'])
plt.plot(fpr_test, tpr_test, color='Darkorange', alpha=ALPHA, linewidth=2, label='ResNet50 : AUC={}'.format(np.round(roc_auc,ROUND)))


fpr_test, tpr_test, _ = roc_curve(DEMOGRAPHICS['y_true'], DEMOGRAPHICS['y_pred'])
roc_auc = roc_auc_score(DEMOGRAPHICS['y_true'], DEMOGRAPHICS['y_pred'])
plt.plot(fpr_test, tpr_test, color='gray',  alpha=ALPHA, linewidth=2, label='Demographics : AUC={}'.format(np.round(roc_auc,ROUND)))


plt.legend(loc='lower right')

plt.savefig('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/PAPER_SESSIONS/Contralateral_subset_{}.png'.format(NAME), dpi=400)
