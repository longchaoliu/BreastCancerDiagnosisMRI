#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:17:33 2023

@author: deeperthought
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



partition = np.load('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/DATA/Axial_Data.npy', allow_pickle=True).item()

scans = list(set([x[:29] for x in partition['test']]))
scans.sort()

pdf_pages = PdfPages('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/DATA/axial_test_QA/' + 'QA_test_831onwards.pdf')

PATH = '/home/deeperthought/kirbyPRO/alignedNiiAxial-May2020-cropped-normed/'

#scans.index(scan)
#
#scans[831]

TOT = len(scans[831:])
n = 0
for scan in scans[831:]:
    n += 1
    print(n,TOT)
    t1post = nib.load(PATH + scan + '/T1_axial_02_01.nii.gz').get_data()
    slope1 = nib.load(PATH + scan + '/T1_axial_slope1.nii.gz').get_data()
    slope2 = nib.load(PATH + scan + '/T1_axial_slope2.nii.gz').get_data()
    
    fig = plt.figure()
    plt.suptitle(scan)
    plt.subplot(1,3,1)
    plt.imshow(t1post[:,:,t1post.shape[2]/2], cmap='gray');plt.xticks([]); plt.yticks([])
    plt.subplot(1,3,2)
    plt.imshow(slope1[:,:,t1post.shape[2]/2], cmap='gray');plt.xticks([]); plt.yticks([])    
    plt.subplot(1,3,3)
    plt.imshow(slope2[:,:,t1post.shape[2]/2], cmap='gray');plt.xticks([]); plt.yticks([])    
    pdf_pages.savefig(fig)
    plt.close()

pdf_pages.close()


#%%


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#
#
#b1 = np.random.normal(0.2,0.1,100)
#m1 = np.random.normal(0.4,0.1,100)
#b2 = np.random.normal(0.6,0.1,100)
#m2 = np.random.normal(0.8,0.1,100)
#
#plt.hist(b1, color='g')
#plt.hist(m1, color='r')
#plt.hist(b2, color='g')
#plt.hist(m2, color='r')
#
#a = pd.DataFrame(zip(b1,[0]*len(b1)))
#b = pd.DataFrame(zip(m1,[1]*len(m1)))
#c = pd.DataFrame(zip(b2,[0]*len(b2)))
#d = pd.DataFrame(zip(m2,[1]*len(m2)))
#
#result = pd.concat([a,b,c,d])
#
#result.columns = ['y_pred','y_true']
#
#result = result.loc[result['y_pred'] > 0]
#result = result.loc[result['y_pred'] < 1]

def draw_ROC(result, path=''):
    
    roc_auc_test_final = roc_auc_score( [int(x) for x in result['y_true'].values],result['y_pred'].values)
    fpr_test, tpr_test, thresholds = roc_curve([int(x) for x in result['y_true'].values],result['y_pred'].values)
    malignants_test = result.loc[result['y_true'] == 1, 'y_pred']
    benigns_test = result.loc[result['y_true'] == 0, 'y_pred']         
    
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(fpr_test, tpr_test, label='ROC curve (area = %0.2f)' % roc_auc_test_final)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")   
    
    plt.subplot(1,3,2)                   
    plt.hist(malignants_test.values, color='r', alpha=1, bins=100)
    plt.hist(benigns_test.values, color='g', alpha=0.5, bins=100)
    plt.yscale('log')
    plt.legend(['Malignants (N={})'.format(len(malignants_test)), 'Benigns  (N={})'.format(len(benigns_test))])
    plt.title('N = {}'.format(len(result)))

    plt.subplot(1,3,3)                   
    plt.hist(malignants_test.values, color='r', alpha=1, bins=100)
    plt.hist(benigns_test.values, color='g', alpha=0.5, bins=100)
    #plt.yscale('log')
    plt.legend(['Malignants (N={})'.format(len(malignants_test)), 'Benigns  (N={})'.format(len(benigns_test))])
    plt.title('N = {}'.format(len(result)))
    
    plt.tight_layout()
    
    if len(path) > 0:
        plt.savefig(path, dpi=400)

    return roc_auc_test_final

#
#result = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/PAPER_SESSIONS/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/test_result.csv')
#draw_ROC(result)
#
#result['breast_ID'] = result['scan'].str[:20] + result['scan'].str[-1]
#
#result = result.drop_duplicates('breast_ID', keep='first')
#draw_ROC(result)


result = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/test_result.csv')
result['Exam'] = result['scan'].str[:-2]
qa = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/DATA/axial_test_QA/QA_results.csv')

result_qa = result.loc[~result['Exam'].isin(qa['Scan'].values)]
result_bad = result.loc[result['Exam'].isin(qa['Scan'].values)]

draw_ROC(result_qa)

result_qa.to_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/test_result_QA.csv')

draw_ROC(result_qa, '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/test_result_QA.png')

draw_ROC(result_bad)
