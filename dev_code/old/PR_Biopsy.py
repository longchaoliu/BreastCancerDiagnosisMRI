#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 13:35:57 2022

@author: deeperthought
"""

""" RISK PREDICTION FIGURE"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.patches as patches
import math

RESULTS_TABLE_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/Contra_LateInt_FreezeHalf1_classifier_train36031_val4038_DataAug_Clinical_Contra_depth6_filters42_L20_batchsize8/test_result.csv'

CONCATENATE = 0

EQUALIZE_HISTOGRAM = 0

USE_ONLY_BIRADS45 = 1

USE_INDIVIDUAL_BREASTS = 0

RESULT_NAME = RESULTS_TABLE_PATH.split('/')[-1].split('.csv')[0] 

PLOT_BIRADS_DISTRIBUTION = 0

save_fig = 1

MASTER = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Data_Partitions.csv')
RISK = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Data_ExamHistory_Labels.csv')

RESULT_OUTPUT = '/'.join(RESULTS_TABLE_PATH.split('/')[:-1])

#%%

if CONCATENATE:
    GLOBAL_RESULTS_TEST1 = pd.read_csv(RESULTS_TABLE_PATH.replace('1-fold.csv', '1-fold.csv'))
    GLOBAL_RESULTS_TEST2 = pd.read_csv(RESULTS_TABLE_PATH.replace('1-fold.csv', '2-fold.csv'))
    GLOBAL_RESULTS_TEST3 = pd.read_csv(RESULTS_TABLE_PATH.replace('1-fold.csv', '3-fold.csv'))
    GLOBAL_RESULTS_TEST4 = pd.read_csv(RESULTS_TABLE_PATH.replace('1-fold.csv', '4-fold.csv'))    
    GLOBAL_RESULTS_TEST5 = pd.read_csv(RESULTS_TABLE_PATH.replace('1-fold.csv', '5-fold.csv'))    
    results = pd.concat([GLOBAL_RESULTS_TEST1, GLOBAL_RESULTS_TEST2, GLOBAL_RESULTS_TEST3, GLOBAL_RESULTS_TEST4, GLOBAL_RESULTS_TEST5])
else:
    results = pd.read_csv(RESULTS_TABLE_PATH)
    if 'scanID' not in results.columns:
        results.columns = [u'scanID', u'y_pred', u'y_true', u'max_slice', u'GT_slice']



if EQUALIZE_HISTOGRAM:
    from skimage import exposure
    results['y_pred'] = exposure.equalize_hist(results['y_pred'].values,nbins=5000)
    RESULT_NAME = RESULT_NAME + '_HistEQ'
# Keep images from individual subjects

#results['breast'] = results['scanID'].str[:20] + '_' + results['scanID'].str[-1]
#breasts = results['breast'].unique()
#results = results.sample(frac=1).drop_duplicates('breast').reset_index(drop=True)
#

noisy_images = ['MSKCC_16-328_1_04596_20110506_r']

results = results.loc[~results['scanID'].isin(noisy_images)]

wrong_side = ['MSKCC_16-328_1_08288_20110610_l']
results = results.loc[~results['scanID'].isin(wrong_side)]


#%%
results.columns

#results_M = results.loc[results['y_true'] == 1]
#results_M['BIRADS'].value_counts()
#if USE_ONLY_BIRADS45:
#    results = results.loc[results['BIRADS'].isin(['4','5'])]
#    RESULT_NAME = RESULT_NAME + '_BIRADS45'
#results = results.loc[results['BIRADS'].isin(['1','2','NR'])]

#%%

#print('removing BIRADS 6...')

#bad_quality = ['MSKCC_16-328_1_05404_20130624_l',
#'MSKCC_16-328_1_11736_20110427_l',
# ',
#'MSKCC_16-328_1_00393_20120508_r',
#'MSKCC_16-328_1_02221_20060509_l',
#'MSKCC_16-328_1_02132_20101004_l',
#'MSKCC_16-328_1_00403_20070227_l']
#
#results = results.loc[~results['scanID'].isin(bad_quality)]


results = results.drop_duplicates('scanID')

results = results.sort_values('y_pred')

#
#results.reset_index(drop=True, inplace=True)
#
#

#test_partition = pd.read_csv('/home/deeperthought/Projects/DGNS/Risk_Prediction/Sessions/FINAL/training_dropout0.5_classWeight1.0_Clinical/risk/naturalPrevalence1_16_LR0.0005_AsymmetricFocalLoss_50Epochs/10_Folds/results_sheet.csv')
#
#results = results.loc[results['scanID'].isin(test_partition['scanID'])]
#
#len(results.loc[results['y_true'] == 0])
#cancer_N = len(results.loc[results['y_true'] == 1])
#healthy_N = int((cancer_N/2.)*100)
#
#healthy = results.loc[results['y_true'] == 0, 'scanID']
#cancer = results.loc[results['y_true'] == 1, 'scanID']
#
#
#healthy_scans = np.random.choice(healthy, size = healthy_N, replace=False)
#
#results = pd.concat([results.loc[results['scanID'].isin(healthy_scans)],results.loc[results['y_true'] == 1]])
#
#results = results.sort_values('y_pred')
if USE_INDIVIDUAL_BREASTS:
    results['breast_ID'] = results['scanID'].str[:20] + '_' + results['scanID'].str[-1]
    unique_breasts = results['breast_ID'].unique()
    
    converts = results.loc[results['y_true'] == 1]
    healthy = results.loc[results['y_true'] == 0]
    
    healthy = healthy.sample(frac=1).drop_duplicates('breast_ID')
    converts = converts.sample(frac=1).drop_duplicates('breast_ID')
    
    results = pd.concat([converts, healthy])
    
    RESULT_NAME = RESULT_NAME + '_IndividualBreasts'

results.reset_index(drop=True, inplace=True)



#%%

if 'BIRADS' not in results.columns:
    

    results = pd.merge(results, MASTER[['Scan_ID','BIRADS','Image_QA']], left_on='scanID', right_on='Scan_ID')
    
    for i, row in results.loc[results['BIRADS'].str.contains(',')].iterrows():
        x = row['BIRADS']
        scanid = row['scanID']
        BIRADS = np.array(x.split(','), dtype='int').max()
        results.loc[results['scanID'] == scanid, 'BIRADS'] = BIRADS
        
results['BIRADS'].value_counts()
#
#results = results.loc[results['BIRADS'] != '4.6']
#results = results.loc[results['BIRADS'] != 'MISSING']
#
if USE_ONLY_BIRADS45:
    results = results.loc[results['BIRADS'].isin(['4','5'])]
    RESULT_NAME = RESULT_NAME + '_BIRADS45'
results.reset_index(drop=True, inplace=True)

results.loc[results['BIRADS'] == 1, 'BIRADS'] = '1'
results.loc[results['BIRADS'] == 2, 'BIRADS'] = '2'
results.loc[results['BIRADS'] == 3, 'BIRADS'] = '3'
results.loc[results['BIRADS'] == 4, 'BIRADS'] = '4'
results.loc[results['BIRADS'] == 5, 'BIRADS'] = '5'
results.loc[results['BIRADS'] == 6, 'BIRADS'] = '6'


if PLOT_BIRADS_DISTRIBUTION:
    plt.figure(figsize=(8,6))
    plt.title('BIRADS')
    BINS = 50
#    plt.hist(results.loc[results['BIRADS'] == '2', 'y_pred'], color='green', alpha=1, label='2', bins=BINS)
#    plt.hist(results.loc[results['BIRADS'] == '1', 'y_pred'], color='lightgreen',alpha=1, label='1', bins=BINS)
#    plt.hist(results.loc[results['BIRADS'] == '3', 'y_pred'], color='orange', alpha=1, label='3', bins=BINS)
#    plt.hist(results.loc[results['BIRADS'] == '4', 'y_pred'], color='red', alpha=0.8, label='4', bins=BINS)
#    plt.hist(results.loc[results['BIRADS'] == '6', 'y_pred'], color='black', alpha=1, label='6', bins=BINS)
#    plt.hist(results.loc[results['BIRADS'] == '5', 'y_pred'], color='darkred', alpha=1, label='5', bins=BINS)

    import seaborn as sns


    sns.distplot(results.loc[results['BIRADS'] == '1', 'y_pred'], kde=True, bins=20, hist=False, color='lightgreen', label='1')
    sns.distplot(results.loc[results['BIRADS'] == '2', 'y_pred'], kde=True, bins=20, hist=False, color='green', label='2')
    sns.distplot(results.loc[results['BIRADS'] == '3', 'y_pred'], kde=True, bins=20, hist=False, color='orange', label='3')
    sns.distplot(results.loc[results['BIRADS'] == '4', 'y_pred'], kde=True, bins=20, hist=False, color='red', label='4')
    sns.distplot(results.loc[results['BIRADS'] == '5', 'y_pred'], kde=True, bins=20, hist=False, color='darkred', label='5')
    sns.distplot(results.loc[results['BIRADS'] == '6', 'y_pred'], kde=True, bins=20, hist=False, color='black', label='6')
    
    plt.yticks([])
    plt.xlabel('Predicted probability')
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(RESULT_OUTPUT +  '/{}_results_BIRADS_distribution.png'.format(RESULT_NAME), dpi=400)   
    plt.close()
    
#results.loc[results['BIRADS'] == '4']

#%%
y_pred_full = results['y_pred']
y_true_full = results['y_true']

x_lim_min = np.min(y_pred_full) 
x_lim_max = np.max(y_pred_full)

####################### Histograms ###############################
radiologist_PPV  = 0.25    # At a prevalence of 
radiologist_NPV =  1.
myBins = np.linspace(0,1,num=5000)

M = [ y_pred_full[x[0]] for x in np.argwhere(np.array(y_true_full) > 0.0)]
B = [ y_pred_full[x[0]] for x in np.argwhere(np.array(y_true_full) == 0.0)]

#B = np.random.choice(B, size=int(len(M)/0.028), replace=True)


M.sort()
B.sort()
M_counts, M_bins = np.histogram(M, bins=myBins)  # Mh[0] = counts, Mh[1] = bins
B_counts, B_bins = np.histogram(B, bins=myBins)

######################### NPV ###################################
recall_healthy = []
precision_healthy = []
Benign = 0
Malignant = 0
total_healthy = np.sum(B_counts)

for i in np.arange(0,len(B_counts),1):
  threshold  = B_bins[i] # Shohuld be == B_bins[i]
  #print(threshold)

  Benign += B_counts[i] # aggregated benigns as I lower the threshold
  Malignant += M_counts[i] # aggregated malignants as I lower the threshold
  
  TP = Benign
  FP = Malignant
  FN = total_healthy - Benign
  
  if TP+FP > 0:
      precision_healthy.append(float(TP)/(TP+FP))

  else:
      precision_healthy.append(np.nan)      

  if TP+FN > 0:      
      recall_healthy.append(float(TP)/(TP+FN))  
  else:
      recall_healthy.append(np.nan)      
  
#plt.plot(precision_healthy)

#---------- Radiologist NPV -------------#
precision_healthy = [0 if math.isnan(x) else x for x in precision_healthy]

X_indx = []
for i in range(len(precision_healthy)):
    if precision_healthy[i] >= radiologist_NPV:
        X_indx.append(i)
if len(X_indx) > 0:
    X_indx = X_indx[-1]
    NPV_Meeting_point = precision_healthy[X_indx]
    NPV_Number_cases_meeting_point  = np.sum(B_counts[:X_indx])
else:
    NPV_Meeting_point = 1
    NPV_Number_cases_meeting_point  = 0
print('{} cases ({}%) can be skipped with NPV of {}.'.format(NPV_Number_cases_meeting_point, round(NPV_Number_cases_meeting_point*100./(len(M) + len(B)),2 ) , NPV_Meeting_point))


NPV_results = results[0:NPV_Number_cases_meeting_point]
NPV_results = NPV_results.loc[NPV_results['BIRADS'] != 'NR']
NPV_results['BIRADS'].value_counts()
#skipped_biopsies = len(NPV_results.loc[NPV_results['BIRADS'].astype(int) > 3])

#print('Skipping {} biopsies'.format(skipped_biopsies))

##---------- PERFECT NPV = 1-------------#
#X_indx = []
#for i in range(len(precision_healthy)):
#    if precision_healthy[i] == 1.:#== 1.:
#        X_indx.append(i)
#        
#if len(X_indx) > 0:
#    X_indx = X_indx[-1]
#    NPV_Meeting_point = precision_healthy[X_indx]
#    NPV_Number_cases_meeting_point  = np.sum(B_counts[:X_indx])
#else:
#    NPV_Meeting_point = 1
#    NPV_Number_cases_meeting_point  = 0
#    
#print('{} cases ({}%) can be skipped with NPV of {}.'.format(NPV_Number_cases_meeting_point, round(NPV_Number_cases_meeting_point*100./(len(B)),2 ) , NPV_Meeting_point))

predicted_risk_healthy_threshold = B_bins[X_indx]


NPV_results = results[0:NPV_Number_cases_meeting_point]
NPV_results = NPV_results.loc[NPV_results['BIRADS'] != 'NR']
NPV_results['BIRADS'].value_counts()
#skipped_biopsies = len(NPV_results.loc[NPV_results['BIRADS'].astype(int) > 3])

#print('Skipping {} biopsies'.format(skipped_biopsies))
#NPV_TEXT = '{} cases ({}%) \n can be skipped \n NPV={}\nSkipping {} biopsies'.format(NPV_Number_cases_meeting_point, round(NPV_Number_cases_meeting_point*100./(len(B)),2 ) , NPV_Meeting_point, skipped_biopsies)

fraction_healthy_cases_recovered = round(NPV_Number_cases_meeting_point*1./len(B),2 ) 


######################### PPV ###################################
recall_cancer = []
precision_cancer = []
Benign = 0
Malignant = 0
Biopsied = 0
TP = []
FN = []
total_cancers = np.sum(M_counts)
total_healthy = np.sum(B_counts)

for i in np.arange(len(M_counts)-1,-1,-1):
  threshold  = M_bins[i] # Shohuld be == B_bins[i]
  #print(threshold)

  Benign += B_counts[i] # aggregated benigns as I lower the threshold
  Malignant += M_counts[i] # aggregated malignants as I lower the threshold
  
  TP = Malignant
  FP = Benign
  FN = total_cancers - Malignant
  if TP+FP > 0:
      precision_cancer.append(float(TP)/(TP+FP))

  else:
      precision_cancer.append(np.nan)      

  if TP+FN > 0:      
      recall_cancer.append(float(TP)/(TP+FN))  
  else:
      recall_cancer.append(np.nan)      
  
#len(precision_cancer)
#plt.plot(precision_cancer)
#    


X_indx = []
while len(X_indx) == 0:

    X_indx = [precision_cancer.index(x) for x in precision_cancer if np.abs(x - radiologist_PPV) < 1][0]
    PPV_Meeting_point = precision_cancer[X_indx]
    PPV_Number_cases_meeting_point  = np.sum(M_counts[-X_indx:])
    
    precision_cancer = [0 if math.isnan(x) else x for x in precision_cancer]
    
    X_indx = []
    for i in range(len(precision_cancer)):
        if precision_cancer[i] >= radiologist_PPV:
            X_indx.append(i)
            
    if len(X_indx) == 0:
        radiologist_PPV = radiologist_PPV - 0.02
        

X_indx = X_indx[-1]

predicted_risk_threshold = M_bins[-X_indx]

PPV_Meeting_point = precision_cancer[X_indx]
PPV_Number_cases_meeting_point  = np.sum(M_counts[-X_indx:])

print('{} cases ({}%) can be early retrieved with PPV of {}.'.format(PPV_Number_cases_meeting_point, round(PPV_Number_cases_meeting_point*100./(len(M)),2 ) , PPV_Meeting_point))

PPV_results = results.loc[results['y_pred'] > predicted_risk_threshold]
#PPV_results = PPV_results.loc[PPV_results['BIRADS'] != 'NR']
PPV_results['BIRADS'].value_counts()
#added_biopsies = len(PPV_results.loc[PPV_results['BIRADS'].astype(int) < 4])

#print('Adding {} biopsies'.format(added_biopsies))    

fraction_cases_recovered = round(PPV_Number_cases_meeting_point*100./len(M),2 ) 

#PPV_TEXT = '{} cancers ({}%)\ncan be caugth\nPPV={}\nAdding {} biopsies'.format(PPV_Number_cases_meeting_point, round(PPV_Number_cases_meeting_point*100./(len(M)),2 ) , round(PPV_Meeting_point,2), added_biopsies)



#%% ####################### plot ######################################  

FIGSIZE = (12,11)
  
FONTSIZE = 14
xlimit = max(M[-1],B[-1])
fig, [[ax0, ax1], [ax2, ax3]] = plt.subplots(nrows=2, ncols=2,figsize=FIGSIZE)


roc_auc_test_final = roc_auc_score(y_true_full, y_pred_full)
fpr_test, tpr_test, _ = roc_curve(y_true_full, y_pred_full)


ax0.plot(fpr_test, tpr_test, color='blue', alpha=1, linewidth=2)
ax0.text(x=0.6,y=0.4,s='AUC = ' + r"$\bf{" + str(round(roc_auc_test_final,2)) + "}$")
ax0.plot([0, 1], [0, 1], 'r--')  # random predictions curve
ax0.set_xlim([-0.1, 1.1])
ax0.set_ylim([-0.1, 1.1])
ax0.set_xlabel('False Positive Rate', fontsize=FONTSIZE)
ax0.set_ylabel('True Positive Rate', fontsize=FONTSIZE)


weights = np.ones_like(M)/float(len(M))
ax1.hist(M,100, alpha=0.9, color='red')
ax1.set_title('MRI classification')
ax1.set_ylabel('# MRI exams', fontsize=FONTSIZE)
ax1.set_xlabel('Predicted Malignancy', fontsize=FONTSIZE)
weights = np.ones_like(B)/float(len(B))
ax1.hist(B,100, alpha=0.5, color='green')
ax1.set_xlim([x_lim_min - 0.05,x_lim_max + 0.05])
ax1.legend(['{} cancers'.format(len(M)), '{} healthy'.format(len(B))])
if not USE_ONLY_BIRADS45:
    ax1.set_yscale('log')
#ax1.set_xscale('logit')
 # NPV
 
ax2.plot(recall_healthy,precision_healthy, 'k-')

ax22 = ax2.twinx()
ax22.plot(np.arange(0,1,1./len(B_counts)), np.cumsum(B_counts), 'b-')

ax2.axhline(radiologist_NPV, color='g', linestyle='dashed')
plt.vlines(x=NPV_Meeting_point,ymin=0,ymax=NPV_Number_cases_meeting_point, color='r', linestyle='dashed')
ax2.set_title('Skipped readings')
ax2.set_ylabel('Negative Predictive Value', color='k', fontsize=FONTSIZE)
ax22.set_ylabel('# Avoided Readings', color='b', fontsize=FONTSIZE)
ax2.set_xlim([x_lim_min,x_lim_max])
ax2.spines['right'].set_color('blue')

ax22.yaxis.label.set_color('blue')
ax22.tick_params(axis='y', colors='blue')

ax2.yaxis.grid(True)
ax2.xaxis.grid(True)
#ax2.set_xlabel('Predicted Risk\n' + NPV_TEXT, fontsize=FONTSIZE)
ax2.set_ylim([0.975,1])


# PPV 

#ax3.plot(myX2[:-1],np.arange(0,1,1./(len(myX2)-1)), 'k-')

ax3.plot(recall_cancer,precision_cancer, 'k-')

ax3.axhline(radiologist_PPV, color='g', linestyle='dashed')
ax33 = ax3.twinx()
#plt.vlines(x=PPV_Meeting_point,ymin=0,ymax=PPV_Number_cases_meeting_point, colors='r', linestyles='dashed')
ax33.plot(np.arange(0,1,1./len(M_counts)),np.cumsum(M_counts), 'b-')
ax3.yaxis.grid(True)
ax3.xaxis.grid(True)
ax3.set_title('Biopsy today')
#ax3.set_xlabel('Predicted Risk\n' + PPV_TEXT, fontsize=FONTSIZE)
ax3.set_ylabel('Positive Predictive Value', color='k', fontsize=FONTSIZE)
ax3.set_ylabel('# Early Detections', color='b', fontsize=FONTSIZE)
ax3.set_xlim([x_lim_min,x_lim_max])
#ax3.set_xlim([0.98,x_lim_max])

ax3.spines['right'].set_color('blue')
ax33.yaxis.label.set_color('blue')
ax33.tick_params(axis='y', colors='blue')

plt.tight_layout()


if save_fig:
  plt.savefig(RESULT_OUTPUT + '/Linear_Scale_Pure_Precision_no_recall_Real_{}.png'.format(RESULT_NAME), dpi=300)  
   
#%% PR CUrve

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(results['y_true'], results['y_pred'])
AUC_PR = auc(recall, precision)
average_precision = average_precision_score(results['y_true'], results['y_pred'])


#create precision recall curve
fig, (ax0, ax1,ax2) = plt.subplots(1,3, figsize=(18,6))


########## HISTOGRAM ##########


weights = np.ones_like(M)/float(len(M))
ax0.hist(M,100, alpha=0.9, color='red')
ax0.set_title('MRI classification')
ax0.set_ylabel('# MRI exams', fontsize=20)
ax0.set_xlabel('Predicted Malignancy', fontsize=20)
weights = np.ones_like(B)/float(len(B))
ax0.hist(B,100, alpha=0.5, color='green')
ax0.set_xlim([x_lim_min - 0.05,x_lim_max + 0.05])
ax0.grid()
ax0.set_yscale('log')
ax0.legend(['Cancer','Benign'])

#width=0.05
#height=width*100
#rect = patches.Ellipse((predicted_risk_threshold, 120), width, height, linewidth=2, edgecolor='k', facecolor='k', alpha=1)
#ax0.add_patch(rect)
#ax0.plot([predicted_risk_threshold, predicted_risk_threshold], [0, 120], linestyle='--', color='k', label='No Skill')
#
#
#width=0.05
#height=width*100
#rect = patches.Rectangle((predicted_risk_healthy_threshold-width/2., 120-height/2.),  width, height, linewidth=2, edgecolor='k', facecolor='k', alpha=1)
#ax0.add_patch(rect)
#ax0.plot([predicted_risk_healthy_threshold, predicted_risk_healthy_threshold], [0, 120], linestyle='--', color='k', label='No Skill')

########## PPV PLOT ###############

ax2.plot(recall, precision, color='purple')
#add ax2is labels to plot
ax2.set_title('Predicting Cancer')
ax2.set_ylabel('PPV', fontsize=20)
ax2.set_xlabel('Fraction of biopsy referrals', fontsize=20)
ax2.plot([0, 1], [radiologist_PPV, radiologist_PPV], linestyle='-', label='Radiologist PPV')
#no_skill = np.sum(results['y_true']==1) / float(len(results))


#side = 0.025
#rect = patches.Circle((fraction_cases_recovered/100. - side/4, PPV_Meeting_point - side/2.),side, linewidth=2, edgecolor='k', facecolor='k', alpha=1)
#ax2.add_patch(rect)


#ax2.plot([fraction_cases_recovered/100., fraction_cases_recovered/100.], [0, PPV_Meeting_point], linestyle='--', color='k', label='No Skill')
#ax2.legend(['AUC-PR: ' + str(np.round(average_precision,2)),'Radiologist PPV', 'Operating point'])

#ax2.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

ax2.grid()



########## NPV PLOT ###############

results['y_true_neg'] = 1 - results['y_true']
results['y_pred_neg'] = 1 - results['y_pred']
precision, recall, thresholds = precision_recall_curve(results['y_true_neg'], results['y_pred_neg'])
AUC_PR = auc(recall, precision)
average_precision = average_precision_score(results['y_true_neg'], results['y_pred_neg'])


ax1.plot(recall, precision, color='purple')
#add ax1is labels to plot
ax1.set_title('Predicting healthy scans')
ax1.set_ylabel('NPV', fontsize=20)
ax1.set_xlabel('Fraction of omitted readings', fontsize=20)
no_skill = np.sum(results['y_true']==0) / float(len(results))
ax1.plot([0, 1], [radiologist_NPV, radiologist_NPV], linestyle='-', label='Radiologist NPV')

#side1=0.05
#side2=0.002
#rect = patches.Rectangle((fraction_healthy_cases_recovered-side1/2., NPV_Meeting_point-side2/2), side1,side2, linewidth=2, edgecolor='k', facecolor='k', alpha=1)
#ax1.add_patch(rect)


#ax1.plot([fraction_healthy_cases_recovered, fraction_healthy_cases_recovered], [no_skill, NPV_Meeting_point], linestyle='--', color='k', label='No Skill')
#ax1.legend(['AUC-PR: ' + str(np.round(average_precision,2)),'Radiologist NPV',  'Operating point'])

#ax1.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

ax1.grid()

ax1.set_yticks([0.97,0.98,0.99,1])

plt.tight_layout()

plt.subplots_adjust(wspace=0.23)


#display plot
if save_fig:
    plt.savefig(RESULT_OUTPUT +  '/{}_results_PR_Curve.png'.format(RESULT_NAME), dpi=400)   
    results.to_csv(RESULT_OUTPUT + '/{}_results_sheet.csv'.format(RESULT_NAME), index=False)


#%%
#%%
#%%
def SENS(TP,FN):
    return TP / (TP + FN)

def SPEC(TN, FP):
    'TNR'
    return TN / (TN + FP)

def PPV(TP,FP):
    return TP / (TP + FP)


def NPV(TN,FN):
    if (TN + FN) == 0:
        return 1
    else:
        return TN / (TN + FN) 
    
######## HIGH RISK THRESHOLD (CIRCLE) ##################################
vals_above = results.loc[results['y_pred'] > predicted_risk_threshold, 'y_true'].value_counts()    
vals_above = vals_above.astype(float)
vals_below = results.loc[results['y_pred'] <= predicted_risk_threshold, 'y_true'].value_counts()    
vals_below = vals_below.astype(float)

SENS(TP=vals_above[1], FN=vals_below[1])  #0.80
SPEC(TN=vals_below[0], FP=vals_above[0])  #0.89

######## LOW RISK THRESHOLD (SQUARE) ##################################

predicted_risk_healthy_threshold = 0.25

vals_above = results.loc[results['y_pred'] > predicted_risk_healthy_threshold, 'y_true'].value_counts()    
vals_above = vals_above.astype(float)
vals_below = results.loc[results['y_pred'] <= predicted_risk_healthy_threshold, 'y_true'].value_counts()    
vals_below = vals_below.astype(float)

SENS(TP=vals_above[1], FN=vals_below[1])  #0.05
SPEC(TN=vals_below[0], FP=vals_above[0])  #0.05
NPV(TN=vals_below[0], FN=vals_above[0])



#%%

df = pd.DataFrame(columns=['threshold','TP','TN','FP','FN','PPV','NPV','SENS','SPEC','TOTAL'])

for risk_threshold in np.arange(0,1,0.001):
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
    ppv_threshold = PPV(TP,FP)
    total_threshold = TP+FP+TN+FN
    
    df = df.append({'threshold':risk_threshold,'TP':TP,'TN':TN,'FP':FP,'FN':FN,
                    'PPV':PPV(TP,FP),'NPV':NPV(TN,FN),
                    'SENS':SENS(TP,FN),'SPEC':SPEC(TN, FP),
                    'TOTAL':total_threshold}, ignore_index=True)    

plt.figure(figsize=(5,5))
plt.title('ROC')
plt.plot(1-df['SPEC'],df['SENS'])
plt.xlabel('1-SPEC')
plt.ylabel('SENS')
plt.plot([0,1],[0,1])


def SENS(TP,FN):
    return TP / (TP + FN)
def SPEC(TN, FP):
    'TNR'
    return TN / (TN + FP)

plt.figure(figsize=(5,5))
df = df.sort_values('threshold', ascending=True)
#plt.plot(((df['TN']+df['FN'])/df['TOTAL']), 1-(df['FN']/(df['FN']+df['TP'])))
plt.plot(((df['TN']+df['FN'])/df['TOTAL']), df['SENS'])

plt.ylabel('Fraction cancers detected')
plt.xlabel('Fraction omitted readings')
plt.grid()
#plt.xlim([0,0.1])
#plt.ylim([0,0.01])


plt.plot(df['SENS'],df['PPV'])
plt.plot(df['SPEC'],df['NPV'])


plt.plot(df['SPEC'],(df['TN']+df['FN'])/df['TOTAL'])
plt.plot([0,1],[0,1])
