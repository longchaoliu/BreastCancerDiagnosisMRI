#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:48:01 2024

@author: deeperthought
"""

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
import os



# RESULTS_TABLE_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Convert_finetuning/model_pretrained/DGNS_notFineTuned_Baseline_Converts_TestSet.csv'
# results = pd.read_csv(RESULTS_TABLE_PATH)

# RESULTS_TABLE_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Convert_finetuning/model_pretrained/DGNS_notFineTuned_Baseline_Converts_TestSet.csv'
# results = pd.read_csv(RESULTS_TABLE_PATH)

# PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Convert_finetuning/Sessions/Seeded_25epochs_Segmented_and_NotSegmented_10Folds_LastLayerFineTune_BalancedClasses/'
# PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Convert_finetuning/Sessions/Seeded_50epochs_Segmented_and_NotSegmented_10Folds_LastTwoLayersFineTune_BalancedClasses/'

PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Convert_finetuning/Sessions/Seeded_50epochs_Segmented_and_NotSegmented_10Folds_LastTwoLayersFineTune_TrainPrevalence/'


# PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Convert_finetuning/Sessions/Seeded_50_true_epochs_Segmented_and_NotSegmented_10Folds_LastTwoLayersFineTune_TrainPrevalence/'
results_list = []
for fold in os.listdir(PATH):
    
    
    # if os.path.exists(PATH + fold + '/Test_NoDemographics_result.csv'):
    #     results_list.append(pd.read_csv(PATH + fold + '/Test_NoDemographics_result.csv'))
        
    if os.path.exists(PATH + fold + '/test_result.csv'):
        results_list.append(pd.read_csv(PATH + fold + '/test_result.csv'))
        
results = pd.concat(results_list)

    
save_fig = 0

MASTER = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Data_Partitions.csv')

RESULT_OUTPUT = PATH

NAME = 'DGNS_Finetuned_Converts_TestSet'

LOAD_REAL_RESULTS = 1

LOAD_SIMULATED_RESULTS = 0

ADJUST_PREVALENCE = 1

DO_PREVALENCE_SIMS = 0

PAPER_RED = '#a3238e'

PAPER_BLUE = '#4f81bd'

# YEAR = 2014

df_birad_features = pd.read_csv('/home/deeperthought/Documents/Papers_and_grants/Risk_paper/data/RISK/BIRADS_features/NOW_scans_BIRADS_Features_CLEAN.csv')

UNIQUE_BREASTS = 0

#%%

if LOAD_REAL_RESULTS:
    
    # results1 = pd.read_csv(RESULTS_TABLE_PATH1)
    # results2 = pd.read_csv(RESULTS_TABLE_PATH2)

    # results = results.loc[results['y_true'] == 1]    
    # results1 = results1.loc[results1['y_true'] == 0]
    # results2 = results2.loc[results2['y_true'] == 0]
    
    # results2 = results2.loc[results2['scan'].isin(results1['scan'])]

    # results = pd.concat([results, results2])
    
    
    
    
    results.reset_index(drop=True, inplace=True)

    # results['year'] = results['scan'].apply(lambda x : int(x[-10:-6]))
    
    #results = results.loc[results['year'] == YEAR]
    
elif LOAD_SIMULATED_RESULTS:
    N_benigns = 10000
    N_malignants = 200
    mean_benigns = 0.5
    mean_malignants = 0.5
    
    y_pred = np.random.normal(loc=mean_benigns,scale=0.1,size=N_benigns)
    y_pred = y_pred[y_pred > 0]
    y_pred = y_pred[y_pred < 1]
    y_true = [0]*len(y_pred)
    
    benigns = pd.DataFrame(zip(y_pred, y_true), columns=['y_pred','y_true'])
    
    y_pred = np.random.normal(loc=mean_malignants,scale=0.1,size=N_malignants)
    y_pred = y_pred[y_pred > 0]
    y_pred = y_pred[y_pred < 1]
    y_true = [1]*len(y_pred)
    
    malignants = pd.DataFrame(zip(y_pred, y_true), columns=['y_pred','y_true'])

    results = pd.concat([malignants, benigns])    
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
    
def get_triage_curves(results, steps=1000):
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

results['subject'] = results['scan'].str[:20]
results['breast'] = results['scan'].str[:20] + results['scan'].str[-1]

print(len(results['subject'].unique()))
print(len(results['breast'].unique()))

print(len(results.loc[results['y_true'] == 1, 'subject'].unique()))
print(len(results.loc[results['y_true'] == 1, 'breast'].unique()))
print(len(results.loc[results['y_true'] == 1, 'scan'].unique()))

print(len(results.loc[results['y_true'] == 0, 'subject'].unique()))
print(len(results.loc[results['y_true'] == 0, 'breast'].unique()))
print(len(results.loc[results['y_true'] == 0, 'scan'].unique()))



    
if save_fig:
    results.to_csv(RESULT_OUTPUT + NAME + '.csv')


results.columns


# results = pd.concat([resultB.sample(n=167*50, replace=True),resultM])
if UNIQUE_BREASTS:
    results = results.drop_duplicates('breast')
    
    resultB = results.loc[results['y_true'] == 0]
    
    resultM = results.loc[results['y_true'] == 1]

    results = pd.concat([resultB, resultB,resultB, resultB,resultB, resultM])
    
else:
    
    resultB = results.loc[results['y_true'] == 0]
    
    resultM = results.loc[results['y_true'] == 1]
    
    results = pd.concat([resultB, resultB, resultM])


results.reset_index(inplace=True, drop=True)



#%%

triage_df = []
for i in range(20):
    print(i)
    triage_df.append(get_triage_curves(results.sample(frac=1, replace=True), steps=1000))



triage_df_mean = triage_df[0].copy()
triage_df_max = triage_df[0].copy()
triage_df_min = triage_df[0].copy()

triage_df_95perct = triage_df[0].copy()
triage_df_05perct = triage_df[0].copy()


for col in ['N_below_threshold', 'N_above_threshold', 'TP', 'TN', 'FP','FN', 'PPV', 'NPV', 'SENS', 'SPEC', 'TOTAL']:
    triage_df_mean[col] = np.mean([ x[col].values for x in triage_df], axis=0)
    triage_df_max[col] = np.max([ x[col].values for x in triage_df], axis=0)
    triage_df_min[col] = np.min([ x[col].values for x in triage_df], axis=0)
    triage_df_95perct[col] = np.percentile([ x[col].values for x in triage_df], 97.5, axis=0)
    triage_df_05perct[col] = np.percentile([ x[col].values for x in triage_df], 2.5,  axis=0)

# for i in range(len(triage_df)):
#     plt.plot( (triage_df[i]['TN']+triage_df[i]['FN'])/(triage_df[i]['TN']+triage_df[i]['FN']+triage_df[i]['TP']+triage_df[i]['FP']), triage_df[i]['NPV'], color='dodgerblue', alpha=0.5);

X_SPEC = triage_df_mean['SPEC']#(triage_df_mean['TN']+triage_df_mean['FN'])/(triage_df_mean['TN']+triage_df_mean['FN']+triage_df_mean['TP']+triage_df_mean['FP'])
Y_FOR = 1-triage_df_mean['NPV']
Y_FOR_max = 1-triage_df_max['NPV']
Y_FOR_min = 1-triage_df_min['NPV']



X_SENS = triage_df_mean['SENS']#(triage_df_mean['TP']+triage_df_mean['FP'])/(triage_df_mean['TN']+triage_df_mean['FN']+triage_df_mean['TP']+triage_df_mean['FP'])
Y_FDR = 1-triage_df_mean['PPV']
Y_FDR_max = 1-triage_df_95perct['PPV']
Y_FDR_min = 1-triage_df_05perct['PPV']

X_SENS_top95 = triage_df_95perct['SENS']
X_SENS_bottom95 = triage_df_05perct['SENS']


from scipy.ndimage import gaussian_filter




triage_df = get_triage_curves(results, steps=1000)

# plt.figure(figsize=(20,20))
# plt.subplot(2,1,1)
# plt.plot(triage_df['threshold'], triage_df['TP'], label='TP', color='r'); 
# plt.plot(triage_df['threshold'], triage_df['FN'], label='FN', color='orange'); 
# plt.legend(); plt.grid()
# plt.subplot(2,1,2)
# plt.plot(triage_df['threshold'], triage_df['TN'], label='TN', color='g'); 
# plt.plot(triage_df['threshold'], triage_df['FP'], label='FP', color='b');
# plt.legend(); plt.grid()
#%% change prevalence
if DO_PREVALENCE_SIMS:
    
    triage_curves_prevalence = {}
    
    STEPS = 0.02
    
    for PREVALENCE in [0.005,0.01,0.02,0.05,0.1]:#np.arange(0.01,0.1+STEPS,STEPS):
        print(PREVALENCE)
    
        N = int(9100*PREVALENCE)
        
        
        M_draw = results.loc[results['y_true']==1].sample(replace=True, n=N)
    
        tmp = pd.concat([results.loc[results['y_true'] == 0], M_draw])
        
        triage_df_temp = get_triage_curves(tmp, steps=1000)
    
        triage_curves_prevalence[PREVALENCE] = triage_df_temp
    

#%%

auc = roc_auc_score(results['y_true'], results['y_pred'])

#%%



SIGMA = 5
y_for_smooth = gaussian_filter(Y_FOR,sigma=SIGMA, order=0, mode='nearest')
y_for_max_smooth = gaussian_filter(Y_FOR_max,sigma=SIGMA, order=0, mode='nearest')
y_for_min_smooth = gaussian_filter(Y_FOR_min,sigma=SIGMA, order=0, mode='nearest')


y_fdr_smooth = gaussian_filter(Y_FDR,sigma=SIGMA, order=0, mode='nearest')
y_fdr_max_smooth = gaussian_filter(Y_FDR_max,sigma=SIGMA, order=0, mode='nearest')
y_fdr_min_smooth = gaussian_filter(Y_FDR_min,sigma=SIGMA, order=0, mode='nearest')



import matplotlib
matplotlib.rcParams.update({'font.size': 11})


plt.figure(1, figsize=(4,2))

plt.hist(results.loc[results['y_true'] == 0, 'y_pred'], color='g', alpha=0.8, bins=100, label= 'no cancer' + str(len(results.loc[results['y_true'] == 0, 'y_pred'])))
plt.hist(results.loc[results['y_true'] == 1, 'y_pred'], color='r', alpha=0.8, bins=100, label= 'Cancer' + str(len(results.loc[results['y_true'] == 1, 'y_pred'])))
plt.yscale('log')
plt.xlim([0,1.1])


plt.figure(2, figsize=(4,8))

plt.subplot(2,1,2)
plt.plot(triage_df['SENS'],triage_df['threshold'], color=PAPER_RED); plt.ylabel('Threshold'); plt.xlabel('Sensitivity'); plt.grid()

plt.subplot(2,1,1) 
plt.plot(X_SENS,y_fdr_smooth, color=PAPER_RED, linewidth=2)
plt.fill_between(X_SENS, y_fdr_min_smooth, y_fdr_max_smooth, interpolate=True, color=PAPER_RED, alpha=0.25)
plt.legend()    
plt.ylabel('Cost (FDR)'); plt.xlabel('Benefit (Sensitivity)'); plt.grid()
plt.ylim([0.7,1])


plt.tight_layout()

if save_fig:
    plt.savefig(RESULT_OUTPUT + NAME + '_EarlyDetection.png')
#--------------------------------------------------------------------------------------


FIGSIZE = (8,8)
import matplotlib
matplotlib.rcParams.update({'font.size': 11})



plt.figure(3, figsize=(4,2))

plt.hist(results.loc[results['y_true'] == 0, 'y_pred'], color='g', alpha=0.8, bins=100, label= 'no cancer' + str(len(results.loc[results['y_true'] == 0, 'y_pred'])))
plt.hist(results.loc[results['y_true'] == 1, 'y_pred'], color='r', alpha=0.8, bins=100, label= 'Cancer' + str(len(results.loc[results['y_true'] == 1, 'y_pred'])))
plt.yscale('log')
plt.xlim([0,1])


plt.figure(4, figsize=(4,8))

plt.subplot(2,1,2)
plt.plot(triage_df['SPEC'],triage_df['threshold'], color=PAPER_BLUE); plt.ylabel('threshold'); plt.xlabel('Specificity'); plt.grid()

plt.subplot(2,1,1) 
plt.plot(X_SPEC,y_for_smooth, color=PAPER_BLUE, linewidth=2)
plt.fill_between(X_SPEC, y_for_min_smooth, y_for_max_smooth, interpolate=True, color='dodgerblue', alpha=0.25)
plt.legend()    
plt.ylabel('Cost (FOR)'); plt.xlabel('Benefit (Specificity)'); plt.grid()


plt.tight_layout()

if save_fig:
    plt.savefig(RESULT_OUTPUT + NAME + '_SkippingExams.png')
    
#%%


FONTSIZE = 15

# PAPER FIGURE 4:
    
plt.figure(4, figsize=((10,5)))
N,M=1,2



plt.subplot(N,M,1); plt.title('Diagnosis', fontsize=FONTSIZE)

BINS=90
plt.hist(results.loc[results['y_true'] == 0, 'y_pred'], color='g', alpha=0.8, bins=BINS, label= 'no cancer (n=' + str(len(results.loc[results['y_true'] == 0, 'y_pred'])) + ')')
plt.hist(results.loc[results['y_true'] == 1, 'y_pred'], color='r', alpha=0.8, bins=BINS, label= 'cancer (n=' + str(len(results.loc[results['y_true'] == 1, 'y_pred'])) + ')')
plt.ylim([0.6,500])
plt.xlim([0,1])
# plt.legend(loc='upper right')

plt.yscale('log')


plt.subplot(N,M,2); plt.title('Diagnosis', fontsize=FONTSIZE)

# for i in range(len(tprs)):
#     plt.plot(tprs[i], fprs[i], color='dodgerblue', alpha=0.01)

plt.plot(triage_df['SPEC'], triage_df['SENS'], color='blue')
plt.fill_between(X_SPEC, X_SENS_bottom95, X_SENS_top95, interpolate=True, color='dodgerblue', alpha=0.25)

plt.text(x=0.25,y=0.1,s='AUC = ' + str(round(auc,2)))
plt.plot([0, 1], [1, 0], linestyle='--', color='red')  # random predictions curve
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('Specificity', fontsize=FONTSIZE)
plt.ylabel('Sensitivity', fontsize=FONTSIZE)

# (sensitivity and specificity for circle: 96%, 17% and for square: 33%, 86%



#%%


plt.xlabel('Predicted risk', fontsize=FONTSIZE)

#----- benefit-cost -----------------
plt.subplot(N,M,3); plt.title('Omit', fontsize=FONTSIZE)
plt.plot(X_SPEC, y_for_smooth, color=PAPER_BLUE, linewidth=2)
plt.fill_between(X_SPEC, y_for_min_smooth, y_for_max_smooth, interpolate=True, color='dodgerblue', alpha=0.25)
plt.ylabel('Cost (False Omission Rate)', fontsize=FONTSIZE)
plt.xlabel('Benefit (Specificity)', fontsize=FONTSIZE)
plt.grid()
plt.hlines(y=0.005, xmin=0, xmax=1, linestyle='--', color='k', alpha=0.75)

plt.subplot(N,M,4); plt.title('Re-evaluate', fontsize=FONTSIZE)
plt.plot(X_SENS, y_fdr_smooth, color=PAPER_RED, linewidth=2)
plt.fill_between(X_SENS, y_fdr_min_smooth, y_fdr_max_smooth, interpolate=True, color=PAPER_RED, alpha=0.25)
plt.ylim([0.7,1])
plt.grid()
plt.hlines(y=0.95, xmin=0, xmax=1, linestyle='--', color='k', alpha=0.75)
plt.ylabel('Cost (False re-evaluation rate)', fontsize=FONTSIZE)
plt.xlabel('Benefit (Sensitivity)', fontsize=FONTSIZE)

plt.tight_layout()

# #---------- cost-benefit --------------
# plt.subplot(N,M,3); plt.title('Omit', fontsize=FONTSIZE)
# plt.plot(y_for_smooth, X_SPEC, color=PAPER_BLUE, linewidth=2)
# plt.fill_between(X_SPEC, y_for_min_smooth, y_for_max_smooth, interpolate=True, color='dodgerblue', alpha=0.25)
# plt.ylabel('Cost (FOR)', fontsize=FONTSIZE)
# plt.xlabel('Benefit (SPEC)', fontsize=FONTSIZE)
# plt.grid()

# plt.subplot(N,M,4); plt.title('Early detection', fontsize=FONTSIZE)
# plt.plot(X_SENS, y_fdr_smooth, color=PAPER_RED, linewidth=2)
# plt.fill_between(X_SENS, y_fdr_min_smooth, y_fdr_max_smooth, interpolate=True, color='darkorange', alpha=0.25)
# plt.ylim([0.5,1])
# plt.grid()



if save_fig:
    plt.savefig(RESULT_OUTPUT + NAME + '_SummaryFigure.png')


#%%

print('worse converts:')


rm = results.loc[results['y_true'] == 1][['scan','y_pred']]
rm = rm.sort_values(by='y_pred')

print(rm.loc[rm['y_pred'] < 0.45])

#%%



x = (triage_df['TP']+triage_df['FP'])/(triage_df['TN']+triage_df['FN']+triage_df['TP']+triage_df['FP']) 
x = x.values

y = triage_df['PPV']
y = y.values

thr = triage_df['threshold'].values

earlyCancers = pd.DataFrame(zip(thr,x,y), columns=['threshold','fraction','precision'])

FRACTION_REEVALUATION = 0.1

OP_PPV = earlyCancers[earlyCancers['fraction'] >= FRACTION_REEVALUATION].iloc[-1]['precision']
OP_THRESHOLD = earlyCancers[earlyCancers['fraction'] >= FRACTION_REEVALUATION].iloc[-1]['threshold']

N_cancers = len(results.loc[results['y_true'] == 1])
OP_SENS = triage_df.loc[triage_df['threshold'] == OP_THRESHOLD]['SENS'].values[0]
OP_SPEC = triage_df.loc[triage_df['threshold'] == OP_THRESHOLD]['SPEC'].values[0]
OP_N = triage_df.loc[triage_df['threshold'] == OP_THRESHOLD]['N_above_threshold'].values[0]
OP_TP = triage_df.loc[triage_df['threshold'] == OP_THRESHOLD]['TP'].values[0]
OP_FN = triage_df.loc[triage_df['threshold'] == OP_THRESHOLD]['FN'].values[0]
OP_FP = triage_df.loc[triage_df['threshold'] == OP_THRESHOLD]['FP'].values[0]


print('\nOperating point for early detection:')
print(f'\nAt the cost of evaluating {FRACTION_REEVALUATION*100}% of data: ')
print(f'\nWe evaluate everything above threshold={OP_THRESHOLD}, the operating point has SENS={OP_SENS}, SPEC={OP_SPEC} ')
print(f'\nWe have a PPV={OP_PPV}')
print(f'\nWe evaluate {OP_N} images, out of which we have {OP_TP} early detections (out of {N_cancers}, or {OP_SENS}%)')


triage_df.loc[triage_df['threshold'] == OP_THRESHOLD]


#%%


plt.plot((1-triage_df['NPV']))

triage_df['FOR'] = 1-triage_df['NPV']

COST_THRESHOLD_FOR = 0.005

OP_THRESHOLD = triage_df.loc[triage_df['FOR'] < COST_THRESHOLD_FOR, 'threshold'].values[-1]

N_cancers = len(results.loc[results['y_true'] == 1])
OP_SENS = triage_df.loc[triage_df['threshold'] == OP_THRESHOLD]['SENS'].values[0]
OP_SPEC = triage_df.loc[triage_df['threshold'] == OP_THRESHOLD]['SPEC'].values[0]
OP_N = triage_df.loc[triage_df['threshold'] == OP_THRESHOLD]['N_below_threshold'].values[0]
OP_TP = triage_df.loc[triage_df['threshold'] == OP_THRESHOLD]['TP'].values[0]
OP_FN = triage_df.loc[triage_df['threshold'] == OP_THRESHOLD]['FN'].values[0]
OP_FP = triage_df.loc[triage_df['threshold'] == OP_THRESHOLD]['FP'].values[0]


print('\nOperating point for risk adjusted screening:')
print(f'\nWe evaluate everything below threshold={OP_THRESHOLD}, the operating point has SENS={OP_SENS}, SPEC={OP_SPEC} ')
print(f'\nWe skip {OP_N} images, out of which we have {OP_FN} cancers')


triage_df.loc[triage_df['threshold'] == OP_THRESHOLD]


#%%

'''
For the cost-benefit analysis of omitting exams we propose to use:

Cost: False omission rate = FN/(TN+FN) 				(3)
Benefit: Fraction correctly omitted = TN/(TN+FN+TP+FP)			(4)

For the cost-benefit analysis in early detection we propose to use (Fig. S1C):

Cost: Flase detection rate  = FP/(TP+FP) 				(5)
Benefit: Fraction correctly re-evaluated= TP/(TN+FN+TP+FP)		(6)
'''


cost_low = triage_df['FN']/(triage_df['TN'] + triage_df['FN'])
benefit_low = triage_df['SPEC']#(triage_df['TN'])/(triage_df['TN']+triage_df['FN']+triage_df['TP']+triage_df['FP']) 

cost_high = triage_df['FP']/(triage_df['TP'] + triage_df['FP'])
benefit_high = triage_df['SENS']#(triage_df['TP'])/(triage_df['TN']+triage_df['FN']+triage_df['TP']+triage_df['FP']) 



plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(benefit_low,cost_low, color='blue')

plt.xlabel('SPEC')

#plt.xlabel('Fraction correctly omitted')
plt.ylabel('False omission rate')
plt.grid()
plt.hlines(y=0.005,xmin=0, xmax=1, alpha=0.5, linestyle='--', color='k')

plt.subplot(122)
plt.plot(benefit_high,cost_high, color='r')
plt.xlabel('SENS')
#plt.xlabel('Fraction correctly re-evaluated')
plt.ylabel('Flase detection rate')
plt.grid()
plt.hlines(y=0.95,xmin=0, xmax=1, alpha=0.5, linestyle='--', color='k')








#%%


triage_df.loc[triage_df['NPV'] >= 0.995].iloc[-1]




#%%

years = list(set([x.split('_')[-2][:4] for x in results['scan']]))
years.sort()

results['year'] = results['scan'].apply(lambda x : x.split('_')[-2][:4])

prevalence_years = {}
n_years = {}
for YEAR in years[:-1]:
    result_year = results.loc[results['year'] == YEAR]
    N_year = len(result_year)
    N_reevaluated = len(result_year.loc[result_year['y_pred'] >= OP_THRESHOLD])
    pathologies_in_year = result_year.loc[result_year['y_pred'] >= OP_THRESHOLD]['y_true'].value_counts()
    N_evaluated_benigns = pathologies_in_year[0]
    try:
        N_evaluated_converts = pathologies_in_year[1]
    except:
        N_evaluated_converts = 0
    
    not_evaluated_pathologies_in_year = result_year.loc[result_year['y_pred'] < OP_THRESHOLD]['y_true'].value_counts()
    try:
        N_not_evaluated_converts = not_evaluated_pathologies_in_year[1]
    except:
        N_not_evaluated_converts = 0
        
    print(f'\n----- year {YEAR} ------')
    print(f'From {N_year} exams, we proposed re-evaluating {N_reevaluated}, out of which {N_evaluated_benigns} are benigns, but {N_evaluated_converts} will turn into cancer within a year')
        
    prevalence_year = result_year['y_true'].value_counts().values
    n_years[YEAR] = prevalence_year
    if len(prevalence_year) == 2:
        prevalence_year = prevalence_year[1]/prevalence_year[0]
        prevalence_years[YEAR] = prevalence_year
