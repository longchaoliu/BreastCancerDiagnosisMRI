#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:53:07 2024

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


result_path = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/best_model_results/test_result.csv'

resultsM_deltas = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/test_result_with_deltas.csv'


results = pd.read_csv(result_path)

resM = pd.read_csv(resultsM_deltas)


resB = results.loc[results['y_true']==0]


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

#%%

triage_df = get_triage_curves(results, steps=1000)

auc = roc_auc_score(results['y_true'], results['y_pred'])


#%%

# results0 = pd.concat([resB, resM.loc[resM['min_distance'] == 0]])
# triage_df_nrSlices_0 = get_triage_curves(results0, steps=1000)

# results1 = pd.concat([resB, resM.loc[resM['min_distance'] < 2]])
# triage_df_nrSlices_1 = get_triage_curves(results1, steps=1000)


# results2 = pd.concat([resB, resM.loc[resM['min_distance'] < 3]])
# triage_df_nrSlices_2 = get_triage_curves(results2, steps=1000)

# results3 = pd.concat([resB, resM.loc[resM['min_distance'] < 4]])
# triage_df_nrSlices_3 = get_triage_curves(results3, steps=1000)

# results4 = pd.concat([resB, resM.loc[resM['min_distance'] < 5]])
# triage_df_nrSlices_4 = get_triage_curves(results4, steps=1000)

# results5 = pd.concat([resB, resM.loc[resM['min_distance'] < 6]])
# triage_df_nrSlices_5 = get_triage_curves(results5, steps=1000)

# results6 = pd.concat([resB, resM.loc[resM['min_distance'] < 7]])
# triage_df_nrSlices_6 = get_triage_curves(results6, steps=1000)

# results7 = pd.concat([resB, resM.loc[resM['min_distance'] < 8]])
# triage_df_nrSlices_7 = get_triage_curves(results7, steps=1000)



#%%  Inverse: triage curves with misses. So everything that is not a HIT


results0 = pd.concat([resB, resM.loc[resM['min_distance'] > 0]])  # Here malignants are all that are missed by looking a the top slice
triage_df_nrSlices_0 = get_triage_curves(results0, steps=1000)

results1 = pd.concat([resB, resM.loc[resM['min_distance'] > 1]]) # Here malignants are all that are missed by looking a the top slice +/- 1
triage_df_nrSlices_1 = get_triage_curves(results1, steps=1000)


results2 = pd.concat([resB, resM.loc[resM['min_distance'] > 2]])# Here malignants are all that are missed by looking a the top slice +/- 2
triage_df_nrSlices_2 = get_triage_curves(results2, steps=1000)

results3 = pd.concat([resB, resM.loc[resM['min_distance'] > 3]])
triage_df_nrSlices_3 = get_triage_curves(results3, steps=1000)

results4 = pd.concat([resB, resM.loc[resM['min_distance'] > 4]])
triage_df_nrSlices_4 = get_triage_curves(results4, steps=1000)

results5 = pd.concat([resB, resM.loc[resM['min_distance'] > 5]])
triage_df_nrSlices_5 = get_triage_curves(results5, steps=1000)

results6 = pd.concat([resB, resM.loc[resM['min_distance'] > 6]])
triage_df_nrSlices_6 = get_triage_curves(results6, steps=1000)

results7 = pd.concat([resB, resM.loc[resM['min_distance'] > 7]])
triage_df_nrSlices_7 = get_triage_curves(results7, steps=1000)


#%%

resM['min_distance'].value_counts()

BINS = len(resM['min_distance'].unique())

plt.hist(resM['min_distance'], bins=BINS)
plt.xticks(np.arange(BINS))

from scipy.stats import pearsonr
plt.figure(figsize=(5,5))
plt.plot([0,55],[0,55], alpha=0.5, color='gray', linestyle='--')
plt.scatter(resM['GT_slice'], resM['max_slice'], alpha=0.35)
plt.xlabel('Index Slice (Radiologist)', fontsize=14)
plt.ylabel('Top predicted slice (Machine)', fontsize=14)
plt.xlim([5,55]); plt.ylim([5,55])

pearsonr(resM['GT_slice'], resM['max_slice'])

plt.figure(figsize=(6,6))

plt.plot(triage_df_nrSlices_0['SPEC'], triage_df_nrSlices_0['SENS'], label='1')
plt.plot(triage_df_nrSlices_1['SPEC'], triage_df_nrSlices_1['SENS'], label='2')
plt.plot(triage_df_nrSlices_2['SPEC'], triage_df_nrSlices_2['SENS'], label='3')
plt.plot(triage_df_nrSlices_3['SPEC'], triage_df_nrSlices_3['SENS'], label='4')
plt.plot(triage_df_nrSlices_4['SPEC'], triage_df_nrSlices_4['SENS'], label='5')
plt.plot(triage_df_nrSlices_5['SPEC'], triage_df_nrSlices_5['SENS'], label='6')
plt.plot(triage_df_nrSlices_6['SPEC'], triage_df_nrSlices_6['SENS'], label='7')
plt.plot(triage_df_nrSlices_7['SPEC'], triage_df_nrSlices_7['SENS'], label='8')

plt.plot([1,0],[0,1], 'k--', alpha=0.2)
plt.xlabel('SPEC'); plt.ylabel('SENS')
# plt.xlim([0.85,1])
plt.legend()

plt.figure(figsize=(6,6))

plt.plot(triage_df_nrSlices_0['SENS'], triage_df_nrSlices_0['PPV'], label='1')
plt.plot(triage_df_nrSlices_1['SENS'], triage_df_nrSlices_1['PPV'], label='2')
plt.plot(triage_df_nrSlices_2['SENS'], triage_df_nrSlices_2['PPV'], label='3')
plt.plot(triage_df_nrSlices_3['SENS'], triage_df_nrSlices_3['PPV'], label='4')
plt.plot(triage_df_nrSlices_4['SENS'], triage_df_nrSlices_4['PPV'], label='5')
plt.plot(triage_df_nrSlices_5['SENS'], triage_df_nrSlices_5['PPV'], label='6')
plt.plot(triage_df_nrSlices_6['SENS'], triage_df_nrSlices_6['PPV'], label='7')
plt.plot(triage_df_nrSlices_7['SENS'], triage_df_nrSlices_7['PPV'], label='8')
plt.xlabel('SENS'); plt.ylabel('PPV')
# plt.xlim([0.85,1])
plt.legend()




plt.plot(triage_df_nrSlices_1['SENS'], triage_df_nrSlices_1['TN'], label='2')

#%%


triage_df_nrSlices_0.loc[triage_df_nrSlices_0['SENS'] == 1].iloc[-1]


'''
at SENS=1, we can do abbreviated readings of the top slice +/- 1 slice of 10% of the data, while putting eyes on target.
'''



triage_df_nrSlices_1.loc[triage_df_nrSlices_1['SENS'] == 1].iloc[-1]

'''
at SENS=1, we can do abbreviated readings of the top slice +/- 1 slice of 10% of the data, while putting eyes on target.
'''


triage_df_nrSlices_2.loc[triage_df_nrSlices_2['SENS'] == 1].iloc[-1]

'''
at SENS=1, we can do abbreviated readings of the top slice +/- 1 slice of 10% of the data, while putting eyes on target.
'''



triage_df_nrSlices_6.loc[triage_df_nrSlices_6['SENS'] == 1].iloc[-1]

'''
at SENS=1, we can do abbreviated readings of the top slice +/- 6 slice of 66% of the data, while putting eyes on target.
'''




'''
BUT: This is only focusing on the top slice. What about first N top slices??
'''


#%% Benefit cost abbreviated readings

plt.figure(figsize=(10,10))
m,n=2,2
plt.subplot(m,n,1); plt.title('Top slice')
plt.hist(resB['y_pred'], color='g', bins=90)
plt.hist(resM.loc[resM['min_distance'] > 0, 'y_pred'], color='r', bins=90)
plt.yscale('log')

plt.subplot(m,n,2); plt.title('Top slice + 1')
plt.hist(resB['y_pred'], color='g', bins=90)
plt.hist(resM.loc[resM['min_distance'] > 1, 'y_pred'], color='r', bins=90)
plt.yscale('log')

plt.subplot(m,n,3); plt.title('Top slice + 2')
plt.hist(resB['y_pred'], color='g', bins=90)
plt.hist(resM.loc[resM['min_distance'] > 2, 'y_pred'], color='r', bins=90)
plt.yscale('log')

plt.subplot(m,n,4); plt.title('Top slice + 3')
plt.hist(resB['y_pred'], color='g', bins=90)
plt.hist(resM.loc[resM['min_distance'] > 3, 'y_pred'], color='r', bins=90)
plt.yscale('log')

plt.hist(resB['y_pred'], color='g', bins=90)
plt.hist(resM.loc[resM['min_distance'] > 4, 'y_pred'], color='r', bins=90)
plt.yscale('log')
#%%

FONTSIZE = 12

plt.figure(4, figsize=((10,5)))
N,M=1,2

plt.subplot(N,M,1); 

plt.plot(triage_df['SPEC'], triage_df['SENS'], color='blue')
plt.fill_between(X_SPEC, X_SENS_bottom95, X_SENS_top95, interpolate=True, color='dodgerblue', alpha=0.25)

plt.text(x=0.25,y=0.1,s='AUC = ' + str(round(auc,2)))
plt.plot([0, 1], [1, 0], linestyle='--', color='red')  # random predictions curve
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('Specificity', fontsize=FONTSIZE)
plt.ylabel('Sensitivity', fontsize=FONTSIZE)

# (sensitivity and specificity for circle: 96%, 17% and for square: 33%, 86%


plt.subplot(N,M,2); plt.title('Diagnosis', fontsize=FONTSIZE)
# plt.hist(results.loc[(results['y_true'] == 1)*( ~ results['scan'].isin(visible_lesion_scans)), 'y_pred'], color='dodgerblue', alpha=0.9, bins=90)
# plt.hist(results.loc[(results['y_true'] == 1)*(results['scan'].isin(visible_lesion_scans)), 'y_pred'], color='r', alpha=0.9, bins=90)

BINS=90
plt.hist(results.loc[results['y_true'] == 0, 'y_pred'], color='g', alpha=0.8, bins=BINS, label= 'no cancer (n=' + str(len(results.loc[results['y_true'] == 0, 'y_pred'])) + ')')
plt.hist(results.loc[results['y_true'] == 1, 'y_pred'], color='r', alpha=0.8, bins=BINS, label= 'cancer (n=' + str(len(results.loc[results['y_true'] == 1, 'y_pred'])) + ')')
plt.ylim([0.6,500])
plt.xlim([0,1])
# plt.legend(loc='upper right')

plt.yscale('log')

#%%

triage_df.loc[triage_df['SPEC'] >= 0.9].iloc[0]

triage_df_95perct.loc[triage_df_95perct['SPEC'] >= 0.9].iloc[0]
triage_df_05perct.loc[triage_df_05perct['SPEC'] >= 0.9].iloc[0]


triage_df.loc[triage_df['NPV'] == 1].iloc[-1]


plt.plot(triage_df['SENS'], triage_df['PPV'])
