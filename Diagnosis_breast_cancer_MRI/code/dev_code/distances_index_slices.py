#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:33:32 2023

@author: deeperthought
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

result_path = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/test_result_with_deltas.csv'

OUTPUT_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/localization/'

res = pd.read_csv(result_path)

res.columns

resM = res.loc[res['y_true'] == 1]

resM['min_distance'] = 99

distances = {}

slices_until_hit = []

i = 1
for row in resM.iterrows():
    print(i)
    i += 1   
    
    scan = row[1]['scan']
    segmented_slice = row[1]['GT_slice']
    preds = np.array(row[1]['slice_preds'].replace('[','').replace(']','').split(', '), dtype='float')

    ranked = np.flip(np.argsort(preds))

 
    #----- TOP 95% slice ----------
    # prc = np.max(preds)#, 99)
    # top_slice = int(np.squeeze(np.argwhere(preds >= prc)))
    
    #----- TOP 1 slice ----------
    # top_slice = np.argmax(preds)
    
    
    number_of_slices_until_hit = int(np.squeeze(np.argwhere(ranked == segmented_slice)))
    
    # n = 0
    # for sl in ranked:
        
    #     if (segmented_slice < sl-2) or (segmented_slice > sl+2):
    #         n+=1
    #         continue
    #     else:
    #         number_of_slices_until_hit = n
        
    #         break
        
    
    slices_until_hit.append(number_of_slices_until_hit)
    
    # delta = np.min(np.abs(top_slice - segmented_slice))
    
    # distances[scan] = delta
    
    # resM.loc[resM['scan'] == scan, 'min_distance'] = delta
    # resM.loc[resM['scan'] == scan, 'max_slice'] = top_slice
    


plt.figure(figsize=(4,5))
plt.hist(slices_until_hit, bins=30, alpha=0.9)
plt.vlines(x=np.mean(slices_until_hit), ymin=0, ymax=120, color='r', linestyle='--')
plt.xlabel('Model ranked slices', fontsize=11)

np.mean(slices_until_hit)

np.median(slices_until_hit)


plt.hist(distances.values(), bins=np.max(list(distances.values())))
plt.xlabel('number of slices away from 95%-peak')
plt.ylabel('images with cancer')


distances.values()


resM['min_distance'].value_counts()



resM.to_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/test_result_with_deltas.csv')
