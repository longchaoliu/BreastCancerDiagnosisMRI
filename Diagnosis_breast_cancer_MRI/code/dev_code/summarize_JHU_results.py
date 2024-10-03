#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:05:54 2024

@author: deeperthought
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.transform import resize

res = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/JHU/JHU_results.csv')


SEG_PATH = '/home/deeperthought/kirbyPRO/jhuData/manSegRes/Segmentation_Results-Babita50cases/segmentation/'
# SEG_PATH = '/home/deeperthought/kirbyPRO/jhuData/manSegRes/Segmentation_Results-Emily50cases/segmentation/'
# SEG_PATH = '/home/deeperthought/kirbyPRO/jhuData/manSegRes/Segmentation_Results-Kelly50cases/segmentation/'
# SEG_PATH = '/home/deeperthought/kirbyPRO/jhuData/manSegRes/Segmentation_Results-Philip50cases/segmentation/'
segmentations = os.listdir(SEG_PATH)

res['Hit'] = -1
res['X1'] = -1
res['X2'] = -1

for row in res.iterrows():
    
    exam = row[1]['exam']
    
    segmented = [x for x in segmentations if x.startswith(exam)]
    
    if len(segmented) == 0:
         print('exam not segmented.. skip')
         continue
     
    segmented = segmented[0]
    img = nib.load(SEG_PATH + segmented).get_fdata()
    
    img = np.flip(img, axis=1)
    
    # t1post = nib.load(f'/home/deeperthought/kirbyPRO/jhuData/alignedNii-normed/{exam}/T1_axial_slope1.nii').get_fdata()
    
    
    # plt.figure(figsize=(10,5))
    # plt.subplot(1,2,1); plt.suptitle(exam)
    # plt.imshow(np.max(t1post,-1))
    # plt.subplot(1,2,2)
    # plt.imshow(np.max(t1post,-1) + img[5]*10)
    

    x = np.argwhere(img >0)[:,1]
    y = np.argwhere(img >0)[:,2]

    x1 = np.min(x)
    x2 = np.max(x)

    ypred = row[1]['max_slice']

    if ypred >= x1 and ypred <= x2:
        HIT = 1
    else:
        HIT = 0
        
    res.loc[res['exam'] == exam, 'Hit'] = HIT
    res.loc[res['exam'] == exam, 'X1'] = x1
    res.loc[res['exam'] == exam, 'X2'] = x2
    
res = res.loc[res['Hit'] != -1]

loc = res['Hit'].value_counts()

print(f'Accuracy = {loc[1]*100./(loc[0]+loc[1])}% ({loc[1]}/{loc[1]+loc[0]})')
 
res['width'] = res['X2'] - res['X1']

np.mean(res['width']*0.6*0.1)

res.loc[res['Hit'] == 0, 'pred'].mean()
res.loc[res['Hit'] == 1, 'pred'].mean()

plt.hist(res.loc[res['Hit'] == 0, 'pred'], label='miss', alpha=0.5)
plt.hist(res.loc[res['Hit'] == 1, 'pred'], label='hit', alpha=0.5)
plt.legend()

#%%
