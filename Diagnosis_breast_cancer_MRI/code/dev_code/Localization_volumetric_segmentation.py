#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:26:17 2024

@author: deeperthought
"""


import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.ndimage.measurements import label 
from skimage import measure


SEG_PATH = '/home/deeperthought/Projects/Segmentation_breast_cancer_MRI-main/predictions/'

# SEG_PATH = r'Z:\Projects\Segmentation_breast_cancer_MRI-main\predictions\\'

RAD_SEG_PATH = '/home/deeperthought/kirbyPRO/Saggittal_segmentations_clean/nifti/' #'/home/deeperthought/kirbyPRO/Saggittal_segmentations_clean/raw/'

# RAD_SEG_PATH = r"Z:\kirbyPRO\Saggittal_segmentations_clean\nifti\\"

result = pd.read_csv('/home/deeperthought/Documents/Papers_and_grants/Diagnosis_paper/data/sagittal_test_results.csv')

# result = pd.read_csv(r"Z:\Documents\Papers_and_grants\Diagnosis_paper\data\sagittal_test_results.csv")


segmented_volumetric = os.listdir(SEG_PATH)

radiologist_segmentations = os.listdir(RAD_SEG_PATH)


output_path = '/home/deeperthought/Documents/Papers_and_grants/Diagnosis_paper/data/sagittal_volumetric_localization_results.csv'

# output_path = r"Z:\Documents\Papers_and_grants\Diagnosis_paper\data\sagittal_volumetric_localization_results.csv"

if os.path.exists(output_path):
    df = pd.read_csv(output_path)#'/home/deeperthought/Documents/Papers_and_grants/Diagnosis_paper/data/sagittal_volumetric_localization_results.csv')
else:
    df = pd.DataFrame(columns=['scan','max_slice','sagittal_slices_GT','Hit', 'segmented_slice', 'previous_HIT'])



def keep_overlapping_components(arr1, arr2, segmented_slice): 
    
    labeled, ncomponents = label(arr2)
    
    # plt.imshow(labeled[segmented_slice[0]])
    
    overlap = np.logical_and(arr1, arr2)
    
    if not overlap.any():
        print('no overlap!')
        
        if len(segmented_slice) == 1:
        
            return [segmented_slice[0]-1, segmented_slice[0], segmented_slice[0]+1]
        
        elif len(segmented_slice) > 1:
            sagittal_segmented_slices = []
            for segsl in segmented_slice:
                sagittal_segmented_slices.extend([segsl[0]-1, segsl[0], segsl[0]+1])
            return sagittal_segmented_slices
    
    connected_component_index_overlap = list(set(labeled[overlap]))
    
    filtered = np.zeros(seg_img.shape)    
    
    for cc_index in connected_component_index_overlap:
        
        filtered[labeled == cc_index] = 1
        
    sagittal_segmented_slices = list(set(np.argwhere(filtered == 1)[:,0]))

    return sagittal_segmented_slices



# exam = 'MSKCC_16-328_1_14121_20100323_l'
# scan = 'MSKCC_16-328_1_14121_20100323_T1_left_02_01_epoch0.nii.gz'

N = len(segmented_volumetric)
i = 1
for scan in segmented_volumetric:
    print(f'{i}/{N}')
    i += 1
    exam = scan[:29] + scan[32:34]
    
    #if exam =='MSKCC_16-328_1_14121_20100323_l':
    #    print('Faulty file!')
    #    continue
    
    print(scan)
    
    if exam in df['scan'].values:
        print('already_done. Skip!')
        continue
    
    maxslice = result.loc[result['scan'] == exam, 'max_slice'].values[0]
    GTslice = result.loc[result['scan'] == exam, 'GT_slice'].values[0]
    
    
    rad_seg = [x for x in radiologist_segmentations if x.startswith(exam)][0]

    try:
        seg_img = nib.load(SEG_PATH + scan).get_fdata()
    except:
        print('')
    
    rad_img = nib.load(RAD_SEG_PATH + rad_seg).get_fdata()

    seg_img.shape
    rad_img.shape    

    segmented_slice = list(set(np.argwhere(rad_img >0)[:,0]))

    # plt.imshow(rad_img[GTslice])
    # plt.imshow(seg_img[segmented_slice[0]])

    if seg_img.shape[0] != rad_img.shape[0]:
        print('wrong segmentation file??')
        break

    if seg_img.shape[1] != rad_img.shape[1]:
        seg_img = resize(seg_img, output_shape = rad_img.shape, order=2, anti_aliasing=True)


    MAX = seg_img[segmented_slice[0]].max()

    seg_img[seg_img > MAX/2] = 1
    seg_img[seg_img < 1] = 0

    sagittal_segmented_slices = keep_overlapping_components(rad_img, seg_img, segmented_slice)

    if sagittal_segmented_slices == 0:
        print('no overlap!')
        break


    preHIT =0 
    for segslice in segmented_slice:
        if abs(segslice - maxslice) <= 2:
            preHIT = 1
        

    HIT = 0
    if maxslice in sagittal_segmented_slices:
        HIT = 1

    df = df.append({'scan':exam,'max_slice':maxslice,'sagittal_slices_GT':sagittal_segmented_slices,'Hit':HIT, 'segmented_slice':segmented_slice, 'previous_HIT':preHIT}, ignore_index=True)

    df.to_csv(output_path)#'/home/deeperthought/Documents/Papers_and_grants/Diagnosis_paper/data/sagittal_volumetric_localization_results.csv', index=False)



#%%



hits = df['Hit'].value_counts()

prehits = df['previous_HIT'].value_counts()

hits[1]*100./(hits[0] + hits[1])

prehits[1]*100./(prehits[0] + prehits[1])



df['previous_HIT']


df['sagittal_slices_GT']



pre = df['previous_HIT'].value_counts()

print(pre[1]/(pre[1]+pre[0]))


now = df['Hit'].value_counts()

print(now[1]/(now[1]+now[0]))


df.columns

df['width'] = 0

for row in df.iterrows():
    slices = row[1]['sagittal_slices_GT']
    slices = slices.replace('[','').replace(']','').split(', ')
    len(slices)
    df.loc[df['scan'] == row[1]['scan'], 'width'] = len(slices)


df['width'].mean()*3
df['width'].std()*3
df['width'].median()*3
df['width'].min()*3
df['width'].max()*3
df['width'].mode()*3

df['width'].value_counts()

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.hist(df['width'], bins=42)
plt.xticks(np.arange(0,42))
