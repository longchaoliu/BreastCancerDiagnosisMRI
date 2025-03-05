#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:54:15 2024

@author: deeperthought
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize



MASTER = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Data_Partitions.csv')
RISK = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Data_ExamHistory_Labels.csv')


RISK = RISK[['Scan_ID', 'BIRADS', 'Pathology', 'Images Available', 'Convert','Breast_Develops_Cancer', 'Pathology_1year', 'Pathology_2year','Pathology_3year', 'Pathology_4year', 'Pathology_5year']]
 
MASTER = MASTER[['Scan_ID', 'BIRADS', 'Pathology', 'Segmentation_Path', 'Convert','Images Available']]
      
      
segmented = MASTER.loc[MASTER['Segmentation_Path'].str.startswith('/', na=False), ['Scan_ID','Segmentation_Path']]



df_merged = RISK.merge(segmented, on='Scan_ID', how='left')


df_merged['Segmented'] = 0

df_merged.loc[df_merged['Segmentation_Path'].str.startswith('/', na=False), 'Segmented'] = 1

df_merged['Segmented'].value_counts()

df_merged.columns

df_merged = df_merged[['Scan_ID', 'BIRADS', 'Pathology', 'Convert',  'Segmented', 'Pathology_1year', 'Breast_Develops_Cancer','Pathology_2year', 'Pathology_3year', 'Pathology_4year', 'Pathology_5year' ,'Segmentation_Path', 'Images Available']]

df_merged.to_csv('/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/metadata/curated/pathology.csv', index=False)



#%%

axial = np.load("/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/DATA/Axial_Data.npy", allow_pickle=True).item()
labels = np.load("/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/DATA/Axial_Labels.npy", allow_pickle=True).item()

segmentations = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/DATA/Axial_segmented_annotations_lesions.csv')

df = pd.DataFrame(columns=['scanID', 'Pathology','BIRADS','segmented_slices','modality','institute','notes', 't1post_path','slope1_path','slope2_path','segmentation_path'])



axial_scanid = [x[:31] for x in axial['train']]
axial_scanid.extend([x[:31] for x in axial['validation']])
axial_scanid.extend([x[:31] for x in axial['test']])

len(set(axial_scanid))

axial_label_keys = list(set(labels.keys()))

len(axial_label_keys)


########################################


Segmented_axials_path1 = '/home/deeperthought/kirby_MSK/250caseSegExtd-May2020-cropped/'
Segmented_axials_path2 = '/home/deeperthought/kirby_MSK/segExtd-Mar2021/'

seg1 = [x[:29] for x in os.listdir(Segmented_axials_path1)]
seg2 = [x[:29] for x in os.listdir(Segmented_axials_path2)]

AXIAL_PATH1 = '/home/deeperthought/kirbyPRO/alignedNiiAxial-May2020-cropped-normed/'

axial_scans = os.listdir(AXIAL_PATH1)


axial_scans = list(set(axial_scans))


for ax in axial_scans:
    
    for side in ['r','l']:
        scanid = ax + '_' + side
        segmentation_path = ''
        segmented_slices = ''
        try:
            pathology = labels[scanid]
        except KeyError:
            pathology = 'Unknown'

        if scanid in segmentations['Exam'].values:
            print('segmented exam')
            z1 = segmentations.loc[segmentations['Exam'] == scanid, 'z1'].values[0]
            z2 = segmentations.loc[segmentations['Exam'] == scanid, 'z2'].values[0]
            segmented_slices = list(np.arange(z1,z2))
            if ax in seg1:
                segmentation_path = '/home/deeperthought/kirby_MSK/250caseSegExtd-May2020-cropped/'
            elif ax in seg2:
                segmentation_path = '/home/deeperthought/kirby_MSK/segExtd-Mar2021/'
                
        
        row = {'scanID':scanid, 
         'Pathology':pathology,
         'BIRADS':'Unknown',
         'segmented_slices':segmented_slices,
         'modality':'axial',
         'institute':'MSKCC',
         'notes':'',
         't1post_path':AXIAL_PATH1 + ax + '/T1_axial_02_01.nii.gz',
         'slope1_path':AXIAL_PATH1 + ax + '/T1_axial_slope1.nii.gz',
         'slope2_path':AXIAL_PATH1 + ax + '/T1_axial_slope2.nii.gz',
         'segmentation_path':segmentation_path}
        
        df = df.append(row, ignore_index=True)


df.to_csv('/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/metadata/curated/pathology_axials.csv', index=False)

df['Pathology'].value_counts()
