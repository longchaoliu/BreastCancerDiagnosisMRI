#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 18:01:55 2024

@author: deeperthought
"""


import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize


#%%    USER INPUT

MASTER = pd.read_csv('/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/metadata/curated/pathology_axials.csv')

OUTPUT_FOLDER_2D = '/media/HDD/Diagnosis_2D_slices/X_1neighbor/' 

NEIGHBOR_SLICES = 0

MODALITY = 'sagittal'

N_BENIGN_SLICES = 3

DTYPE = 'float16'

#%% Already done

if not os.path.exists(OUTPUT_FOLDER_2D):
    os.mkdir(OUTPUT_FOLDER_2D)
    os.mkdir(OUTPUT_FOLDER_2D  + 'BENIGN')
    os.mkdir(OUTPUT_FOLDER_2D + 'MALIGNANT')

DONE = os.listdir(OUTPUT_FOLDER_2D + 'BENIGN')
DONE.extend( os.listdir(OUTPUT_FOLDER_2D + 'MALIGNANT'))

DONE_SCANID = list(set([x[:31] for x in DONE]))
DONE_EXAMS = list(set([x[:-2] for x in DONE_SCANID]))

len(DONE_EXAMS)


#%% Removing scans that have invalid entries (incomplete series, etc)


no_slope1 = []


NaNs = []


#%% short clean up

# Remove unknown pathology
MASTER.loc[MASTER['Pathology'] == '0', 'Pathology'] = 'Benign'
MASTER.loc[MASTER['Pathology'] == '1', 'Pathology'] = 'Malignant'
MASTER = MASTER.loc[MASTER['Pathology'].isin(['Benign','Malignant'])]

# remove unsegmented malignants

MASTER_B = MASTER.loc[MASTER['Pathology'] == 'Benign']
MASTER_M = MASTER.loc[MASTER['Pathology'] == 'Malignant']


MASTER__segmented = MASTER_M.loc[MASTER_M['segmented_slices'].str.startswith('[', na=False)]

MASTER = MASTER__segmented#pd.concat([MASTER__segmented, MASTER_B])

MASTER = MASTER.sort_values('scanID')

MASTER['Exam'] = MASTER['scanID'].str[:29]

#%%
import sys
sys.path.append('/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/code/')
from utils import load_and_preprocess


EXAMS = list((MASTER['Exam'].values))


EXAMS = set(EXAMS) - set(DONE_EXAMS)

TOT = len(EXAMS)

#%%
counter = 0
for exam in EXAMS: #TOT
    print(f'{counter}/{TOT}')
    counter += 1
    patient = exam[:20]
    
    all_subject_channels = [ MASTER.loc[MASTER['Exam'] == exam, 't1post_path'].values[0],
                           MASTER.loc[MASTER['Exam'] == exam, 'slope1_path'].values[0],
                           MASTER.loc[MASTER['Exam'] == exam, 'slope2_path'].values[0]]

    T1_pre_nii_path = '' #MRI_PATH + exam + '/T1_{}_01_01.nii'.format(side)
        
    #-------------------- Load and preprocess DCE MRI  ---------------------------------

    X, shape = load_and_preprocess(all_subject_channels, T1_pre_nii_path=T1_pre_nii_path, imaging_protocol='axial', debug=False) 

    if X is None:
        print('File not found error, skip!')
        no_slope1.append(exam)
        continue

    if DTYPE == 'float16':
        X = X.astype(np.float16)

    if np.any(np.isnan(X)):
        print('NaNs found, skip..')
        NaNs.append(exam)    
        continue
    
    for side in ['r','l']:
        
             
        scanID = exam + '_' + side
        
        if scanID not in MASTER['scanID'].values:
            continue
        
        pathology = MASTER.loc[MASTER['scanID'] == scanID, 'Pathology'].values[0]
        
        if pathology == 'Benign':
           
            FOLDER = 'BENIGN'
            
            if side == 'l':
                breast_slices = np.arange(0,X.shape[0]//2)
                middle_slice = breast_slices[len(breast_slices)//2]
            else:
                breast_slices = np.arange(X.shape[0]//2,X.shape[0])
                middle_slice = breast_slices[len(breast_slices)//2]

            random = list(np.random.choice(list(set(breast_slices) - set([middle_slice])), size=N_BENIGN_SLICES-1, replace=False))
            selected_slices = random + [middle_slice]
                        
        elif pathology == 'Malignant':
           
            FOLDER = 'MALIGNANT'
            
            selected_slices = MASTER.loc[MASTER['scanID'] == scanID, 'segmented_slices'].values[0]
               
            selected_slices = [int(x) for x in selected_slices.replace('[','').replace(']','').split(', ')]

            selected_slices = selected_slices[1::3]   
            

        #-------------------- Store Slices  ---------------------------------
        print("####### STORE ##############")

        for i in range(len(selected_slices)):
            np.save(f'{OUTPUT_FOLDER_2D}/{FOLDER}/{scanID}_{selected_slices[i]}.npy', X[selected_slices[i]-NEIGHBOR_SLICES:selected_slices[i]+NEIGHBOR_SLICES+1])
                    

