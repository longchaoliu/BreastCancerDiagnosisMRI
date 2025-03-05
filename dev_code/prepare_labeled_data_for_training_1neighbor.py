#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:51:22 2024

@author: deeperthought
"""


 
import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize

os.chdir('/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/code/')

#%%    USER INPUT

MRI_PATH = '/home/deeperthought/kirby_MSK/alignedNii-Nov2019/'


OUTPUT_FOLDER = '/media/HDD/Diagnosis_2D_slices/X_2.5D/' #'/home/deeperthought/kirbyPRO/Saggittal_Full_Slices/2D_slices/X/'
OUTPUT_SEGMENTATIONS_2D = '/home/deeperthought/kirbyPRO/Saggittal_segmentations_clean/2D/'

NEIGHBOR_SLICES = 1

MODALITY = 'sagittal'

N_BENIGN_SLICES = 3

DTYPE = 'float16'

#%% Available scans

extracted_benigns = os.listdir('/media/HDD/Diagnosis_2D_slices/X/BENIGN/')
extracted_malignants = os.listdir('/media/HDD/Diagnosis_2D_slices/X/MALIGNANT/')

extracted_all = extracted_malignants + extracted_benigns

benign_scanIDs = list(set([x[:31] for x in extracted_benigns]))
malignant_scanIDs = list(set([x[:31] for x in extracted_malignants]))


SCANIDS = benign_scanIDs   ### CHANGE TO MALIGNANT 
FOLDER = 'BENIGN'


# SCANIDS = malignant_scanIDs   
# FOLDER = 'MALIGNANT'

#%% Already done

DONE = os.listdir(OUTPUT_FOLDER + 'BENIGN')
DONE.extend( os.listdir(OUTPUT_FOLDER + 'MALIGNANT'))

DONE_SCANID = list(set([x[:31] for x in DONE]))
DONE_EXAMS = list(set([x[:-2] for x in DONE_SCANID]))

SCANIDS = list(set(SCANIDS) - set(DONE_SCANID))
print('{} exams still to extract 2D slices'.format(len(SCANIDS)))




#%%

import sys
sys.path.append('/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/code/')
from utils import load_and_preprocess


TOT = len(SCANIDS)

NaNs = []
noslope1 = []


for SUBJECT_INDEX in range(TOT): #TOT
    print(SUBJECT_INDEX, TOT)
    scanID = SCANIDS[SUBJECT_INDEX]
    
    patient = scanID[:20]
    exam = scanID[:29]
    side = 'right'
    contra_side = 'left'
    if scanID[-1] == 'l': 
        side = 'left'
        contra_side = 'right'
        

    #-------------------- Get paths of DCE-MRI. T1 if not normalized before  ---------------------------------

    print("####### IPSILATERAL ##############")
    all_subject_channels = [MRI_PATH + exam + '/T1_{}_02_01.nii'.format(side),
                           MRI_PATH + exam + '/T1_{}_slope1.nii'.format(side),
                           MRI_PATH + exam + '/T1_{}_slope2.nii'.format(side)]

    T1_pre_nii_path = MRI_PATH + exam + '/T1_{}_01_01.nii'.format(side)
        
    #-------------------- Load and preprocess DCE MRI  ---------------------------------

    X, shape = load_and_preprocess(all_subject_channels, T1_pre_nii_path=T1_pre_nii_path, side=side, imaging_protocol=MODALITY, debug=False)

    if X is None:
        print('File not found error, skip!')
        noslope1.append(scanID)
        continue

    if DTYPE == 'float16':
        X = X.astype(np.float16)

    if np.any(np.isnan(X)):
        NaNs.append(scanID)
        continue
        
    #-------------------- Get pathology. If cancer, get slice number, else random slice ---------------------------------
        'Dont do this anymore. Load prepared sheet with slice numbers.'

    # pathology = RISK.loc[RISK['Scan_ID'] == scanID, 'Pathology'].values[0]
 
    selected_slices = [int(x.split(scanID)[-1].split('_')[-1].split('.npy')[0]) for x in extracted_all if x.startswith(scanID)]
 
    
    #-------------------- Store Slices  ---------------------------------
    print("####### STORE ##############")


    for i in range(len(selected_slices)):
        #if not os.path.exists(OUTPUT_FOLDER + '/{}_{}.npy'.format(scanID, selected_slices[i])):
        np.save(f'{OUTPUT_FOLDER}/{FOLDER}/{scanID}_{selected_slices[i]}.npy', X[selected_slices[i]-NEIGHBOR_SLICES:selected_slices[i]+NEIGHBOR_SLICES+1])
                
        # if segmentation_available:  
        # #     if not os.path.exists(OUTPUT_SEGMENTATIONS + '/{}_{}.npy'.format(scanID, selected_slices[i])):
    
        # #         groundtruth_crop = groundtruth[selected_slices[i]]
        #     np.save(OUTPUT_SEGMENTATIONS + '/{}_{}.npy'.format(scanID, selected_slices[i]), groundtruth_crop)
            
                
        
        
    #sl = np.load('/media/HDD/Diagnosis_2D_slices/X_2.5D/MALIGNANT/MSKCC_16-328_1_05199_20030621_r_11.npy', allow_pickle=True)

    # sl.min()
    # plt.imshow(sl[0,:,:,2])
