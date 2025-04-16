#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Input: DCE MRI (aligned), slice numbers or segmentations of cancer, or breast/exam-wide pathology for benigns

Output: Sagittal slices 


Should work for both Sagittal and axial

Should use same pre-processing script as with inference.

Add option to include T2 or other modalities.

Add option to extract contralateral or aligned previous exam?

Add option to extract neighbor slices?


@author: deeperthought
"""

 
import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize



#%%    USER INPUT

MRI_PATH = '/path/to/alignedNii-Nov2019/'

MASTER = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Data_Partitions.csv')

RISK = pd.read_csv('/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/metadata/curated/pathology.csv')

OUTPUT_FOLDER_2D = '/home/deeperthought/kirbyPRO/Saggittal_Full_Slices/2D_slices/X/'
OUTPUT_SEGMENTATIONS_2D = '/home/deeperthought/kirbyPRO/Saggittal_segmentations_clean/2D/'

NEIGHBOR_SLICES = 1

MODALITY = 'sagittal'

N_BENIGN_SLICES = 3

DTYPE = 'float16'

'''
There are 140 converts from which I already extracted slices, but 2 random slices as if they were benigns. 

I CANNOT USE THESE. Better to remove.

Instead use the spreadsheet CONVERT_SLICES  to extract new slices. These will have label == 1 for 1 year cancer.

'''


#%% Available scans

AVAILABLE_DATA = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/aligned-Nov2019_Triples.csv')
AVAILABLE_SCANID = list(set(AVAILABLE_DATA['Scan_ID'].values))
len(AVAILABLE_SCANID)

EXAMS = os.listdir(MRI_PATH)
len(EXAMS)

SCANIDS = list(set(AVAILABLE_DATA.loc[AVAILABLE_DATA['Exam'].isin(EXAMS), 'Scan_ID'].values))

SCANIDS = set(SCANIDS).intersection(set(RISK['Scan_ID'].values))


partition_sagittal = np.load("/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/DATA/Data.npy", allow_pickle=True).item()
labels_sagittal = np.load("/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/DATA/Labels.npy", allow_pickle=True).item()

partition_sagittal.keys()

train_data = list(set([x[:31] for x in partition_sagittal['train']]))
val_data = list(set([x[:31] for x in partition_sagittal['validation']]))
test_data = list(set([x[:31] for x in partition_sagittal['test']]))

SCANIDS =  list(set(train_data + val_data + test_data))

len(SCANIDS)



#%% Already done

DONE = os.listdir(OUTPUT_FOLDER_2D + 'BENIGN')
DONE.extend( os.listdir(OUTPUT_FOLDER_2D + 'MALIGNANT'))

DONE_SCANID = list(set([x[:31] for x in DONE]))
DONE_EXAMS = list(set([x[:-2] for x in DONE_SCANID]))

SCANIDS = list(set(SCANIDS) - set(DONE_SCANID))
print('{} exams still to extract 2D slices'.format(len(SCANIDS)))

#%%

OUTPUT_FOLDER = OUTPUT_FOLDER_2D
OUTPUT_FOLDER_CONTRA = OUTPUT_FOLDER.replace('/X/','/Contra/')
OUTPUT_FOLDER_PREVIOUS = OUTPUT_FOLDER.replace('/X/','/Previous/')

OUTPUT_SEGMENTATIONS = OUTPUT_SEGMENTATIONS_2D

if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)
if not os.path.exists(OUTPUT_FOLDER_CONTRA):
    os.mkdir(OUTPUT_FOLDER_CONTRA)
if not os.path.exists(OUTPUT_FOLDER_PREVIOUS):
    os.mkdir(OUTPUT_FOLDER_PREVIOUS)
if not os.path.exists(OUTPUT_SEGMENTATIONS):
    os.mkdir(OUTPUT_SEGMENTATIONS)


#patients_with_aligned_previous = os.listdir(MRI_ALIGNED_HISTORY_PATH)


NaNs = []

#%%
print('Removing un-segmented malignants:')

segmented_slices = os.listdir('/home/deeperthought/kirbyPRO/Saggittal_segmentations_clean/2D/')
len(set(DONE).intersection(set(segmented_slices)))
len(set(segmented_slices) - set(DONE))
segmented_scanid = list(set([x[:31] for x in segmented_slices]))

malignants = list(set(RISK.loc[RISK['Pathology'] == 'Malignant', 'Scan_ID'].values))

segmented = list(set(MASTER.loc[MASTER['Segmented'] == 1, 'Scan_ID'].values))

unsegmented_malignants = list(set(malignants) - set(segmented))

print('Malignants | segmented | not-segmented')
print(f'{len(malignants)}       | {len(segmented)}      | {len(unsegmented_malignants)}')

SCANIDS = list(set(SCANIDS) - set(unsegmented_malignants))
print('{} exams still to extract 2D slices'.format(len(EXAMS)))

#%% Removing scans that have invalid entries (incomplete series, etc)


no_slope1 = ['MSKCC_16-328_1_03326_20080421_l', 'MSKCC_16-328_1_01232_20060321_r', 'MSKCC_16-328_1_01232_20060321_l', 'MSKCC_16-328_1_01920_20021119_l']

no_slope1 = ['MSKCC_16-328_1_01920_20021119_l',
 'MSKCC_16-328_1_06063_20060521_r',
 'MSKCC_16-328_1_04950_20090515_r',
 'MSKCC_16-328_1_02868_20060206_l',
 'MSKCC_16-328_1_09810_20070624_r',
 'MSKCC_16-328_1_09810_20070624_l',
 'MSKCC_16-328_1_11731_20090620_r',
 'MSKCC_16-328_1_08255_20060303_r',
 'MSKCC_16-328_1_10093_20070220_l',
 'MSKCC_16-328_1_04168_20070223_l',
 'MSKCC_16-328_1_01920_20021119_r',
 'MSKCC_16-328_1_09172_20070110_l',
 'MSKCC_16-328_1_03422_20060216_l',
 'MSKCC_16-328_1_11443_20090526_l',
 'MSKCC_16-328_1_01432_20090523_r']

SCANIDS = list(set(SCANIDS) - set(no_slope1))

NaNs = ['MSKCC_16-328_1_00120_20021203_l',
 'MSKCC_16-328_1_04455_20060216_r',
 'MSKCC_16-328_1_03539_20090427_r',
 'MSKCC_16-328_1_01816_20020919_l',
 'MSKCC_16-328_1_00944_20070609_r',
 'MSKCC_16-328_1_07558_20060124_l',
 'MSKCC_16-328_1_08337_20060214_r',
 'MSKCC_16-328_1_00707_20030427_l',
 'MSKCC_16-328_1_08095_20120721_l',
 'MSKCC_16-328_1_00808_20060123_r',
 'MSKCC_16-328_1_04528_20040117_r',
 'MSKCC_16-328_1_02035_20020317_r',
 'MSKCC_16-328_1_03790_20071230_l',
 'MSKCC_16-328_1_04436_20061219_r',
 'MSKCC_16-328_1_07152_20060628_l',
 'MSKCC_16-328_1_02956_20031028_r',
 'MSKCC_16-328_1_14115_20130830_l',
 'MSKCC_16-328_1_00812_20020930_l',
 'MSKCC_16-328_1_09667_20080501_l',
 'MSKCC_16-328_1_05448_20080301_l',
 'MSKCC_16-328_1_03510_20100122_l',
 'MSKCC_16-328_1_01565_20021017_l',
 'MSKCC_16-328_1_05817_20060215_l',
 'MSKCC_16-328_1_03230_20080220_l',
 'MSKCC_16-328_1_08817_20060127_l',
 'MSKCC_16-328_1_05817_20060215_r']

SCANIDS = list(set(SCANIDS) - set(NaNs))


#%% Remove Converts


#%% Remove Reader Study Scans




#%%
import sys
sys.path.append('/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/code/')
from utils import load_and_preprocess


# ONLY MALIGNANTS 
# SCANIDS = segmented


# ONLY BENIGNS
# SCANIDS = list(set(SCANIDS) - set(segmented))

TOT = len(SCANIDS)

NaNs = []
noslope1 = []




for SUBJECT_INDEX in range(TOT): #TOT
    print(SUBJECT_INDEX, TOT)
    FOLDER = 'BENIGN'
    scanID = SCANIDS[SUBJECT_INDEX]
    
    patient = scanID[:20]
    exam = scanID[:29]
    side = 'right'
    contra_side = 'left'
    if scanID[-1] == 'l': 
        side = 'left'
        contra_side = 'right'
        
        
    if len(RISK.loc[RISK['Scan_ID'] == scanID, 'Pathology']) == 0:
        print('No pathology, weirdly..')
        
        if scanID in labels_sagittal.keys():
            print('but found in labels_sagittal..')
            
            label = labels_sagittal[scanID]
            
            if label == 0:
                pathology = 'Benign'
            elif label == 1:
                pathology = 'Malignant'
                

        else:
            continue


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
        
    #-------------------- Get pathology. If cancer, get slice number, else random slice ---------------------------------
        'Dont do this anymore. Load prepared sheet with slice numbers.'

 
    if pathology == 'Malignant':
        FOLDER = 'MALIGNANT'
        if scanID in segmented_scanid:
    

        #if segmentation_available: # What do I do with malignants that have no segmentation? These are not part of the training set of this pipeline!
            segmentation_path = MASTER.loc[MASTER['Scan_ID'] == scanID, 'Segmentation_Path'].values[0]    
            groundtruth = nib.load(segmentation_path).get_data()
            selected_slices = list(set(np.where(groundtruth > 0)[0]))        
        else:
            print('Unsegmented malignant, skip')
            continue
        
    if pathology == 'Benign': 
        random = list(np.random.choice(list(range(2,X.shape[0]//2)) + list(np.arange(X.shape[0]//2+1,X.shape[0]-2)), size=N_BENIGN_SLICES-1, replace=False))
        selected_slices = random + [X.shape[0]//2]
        
    #-------------------- Store Slices  ---------------------------------
    print("####### STORE ##############")


    for i in range(len(selected_slices)):
        #if not os.path.exists(OUTPUT_FOLDER + '/{}_{}.npy'.format(scanID, selected_slices[i])):
        np.save(f'{OUTPUT_FOLDER}/{FOLDER}/{scanID}_{selected_slices[i]}.npy', X[selected_slices[i]-NEIGHBOR_SLICES:selected_slices[i]+NEIGHBOR_SLICES+1])
             
        
