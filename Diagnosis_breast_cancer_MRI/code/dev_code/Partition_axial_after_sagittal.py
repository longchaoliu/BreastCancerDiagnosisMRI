#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:38:16 2023

@author: deeperthought
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

'''
Prepare axial set!!

Train/val/test

Respect patient partition in sagittal.

Data in: /home/deeperthought/kirby_MSK/alignedNiiAxial-Nov2019/

'''

DATA_PATH = '/home/deeperthought/kirby_MSK/alignedNiiAxial-Nov2019/'

REDCAP = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/REDCAP/2023/REDCAP_EZ.csv') 


exams = os.listdir(DATA_PATH)

len(exams)

REDCAP = REDCAP.loc[REDCAP['Exam'].isin(exams)]

REDCAP['bi_rads_assessment_for_stu'].value_counts()

REDCAP.loc[REDCAP['bi_rads_assessment_for_stu'] < '4', 'overall_study_assessment'].value_counts()
REDCAP.loc[REDCAP['bi_rads_assessment_for_stu'] >= '4', 'overall_study_assessment'].value_counts()

REDCAP.loc[REDCAP['bi_rads_assessment_for_stu'] < '4', 'right_breast_tumor_status'].value_counts()
REDCAP.loc[REDCAP['bi_rads_assessment_for_stu'] < '4', 'left_breast_tumor_status'].value_counts()

REDCAP.loc[REDCAP['bi_rads_assessment_for_stu'] >= '4', 'right_breast_tumor_status'].value_counts()
REDCAP.loc[REDCAP['bi_rads_assessment_for_stu'] >= '4', 'left_breast_tumor_status'].value_counts()

# GATHER BENIGNS
REDCAP_B123 = REDCAP.loc[REDCAP['bi_rads_assessment_for_stu'] < '4']
REDCAP_B123 = REDCAP_B123.loc[(REDCAP_B123['right_breast_tumor_status'] != 'Malignant') * (REDCAP_B123['left_breast_tumor_status'] != 'Malignant')]
REDCAP_benigns = REDCAP_B123.loc[REDCAP_B123['true_negative_mri_no_cance'] == 'Yes']

# GATHER MALIGNANTS
REDCAP_B45 = REDCAP.loc[REDCAP['bi_rads_assessment_for_stu'] > '3']
REDCAP_malignants = REDCAP_B45.loc[(REDCAP_B45['right_breast_tumor_status'] == 'Malignant') + (REDCAP_B45['left_breast_tumor_status'] == 'Malignant')]


REDCAP_malignants.index = np.arange(0,len(REDCAP_malignants)*2, 2)
REDCAP_benigns.index = np.arange(1,len(REDCAP_benigns)*2, 2)


AXIAL_SCANS = pd.concat([REDCAP_malignants,REDCAP_benigns])
AXIAL_SCANS.sort_index(inplace=True)

len(REDCAP_malignants), len(REDCAP_benigns)

benign_subjects_axial = list(set(REDCAP_benigns['Subject_ID']))
malignant_subjects_axial = list(set(REDCAP_malignants['Subject_ID']))

len(malignant_subjects_axial), len(benign_subjects_axial)

ALL_AXIAL_SUBJECTS = malignant_subjects_axial + benign_subjects_axial

#%% sagittal partitions

PRESAVED_DATA = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/DATA/Data.npy'
PRESAVED_LABELS = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/DATA/Labels.npy'
PRESAVED_CLINICAL_INFO = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/DATA/Clinical_Data_Train_Val.csv'

partition = np.load(PRESAVED_DATA, allow_pickle=True).item()
labels = np.load(PRESAVED_LABELS, allow_pickle=True).item()
clinical_features = pd.read_csv(PRESAVED_CLINICAL_INFO)


sagital_subjects_train = list(set([ x[:20] for x in partition['train']]))
sagital_subjects_train.extend( list(set([ x[:20] for x in partition['validation']])))

sagital_subjects_test = list(set([ x[:20] for x in partition['test']]))


# Do the subjects in sagittal and axial overlap fully? Or do I have some unique ones?

ALL_SAGITTAL_SUBJECTS = sagital_subjects_train + sagital_subjects_test

len(ALL_AXIAL_SUBJECTS)
len(set(ALL_AXIAL_SUBJECTS).intersection(set(ALL_SAGITTAL_SUBJECTS)))

unique_axial_subjects = list(set(ALL_AXIAL_SUBJECTS) - set(ALL_SAGITTAL_SUBJECTS))


#%%

axial_train_subjects = [x for x in ALL_AXIAL_SUBJECTS if x in sagital_subjects_train]
axial_test_subjects = [x for x in ALL_AXIAL_SUBJECTS if x in sagital_subjects_test]
axial_free_subjects = [x for x in ALL_AXIAL_SUBJECTS if x not in sagital_subjects_test and x not in sagital_subjects_train]

len(axial_train_subjects), len(axial_test_subjects), len(axial_free_subjects)

set(axial_train_subjects).intersection(set(axial_test_subjects))
set(axial_train_subjects).intersection(set(axial_free_subjects))
set(axial_test_subjects).intersection(set(axial_free_subjects))


REDCAP_malignants.loc[REDCAP_malignants['Subject_ID'].isin(axial_test_subjects)]
REDCAP_malignants.loc[REDCAP_malignants['Subject_ID'].isin(axial_train_subjects)]
REDCAP_malignants.loc[REDCAP_malignants['Subject_ID'].isin(axial_free_subjects)]

REDCAP_malignants.loc[REDCAP_malignants['Subject_ID'].isin(unique_axial_subjects)]


#%% Need to link the subjects to the actual available scans. Also need the segmentations of the malignants for training!

'''
Exams are named with protocol ID: MSKCC_16-328_1_  and RIA_19_093

Folder with Axials has naming convention with MSKCC_16-328_1_

Need to map these 

Or, maybe I dont have any RIA_19 in that folder??


'''

#%%

REDCAP_raw = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/REDCAP/2023/20425MachineLearning_DATA_2023-01-24_1046.csv')

groups = REDCAP_raw.groupby('mrn_id')

REDCAP_raw['MSKCC_16_ID'] = ''

for pat in groups:
    print(pat[0])
    
    patient_IDs = [x for x in pat[1]['x_nat_mri_anonymized_mrn'].values if len(x)>1]

    mskcc_16_ID = list(set([x for x in patient_IDs if x.startswith('MSKCC')]))
    if len(mskcc_16_ID) > 0:
        
        REDCAP_raw.loc[REDCAP_raw['mrn_id'] == pat[0], 'MSKCC_16_ID'] = mskcc_16_ID
    else:
        REDCAP_raw.loc[REDCAP_raw['mrn_id'] == pat[0], 'MSKCC_16_ID'] = 'Unknown'
        
    
#%%

segmented_axials = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/axial/150segLizConfirmed.csv')

#REDCAP_raw = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/REDCAP/2023/20425MachineLearning_DATA_2023-01-24_1046.csv')

REDCAP_raw[['mri_date', 'x_nat_mri_anonymized_mrn']] = REDCAP_raw[['mri_date','x_nat_mri_anonymized_mrn']].fillna('')
REDCAP_raw['Exam'] = REDCAP_raw.apply(lambda x : x['x_nat_mri_anonymized_mrn'] + '_' + x['mri_date'].replace('-',''), axis=1)

REDCAP_raw['Exam'] = REDCAP_raw.apply(lambda x : x['MSKCC_16_ID'] + '_' + x['mri_date'].replace('-',''), axis=1)


REDCAP_raw.loc[REDCAP_raw['Exam'].isin(segmented_axials['DE_ID'].values)]

segmented_axials['DE_ID']
REDCAP.loc[REDCAP['Exam'].isin(segmented_axials['DE_ID'])]

segmented_cancer_axials =  # these go into the training set

unsegmented_cancer_axials = # these go into the test set.


#%%

AXIAL_PATH = '/home/deeperthought/kirbyPRO/alignedNiiAxial-May2020-cropped-normed/'




