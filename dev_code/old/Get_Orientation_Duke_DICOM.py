#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:48:30 2024

@author: deeperthought
"""

   
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.transform import resize



DATA_PATH = '/home/deeperthought/kirby_MSK/dukePublicData/alignedNii-normed/'      

subj = 'Breast_MRI_001'; selected_slice=61
# subj = 'Breast_MRI_005'; selected_slice=75
# subj = 'Breast_MRI_010'; selected_slice=34
# subj = 'Breast_MRI_016'; selected_slice=91
# subj = 'Breast_MRI_019'; selected_slice=75
# subj = 'Breast_MRI_023'; selected_slice=88


for subj in ['Breast_MRI_001','Breast_MRI_002','Breast_MRI_003','Breast_MRI_004']:
    print('\n')
    print(subj)
    DATA = [DATA_PATH + subj + '/T1_axial_01.nii.gz', 
            DATA_PATH + subj + '/T1_axial_02.nii.gz',
            DATA_PATH + subj + '/T1_axial_slope1.nii.gz',
            DATA_PATH + subj + '/T1_axial_slope2.nii.gz']
    
    
    # seg = nib.load(f'/home/deeperthought/kirby_MSK/dukePublicData/manSegRes/Segmentation_Results-Lars50cases (1)/Segmentation_Results/segmentation/{subj}_T1post.nii.gz').get_fdata()
    
    t1post = nib.load(DATA[1])
    
    
    np.array([t1post.header['srow_x'],t1post.header['srow_y'],t1post.header['srow_z']])
    
    print(t1post.header['sform_code'])
    print(np.diag(np.array([t1post.header['srow_x'],t1post.header['srow_y'],t1post.header['srow_z']])))


    t1post.header['slice_start']

#%%

import pydicom


dicom_path = '/home/deeperthought/kirby/homes/lukas/Duke_Data/manifest-1607053360376/Duke-Breast-Cancer-MRI/'

subjects = [x for x in os.listdir(dicom_path) if x.startswith('Breast')]

subjects.sort()

# df = pd.DataFrame(columns=['subj','sequence','z1','z2','orientation'])
df = pd.DataFrame(columns=['subj','orientation'])

for subj in subjects:
    print(subj)
    fold1 = os.listdir(dicom_path + subj)
    sequences = os.listdir(dicom_path + subj + '/' + fold1[0])
    for seq in sequences[0:1]:
        dcm_files = os.listdir(dicom_path + subj + '/' + fold1[0] + '/' + seq)
        
        dcm1 = dicom_path + subj + '/' + fold1[0] + '/' + seq + '/' + dcm_files[0]
        dcm2 = dicom_path + subj + '/' + fold1[0] + '/' + seq + '/' + dcm_files[1]
        
        z1 = pydicom.filereader.dcmread(dcm1).ImagePositionPatient[-1]
        z2 = pydicom.filereader.dcmread(dcm2).ImagePositionPatient[-1]
        
        ori = pydicom.filereader.dcmread(dcm1).ImageOrientationPatient

        # df = df.append({'subj':subj, 'sequence':seq, 'z1':z1, 'z2':z2, 'orientation':round(ori[0])}, ignore_index=True)

        df = df.append({'subj':subj, 'orientation':round(ori[0])}, ignore_index=True)



df = df.sort_values('subj')


df.to_csv('/home/deeperthought/kirby/homes/lukas/Duke_Data/manifest-1607053360376/dicom_ImageOrientation.csv')



pydicom.filereader.dcmread(dcm1)


lukas_flip = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Duke_predictions/Annotation_Boxes.csv')

df = pd.merge(lukas_flip[['Patient ID','Flip']], df, left_on='Patient ID', right_on='subj')

plt.scatter(df['orientation'],df['Flip'])

import sklearn.metrics

df = df.loc[df['Flip'].isin([0,1])]

df['Flip'].value_counts()
df['orientation'].value_counts()

df.loc[df['orientation'] == -1, 'orientation'] = 0 

sklearn.metrics.confusion_matrix(np.array(df['orientation'].values, dtype='int'), np.array(df['Flip'].values, dtype='int'))


sklearn.metrics.classification_report(np.array(df['orientation'].values, dtype='int'), np.array(df['Flip'].values, dtype='int'))
