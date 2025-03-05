#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:32:55 2023

@author: deeperthought
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize

AXIAL_PATH = '/home/deeperthought/kirbyPRO/alignedNiiAxial-May2020-cropped-normed/'

segmented_axials = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/axial/150segLizConfirmed.csv')

Segmented_axials_path1 = '/home/deeperthought/kirby_MSK/250caseSegExtd-May2020-cropped/'

Segmented_axials_path2 = '/home/deeperthought/kirby_MSK/segExtd-Mar2021/'

REDCAP = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/REDCAP/2023/demographics/20425MachineLearning_DATA_2023-05-11_1234.csv')
REDCAP[['mri_date', 'x_nat_mri_anonymized_mrn']] = REDCAP[['mri_date','x_nat_mri_anonymized_mrn']].fillna('')
REDCAP['Exam'] = REDCAP.apply(lambda x : x['x_nat_mri_anonymized_mrn'] + '_' + x['mri_date'].replace('-',''), axis=1)

axial_exams = os.listdir(AXIAL_PATH)



PATHOLOGY_AXIAL = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/REDCAP/2023/Axials_pathology_assigned_partition.csv')

benign_training = PATHOLOGY_AXIAL.loc[(PATHOLOGY_AXIAL['partition'] == 'Training')*(PATHOLOGY_AXIAL['pathology'] == 'Benign')]

benign_training = benign_training[6000:]

OUTPUT_PATH = '/media/SD/Axial_Slices/'

already_done = os.listdir(OUTPUT_PATH + 'X/')

exams_already_done = [x[:29] for x in already_done]

benign_training = benign_training.loc[ ~ benign_training['Exam'].isin(exams_already_done)]

done_exams_on_the_fly = []

TOT = len(benign_training['Exam'].unique())

resolutions = pd.DataFrame(columns=['scan','res_in','res_out'])

for row in benign_training.iterrows():
    print(len(done_exams_on_the_fly))
    #break
    exam = row[1]['Exam']
    if exam in done_exams_on_the_fly:
        print('already done. skip')
        continue
    scanID = row[1]['scanID']
    

    sides = ['right']
    if scanID[-1] == 'l':
        sides = ['left']
       
    if len(benign_training.loc[benign_training['Exam'] == exam, 'pathology'].values) == 2:
        if (benign_training.loc[benign_training['Exam'] == exam, 'pathology'].values[0] == 'Benign') and (benign_training.loc[benign_training['Exam'] == exam, 'pathology'].values[1] == 'Benign'):
            sides = ['right', 'left']
        
    hdr = nib.load(AXIAL_PATH + exam + '/T1_axial_02_01.nii.gz')
    
    
    t1post = hdr.get_data()
    slope1 = nib.load(AXIAL_PATH + exam + '/T1_axial_slope1.nii.gz').get_data()
    slope2 = nib.load(AXIAL_PATH + exam + '/T1_axial_slope2.nii.gz').get_data()

    t1post = t1post/float(40)
    slope1 = slope1/float(0.3)
    slope2 = slope2/float(0.12)           

#    plt.imshow(t1post[150])

    if not np.all(np.isfinite(t1post)):
        print('Nans! skip')
        continue
    if not np.all(np.isfinite(slope1)):
        print('Nans! skip')
        continue
    if not np.all(np.isfinite(slope2)):
        print('Nans! skip')
        continue
    
    resolution = np.diag(hdr.affine)
    
    resolution_factor_X = resolution[1]/0.5
    resolution_factor_Y = resolution[2]/0.5
    
    output_res =  (resolution[0], resolution[1]/resolution_factor_X, resolution[2]/resolution_factor_Y)  #'THIS SEEMS CORRECT THOUGH???'
    output_shape = (t1post.shape[0], int(t1post.shape[1]*resolution_factor_X), int(t1post.shape[2]*resolution_factor_Y))

    t1post = resize(t1post, output_shape=output_shape, preserve_range=True, anti_aliasing=False)
    slope1 = resize(slope1, output_shape=output_shape, preserve_range=True, anti_aliasing=False)
    slope2 = resize(slope2, output_shape=output_shape, preserve_range=True, anti_aliasing=False)


    if t1post.shape[1] < 512:
        print('image too small. Pad')
        border = 512 - t1post.shape[1]

        t1post = np.pad(t1post, ((0,0),(0,border),(0,0)), 'constant')
        slope1 = np.pad(slope1, ((0,0),(0,border),(0,0)), 'constant')
        slope2 = np.pad(slope2, ((0,0),(0,border),(0,0)), 'constant')
        
    elif t1post.shape[1] > 512:    
        
        length = t1post.shape[1]
        
        extra = length - 512
        
        first_half = extra/2
        second_half = extra - first_half  
                
        t1post = t1post[:, first_half:-second_half,:]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
        slope1 = slope1[:, first_half:-second_half,:]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
        slope2 = slope2[:, first_half:-second_half,:]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    
        
    
    if t1post.shape[2] < 512:
        border = 512 - t1post.shape[2]
        first_half = border/2
        second_half = border - first_half  
                
        t1post = np.pad(t1post, ((0,0),(0,0),(first_half,second_half)), 'constant')
        slope1 = np.pad(slope1, ((0,0),(0,0),(first_half,second_half)), 'constant')
        slope2 = np.pad(slope2, ((0,0),(0,0),(first_half,second_half)), 'constant')
        
    elif t1post.shape[2] > 512:    
        
        length = t1post.shape[2]
        
        extra = length - 512
        
        first_half = extra/2
        second_half = extra - first_half  
                
        t1post = t1post[:, :, first_half:-second_half]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
        slope1 = slope1[:, :, first_half:-second_half]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
        slope2 = slope2[:, :, first_half:-second_half]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    

    assert (t1post.shape[1],t1post.shape[2]) == (512,512), 'Something went wrong with dimensions. NOT 512 x 512!!'
    
    number_slices = t1post.shape[0]
    
    left_breast = np.flip(t1post[:number_slices/2], axis=0)
    left_breast_slope1 = np.flip(slope1[:number_slices/2], axis=0)
    left_breast_slope2 = np.flip(slope2[:number_slices/2], axis=0)

    right_breast = t1post[number_slices/2:]
    right_breast_slope1 = slope1[number_slices/2:]
    right_breast_slope2 = slope2[number_slices/2:]
    
    for side in sides:
        if side == 'right':
            contra_side = 'left'            
                
            middle_slice = int(right_breast.shape[0]/2)
            random_slice = list(np.random.choice(range(0,middle_slice) + range(middle_slice+1,right_breast.shape[0]-1), size=2, replace=False))
            sagittal_slices_of_interest = [middle_slice] + random_slice
        
            for sl in sagittal_slices_of_interest:
                X = np.array([np.stack([right_breast[sl], right_breast_slope1[sl], right_breast_slope2[sl]], axis=-1)])
                
                X_contra = np.array([left_breast[sl]])
                
                np.save(OUTPUT_PATH  + 'X/' + exam + '_' + side[0] + '_' + str(sl) + '.npy', X)
                np.save(OUTPUT_PATH + 'Contralateral/' + exam + '_' + contra_side[0] + '_' + str(sl) + '.npy', X_contra)

 
    
        if side == 'left':
            contra_side = 'right'
            
            middle_slice = int(right_breast.shape[0]/2)
            random_slice = list(np.random.choice(range(0,middle_slice) + range(middle_slice+1,right_breast.shape[0]-1), size=2, replace=False))
            sagittal_slices_of_interest = [middle_slice] + random_slice
        
            for sl in sagittal_slices_of_interest:
                
                X = np.array([np.stack([left_breast[sl], left_breast_slope1[sl], left_breast_slope2[sl]], axis=-1)])
                
                X_contra = np.array([right_breast[sl]])
                
                np.save(OUTPUT_PATH  + 'X/' + exam + '_' + side[0] + '_' + str(sl) + '.npy', X)
                np.save(OUTPUT_PATH + 'Contralateral/' + exam + '_' + contra_side[0] + '_' + str(sl) + '.npy', X_contra)

    done_exams_on_the_fly.append(exam)