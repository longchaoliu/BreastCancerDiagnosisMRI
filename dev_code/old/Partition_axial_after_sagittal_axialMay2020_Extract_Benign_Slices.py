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

#benign_training = benign_training[6000:]

OUTPUT_PATH = '/media/SD/Axial_Slices/'

already_done = os.listdir(OUTPUT_PATH + 'X/')

exams_already_done = [x[:29] for x in already_done]

benign_training = benign_training.loc[ ~ benign_training['Exam'].isin(exams_already_done)]

done_exams_on_the_fly = []

TOT = len(benign_training['Exam'].unique())

resolutions = pd.DataFrame(columns=['scan','res_in','res_out'])

for row in benign_training.iterrows():
#    print(len(done_exams_on_the_fly))
#    break
    exam = row[1]['Exam']
#    if exam in done_exams_on_the_fly:
#        print('already done. skip')
#        continue
    scanID = row[1]['scanID']
    

    sides = ['right']
    if scanID[-1] == 'l':
        sides = ['left']
       
    if len(benign_training.loc[benign_training['Exam'] == exam, 'pathology'].values) == 2:
        if (benign_training.loc[benign_training['Exam'] == exam, 'pathology'].values[0] == 'Benign') and (benign_training.loc[benign_training['Exam'] == exam, 'pathology'].values[1] == 'Benign'):
            sides = ['right', 'left']
        
    hdr = nib.load(AXIAL_PATH + exam + '/T1_axial_02_01.nii.gz')
    
    #%%
    print(scanID)
    t1post = hdr.get_data()
    resolution = np.diag(hdr.affine)

    if resolution[0] > 0.5: # 'THIS IS ALREADY WRONG! I DONT CARE ABOUT FIRST RES'
        output_res =  (resolution[0], resolution[1]/2, resolution[2]/ (resolution[2]/0.33) )  #'THIS SEEMS CORRECT THOUGH???'
        output_shape = (t1post.shape[0], t1post.shape[1]*2, int(t1post.shape[2]* (resolution[2]/0.33) ))
    
    resolutions = resolutions.append({'scan':scanID,'res_in':resolution, 'res_out':output_res},ignore_index=True)
    
    
    resolutions.to_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/DATA/Axial_Resolutions.csv')
    
    
    set([x[0] for x in resolutions['res_in'].values])

    set([x[0] for x in resolutions['res_out'].values])
    set([x[1] for x in resolutions['res_out'].values])
    set([x[2] for x in resolutions['res_out'].values])

    resolutions['res_in_X'] = [x[0] for x in resolutions['res_in'].values]
    resolutions['res_in_Y'] = [x[1] for x in resolutions['res_in'].values]
    resolutions['res_in_Z'] = [x[2] for x in resolutions['res_in'].values]

    resolutions['res_out_X'] = [x[0] for x in resolutions['res_out'].values]
    resolutions['res_out_Y'] = [x[1] for x in resolutions['res_out'].values]
    resolutions['res_out_Z'] = [x[2] for x in resolutions['res_out'].values]
    
    resolutions.loc[resolutions['res_in_X'] < 0.5, 'res_out']

    resolutions.loc[resolutions['res_out_Y'] > 0.4, 'res_in']

    #%%
    
    
    t1post = hdr.get_data()
    slope1 = nib.load(AXIAL_PATH + exam + '/T1_axial_slope1.nii.gz').get_data()
    slope2 = nib.load(AXIAL_PATH + exam + '/T1_axial_slope2.nii.gz').get_data()

    t1post = t1post/float(40)
    slope1 = slope1/float(0.3)
    slope2 = slope2/float(0.12)           

    if t1post.shape[1] < 256:
        print('image too small. Pad')
        border = 256 - t1post.shape[1]
        t1post = np.pad(t1post, ((0,0),(0,border),(0,0)), 'minimum')
        slope1 = np.pad(slope1, ((0,0),(0,border),(0,0)), 'minimum')
        slope2 = np.pad(slope2, ((0,0),(0,border),(0,0)), 'minimum')
        
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
    
#    t1post.shape
#       
#    t1post[150].shape
#    
#    '[sagittal ,coronal, axial]'
#    '[0.6, 0.6, 1.1]'
#    
#    '''
#    
#    I dont care about the first resolution, that is between sagittal slices
#    
#    I care about the 2nd and 3rd, I want those close to 0.3, 0.3
#    
#    '''
#    
#    plt.imshow(t1post[150])
#    plt.imshow(t1post[:,150]) 
#    plt.imshow(t1post[:,:,150])
    
    projection_1d = np.max(np.max(t1post, 0),1)

    breast_end = np.argmin(np.diff(projection_1d[np.arange(0,len(projection_1d),5)]))*5
    breast_end = breast_end + 10 # add some border
    breast_end = np.max([breast_end, 256]) # if breast is small, just crop to 256
        
    t1post = t1post[:, (breast_end-256):breast_end, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    slope1 = slope1[:, (breast_end-256):breast_end, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    slope2 = slope2[:, (breast_end-256):breast_end, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    
    if resolution[0] > 0.5: # 'THIS IS ALREADY WRONG! I DONT CARE ABOUT FIRST RES'
        output_res =  (resolution[0], resolution[1]/2, resolution[2]/ (resolution[2]/0.33) )  #'THIS SEEMS CORRECT THOUGH???'
        output_shape = (t1post.shape[0], t1post.shape[1]*2, int(t1post.shape[2]* (resolution[2]/0.33) ))
    #        output_shape = (512*2, 512*2, int(192*3.3))
    
    else:
        print('new res, inspect.')
        continue   
    
    # RESIZE to match resolutions.  I need final resolution: (whatever, 0.3, 0.3)

    t1post = resize(t1post, output_shape=output_shape, preserve_range=True, anti_aliasing=False)
    slope1 = resize(slope1, output_shape=output_shape, preserve_range=True, anti_aliasing=False)
    slope2 = resize(slope2, output_shape=output_shape, preserve_range=True, anti_aliasing=False)
    
    if t1post.shape[2] < 512:
        border = 512 - t1post.shape[2]
        t1post = np.pad(t1post, ((0,0),(0,0),(0,border)), 'minimum')
        slope1 = np.pad(slope1, ((0,0),(0,0),(0,border)), 'minimum')
        slope2 = np.pad(slope2, ((0,0),(0,0),(0,border)), 'minimum')
        
    else:    
        t1post = t1post[:, :, 0:512]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
        slope1 = slope1[:, :, 0:512]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
        slope2 = slope2[:, :, 0:512]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    

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