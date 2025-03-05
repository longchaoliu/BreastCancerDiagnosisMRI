#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 16:24:41 2023

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

#%% Segmented axials

axial_segmentations1 = [Segmented_axials_path1 + x for x in os.listdir(Segmented_axials_path1)]
axial_segmentations2 = [Segmented_axials_path2 + x for x in os.listdir(Segmented_axials_path2)]

segmented_exams = [x.split('/')[-1].split('_seg')[0] for x in axial_segmentations1]
segmented_exams.extend( [x.split('/')[-1].split('_seg')[0] for x in axial_segmentations2])

segmented_axials.loc[segmented_axials['DE_ID'].isin(axial_exams)]


'''

some segmented exams have a full identifier (MSKCC_16.. or RIA_19-093... )

BUT some have only what I guess is an MRN number
e.g:
    '31779416'
    '35591352mrn'
    
This might not be enough to identify the exact exam. 
Unless those patients have only a single malignant exam..




Cant match those with any available MRN from REDCAP.

These are 101 segmentations, vs 645 with a clear exam, so I will just not use them!


'''
#segmented_exams_with_clear_ID = [x for x in segmented_exams if x.startswith('RIA') or x.startswith('MSK')]
#segmented_exams_with_UNclear_ID = [x for x in segmented_exams if x not in segmented_exams_with_clear_ID]
#len(segmented_exams_with_clear_ID), len(segmented_exams_with_UNclear_ID)
#mrn_segmented_exam_UNclear_ID = [x.replace('mrn','') for x in segmented_exams_with_UNclear_ID]
#REDCAP.loc[REDCAP['mrn_id'].isin(mrn_segmented_exam_UNclear_ID)]  # --> No MRNs with these numbers!! Unknown what these 101 segmentations are

segmented_exams = [x for x in axial_segmentations1 if x.split('/')[-1].startswith('RIA') or x.split('/')[-1].startswith('MSK')]
segmented_exams.extend([x for x in axial_segmentations2 if x.split('/')[-1].startswith('RIA') or x.split('/')[-1].startswith('MSK')])

segmented_exam_IDs = [x.split('/')[-1].split('_seg')[0] for x in segmented_exams]

'From 645 segmentations, I have REDCAP entries for 641'
REDCAP.loc[REDCAP['Exam'].isin(segmented_exam_IDs)]

segmented_exam_IDs = [x for x in segmented_exam_IDs if x in REDCAP['Exam'].values]

'From those 640 segmentations with REDCAP entries, I have 644 available images in ' + AXIAL_PATH

segmented_axials_with_REDCAP_and_images = list(set(axial_exams).intersection(set(segmented_exam_IDs)))

'I have 644 segmented axials to work with'

len(segmented_axials_with_REDCAP_and_images)

REDCAP.loc[REDCAP['Exam'].isin(segmented_axials_with_REDCAP_and_images), ['overall_study_assessment','bi_rads_assessment_for_stu','right_breast_tumor_status','left_breast_tumor_status']]

REDCAP.loc[REDCAP['Exam'].isin(segmented_axials_with_REDCAP_and_images), 'overall_study_assessment'].value_counts()

REDCAP.loc[REDCAP['Exam'].isin(segmented_axials_with_REDCAP_and_images), 'bi_rads_assessment_for_stu'].value_counts()

RECAP_segmented_uncertain_pathology = REDCAP.loc[(REDCAP['Exam'].isin(segmented_axials_with_REDCAP_and_images)) * (REDCAP['right_breast_tumor_status'] != 'Malignant') * (REDCAP['left_breast_tumor_status'] != 'Malignant')  ]

RECAP_segmented_uncertain_pathology[['right_breast_tumor_status','left_breast_tumor_status']]

'There are 100 with uncertain pathology: Either both breasts Benign, or Cancelled, or Unknown, or NaN'

segmented_axials_with_REDCAP_and_images = [x for x in segmented_axials_with_REDCAP_and_images if x not in RECAP_segmented_uncertain_pathology['Exam'].values]

len(segmented_axials_with_REDCAP_and_images)

print('Remain with 540 segmented axials, that have a REDCAP entry, with Malignant pathology, with available images')

#%%  Plot for QA:
#
#OUTPUT_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/axial_segmentations_images/'
#
#available_segmentation = []
#for exam in segmented_axials_with_REDCAP_and_images:
#
#    print exam
#    
#    pathology_summary = REDCAP.loc[REDCAP['Exam'] == exam, ['overall_study_assessment','bi_rads_assessment_for_stu','right_breast_tumor_status','left_breast_tumor_status']].values[0]
#
#    print(pathology_summary)
#    
#    PATHOLOGY = pathology_summary[0]
#    BIRADS = pathology_summary[1]
#    PATHOLOGY_RIGHT_BREAST = pathology_summary[2]
#    PATHOLOGY_LEFT_BREAST = pathology_summary[3]
#    
#    sides = []
#    if PATHOLOGY_RIGHT_BREAST == 'Malignant':
#        sides.append('right')
#    if PATHOLOGY_LEFT_BREAST == 'Malignant':
#        sides.append('left')    
#    
#    img = nib.load(AXIAL_PATH + exam + '/T1_axial_02_01.nii.gz').get_data()
#    slope1 = nib.load(AXIAL_PATH + exam + '/T1_axial_slope1.nii.gz').get_data()
#    slope2 = nib.load(AXIAL_PATH + exam + '/T1_axial_slope2.nii.gz').get_data()
#    
#    seg_path = [x for x in segmented_exams if x.split('/')[-1].startswith(exam)][0]
#    seg = nib.load(seg_path).get_data()
#
#    assert seg.shape == img.shape, 'unequal shapes!'
#      
#
#    
#    # split into right and left breast. In the middle..
#    number_slices = img.shape[0]
#    
#    left_breast = np.flip(img[:number_slices/2], axis=0)
#    left_breast_segmentation = np.flip(seg[:number_slices/2], axis=0)
#    left_breast_slope1 = np.flip(slope1[:number_slices/2], axis=0)
#    left_breast_slope2 = np.flip(slope2[:number_slices/2], axis=0)
#
#    right_breast = img[number_slices/2:]
#    right_breast_segmentation = seg[number_slices/2:]
#    right_breast_slope1 = slope1[number_slices/2:]
#    right_breast_slope2 = slope2[number_slices/2:]
#    
#    for side in sides:
#        if side == 'right':
#            
#            seg_coords = np.argwhere(right_breast_segmentation > 0)
#            sagittal_slices = list(set(seg_coords[:,0])) 
#            segmented_area_per_sag_slice = np.array([np.sum(right_breast_segmentation[x]) for x in sagittal_slices]) 
#            index_of_slices_with_larger_segmented_area = (np.argwhere(segmented_area_per_sag_slice > 150)).flatten() 
#            sagittal_slices_of_interest = [sagittal_slices[x] for x in index_of_slices_with_larger_segmented_area]   
#
#                            
#            fig_index = 1
#            for sl in sagittal_slices_of_interest:
#                plt.figure(fig_index, figsize=(10,10))
#                plt.subplot(1,5,1); plt.imshow(right_breast[sl]); plt.xticks([]); plt.yticks([])# + 5*right_breast_segmentation[sl])
#                plt.subplot(1,5,2); plt.imshow(right_breast_segmentation[sl]); plt.xticks([]); plt.yticks([])
#                plt.subplot(1,5,3); plt.imshow(right_breast_slope1[sl]); plt.xticks([]); plt.yticks([])# + 5*left_breast_segmentation[sl])
#                plt.subplot(1,5,4); plt.imshow(right_breast_slope2[sl]); plt.xticks([]); plt.yticks([])# + 5*left_breast_segmentation[sl])
#                plt.subplot(1,5,5); plt.imshow(left_breast[sl]); plt.xticks([]); plt.yticks([])# + 5*left_breast_segmentation[sl])
#
#                fig_index += 1
#                plt.savefig(OUTPUT_PATH + exam + '_' + side + '_slice_' + str(sl) + '.png')
#                plt.close()
#                
#                'Here I can store the sliuces as np array in kirbyPRO'
#                'but first do correct preprocessing:'
#                'resizing, intensity normalization (if required)'
#                
#        if side == 'left':
#           
#            seg_coords = np.argwhere(left_breast_segmentation > 0)
#            sagittal_slices = list(set(seg_coords[:,0])) 
#            segmented_area_per_sag_slice = np.array([np.sum(left_breast_segmentation[x]) for x in sagittal_slices]) 
#            index_of_slices_with_larger_segmented_area = (np.argwhere(segmented_area_per_sag_slice > 150)).flatten() 
#            sagittal_slices_of_interest = [sagittal_slices[x] for x in index_of_slices_with_larger_segmented_area]   
#                
#            fig_index = 1
#            for sl in sagittal_slices_of_interest:
#                plt.figure(fig_index, figsize=(10,10))
#                plt.subplot(1,5,1); plt.imshow(left_breast[sl]); plt.xticks([]); plt.yticks([])# + 5*right_breast_segmentation[sl])
#                plt.subplot(1,5,2); plt.imshow(left_breast_segmentation[sl]); plt.xticks([]); plt.yticks([])
#                plt.subplot(1,5,3); plt.imshow(left_breast_slope1[sl]); plt.xticks([]); plt.yticks([])# + 5*left_breast_segmentation[sl])
#                plt.subplot(1,5,4); plt.imshow(left_breast_slope2[sl]); plt.xticks([]); plt.yticks([])# + 5*left_breast_segmentation[sl])
#                plt.subplot(1,5,5); plt.imshow(right_breast[sl]); plt.xticks([]); plt.yticks([])# + 5*left_breast_segmentation[sl])
#
#                fig_index += 1
#                plt.savefig(OUTPUT_PATH + exam + '_' + side + '_slice_' + str(sl) + '.png')
#                plt.close()
#
#    

 
    
'''
Just need locations of segmentations so I can grab sagittal slices
'''

#%% Save segmented slices:

blacklisted = pd.read_csv('/home/deeperthought/kirbyPRO/Blacklisted_Axial_Segmented_Images_alignedNiiAxial-May2020-cropped-normed.csv')

blacklisted['exam'] = [ x[:-2] for x in blacklisted['scanID'].values]

len(segmented_axials_with_REDCAP_and_images)
# Remove blacklisted from visual QA
segmented_axials_with_REDCAP_and_images = [x for x in segmented_axials_with_REDCAP_and_images if x not in blacklisted['exam'].values]

print('After QA, we have 482 exams.')

OUTPUT_PATH = '/media/SD/Axial_Slices/'

already_done = os.listdir(OUTPUT_PATH + 'X/')

exams_already_done = [x[:29] for x in already_done]

segmented_axials_with_REDCAP_and_images = [x for x in segmented_axials_with_REDCAP_and_images if x not in exams_already_done]

for exam in segmented_axials_with_REDCAP_and_images:

    print exam
    
    pathology_summary = REDCAP.loc[REDCAP['Exam'] == exam, ['overall_study_assessment','bi_rads_assessment_for_stu','right_breast_tumor_status','left_breast_tumor_status']].values[0]

    print(pathology_summary)
    
    PATHOLOGY = pathology_summary[0]
    BIRADS = pathology_summary[1]
    PATHOLOGY_RIGHT_BREAST = pathology_summary[2]
    PATHOLOGY_LEFT_BREAST = pathology_summary[3]
    
    sides = []
    if PATHOLOGY_RIGHT_BREAST == 'Malignant':
        sides.append('right')
    if PATHOLOGY_LEFT_BREAST == 'Malignant':
        sides.append('left')    
    
    hdr = nib.load(AXIAL_PATH + exam + '/T1_axial_02_01.nii.gz')

    t1post = hdr.get_data()
    slope1 = nib.load(AXIAL_PATH + exam + '/T1_axial_slope1.nii.gz').get_data()
    slope2 = nib.load(AXIAL_PATH + exam + '/T1_axial_slope2.nii.gz').get_data()

    t1post = t1post/float(40)
    slope1 = slope1/float(0.3)
    slope2 = slope2/float(0.12)           

#    plt.imshow(slope1[150])

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
    else:    
        
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
        
    else:    
        
        length = t1post.shape[2]
        
        extra = length - 512
        
        first_half = extra/2
        second_half = extra - first_half  
                
        t1post = t1post[:, :, first_half:-second_half]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
        slope1 = slope1[:, :, first_half:-second_half]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
        slope2 = slope2[:, :, first_half:-second_half]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    

    assert (t1post.shape[1],t1post.shape[2]) == (512,512), 'Something went wrong with dimensions. NOT 512 x 512!!'
    
    
#    
    seg_path = [x for x in segmented_exams if x.split('/')[-1].startswith(exam)][0]
    seg = nib.load(seg_path).get_data()

#      
#    '''
#    I need the segmented slices first... then do all the reshaping stuff....
#    
#    ACTUALLY I dont. I just need sagittal slices. I am not chaning that dimension. So can grab the slices anyway.
#    
#    '''
    

    # split into right and left breast. In the middle..
    number_slices = t1post.shape[0]
    
    left_breast = np.flip(t1post[:number_slices/2], axis=0)
    left_breast_segmentation = np.flip(seg[:number_slices/2], axis=0)
    left_breast_slope1 = np.flip(slope1[:number_slices/2], axis=0)
    left_breast_slope2 = np.flip(slope2[:number_slices/2], axis=0)

    right_breast = t1post[number_slices/2:]
    right_breast_segmentation = seg[number_slices/2:]
    right_breast_slope1 = slope1[number_slices/2:]
    right_breast_slope2 = slope2[number_slices/2:]
    
    for side in sides:
        if side == 'right':
            contra_side = 'left'            
            seg_coords = np.argwhere(right_breast_segmentation > 0)
            sagittal_slices = list(set(seg_coords[:,0])) 
            segmented_area_per_sag_slice = np.array([np.sum(right_breast_segmentation[x]) for x in sagittal_slices]) 
            index_of_slices_with_larger_segmented_area = (np.argwhere(segmented_area_per_sag_slice > 150)).flatten() 
            sagittal_slices_of_interest = [sagittal_slices[x] for x in index_of_slices_with_larger_segmented_area]   
                            
            for sl in sagittal_slices_of_interest:
                X = np.array([np.stack([right_breast[sl], right_breast_slope1[sl], right_breast_slope2[sl]], axis=-1)])
                
                X_contra = np.array([left_breast[sl]])
                
                np.save(OUTPUT_PATH  + 'X/' + exam + '_' + side[0] + '_' + str(sl) + '.npy', X)
                np.save(OUTPUT_PATH + 'Contralateral/' + exam + '_' + contra_side[0] + '_' + str(sl) + '.npy', X_contra)

 
    
        if side == 'left':
            contra_side = 'right'
            seg_coords = np.argwhere(left_breast_segmentation > 0)
            sagittal_slices = list(set(seg_coords[:,0])) 
            segmented_area_per_sag_slice = np.array([np.sum(left_breast_segmentation[x]) for x in sagittal_slices]) 
            index_of_slices_with_larger_segmented_area = (np.argwhere(segmented_area_per_sag_slice > 150)).flatten() 
            sagittal_slices_of_interest = [sagittal_slices[x] for x in index_of_slices_with_larger_segmented_area]   
                
            for sl in sagittal_slices_of_interest:
                
                X = np.array([np.stack([left_breast[sl], left_breast_slope1[sl], left_breast_slope2[sl]], axis=-1)])
                
                X_contra = np.array([right_breast[sl]])
                
                np.save(OUTPUT_PATH  + 'X/' + exam + '_' + side[0] + '_' + str(sl) + '.npy', X)
                np.save(OUTPUT_PATH + 'Contralateral/' + exam + '_' + contra_side[0] + '_' + str(sl) + '.npy', X_contra)

 
    
    
#%% Get pathology of all these axial scans using REDCAP


REDCAP.columns

REDCAP = REDCAP.loc[REDCAP['Exam'].isin(axial_exams)]

REDCAP['bi_rads_assessment_for_stu'].value_counts()

REDCAP.loc[REDCAP['bi_rads_assessment_for_stu'] < '4', 'overall_study_assessment'].value_counts()
REDCAP.loc[REDCAP['bi_rads_assessment_for_stu'] >= '4', 'overall_study_assessment'].value_counts()

REDCAP.loc[REDCAP['bi_rads_assessment_for_stu'] < '4', 'right_breast_tumor_status'].value_counts()
REDCAP.loc[REDCAP['bi_rads_assessment_for_stu'] < '4', 'left_breast_tumor_status'].value_counts()

REDCAP.loc[REDCAP['bi_rads_assessment_for_stu'] >= '4', 'right_breast_tumor_status'].value_counts()
REDCAP.loc[REDCAP['bi_rads_assessment_for_stu'] >= '4', 'left_breast_tumor_status'].value_counts()


segmented_axials_with_REDCAP_and_images


REDCAP.loc[REDCAP['Exam'].isin(segmented_axials_with_REDCAP_and_images), 'overall_study_assessment'].value_counts()
REDCAP.loc[REDCAP['Exam'].isin(segmented_axials_with_REDCAP_and_images), 'right_breast_tumor_status'].value_counts()
REDCAP.loc[REDCAP['Exam'].isin(segmented_axials_with_REDCAP_and_images), 'left_breast_tumor_status'].value_counts()
REDCAP.loc[REDCAP['Exam'].isin(segmented_axials_with_REDCAP_and_images), 'bi_rads_assessment_for_stu'].value_counts()

#%% Grab all subjects - by MRN ?


REDCAP = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/REDCAP/2023/demographics/20425MachineLearning_DATA_2023-05-11_1234.csv')
REDCAP[['mri_date', 'x_nat_mri_anonymized_mrn']] = REDCAP[['mri_date','x_nat_mri_anonymized_mrn']].fillna('')
REDCAP['Exam'] = REDCAP.apply(lambda x : x['x_nat_mri_anonymized_mrn'] + '_' + x['mri_date'].replace('-',''), axis=1)



axial_mrn = list(set(REDCAP.loc[REDCAP['Exam'].isin(axial_exams), 'mrn_id']))

partitions_sagital = np.load('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/DATA/Data.npy', allow_pickle=True).item()

de_ids = list(set([ x[:20] for x in partitions_sagital['train']]))
de_ids.extend(list(set([ x[:20] for x in partitions_sagital['validation']])))
test_de_ids = list(set([ x[:20] for x in partitions_sagital['test']]))


map_exam_mrn_sagital_training_set = REDCAP.loc[REDCAP['x_nat_mri_anonymized_mrn'].isin(de_ids), ['x_nat_mri_anonymized_mrn','mrn_id']]
map_exam_mrn_sagital_test_set = REDCAP.loc[REDCAP['x_nat_mri_anonymized_mrn'].isin(test_de_ids), ['x_nat_mri_anonymized_mrn','mrn_id']]

available_axials_mrn = REDCAP.loc[REDCAP['Exam'].isin(axial_exams), ['x_nat_mri_anonymized_mrn','mrn_id']]
segmented_axials_mrn = REDCAP.loc[REDCAP['Exam'].isin(segmented_axials_with_REDCAP_and_images), ['x_nat_mri_anonymized_mrn','mrn_id']]

map_exam_mrn_sagital_training_set = map_exam_mrn_sagital_training_set.drop_duplicates()
map_exam_mrn_sagital_test_set = map_exam_mrn_sagital_test_set.drop_duplicates()
available_axials_mrn = available_axials_mrn.drop_duplicates()
segmented_axials_mrn = segmented_axials_mrn.drop_duplicates()

'''
Here I have all subjects in training set of sagitals, and available images in axial (including segmented ones)

also their MRN for better matching (so there are no problems with subjetcs with both MSKCC- and RIA- protocols  --> not unique identifiers for subjects)

'''


available_axials_used_in_sagittal_training = pd.merge(available_axials_mrn, map_exam_mrn_sagital_training_set['mrn_id'], on='mrn_id')
available_axials_used_in_sagittal_test = pd.merge(available_axials_mrn, map_exam_mrn_sagital_test_set['mrn_id'], on='mrn_id')

print('{} subjects used in training set, that have axial images available'.format(len(set(available_axials_used_in_sagittal_training['mrn_id'].values))))

available_axials_NOT_used_in_sagittal_training_or_test = available_axials_mrn.loc[~ available_axials_mrn['mrn_id'].isin(map_exam_mrn_sagital_training_set['mrn_id'].values)]
available_axials_NOT_used_in_sagittal_training_or_test = available_axials_NOT_used_in_sagittal_training_or_test.loc[~ available_axials_NOT_used_in_sagittal_training_or_test['mrn_id'].isin(available_axials_used_in_sagittal_test['mrn_id'].values)]

print('{} subjects NOT used in sagittal training nor test, that have axial images available'.format(len(set(available_axials_NOT_used_in_sagittal_training_or_test['mrn_id'].values))))



segmented_axials_in_training_set = pd.merge(available_axials_used_in_sagittal_training, segmented_axials_mrn['mrn_id'], on='mrn_id')
segmented_axials_in_test_set = pd.merge(available_axials_used_in_sagittal_test, segmented_axials_mrn['mrn_id'], on='mrn_id')
segmented_axials_NOT_used_in_sagittal_training_or_test = pd.merge(available_axials_NOT_used_in_sagittal_training_or_test, segmented_axials_mrn['mrn_id'], on='mrn_id')

'All 362 subjects with segmented axials were NOT used in the sagittal training set NOR test!'

'''
all subjects that are not part of the sagittal set are easy, can partition however I want. Just get the pathology and segmentation

'''

available_axials_used_in_sagittal_training               # 3,620 subjects from training
available_axials_used_in_sagittal_test                   # 400 subjects from test 
available_axials_NOT_used_in_sagittal_training_or_test   # 8,391 subjects for free !

# Training set axials summary:
REDCAP.loc[REDCAP['mrn_id'].isin(available_axials_used_in_sagittal_training['mrn_id']), 'overall_study_assessment'].value_counts()
REDCAP.loc[REDCAP['mrn_id'].isin(available_axials_used_in_sagittal_training['mrn_id']), 'bi_rads_assessment_for_stu'].value_counts()
REDCAP.loc[REDCAP['mrn_id'].isin(available_axials_used_in_sagittal_training['mrn_id']), 'right_breast_tumor_status'].value_counts()
REDCAP.loc[REDCAP['mrn_id'].isin(available_axials_used_in_sagittal_training['mrn_id']), 'left_breast_tumor_status'].value_counts()

# Test set axials summary:
REDCAP.loc[REDCAP['mrn_id'].isin(available_axials_used_in_sagittal_test['mrn_id']), 'overall_study_assessment'].value_counts()
REDCAP.loc[REDCAP['mrn_id'].isin(available_axials_used_in_sagittal_test['mrn_id']), 'bi_rads_assessment_for_stu'].value_counts()
REDCAP.loc[REDCAP['mrn_id'].isin(available_axials_used_in_sagittal_test['mrn_id']), 'right_breast_tumor_status'].value_counts()
REDCAP.loc[REDCAP['mrn_id'].isin(available_axials_used_in_sagittal_test['mrn_id']), 'left_breast_tumor_status'].value_counts()

# Free Axials
REDCAP.loc[REDCAP['mrn_id'].isin(available_axials_NOT_used_in_sagittal_training_or_test['mrn_id']), 'overall_study_assessment'].value_counts()
REDCAP.loc[REDCAP['mrn_id'].isin(available_axials_NOT_used_in_sagittal_training_or_test['mrn_id']), 'bi_rads_assessment_for_stu'].value_counts()
REDCAP.loc[REDCAP['mrn_id'].isin(available_axials_NOT_used_in_sagittal_training_or_test['mrn_id']), 'right_breast_tumor_status'].value_counts()
REDCAP.loc[REDCAP['mrn_id'].isin(available_axials_NOT_used_in_sagittal_training_or_test['mrn_id']), 'left_breast_tumor_status'].value_counts()


'''
Do I need to split the pethology information by side and breast?

Make a scanID column, and fill in considering BIRADS and pathology entry...

Remove anything unclear.

I have more than enough data here to work with.

'''


'''

Gather all REDCAP rows from axials with available images:

First, get breasts that have a clear pathology entry

Check that all segmented ones have a clear malignant pathology on correct breat (already did this before so its just confirmation)

Then assign Benign to all breasts with (all) BIRADS < 4 and True Negative

Dismiss the rest.

'''


'''
Ok only problem:
    
I am partitioning by subjects, but I really am interested in exams 

So now that I have the subjects (MRN), continue by selecting the EXAMS I have available in axials

Only after that start filtering by pathology

'''

# THese are subjects (MRN) and ALL their exams (87k)
REDCAP = REDCAP.loc[(REDCAP['mrn_id'].isin(available_axials_used_in_sagittal_training['mrn_id'])) + (REDCAP['mrn_id'].isin(available_axials_used_in_sagittal_test['mrn_id'])) + (REDCAP['mrn_id'].isin(available_axials_NOT_used_in_sagittal_training_or_test['mrn_id']))]

# These are the specific available exams (20k)
REDCAP = REDCAP.loc[REDCAP['Exam'].isin(axial_exams)]


# 1) Get clear pathologies
REDCAP['right_breast_tumor_status'].value_counts()
REDCAP['left_breast_tumor_status'].value_counts()

# Remove rows where both breasts have uncertain pathology:

REDCAP = REDCAP.loc[~ ((REDCAP['right_breast_tumor_status'].isin(['Unknown', 'Cancelled'])) * (REDCAP['left_breast_tumor_status'].isin(['Unknown', 'Cancelled'])) )]

REDCAP['scanID'] = ''
REDCAP['pathology'] = ''

REDCAP_malignant_right_breasts = REDCAP.loc[REDCAP['right_breast_tumor_status'] == 'Malignant']
REDCAP_malignant_right_breasts['scanID'] = REDCAP_malignant_right_breasts['Exam'] + '_r'
REDCAP_malignant_right_breasts['pathology'] = 'Malignant'

REDCAP_malignant_left_breasts = REDCAP.loc[REDCAP['left_breast_tumor_status'] == 'Malignant']
REDCAP_malignant_left_breasts['scanID'] = REDCAP_malignant_left_breasts['Exam'] + '_l'
REDCAP_malignant_left_breasts['pathology'] = 'Malignant'

REDCAP_benign_right_breasts = REDCAP.loc[REDCAP['right_breast_tumor_status'] == 'Benign']
REDCAP_benign_right_breasts['scanID'] = REDCAP_benign_right_breasts['Exam'] + '_r'
REDCAP_benign_right_breasts['pathology'] = 'Benign'

REDCAP_benign_left_breasts = REDCAP.loc[REDCAP['left_breast_tumor_status'] == 'Benign']
REDCAP_benign_left_breasts['scanID'] = REDCAP_benign_left_breasts['Exam'] + '_l'
REDCAP_benign_left_breasts['pathology'] = 'Benign'


PATHOLOGY_AXIAL = pd.concat([REDCAP_malignant_right_breasts[['mrn_id','scanID','Exam', 'overall_study_assessment', 'bi_rads_assessment_for_stu', 'pathology']],
                             REDCAP_malignant_left_breasts[['mrn_id','scanID','Exam', 'overall_study_assessment', 'bi_rads_assessment_for_stu', 'pathology']],
                             REDCAP_benign_right_breasts[['mrn_id','scanID','Exam', 'overall_study_assessment', 'bi_rads_assessment_for_stu', 'pathology']],
                             REDCAP_benign_left_breasts[['mrn_id','scanID','Exam', 'overall_study_assessment', 'bi_rads_assessment_for_stu', 'pathology']]])


# Process missing ones without clear pathology:
REDCAP_others = REDCAP.loc[~ REDCAP['Exam'].isin(PATHOLOGY_AXIAL['Exam'])]
    
# no pathology entries
REDCAP_others['right_breast_tumor_status'].value_counts()
REDCAP_others['left_breast_tumor_status'].value_counts()

# just make it based on BIRADS:
REDCAP_others['bi_rads_assessment_for_stu'].value_counts()

# Assign Benign to anything BIRADS < 6, single BIRADS only

REDCAP_others = REDCAP_others.loc[REDCAP_others['bi_rads_assessment_for_stu'].isin(['1','2','3','4','5'])]

REDCAP_others_left = REDCAP_others.copy()
REDCAP_others_right = REDCAP_others.copy()

REDCAP_others_left['scanID'] = REDCAP_others_left['Exam'] + '_l'
REDCAP_others_left['pathology'] = 'Benign'

REDCAP_others_right['scanID'] = REDCAP_others_right['Exam'] + '_r'
REDCAP_others_right['pathology'] = 'Benign'

PATHOLOGY_AXIAL = pd.concat([REDCAP_malignant_right_breasts[['mrn_id','scanID','Exam', 'overall_study_assessment', 'bi_rads_assessment_for_stu', 'pathology']],
                             REDCAP_malignant_left_breasts[['mrn_id','scanID','Exam', 'overall_study_assessment', 'bi_rads_assessment_for_stu', 'pathology']],
                             REDCAP_benign_right_breasts[['mrn_id','scanID','Exam', 'overall_study_assessment', 'bi_rads_assessment_for_stu', 'pathology']],
                             REDCAP_benign_left_breasts[['mrn_id','scanID','Exam', 'overall_study_assessment', 'bi_rads_assessment_for_stu', 'pathology']], 
                            REDCAP_others_left[['mrn_id','scanID','Exam', 'overall_study_assessment', 'bi_rads_assessment_for_stu', 'pathology']],
                            REDCAP_others_right[['mrn_id','scanID','Exam', 'overall_study_assessment', 'bi_rads_assessment_for_stu', 'pathology']]])


PATHOLOGY_AXIAL = PATHOLOGY_AXIAL.reset_index(drop=True)    
    
###### CONFIRM EVERYTHING LOOKS RIGHT:

PATHOLOGY_AXIAL = PATHOLOGY_AXIAL.sort_values('scanID')

PATHOLOGY_AXIAL['scanID'].value_counts()

PATHOLOGY_AXIAL['pathology'].value_counts()

# Segmentations are only refering to exam. There is one "segmented" exam, benign breast. The benign refers to the other breast. 
# Why doesnt this happen to the rest? Because they have BIRADS 6.
PATHOLOGY_AXIAL.loc[PATHOLOGY_AXIAL['Exam'].isin(segmented_axials_with_REDCAP_and_images), 'pathology'].value_counts()
#example: (why doesnt the contralateral breast get assigned benign? A: no clear pathology on right breast, and BIRADS 6 on exam.)
REDCAP.loc[REDCAP['Exam'] == 'RIA_19-093_000_08373_20160820', ['left_breast_tumor_status','right_breast_tumor_status','bi_rads_assessment_for_stu']]

'''
Looking good! Now check pathologies and segmentations per sagittal partition

this is by subject
'''

PATHOLOGY_AXIAL['partition'] = ''
PATHOLOGY_AXIAL.loc[PATHOLOGY_AXIAL['mrn_id'].isin(available_axials_used_in_sagittal_training['mrn_id']), 'pathology'].value_counts()
PATHOLOGY_AXIAL.loc[PATHOLOGY_AXIAL['mrn_id'].isin(available_axials_used_in_sagittal_test['mrn_id']), 'pathology'].value_counts()
PATHOLOGY_AXIAL.loc[PATHOLOGY_AXIAL['mrn_id'].isin(available_axials_NOT_used_in_sagittal_training_or_test['mrn_id']), 'pathology'].value_counts()


'''
Unused      22,755
Training    13,008
Test         1,452

Axials from subjects used in training set of sagittals:
    12,921 Benigns
    87 Malignants
    
Axials from subjects used in test set of sagittals:
    1,449 Benigns
    3 Malignants
    
Axials from subjects not used in sagittals:
    19,432 Benigns
    3,323 Malignants
    
'''

PATHOLOGY_AXIAL.loc[PATHOLOGY_AXIAL['mrn_id'].isin(available_axials_used_in_sagittal_training['mrn_id']), 'partition'] = 'Training'
PATHOLOGY_AXIAL.loc[PATHOLOGY_AXIAL['mrn_id'].isin(available_axials_used_in_sagittal_test['mrn_id']), 'partition'] = 'Test'
PATHOLOGY_AXIAL.loc[PATHOLOGY_AXIAL['mrn_id'].isin(available_axials_NOT_used_in_sagittal_training_or_test['mrn_id']), 'partition'] = 'Unused'

PATHOLOGY_AXIAL['segmented'] = 0

PATHOLOGY_AXIAL.loc[PATHOLOGY_AXIAL['Exam'].isin(segmented_axials_with_REDCAP_and_images), 'segmented'] = 1

AXIAL_TRAIN = PATHOLOGY_AXIAL.loc[PATHOLOGY_AXIAL['partition'] == 'Training']
AXIAL_TEST = PATHOLOGY_AXIAL.loc[PATHOLOGY_AXIAL['partition'] == 'Test']
AXIAL_UNUSED = PATHOLOGY_AXIAL.loc[PATHOLOGY_AXIAL['partition'] == 'Unused']

# check no overlap
set(AXIAL_TRAIN['mrn_id'].values).intersection(set(AXIAL_TEST['mrn_id'].values))
set(AXIAL_TEST['mrn_id'].values).intersection(set(AXIAL_UNUSED['mrn_id'].values))
set(AXIAL_TRAIN['mrn_id'].values).intersection(set(AXIAL_UNUSED['mrn_id'].values))

set(AXIAL_TRAIN['Exam'].values).intersection(set(AXIAL_TEST['Exam'].values))
set(AXIAL_TEST['Exam'].values).intersection(set(AXIAL_UNUSED['Exam'].values))
set(AXIAL_TRAIN['Exam'].values).intersection(set(AXIAL_UNUSED['Exam'].values))




PATHOLOGY_AXIAL.to_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/REDCAP/2023/Axials_pathology.csv', index=False)


'''
Send all segmented into training

send all malignants not segmented into test

'''

PATHOLOGY_AXIAL.loc[PATHOLOGY_AXIAL['segmented'] == 1, 'partition'] = 'Training'

PATHOLOGY_AXIAL.loc[(PATHOLOGY_AXIAL['pathology'] == 'Malignant')*(PATHOLOGY_AXIAL['partition'] == 'Unused')*(PATHOLOGY_AXIAL['segmented'] == 0), 'partition'] = 'Test'

PATHOLOGY_AXIAL.loc[PATHOLOGY_AXIAL['partition'] == 'Training', 'pathology'].value_counts()
PATHOLOGY_AXIAL.loc[PATHOLOGY_AXIAL['partition'] == 'Test', 'pathology'].value_counts()
PATHOLOGY_AXIAL.loc[PATHOLOGY_AXIAL['partition'] == 'Unused', 'pathology'].value_counts()

len(PATHOLOGY_AXIAL['Exam'].unique())

set(AXIAL_TRAIN['mrn_id'].values).intersection(set(AXIAL_TEST['mrn_id'].values))
set(AXIAL_TEST['mrn_id'].values).intersection(set(AXIAL_UNUSED['mrn_id'].values))
set(AXIAL_TRAIN['mrn_id'].values).intersection(set(AXIAL_UNUSED['mrn_id'].values))

set(AXIAL_TRAIN['Exam'].values).intersection(set(AXIAL_TEST['Exam'].values))
set(AXIAL_TEST['Exam'].values).intersection(set(AXIAL_UNUSED['Exam'].values))
set(AXIAL_TRAIN['Exam'].values).intersection(set(AXIAL_UNUSED['Exam'].values))


PATHOLOGY_AXIAL.to_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/REDCAP/2023/Axials_pathology_assigned_partition.csv', index=False)


#%%

PATHOLOGY_AXIAL = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/REDCAP/2023/Axials_pathology_assigned_partition.csv')

benign_training = PATHOLOGY_AXIAL.loc[(PATHOLOGY_AXIAL['partition'] == 'Training')*(PATHOLOGY_AXIAL['pathology'] == 'Benign')]

#benign_training = benign_training[6000:]

OUTPUT_PATH = '/media/SD/Axial_Slices/'

already_done = os.listdir(OUTPUT_PATH + 'X/')

exams_already_done = [x[:29] for x in already_done]

benign_training = benign_training.loc[ ~ benign_training['Exam'].isin(exams_already_done)]

done_exams_on_the_fly = []

TOT = len(benign_training['Exam'].unique())

for row in benign_training.iterrows():
    print(len(done_exams_on_the_fly))
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

    if t1post.shape[1] < 256:
        print('image too small. Skip')
        continue

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
       
#    t1post[150].shape
#    
#    plt.imshow(t1post[150])
    
    projection_1d = np.max(np.max(t1post, 0),1)

    breast_end = np.argmin(np.diff(projection_1d[np.arange(0,len(projection_1d),5)]))*5
    breast_end = breast_end + 10 # add some border
    breast_end = np.max([breast_end, 256]) # if breast is small, just crop to 256
        
    t1post = t1post[:, (breast_end-256):breast_end, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    slope1 = slope1[:, (breast_end-256):breast_end, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    slope2 = slope2[:, (breast_end-256):breast_end, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    
    if resolution[0] > 0.5:
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
