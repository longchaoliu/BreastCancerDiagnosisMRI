
import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize

AXIAL_PATH = '/home/deeperthought/kirbyPRO/MSKdata/alignedNiiAxial-May2020-cropped-normed/'

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

 