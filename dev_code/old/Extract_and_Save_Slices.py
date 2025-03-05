#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:56:33 2022

Store full slices for direct classification and segmentation models.

2D and 2.5D (1 slice neighbor)

Store all in one folder. Labels are stored in master spreadsheet.

In order to make process faster for benigns, use partitions spreadsheet to focus extraction of slices by partition.

@author: deeperthought
"""


import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize

#%%    USER INPUT

MRI_PATH = '/home/deeperthought/kirby_MSK/alignedNii-Nov2019/'
MRI_ALIGNED_HISTORY_PATH = '/home/deeperthought/kirbyPRO/saggital_Nov2019_alignedHistory/'

MASTER = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Data_Partitions.csv')
RISK = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Data_ExamHistory_Labels.csv')

CONVERT_SLICES = pd.read_csv('/media/HDD/segmented_slice_on_previous_exam.csv')

OUTPUT_FOLDER_2D = '/home/deeperthought/kirbyPRO/Saggittal_Full_Slices/2D_slices/X/'
OUTPUT_SEGMENTATIONS_2D = '/home/deeperthought/kirbyPRO/Saggittal_segmentations_clean/2D/'

NEIGHBOR_SLICES = 0


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

#%% Already done

DONE = os.listdir(OUTPUT_FOLDER_2D)
DONE_SCANID = list(set([x[:31] for x in DONE]))
DONE_EXAMS = list(set([x[:-2] for x in DONE_SCANID]))

len(DONE_EXAMS)


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
#%%

def percentile95_normalizeMRI(data, p95=0):
    if p95 == 0:
        p95 = np.percentile(data,95)
    data1 = data/p95
    return(data1)

def normalizeMRI(data, mean=0, std=1):
    if (mean == 0) and (std == 1):
        mean = np.mean(data)
        std = np.std(data)
    data1 = (data - mean)/std
    return(data1)

#
#def align_contralateral():
#    
#    FILE_DAMAGED_FLAG = False
#    if contra_image_available:
#        #### Align contralateral scan to reference image ####       
#        all_contralateral_subject_channels_aligned = []
#        
#        all_contralateral_subject_channels_aligned = [x.replace('.nii','_contralateral_aligned.nii') for x in all_contralateral_subject_channels]
#        if os.path.exists(all_contralateral_subject_channels_aligned[0]) and os.path.exists(all_contralateral_subject_channels_aligned[1]) and os.path.exists(all_contralateral_subject_channels_aligned[2]):
#                    print('Contralaterals already aligned..skip')
#                        
#        else:
#            for ii in range(len(all_subject_channels)):
#                print(ii)
#                
#                reference_scan = all_subject_channels[ii]
#                contralateral_scan = all_contralateral_subject_channels[ii]
#                contralateral_scan_mirrored = contralateral_scan.replace('.nii','_mirrored.nii')
#                contra_nii = nib.load(contralateral_scan)
#                try:
#                    contra = contra_nii.get_data()
#                except:
#                    print('Loading error.. File damaged.. skip')
#                    FILE_DAMAGED_FLAG = True
#                    break
#                print('\nMirroring:\n{} --> {}'.format(contralateral_scan.replace('/home/deeperthought/kirby_MSK/alignedNii-Nov2019/',''), contralateral_scan_mirrored.replace('/home/deeperthought/kirby_MSK/alignedNii-Nov2019/','')))
#                contra_mirrored = np.flip(contra, 0 )
#                out = nib.Nifti1Image(contra_mirrored, contra_nii.affine)    
#                nib.save(out, contralateral_scan_mirrored)    
#                
#                output_name = contralateral_scan.replace('.nii','_contralateral_aligned.nii')
#                #all_contralateral_subject_channels_aligned.append(output_name)    
#                
#                cpp_file = '/'.join(reference_scan.split('/')[:-1]) + '/OutputCPP.nii.gz'
#                
#                if ii == 0:
#                    some_command = "reg_f3d -ref {} -flo {} -res {} -be 0.01 -cpp {} -maxit 30".format(reference_scan, contralateral_scan_mirrored, output_name, cpp_file)
#                    print('\nAligning contralateral scan:\n{} <-- {}'.format(reference_scan.replace('/home/deeperthought/kirby_MSK/alignedNii-Nov2019/',''), contralateral_scan_mirrored.replace('/home/deeperthought/kirby_MSK/alignedNii-Nov2019/','')))
#                    p = subprocess.Popen(some_command, stdout=subprocess.PIPE, shell=True)
#                    (output, err) = p.communicate()  
#                    p_status = p.wait()
#                else:
#                    #reference_scan = all_contralateral_subject_channels[0].replace('.nii','_contralateral_aligned.nii.gz')
#                    #some_command = "reg_f3d -ref {} -flo {} -res {} -be 0.01".format(reference_scan, contralateral_scan_mirrored, output_name)
#                    
#                    some_command = "reg_resample -ref {} -flo {} -res {} -cpp {}".format(reference_scan, contralateral_scan_mirrored, output_name, cpp_file)
#                    print('\nAligning contralateral scan:\n{} <-- {}'.format(reference_scan.replace('/home/deeperthought/kirby_MSK/alignedNii-Nov2019/',''), contralateral_scan_mirrored.replace('/home/deeperthought/kirby_MSK/alignedNii-Nov2019/','')))
#                    p = subprocess.Popen(some_command, stdout=subprocess.PIPE, shell=True)
#                    (output, err) = p.communicate()  
#                    p_status = p.wait()  
#                    
#                os.remove(contralateral_scan_mirrored)
#                    
#                if not os.path.exists(output_name):
#                    print('Alignment failed! Scan : {}'.format(reference_scan))
#                    sys.exit(0)
#    if FILE_DAMAGED_FLAG:
#        continue
#
#
#
#def align_in_time(PREVIOUS_EXAM, NEWEST_EXAM, SIDE, DATA_PATH, OUTPUT_PATH = '/home/deeperthought/kirbyPRO/saggital_Nov2019_alignedHistory/' ):
#    
#    "Only align previouos breast. Dont do whole history."
#   
#    subject = NEWEST_EXAM[15:-9]
#    OUTPUT_PATH_SUBJECT = OUTPUT_PATH + PREVIOUS_EXAM
#    
#    print('Aligning {}: {} --> {}'.format(subject, PREVIOUS_EXAM[-8:], NEWEST_EXAM[-8:]))
#
#
#    OUTPUT_PATH_EXAM = OUTPUT_PATH + '/' + TARGET_EXAM
#    
#    cpp_file = OUTPUT_PATH_EXAM + '/CPP.nii.gz'
#    
#    if not os.path.exists(OUTPUT_PATH_EXAM):
#        os.mkdir(OUTPUT_PATH_EXAM)
#        
#    for MOD in ['02_01', 'slope1', 'slope2']:
#        
#        REF = DATA_PATH + subject + '_' + NEWEST_EXAM + '/T1_{}_{}.nii.gz'.format(SIDE, MOD)
#     
#        TARGET = DATA_PATH + subject + '_' + PREVIOUS_EXAM + '/T1_{}_{}.nii.gz'.format(SIDE, MOD)
#        
#        output_name = OUTPUT_PATH_EXAM + '/T1_' + SIDE + '_' + MOD + '_TimeAlignment_to_{}.nii.gz'.format(NEWEST_EXAM)
#        
#        if os.path.exists(output_name):
#            print('Already done. Skip..')  
#            continue
#        
#        warp_command = "reg_f3d -ref {} -flo {} -res {} -be 0.1 -cpp {}".format(
#                REF, TARGET, output_name, cpp_file)
#        p = subprocess.Popen(warp_command, stdout=subprocess.PIPE, shell=True)
#        (output, err) = p.communicate()  
#        p_status = p.wait()
#            
#    if os.path.exists(cpp_file):
#        os.remove(cpp_file)
#    
#    return breast
#
def load_and_preprocess(all_subject_channels, T1_pre_nii_path):
    IOERROR_FLAG = False
    try:
        t1post = nib.load(all_subject_channels[0]).get_data()
    except IOError:
        IOERROR_FLAG = True
    try:
        slope1 = nib.load(all_subject_channels[1]).get_data()
    except IOError:
        IOERROR_FLAG = True
    try:
        slope2 = nib.load(all_subject_channels[2]).get_data()    
    except IOError:
        IOERROR_FLAG = True
        
    if IOERROR_FLAG:
        return 0,0,0, IOERROR_FLAG
        
    if t1post.shape[1] != 512:
        output_shape = (t1post.shape[0],512,512)
        t1post = resize(t1post, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
        slope1 = resize(slope1, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
        slope2 = resize(slope2, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')

    p95 = np.percentile(nib.load(T1_pre_nii_path).get_data(),95)
        
    t1post = t1post/p95    
    slope1 = slope1/p95    
    slope2 = slope2/p95    

    t1post = t1post/float(40)
    slope1 = slope1/float(0.3)
    slope2 = slope2/float(0.12)     

    return t1post, slope1, slope2, IOERROR_FLAG

def check_nans(X):
    for img in X:  
        if not np.isfinite(slope2).all():
            return 1
    return 0
#%%
    

#patients_with_aligned_previous = os.listdir(MRI_ALIGNED_HISTORY_PATH)

EXAMS = list(set(AVAILABLE_SCANID) - set(DONE_SCANID))
print('{} exams still to extract 2D slices'.format(len(EXAMS)))
NaNs = []

#%%
print('Removing un-segmented malignants:')

segmented_slices = os.listdir('/home/deeperthought/kirbyPRO/Saggittal_segmentations_clean/2D/')
len(set(DONE).intersection(set(segmented_slices)))
len(set(segmented_slices) - set(DONE))
segmented_scanid = [x[:31] for x in segmented_slices]


MASTER.columns

MASTER['Segmented'].value_counts()

MASTER['Pathology'].value_counts()


malignants = list(set(MASTER.loc[MASTER['Pathology'] == 'Malignant', 'Scan_ID'].values))

segmented = list(set(MASTER.loc[MASTER['Segmented'] == 1, 'Scan_ID'].values))

unsegmented_malignants = list(set(malignants) - set(segmented))

len(unsegmented_malignants)

print(len(malignants),len(segmented), len(unsegmented_malignants))

assert set(DONE_SCANID).intersection(set(unsegmented_malignants)) == set()
#
#extracted_slices = os.listdir('/home/deeperthought/kirbyPRO/Saggittal_Full_Slices/2D_slices/X/')
#
#extracted_slices_of_unsegmented_malignants = [x for x in extracted_slices if x[:31] in unsegmented_malignants]
#
#for i in range(len(extracted_slices_of_unsegmented_malignants)):
#    scan_to_remove = '/home/deeperthought/kirbyPRO/Saggittal_Full_Slices/2D_slices/X/' + extracted_slices_of_unsegmented_malignants[i]
#    os.remove(scan_to_remove)


EXAMS = list(set(EXAMS) - set(unsegmented_malignants))
print('{} exams still to extract 2D slices'.format(len(EXAMS)))

#%% Removing scans that have invalid entries (incomplete series, etc)

print('Removing scans that have invalid entries (incomplete series, etc)')
EXAMS = list(set(EXAMS).intersection( set(RISK['Scan_ID'].values)))
print('{} exams still to extract 2D slices'.format(len(EXAMS)))




#%%

SUBJECT_INDEX = 0
TOT = len(EXAMS)



for SUBJECT_INDEX in range(TOT): #TOT
    print(SUBJECT_INDEX, TOT)
 
    scanID = EXAMS[SUBJECT_INDEX]
    
    patient = scanID[:-11]
    exam = scanID[:-2]
    side = 'right'
    contra_side = 'left'
    if scanID[-1] == 'l': 
        side = 'left'
        contra_side = 'right'
        
    pathology = RISK.loc[RISK['Scan_ID'] == scanID, 'Pathology'].values[0]
 
    print("####### IPSILATERAL ##############")
    all_subject_channels = [MRI_PATH + exam + '/T1_{}_02_01.nii'.format(side),
                           MRI_PATH + exam + '/T1_{}_slope1.nii'.format(side),
                           MRI_PATH + exam + '/T1_{}_slope2.nii'.format(side)]

    T1_pre_nii_path = MRI_PATH + exam + '/T1_{}_01_01.nii'.format(side)

    t1post, slope1, slope2, FLAG = load_and_preprocess(all_subject_channels, T1_pre_nii_path)

    if FLAG:
        print('IO Error: Skip')
        continue

    if check_nans([t1post, slope1, slope2]):
        NaNs.append(scanID)
        continue

#    print("####### CONTRALATERAL ##############")
#    contralateral_available = SET.loc[SET['Scan_ID'] == scanID, 'Contralateral Available'].values[0]
#    
#    if contralateral_available:
#        all_contralateral_channels = [MRI_PATH + exam + '/T1_{}_02_01_contralateral_aligned.nii'.format(contra_side),
#                                      MRI_PATH + exam + '/T1_{}_slope1_contralateral_aligned.nii'.format(contra_side),
#                                      MRI_PATH + exam + '/T1_{}_slope2_contralateral_aligned.nii'.format(contra_side)]
#        
#        T1_pre_nii_path_contralateral = MRI_PATH + exam + '/T1_{}_01_01.nii'.format(contra_side)
#
#        if not os.path.exists(all_contralateral_channels[0]):
#            print('previous exam not aligned yet.. skip')
#            continue    
#        
#        t1post_contra, slope1_contra, slope2_contra, FLAG = load_and_preprocess(all_contralateral_channels, T1_pre_nii_path_contralateral)
#        if FLAG:
#            print('IO Error: Skip')
#            continue
#
#        if check_nans([t1post_contra, slope1_contra, slope2_contra]):
#            NaNs.append(scanID)
#            continue
#        
#    print("####### PREVIOUS ##############")
#    previous_available = SET.loc[SET['Scan_ID'] == scanID, 'Previous Available'].values[0]
#    
#    if previous_available:
#
#        breast_history = MASTER.loc[(MASTER['DE-ID'] == patient) & (MASTER['Scan_ID'].str[-1] == scanID[-1]), 'Scan_ID'].values
#        previous_exam = breast_history[np.argwhere(breast_history == scanID)[0][0] - 1][21:-2]        
#        
#        MRI_ALIGNED_HISTORY_PATH + patient + '/' + previous_exam
#        all_previous_channels = [MRI_ALIGNED_HISTORY_PATH + patient + '/' + previous_exam+ '/T1_{}_02_01_TimeAlignment_to_{}.nii.gz'.format(side, exam[21:]),
#                                 MRI_ALIGNED_HISTORY_PATH + patient + '/' + previous_exam+ '/T1_{}_slope1_TimeAlignment_to_{}.nii.gz'.format(side, exam[21:]),
#                                 MRI_ALIGNED_HISTORY_PATH + patient + '/' + previous_exam+ '/T1_{}_slope2_TimeAlignment_to_{}.nii.gz'.format(side, exam[21:])]            
#
#        T1_pre_nii_path_previous = MRI_PATH + patient + '_' + previous_exam + '/T1_{}_01_01.nii'.format(side)
#    
#        if not os.path.exists(all_previous_channels[0]):
#            print('previous exam not aligned yet.. skip')
#            continue
#    
#        t1post_previous, slope1_previous, slope2_previous, FLAG = load_and_preprocess(all_previous_channels, T1_pre_nii_path_previous)
#        if FLAG:
#            print('IO Error: Skip')
#            continue
#        
#        if check_nans([t1post_previous, slope1_previous, slope2_previous]):
#            NaNs.append(scanID)
#            continue
        
    ####### TARGET ##############  
    if pathology == 'Malignant':
        if scanID in segmented_scanid:
        
        #segmentation_available = MASTER.loc[MASTER['Scan_ID'] == scanID, 'Segmented'].values[0] == 1
      
    
        #print("####### TARGET ##############")
    
        #if segmentation_available: # What do I do with malignants that have no segmentation? These are not part of the training set of this pipeline!
            segmentation_path = MASTER.loc[MASTER['Scan_ID'] == scanID, 'Segmentation_Path'].values[0]    
            groundtruth = nib.load(segmentation_path).get_data()
    
            selected_slices = list(set(np.where(groundtruth > 0)[0]))
        
        else:
            print('Unsegmented malignant, skip')
            continue
        
    if pathology == 'Benign': 
        random = np.random.choice(range(2,t1post.shape[0]//2) + range(t1post.shape[0]//2+1,t1post.shape[0]-2) )
        selected_slices = [random , t1post.shape[0]//2]
#        
#        np.argwhere(t1post == np.max(t1post))
#        plt.imshow(t1post[29])
#
#        np.argwhere(slope1 == np.max(slope1))
#        plt.imshow(slope1[30])
        
    ####### STORE ##############    
    print("####### STORE ##############")


    for i in range(len(selected_slices)):
        
        if not os.path.exists(OUTPUT_FOLDER + '/{}_{}.npy'.format(scanID, selected_slices[i])):
            t1post_crop = t1post[selected_slices[i]-NEIGHBOR_SLICES:selected_slices[i]+NEIGHBOR_SLICES+1]
            slope1_crop = slope1[selected_slices[i]-NEIGHBOR_SLICES:selected_slices[i]+NEIGHBOR_SLICES+1]
            slope2_crop = slope2[selected_slices[i]-NEIGHBOR_SLICES:selected_slices[i]+NEIGHBOR_SLICES+1]
            X = np.stack([t1post_crop, slope1_crop, slope2_crop], -1)
            np.save(OUTPUT_FOLDER + '/{}_{}.npy'.format(scanID, selected_slices[i]), X)

#        if contralateral_available:
#            if not os.path.exists(OUTPUT_FOLDER_CONTRA + '/{}_{}.npy'.format(scanID, selected_slices[i])):
#                t1post_crop_contra = t1post_contra[selected_slices[i]-NEIGHBOR_SLICES:selected_slices[i]+NEIGHBOR_SLICES+1]
#                slope1_crop_contra = slope1_contra[selected_slices[i]-NEIGHBOR_SLICES:selected_slices[i]+NEIGHBOR_SLICES+1]
#                slope2_crop_contra = slope2_contra[selected_slices[i]-NEIGHBOR_SLICES:selected_slices[i]+NEIGHBOR_SLICES+1]
#                X_contra = np.stack([t1post_crop_contra, slope1_crop_contra, slope2_crop_contra], -1)
#                np.save(OUTPUT_FOLDER_CONTRA + '/{}_{}.npy'.format(scanID, selected_slices[i]), X_contra)
#
#        if previous_available:
#            if not os.path.exists(OUTPUT_FOLDER_PREVIOUS + '/{}_{}.npy'.format(scanID, selected_slices[i])):
#    
#                t1post_crop_previous = t1post_previous[selected_slices[i]-NEIGHBOR_SLICES:selected_slices[i]+NEIGHBOR_SLICES+1]
#                slope1_crop_previous = slope1_previous[selected_slices[i]-NEIGHBOR_SLICES:selected_slices[i]+NEIGHBOR_SLICES+1]
#                slope2_crop_previous = slope2_previous[selected_slices[i]-NEIGHBOR_SLICES:selected_slices[i]+NEIGHBOR_SLICES+1]
#                X_previous = np.stack([t1post_crop_previous, slope1_crop_previous, slope2_crop_previous], -1)
#                np.save(OUTPUT_FOLDER_PREVIOUS + '/{}_{}.npy'.format(scanID, selected_slices[i]), X_previous)
                
        if segmentation_available:  
            if not os.path.exists(OUTPUT_SEGMENTATIONS + '/{}_{}.npy'.format(scanID, selected_slices[i])):
    
                groundtruth_crop = groundtruth[selected_slices[i]]
                np.save(OUTPUT_SEGMENTATIONS + '/{}_{}.npy'.format(scanID, selected_slices[i]), groundtruth_crop)
            
                
        
#seg = np.load('/home/deeperthought/kirbyPRO/Saggittal_segmentations_clean/2D/MSKCC_16-328_1_00090_20120202_l_32.npy')
#x = np.load('/home/deeperthought/kirbyPRO/Saggittal_Full_Slices/2D_slices/X/MSKCC_16-328_1_00090_20120202_l_32.npy')
#x2 = np.load('/home/deeperthought/kirbyPRO/Saggittal_Full_Slices/2D_slices/Contra/MSKCC_16-328_1_00090_20120202_l_32.npy')
#x3 = np.load('/home/deeperthought/kirbyPRO/Saggittal_Full_Slices/2D_slices/Previous/MSKCC_16-328_1_00090_20120202_l_32.npy')
#
#plt.subplot(1,3,1)
#plt.imshow(x[0,:,:,0])
#plt.subplot(1,3,2)
#plt.imshow(x[0,:,:,0] + seg)
#plt.subplot(1,3,3)
#plt.imshow(x3[0,:,:,0])
