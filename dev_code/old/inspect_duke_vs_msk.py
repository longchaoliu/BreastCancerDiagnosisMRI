#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:38:48 2024

@author: deeperthought
"""

    
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.transform import resize
import pandas as pd


    
DATA_PATH_DUKE = '/home/deeperthought/kirby_MSK/dukePublicData/alignedNii/'      

DATA_PATH_MSK = '/home/deeperthought/kirby_MSK/alignedNii-Nov2019/'

DATA_PATH_MSK_AXIAL = '/home/deeperthought/kirbyPRO/alignedNiiAxial-May2020-cropped-normed/'

#%%
def load_and_preprocess(all_subject_channels, T1_pre_nii_path='', apply_p95=True, t1post_factor=40):
  
    t1post = nib.load(all_subject_channels[0])
    affine = np.diag(t1post.affine)
    # t1post = t1post.get_fdata()
   
    # slope1 = nib.load(all_subject_channels[1]).get_fdata()
   
    # slope2 = nib.load(all_subject_channels[2]).get_fdata()    
    X_res = affine[0]
    Y_res = affine[1]
    Z_res = affine[2]
   
    
    # if (t1post.shape[1] != 512) or ((t1post.shape[2] != 512)):
    #     output_shape = (t1post.shape[0],512,512)
    #    # t1post = resize(t1post, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
    #     # slope1 = resize(slope1, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
    #     # slope2 = resize(slope2, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
        
    #     Y_res = (t1post.shape[1]/512)*affine[1]
    #     Z_res = (t1post.shape[2]/512)*affine[2]
        

    # if apply_p95:
    #     p95 = np.percentile(nib.load(T1_pre_nii_path).get_fdata(),95)
            
    #     t1post = t1post/p95    
    #     slope1 = slope1/p95    
    #     slope2 = slope2/p95    

    # t1post = t1post/float(t1post_factor)
    # slope1 = slope1/float(0.3)
    # slope2 = slope2/float(0.12)     

    return t1post, X_res, Y_res, Z_res#, slope1, slope2, affine, p95
#%%

exams_duke = os.listdir(DATA_PATH_DUKE)
exams_msk = os.listdir(DATA_PATH_MSK)

exams_msk_axial = os.listdir(DATA_PATH_MSK_AXIAL)

# df = pd.DataFrame(columns=['Site','resolution','shape','T1pre_p95','T1post_max','slope1_max','slope2_max', 'T1post_median', 'T1post_average', 'slope1_median', 'slope2_median'])

df = pd.DataFrame(columns=['Site','resolution','shape'])


OUTPUT_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Duke_predictions/predictions/'
OUTPUT_NAME = 'duke_vs_MSK_vs_axial_resolution_preprocessed.csv'

#%%

    
for i in range(100):
    print(i)
    all_subject_channels = [DATA_PATH_DUKE + exams_duke[i] + '/T1_axial_02.nii.gz',
                            DATA_PATH_DUKE + exams_duke[i] + '/T1_axial_slope1.nii.gz',
                            DATA_PATH_DUKE + exams_duke[i] + '/T1_axial_slope2.nii.gz']
    # T1_pre_nii_path = DATA_PATH_DUKE + exams_duke[i] + '/T1_axial_01.nii.gz'   
    t1,X_res, Y_res, Z_res = load_and_preprocess(all_subject_channels, apply_p95=True, t1post_factor=40)
    # try:
    #     t1 = nib.load(all_subject_channels[0])
    # except:
    #     continue
    df = df.append({'Site':'Duke','resolution_x':X_res,'resolution_y':Y_res,'resolution_z':Z_res,
                    'shape_x':t1.shape[0], 'shape_y':t1.shape[1], 'shape_z':t1.shape[2]}, ignore_index=True)#, 'T1pre_p95':p95, 'T1post_max':np.max(t1),'T1post_average':np.mean(t1), 'slope1_max':np.max(s1),'slope2_max':np.max(s2), 'T1post_median':np.median(t1), 'slope1_median':np.median(s1), 'slope2_median':np.median(s2)}, ignore_index=True)
    
    
    
    all_subject_channels = [DATA_PATH_MSK + exams_msk[i] + '/T1_axial_02_01.nii',
                            DATA_PATH_MSK + exams_msk[i] + '/T1_axial_slope1.nii',
                            DATA_PATH_MSK + exams_msk[i] + '/T1_axial_slope2.nii']   
    # T1_pre_nii_path = DATA_PATH_MSK + exams_msk[i] + '/T1_axial_01_01.nii'  
    # t1, s1, s2, affine, p95 = load_and_preprocess(all_subject_channels, T1_pre_nii_path, apply_p95=True, t1post_factor=30)
    t1,X_res, Y_res, Z_res  = load_and_preprocess(all_subject_channels, apply_p95=True, t1post_factor=40)
    # try:
    #     t1 = nib.load(all_subject_channels[0])
    # except:
    #     continue
    df = df.append({'Site':'MSK','resolution_x':X_res,'resolution_y':Y_res,'resolution_z':Z_res,
                    'shape_x':t1.shape[0], 'shape_y':t1.shape[1], 'shape_z':t1.shape[2]}, ignore_index=True)#, 'T1pre_p95':p95, 'T1post_max':np.max(t1),'T1post_average':np.mean(t1), 'slope1_max':np.max(s1),'slope2_max':np.max(s2), 'T1post_median':np.median(t1), 'slope1_median':np.median(s1), 'slope2_median':np.median(s2)}, ignore_index=True)


    
    all_subject_channels = [DATA_PATH_MSK_AXIAL + exams_msk_axial[i] + '/T1_axial_02_01.nii.gz',
                            DATA_PATH_MSK_AXIAL + exams_msk_axial[i] + '/T1_axial_slope1.nii.gz',
                            DATA_PATH_MSK_AXIAL + exams_msk_axial[i] + '/T1_axial_slope2.nii.gz']   
    # T1_pre_nii_path = DATA_PATH_MSK + exams_msk[i] + '/T1_axial_01_01.nii'  
    # t1, s1, s2, affine, p95 = load_and_preprocess(all_subject_channels, T1_pre_nii_path, apply_p95=True, t1post_factor=30)
    
    t1,X_res, Y_res, Z_res = load_and_preprocess(all_subject_channels, apply_p95=True, t1post_factor=40)
    # try:
    #     t1 = nib.load(all_subject_channels[0])
    # except:
    #     continue
    
    df = df.append({'Site':'MSK_axial','resolution_x':X_res,'resolution_y':Y_res,'resolution_z':Z_res,
                    'shape_x':t1.shape[0], 'shape_y':t1.shape[1], 'shape_z':t1.shape[2]}, ignore_index=True)#, 'T1pre_p95':p95, 'T1post_max':np.max(t1),'T1post_average':np.mean(t1), 'slope1_max':np.max(s1),'slope2_max':np.max(s2), 'T1post_median':np.median(t1), 'slope1_median':np.median(s1), 'slope2_median':np.median(s2)}, ignore_index=True)
    
    
    df.to_csv(OUTPUT_PATH + OUTPUT_NAME)
    

#%%

# raw
df = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Duke_predictions/predictions/duke_vs_MSK_vs_axial_resolution.csv')

# pre-processed
df = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Duke_predictions/predictions/duke_vs_MSK_vs_axial_resolution_preprocessed.csv')


df = df.loc[df['resolution_z'] < 4]

import seaborn as sns

plt.figure(figsize=(5,10))

plt.subplot(311)
sns.kdeplot(df.loc[df['Site'] == 'MSK', 'resolution_x'], label='MSK sagittal res_x')
sns.kdeplot(df.loc[df['Site'] == 'MSK_axial', 'resolution_x'], label='MSK axial res_x')
sns.kdeplot(df.loc[df['Site'] == 'Duke', 'resolution_x'], label='Duke axial res_x')

plt.legend()

plt.subplot(312)
sns.kdeplot(df.loc[df['Site'] == 'MSK', 'resolution_y'], label='MSK sagittal res_y')
sns.kdeplot(df.loc[df['Site'] == 'MSK_axial', 'resolution_y'], label='MSK axial res_y')
sns.kdeplot(df.loc[df['Site'] == 'Duke', 'resolution_y'], label='Duke axial res_y')
plt.legend()


plt.subplot(313)
sns.kdeplot(df.loc[df['Site'] == 'MSK', 'resolution_z'], label='MSK sagittal res_z')
sns.kdeplot(df.loc[df['Site'] == 'MSK_axial', 'resolution_z'], label='MSK axial res_z')
sns.kdeplot(df.loc[df['Site'] == 'Duke', 'resolution_z'], label='Duke axial res_z')
plt.legend()

plt.tight_layout()


#%%


plt.figure(figsize=(5,10))

plt.subplot(311)
plt.hist(df.loc[df['Site'] == 'MSK', 'shape_x'], label='MSK sagittal shape_x', alpha=0.5)
plt.hist(df.loc[df['Site'] == 'MSK_axial', 'shape_x'], label='MSK axial shape_x', alpha=0.5)
plt.hist(df.loc[df['Site'] == 'Duke', 'shape_x'], label='Duke axial shape_x', alpha=0.5)
plt.legend()

plt.subplot(312)
plt.hist(df.loc[df['Site'] == 'MSK', 'shape_y'], label='MSK sagittal shape_y', alpha=0.5)
plt.hist(df.loc[df['Site'] == 'MSK_axial', 'shape_y'], label='MSK axial shape_y', alpha=0.5)
plt.hist(df.loc[df['Site'] == 'Duke', 'shape_y'], label='Duke axial shape_y', alpha=0.5)
plt.legend()

plt.subplot(313)
plt.hist(df.loc[df['Site'] == 'MSK', 'shape_z'], label='MSK sagittal shape_z', alpha=0.5)
plt.hist(df.loc[df['Site'] == 'MSK_axial', 'shape_z'], label='MSK axial shape_z', alpha=0.5)
plt.hist(df.loc[df['Site'] == 'Duke', 'shape_z'], label='Duke axial shape_z', alpha=0.5)
plt.legend()

plt.tight_layout()






    #%%
from scipy.stats import ranksums

df = pd.read_csv(OUTPUT_PATH + OUTPUT_NAME)




# df = df[['Site', 'resolution','T1pre_p95', 'T1post_max','T1post_average',
#        'slope1_max', 'slope2_max', 'T1post_median', 'slope1_median',
#        'slope2_median']]

print(len(df))
df.columns
i = 0
n=3
m=4
plt.figure(figsize=(15,15))
for ATTRIBUTE in df.columns[4:]:
    
    w,p1 = ranksums(df.loc[df['Site'] == 'Duke', ATTRIBUTE], df.loc[df['Site'] == 'MSK', ATTRIBUTE])
    
    i+=1
    plt.subplot(n,m,i)
    plt.title(f'\nMSK vs Duke:{round(p1,2)}')
    plt.hist(df.loc[df['Site'] == 'Duke', ATTRIBUTE], label='Duke', alpha=0.5)
    plt.hist(df.loc[df['Site'] == 'MSK', ATTRIBUTE], label='MSK', alpha=0.5)
    plt.xlabel(ATTRIBUTE)
    
    plt.legend()

plt.tight_layout()

