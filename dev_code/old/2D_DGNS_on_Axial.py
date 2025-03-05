#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:47:06 2023

@author: deeperthought
"""


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="1"
tf.keras.backend.set_session(tf.Session(config=config))

import os
import numpy as np
import matplotlib.pyplot as plt
#from skimage.transform import resize
import nibabel as nib
from skimage.transform import resize
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)



#%%
def load_and_preprocess(all_subject_channels, T1_pre_nii_path):
    t1post = nib.load(all_subject_channels[0]).get_data()
    slope1 = nib.load(all_subject_channels[1]).get_data()
    slope2 = nib.load(all_subject_channels[2]).get_data()    
    
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

    return t1post, slope1, slope2

def load_data_prediction(scanID ,labels, MASTER, USE_CONTRALATERAL, USE_PREVIOUS):
    "Takes scanIDs (not paths) and loads raw MRIs from /home/deeperthought/kirby_MSK/alignedNii-Nov2019/, preprocesses and returns"
    exam = scanID[:-2]
    patient = scanID[:20]
    side = 'right'
    contra_side = 'left'
    if scanID[-1] == 'l': 
        side = 'left'
        contra_side = 'right'
    pathology = labels[scanID]
    
    MRI_PATH = '/home/deeperthought/kirby_MSK/alignedNii-Nov2019/'
    MRI_ALIGNED_HISTORY_PATH = '/home/deeperthought/kirbyPRO/saggital_Nov2019_alignedHistory/'
    
    segmentation_GT = MASTER.loc[MASTER['Scan_ID'] == scanID, 'Segmentation_Path'].values[0]
    contralateral_available = MASTER.loc[MASTER['Scan_ID'] == scanID, 'Contralateral Available'].values[0]
    previous_available = MASTER.loc[MASTER['Scan_ID'] == scanID, 'Previous Available'].values[0]
        
    print('DEBUG: {} segmentation_GT: {}'.format(scanID,segmentation_GT))  
    
    all_subject_channels = [MRI_PATH + exam + '/T1_{}_02_01.nii'.format(side),
                           MRI_PATH + exam + '/T1_{}_slope1.nii'.format(side),
                           MRI_PATH + exam + '/T1_{}_slope2.nii'.format(side)]

    T1_pre_nii_path = MRI_PATH + exam + '/T1_{}_01_01.nii'.format(side)

    t1post, slope1, slope2 = load_and_preprocess(all_subject_channels, T1_pre_nii_path)


       
    if USE_CONTRALATERAL:
        t1post_contra, slope1_contra, slope2_contra = np.zeros(t1post.shape),np.zeros(t1post.shape)  ,np.zeros(t1post.shape)      

        if contralateral_available:
            all_contralateral_channels = [MRI_PATH + exam + '/T1_{}_02_01_contralateral_aligned.nii'.format(contra_side),
                                          MRI_PATH + exam + '/T1_{}_slope1_contralateral_aligned.nii'.format(contra_side),
                                          MRI_PATH + exam + '/T1_{}_slope2_contralateral_aligned.nii'.format(contra_side)]
            
            T1_pre_nii_path_contralateral = MRI_PATH + exam + '/T1_{}_01_01.nii'.format(contra_side)
    
            if not os.path.exists(all_contralateral_channels[0]):
                print('previous exam not aligned yet.. skip')
            else:
                t1post_contra, slope1_contra, slope2_contra = load_and_preprocess(all_contralateral_channels, T1_pre_nii_path_contralateral)


    if USE_PREVIOUS:
        t1post_previous, slope1_previous, slope2_previous = np.zeros(t1post.shape),np.zeros(t1post.shape)  ,np.zeros(t1post.shape)      

        if previous_available:
    
            breast_history = MASTER.loc[(MASTER['DE-ID'] == patient) & (MASTER['Scan_ID'].str[-1] == scanID[-1]), 'Scan_ID'].values
            previous_exam = breast_history[np.argwhere(breast_history == scanID)[0][0] - 1][21:-2]        
            
            MRI_ALIGNED_HISTORY_PATH + patient + '/' + previous_exam
            all_previous_channels = [MRI_ALIGNED_HISTORY_PATH + patient + '/' + previous_exam+ '/T1_{}_02_01_TimeAlignment_to_{}.nii.gz'.format(side, exam[21:]),
                                     MRI_ALIGNED_HISTORY_PATH + patient + '/' + previous_exam+ '/T1_{}_slope1_TimeAlignment_to_{}.nii.gz'.format(side, exam[21:]),
                                     MRI_ALIGNED_HISTORY_PATH + patient + '/' + previous_exam+ '/T1_{}_slope2_TimeAlignment_to_{}.nii.gz'.format(side, exam[21:])]            
    
            T1_pre_nii_path_previous = MRI_PATH + patient + '_' + previous_exam + '/T1_{}_01_01.nii'.format(side)
        
            if not os.path.exists(all_previous_channels[0]):
                print('previous exam not aligned yet.. skip')
            else:
                t1post_previous, slope1_previous, slope2_previous = load_and_preprocess(all_previous_channels, T1_pre_nii_path_previous)

        
    if not pd.isnull(segmentation_GT):
        groundtruth = nib.load(segmentation_GT).get_data()
        if np.sum(groundtruth) == 0:
            segmented_slice = 0            
        else:
            segmented_slice = list(set(np.where(groundtruth > 0)[0]))[0]
    else:
        groundtruth = 0
        segmented_slice = 0
        
    if not USE_CONTRALATERAL and not USE_PREVIOUS:
        
        return pathology, segmented_slice, t1post, slope1, slope2

    if not USE_CONTRALATERAL and USE_PREVIOUS:
        
        return pathology, segmented_slice, t1post, slope1, slope2, t1post_previous, slope1_previous, slope2_previous    

    if USE_CONTRALATERAL and not USE_PREVIOUS:

        return pathology, segmented_slice, t1post, slope1, slope2, t1post_contra, slope1_contra, slope2_contra

    if USE_CONTRALATERAL and USE_PREVIOUS:

        return pathology, segmented_slice, t1post, slope1, slope2, t1post_contra, slope1_contra, slope2_contra, t1post_previous, slope1_previous, slope2_previous    



def unfreeze_layers(model):
    model_type = type(model) 
    for i in model.layers:
        i.trainable = True
        if type(i) == model_type:
            unfreeze_layers(i)
    return model


def load_model_frozen(PATH):
    #model.load_weights(FOLDER + NAME + '.h5')
    model_loaded = tf.keras.models.load_model(PATH)
    model = unfreeze_layers(model_loaded)
    adam = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=adam, loss='binary_crossentropy',  metrics=['accuracy'])  
    return model

    
def add_age(clinical_features, clinical_df):
  ages = clinical_df['Unnamed: 1_level_0']
  ages['DE-ID']  = clinical_df['Unnamed: 0_level_0']['DE-ID']
  #ages['DE-ID'] = ages.index
  ages.reset_index(level=0, inplace=True)
  ages = ages[['DE-ID','DOB']]  
  clinical_features2 = clinical_features.copy()
  clinical_features2['DE-ID'] = clinical_features2['exam'].apply(lambda x : x[:20]) 
  clinical_features2 = clinical_features2.merge(ages, on=['DE-ID'])
  clinical_features2['Age'] = clinical_features2.apply(lambda row : int(row['exam'][-8:-4]) - int(row['DOB']), axis=1)
  clinical_features2 = clinical_features2[['exam','Age']]
  return clinical_features2


def add_ethnicity_oneHot(df, clinical):
 
  clinical_df = pd.concat([clinical['Unnamed: 0_level_0']['DE-ID'], clinical['Unnamed: 4_level_0']['ETHNICITY'], clinical['Unnamed: 3_level_0']['RACE']], axis=1)
  
  clinical_df = clinical_df.set_index('DE-ID')
  clinical_df.loc[clinical_df['ETHNICITY'] == 'NO VALUE ENTERED'] = 'UNKNOWN'  
  clinical_df.loc[clinical_df['RACE'] == 'OTHER'] = 'UNKNOWN'  
  clinical_df.loc[clinical_df['RACE'] == 'PT REFUSED TO ANSWER'] = 'UNKNOWN'  
  clinical_df.loc[clinical_df['RACE'] == 'NO VALUE ENTERED'] = 'UNKNOWN'  

  
  clinical_df = pd.get_dummies(clinical_df)

  df['DE-ID'] = df['exam'].str[:20] 
  
  df2 =  pd.merge(df, clinical_df, on='DE-ID')

  return df2

def add_family_hx(df, clinical):
  fam = pd.DataFrame(columns=['DE-ID','Family Hx'])
  fam['Family Hx'] = clinical['Family Hx']['Family Hx']
  fam['Family Hx'] = fam['Family Hx'].apply(lambda x : 1 if x == 'Yes' else 0)
  fam['DE-ID'] = clinical['Unnamed: 0_level_0']['DE-ID']
  fam.reset_index(level=0, inplace=True) 
  df2 = df.copy()
  df2['DE-ID'] = df2['exam'].apply(lambda x : x[:20]) 
  df3 = df2.merge(fam, on=['DE-ID'])
  df3.head()
  df3 = df3[['exam','Family Hx']]
  df4 = df3.merge(df, on=['exam'])
  df4 = df4.loc[df4['exam'].isin(df['exam'])]
  return df4

#%%
PRETRAINED_MODEL = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/best_model.h5'
best_model = load_model_frozen(PRETRAINED_MODEL)
    

clinical = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Clinical_Data_Train.csv')

REDCAP = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/REDCAP/2023/REDCAP_EZ.csv') 


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#%%

labels = np.load('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Labels.npy', allow_pickle=True).item()

#%%
DATA_PATH = '/home/deeperthought/kirby_MSK/alignedNiiAxial-Nov2019/'

exams = os.listdir(DATA_PATH)


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


#%%

clinical_df = pd.read_excel('/home/deeperthought/Projects/MSKCC/MSKCC/Data_spreadsheets/Diamond_and_Gold/CCNY_CLINICAL_4_17_2019.xlsx', header=[0,1])    

clinical_df.columns

X = list(AXIAL_SCANS['Exam'])

clinical_features = pd.DataFrame(columns=['exam'])
clinical_features['exam'] = X
clinical_features = add_age(clinical_features, clinical_df)
clinical_features = add_ethnicity_oneHot(clinical_features, clinical_df)
clinical_features = add_family_hx(clinical_features, clinical_df)
clinical_features['Age'] = clinical_features['Age']/100.
clinical_features = clinical_features.drop_duplicates()
CLINICAL_FEATURE_NAMES = [u'Family Hx',u'Age',u'ETHNICITY_HISPANIC OR LATINO',u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',u'RACE_ASIAN-FAR EAST/INDIAN SUBCONT',u'RACE_BLACK OR AFRICAN AMERICAN',u'RACE_NATIVE AMERICAN-AM IND/ALASKA',u'RACE_NATIVE HAWAIIAN OR PACIFIC ISL',u'RACE_UNKNOWN',u'RACE_WHITE']
   
clinical_features[clinical_features.isnull().any(axis=1)]

np.sum(pd.isnull(clinical_features).any(axis=1))


#%%

#EXAM = 'MSKCC_16-328_1_00001_20150710' # benign
#
#EXAM = 'MSKCC_16-328_1_02914_20140516' # malignant right

RESOLUTIONS = {}
SHAPES ={}

results = pd.DataFrame(columns=['Exam','y_pred','y_true','max_slice'])

OUTPUT_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/axial_results/'

if os.path.exists(OUTPUT_PATH + 'results.csv'):
    results = pd.read_csv(OUTPUT_PATH + 'results.csv')

for row in AXIAL_SCANS.iterrows():
    
    EXAM = row[1]['Exam']
    
    if EXAM in results['Exam'].values:
        print('Already done, continue..')
        continue
    
    all_subject_channels = [DATA_PATH + EXAM + '/T1_axial_02_01.nii',
                            DATA_PATH + EXAM + '/T1_axial_slope1.nii',
                            DATA_PATH + EXAM + '/T1_axial_slope2.nii']
    
    T1_pre_nii_path = DATA_PATH + EXAM + '/T1_axial_01_01.nii'  
    
    try:
        hdr = nib.load(all_subject_channels[0])
        t1pre = nib.load(T1_pre_nii_path).get_data()
        t1post = nib.load(all_subject_channels[0]).get_data()
        slope1 = nib.load(all_subject_channels[1]).get_data()
        slope2 = nib.load(all_subject_channels[2]).get_data()    
    except:
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
    RESOLUTIONS[EXAM] = resolution
    SHAPES[EXAM] = t1post.shape
       
    if t1post.shape[0] < 500:
        print('Weird low res axial. Skip')
        continue
    # CROP
#    plt.imshow(np.max(t1post, 0)) # maximum projection, sagittal.
#    plt.plot(np.max(np.max(t1post, 0),1)) # where it dips its end of breast.
#    
    projection_1d = np.max(np.max(t1post, 0),1)

    breast_end = np.argmin(np.diff(projection_1d[np.arange(0,len(projection_1d),5)]))*5
    breast_end = breast_end + 10 # add some border
    breast_end = np.max([breast_end, 256]) # if breast is small, just crop to 256
        
    t1post = t1post[:, (breast_end-256):breast_end, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    t1pre = t1pre[:, (breast_end-256):breast_end, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    slope1 = slope1[:, (breast_end-256):breast_end, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    slope2 = slope2[:, (breast_end-256):breast_end, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    
#    
#    t1post = t1post[:, 50:512+50, 0:512]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
#    t1pre = t1pre[:, 50:512+50, 0:512]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
#    slope1 = slope1[:, 50:512+50, 0:512]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
#    slope2 = slope2[:, 50:512+50, 0:512]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
#    

    if resolution[0] > 0.5:
        output_shape = (t1post.shape[0], t1post.shape[1]*2, int(t1post.shape[2]* (resolution[2]/0.33) ))
#        output_shape = (512*2, 512*2, int(192*3.3))

    else:
        print('new res, inspect.')
        break   
    
    # RESIZE to match resolutions.  I need final resolution: (whatever, 0.3, 0.3)
    
    t1post = resize(t1post, output_shape=output_shape, preserve_range=True, anti_aliasing=False)
    slope1 = resize(slope1, output_shape=output_shape, preserve_range=True, anti_aliasing=False)
    slope2 = resize(slope2, output_shape=output_shape, preserve_range=True, anti_aliasing=False)
    t1pre = resize(t1pre, output_shape=output_shape, preserve_range=True, anti_aliasing=False)
    

    if t1post.shape[2] < 512:
        border = 512 - t1post.shape[2]
        t1post = np.pad(t1post, ((0,0),(0,0),(0,border)), 'minimum')
        slope1 = np.pad(slope1, ((0,0),(0,0),(0,border)), 'minimum')
        slope2 = np.pad(slope2, ((0,0),(0,0),(0,border)), 'minimum')
        t1pre = np.pad(t1pre, ((0,0),(0,0),(0,border)), 'minimum')
        
    else:    
        t1post = t1post[:, :, 0:512]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
        slope1 = slope1[:, :, 0:512]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
        slope2 = slope2[:, :, 0:512]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
        t1pre = t1pre[:, :, 0:512]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast

    p95 = np.percentile(t1pre,95)
        
    t1post = t1post/p95    
    slope1 = slope1/p95    
    slope2 = slope2/p95    
    
    t1post = t1post/float(40)
    slope1 = slope1/float(0.3)
    slope2 = slope2/float(0.12)             


    clinical_mode = clinical[[u'Family Hx',u'Age',u'ETHNICITY_HISPANIC OR LATINO',u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',u'RACE_ASIAN-FAR EAST/INDIAN SUBCONT',u'RACE_BLACK OR AFRICAN AMERICAN',u'RACE_NATIVE AMERICAN-AM IND/ALASKA',u'RACE_NATIVE HAWAIIAN OR PACIFIC ISL',u'RACE_UNKNOWN',u'RACE_WHITE']].mode().values

    subj_df = clinical_features.loc[clinical_features['DE-ID'] == EXAM[:20]]
    
    if len(subj_df) > 0:
        
        clinic_info_exam = subj_df.loc[subj_df['exam'] == EXAM,[u'Family Hx',u'Age',u'ETHNICITY_HISPANIC OR LATINO',u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',u'RACE_ASIAN-FAR EAST/INDIAN SUBCONT',u'RACE_BLACK OR AFRICAN AMERICAN',u'RACE_NATIVE AMERICAN-AM IND/ALASKA',u'RACE_NATIVE HAWAIIAN OR PACIFIC ISL',u'RACE_UNKNOWN',u'RACE_WHITE']].values

    else:
        print('no clinical info patient, use average')
        clinic_info_exam = clinical_mode


    X = np.stack([t1post, slope1, slope2], axis=-1)
    
    X.shape
    
    number_slices = t1post.shape[0]
    left_breast = X[:number_slices/2]
    right_breast = X[number_slices/2:]
    
    ################## LEFT ########################################################
    preds = []
    for i in range(1,left_breast.shape[0]):
        #print(i)
        pred = best_model.predict([left_breast[i-1:i], clinic_info_exam])
        preds.append(pred[0,1])
        
    global_prediction = np.max(preds)
    max_slice = np.argmax(preds)
    
    side = 'left'
    # SAVE RESULT
    if row[1]['left_breast_tumor_status'] == 'Malignant':
        y_true = 1
    else:
        y_true = 0
        
    results = results.append({'Exam':EXAM, 'Side':side,'y_pred':global_prediction, 'y_true': y_true, 'max_slice':max_slice}, ignore_index=True)
    
    results.to_csv(OUTPUT_PATH + 'results.csv', index=False)
    
    # DISPLAY
        
    axial_projection_t1post = np.max(t1post,-1)
    axial_projection_slope1 = np.max(slope1,-1)
    
    fig, ax = plt.subplots(4,1, sharex=True, figsize=(5,11))    
    plt.suptitle('GT={} , side={}: pred={}'.format(y_true, side, round(global_prediction,2)))
    plt.subplots_adjust(hspace=.0)
    ax[0].plot(preds)
#    ax[1].imshow(np.rot90(t1post[:,:,300]), cmap='gray' )
    ax[1].imshow(np.rot90(axial_projection_t1post), cmap='gray' , vmax=np.percentile(axial_projection_t1post,99.9))
    ax[1].set_aspect('auto')
#    ax[2].imshow(np.rot90(slope1[:,:,300]), cmap='gray' , vmax = np.percentile(slope1,99))
    ax[2].imshow(np.rot90(axial_projection_slope1), cmap='gray' , vmax=np.percentile(axial_projection_slope1,99.9))

    ax[2].set_aspect('auto')
    
    ax[3].imshow(np.rot90(t1post[max_slice]), cmap='gray' )

    plt.savefig(OUTPUT_PATH + 'figures/' + EXAM + '_' + side )
    plt.close()

    ################## RIGHT ########################################################
    preds = []
    for i in range(1,right_breast.shape[0]):
        #print(i)
        pred = best_model.predict([right_breast[i-1:i], clinic_info_exam])
        preds.append(pred[0,1])
        
    global_prediction = np.max(preds)
    max_slice = np.argmax(preds)
    
    side = 'right'
    # SAVE RESULT
    if row[1]['right_breast_tumor_status'] == 'Malignant':
        y_true = 1
    else:
        y_true = 0
        
    results = results.append({'Exam':EXAM, 'Side':side,'y_pred':global_prediction, 'y_true': y_true, 'max_slice':max_slice}, ignore_index=True)
    
    results.to_csv(OUTPUT_PATH + 'results.csv', index=False)
    
    # DISPLAY
        
    axial_projection_t1post = np.max(t1post,-1)
    axial_projection_slope1 = np.max(slope1,-1)
    
    fig, ax = plt.subplots(4,1, sharex=True, figsize=(5,11))    
    plt.suptitle('GT={} , side={}: pred={}'.format(y_true, side, round(global_prediction,2)))
    plt.subplots_adjust(hspace=.0)
    ax[0].plot(([0]*int(number_slices/2)) + list(preds))
#    ax[1].imshow(np.rot90(t1post[:,:,300]), cmap='gray' )
    ax[1].imshow(np.rot90(axial_projection_t1post), cmap='gray' , vmax=np.percentile(axial_projection_t1post,99.9))
    ax[1].set_aspect('auto')
#    ax[2].imshow(np.rot90(slope1[:,:,300]), cmap='gray' , vmax = np.percentile(slope1,99))
    ax[2].imshow(np.rot90(axial_projection_slope1), cmap='gray' , vmax=np.percentile(axial_projection_slope1,99.9))

    ax[2].set_aspect('auto')
    
    ax[3].imshow(np.rot90(t1post[max_slice]), cmap='gray' )

    plt.savefig(OUTPUT_PATH + 'figures/' + EXAM + '_' + side )
    plt.close()
