#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:47:06 2023

@author: deeperthought
"""


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="2"
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


def make_prediction_whole_scan(model, all_data, clinic_info_exam, USE_CONTRALATERAL, USE_PREVIOUS):
    
    t1post, slope1, slope2 = all_data[0], all_data[1], all_data[2]

    if USE_CONTRALATERAL:
        t1post_contra, slope1_contra, slope2_contra = all_data[3], all_data[4], all_data[5]

    if USE_PREVIOUS:
        t1post_previous, slope1_previous, slope2_previous = all_data[-3], all_data[-2], all_data[-1]
    
    slice_preds = []
    for i in range(t1post.shape[0]):

        if not USE_CONTRALATERAL and not USE_PREVIOUS:
            X = np.expand_dims(np.stack([t1post[i],slope1[i],slope2[i]],-1),0)
            
        if USE_CONTRALATERAL and not USE_PREVIOUS:
            X = np.expand_dims(np.stack([t1post[i],slope1[i],slope2[i], t1post_contra[i],slope1_contra[i],slope2_contra[i]],-1),0)

        if not USE_CONTRALATERAL and USE_PREVIOUS:
            X = np.expand_dims(np.stack([t1post[i],slope1[i],slope2[i], t1post_previous[i],slope1_previous[i],slope2_previous[i]],-1),0)

        if USE_CONTRALATERAL and USE_PREVIOUS:
            X = np.expand_dims(np.stack([t1post[i],slope1[i],slope2[i], t1post_contra[i],slope1_contra[i],slope2_contra[i], t1post_previous[i], slope1_previous[i], slope2_previous[i]],-1),0)
              
        if len(clinic_info_exam) == 1:
            yhat = model.predict([X,clinic_info_exam ])
        
        else:
            yhat = model.predict(X)
        slice_preds.append(yhat[0,1])
    return slice_preds

#%%
PRETRAINED_MODEL = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/best_model.h5'
best_model = load_model_frozen(PRETRAINED_MODEL)
    

clinical = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Clinical_Data_Train.csv')

DATA = pd.read_csv('/home/deeperthought/Documents/Papers_and_grants/Risk_paper/data/RISK/results_sheet_BIRADS123.csv')


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#%%

DATA = DATA[['scanID','y_true']]


DATA_converts = DATA.loc[DATA['y_true'] == 1]

partition = np.load('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Data.npy', allow_pickle=True).item()

labels = np.load('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Labels.npy', allow_pickle=True).item()

partition.keys()

train_scanids = [x[:31] for x in partition['train']]
val_scanids = [x[:31] for x in partition['validation']]

len(DATA.loc[DATA['scanID'].isin(train_scanids)])

len(DATA_converts.loc[DATA_converts['scanID'].isin(train_scanids)])

labels = dict(zip(DATA_converts['scanID'],DATA_converts['y_true']))

#%%

clinical_df = pd.read_excel('/home/deeperthought/Projects/MSKCC/MSKCC/Data_spreadsheets/Diamond_and_Gold/CCNY_CLINICAL_4_17_2019.xlsx', header=[0,1])    

clinical_df.columns

X = [x[:29] for x in DATA_converts['scanID']]

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

DATA_PATH = '/home/deeperthought/kirby_MSK/alignedNii-Nov2019/'

OUTPUT_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/converts_results/'

if os.path.exists(OUTPUT_PATH + 'results.csv'):
    results = pd.read_csv(OUTPUT_PATH + 'results.csv')

else:
    results = pd.DataFrame(columns=['scan','y_pred', 'y_true','max_slice'])

for row in DATA_converts.iterrows():
    
    SCANID = row[1]['scanID']
    
    EXAM = SCANID[:-2]
    SIDE = SCANID[-1]
    if SIDE == 'l':
        SIDE = 'left'
    else:
        SIDE = 'right'
    
    all_subject_channels = [DATA_PATH + EXAM + '/T1_{}_02_01.nii'.format(SIDE),
                            DATA_PATH + EXAM + '/T1_{}_slope1.nii'.format(SIDE),
                            DATA_PATH + EXAM + '/T1_{}_slope2.nii'.format(SIDE)]
    
    T1_pre_nii_path = DATA_PATH + EXAM + '/T1_{}_01_01.nii'.format(SIDE)
    
    t1post, slope1, slope2 = load_and_preprocess(all_subject_channels, T1_pre_nii_path)
    
    clinic_info_exam = clinical_features.loc[clinical_features['exam'] == EXAM,[u'Family Hx',u'Age',u'ETHNICITY_HISPANIC OR LATINO',u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',u'RACE_ASIAN-FAR EAST/INDIAN SUBCONT',u'RACE_BLACK OR AFRICAN AMERICAN',u'RACE_NATIVE AMERICAN-AM IND/ALASKA',u'RACE_NATIVE HAWAIIAN OR PACIFIC ISL',u'RACE_UNKNOWN',u'RACE_WHITE']].values
    
    X = np.stack([t1post, slope1, slope2], axis=-1)
    
    slice_preds = []
    for i in range(X.shape[0]):

        yhat = best_model.predict([X[i:i+1],clinic_info_exam ])
    
        slice_preds.append(yhat[0,1])
            
    results = results.append({'scan':SCANID,'y_pred':np.max(slice_preds), 'y_true':1, 
                            'max_slice':np.argmax(slice_preds)}, 
            ignore_index=True)

    results.to_csv(OUTPUT_PATH + '/converts_result.csv', index=False)
    
   