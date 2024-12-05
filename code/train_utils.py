#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 13:09:02 2024

@author: deeperthought
"""


# GPU = 2
import tensorflow as tf
# if tf.__version__[0] == '1':
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     config.gpu_options.visible_device_list="0"
#     tf.keras.backend.set_session(tf.Session(config=config))

# elif tf.__version__[0] == '2':
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if gpus:
#       # Restrict TensorFlow to only use the first GPU
#       try:
#         tf.config.experimental.set_visible_devices(gpus[GPU], 'GPU')
#         tf.config.experimental.set_memory_growth(gpus[GPU], True)
#       except RuntimeError as e:
#         # Visible devices must be set at program startup
#         print(e)

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from skimage.transform import resize

#%%   Data handling functions should be imported from utils and should be the new versions

# def load_and_preprocess(all_subject_channels, T1_pre_nii_path):
#     t1post = nib.load(all_subject_channels[0]).get_data()
#     slope1 = nib.load(all_subject_channels[1]).get_data()
#     slope2 = nib.load(all_subject_channels[2]).get_data()    
    
#     if t1post.shape[1] != 512:
#         output_shape = (t1post.shape[0],512,512)
#         t1post = resize(t1post, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
#         slope1 = resize(slope1, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
#         slope2 = resize(slope2, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')

#     p95 = np.percentile(nib.load(T1_pre_nii_path).get_data(),95)
        
#     t1post = t1post/p95    
#     slope1 = slope1/p95    
#     slope2 = slope2/p95    

#     t1post = t1post/float(40)
#     slope1 = slope1/float(0.3)
#     slope2 = slope2/float(0.12)     

#     return t1post, slope1, slope2

from utils import load_and_preprocess

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

    processed_images, _ = load_and_preprocess(all_subject_channels, T1_pre_nii_path)
    t1post, slope1, slope2 = processed_images[:,:,:,0], processed_images[:,:,:,1], processed_images[:,:,:,2]

       
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


def get_results_on_dataset(model, scans_list, labels, Data_description, NAME, OUT, USE_CLINICAL, USE_CONTRALATERAL, USE_PREVIOUS, clinical_info, name='VAL'):
    
    if os.path.exists(OUT + NAME + '/{}_result.csv'.format(name)):
        print('Found previous results in : {}'.format(OUT + NAME + '/{}_result.csv'.format(name)))
        result = pd.read_csv(OUT + NAME + '/{}_result.csv'.format(name))
        print('Loading previous results.')
        
        print('{} scans already done.'.format(len(result)))
    else:
        result = pd.DataFrame(columns=['scan','y_pred','y_true','max_slice','GT_slice','slice_preds'])
    N = 0
    
    scans_list = [x for x in scans_list if x not in result['scan'].values]
    
    TOT = len(scans_list)
    
    for scan in scans_list:
        #scan = scan[:31]
        N += 1
        print('{}/{}'.format(N,TOT))
    
        all_data = load_data_prediction(scan,labels, Data_description, USE_CONTRALATERAL, USE_PREVIOUS) 
        
        pathology = all_data[0]
        segmented_slice = all_data[1]
        
        clinic_info_exam = 0
        if USE_CLINICAL:
            clinic_info_exam = clinical_info.loc[clinical_info['scan_ID'] == scan,[u'Family Hx',u'Age',u'ETHNICITY_HISPANIC OR LATINO',u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',u'RACE_ASIAN-FAR EAST/INDIAN SUBCONT',u'RACE_BLACK OR AFRICAN AMERICAN',u'RACE_NATIVE AMERICAN-AM IND/ALASKA',u'RACE_NATIVE HAWAIIAN OR PACIFIC ISL',u'RACE_UNKNOWN',u'RACE_WHITE']].values

        slice_preds = make_prediction_whole_scan(model, all_data[2:], clinic_info_exam, USE_CONTRALATERAL, USE_PREVIOUS)
        
        result = result.append({'scan':scan,'y_pred':np.max(slice_preds), 'y_true':pathology, 
                                'max_slice':np.argmax(slice_preds), 'GT_slice':segmented_slice, 'slice_preds':slice_preds}, 
                ignore_index=True)

        print('{} - PATHOLOGY: {}\nsegmented_slice: {}, max_slice: {} = {}:'.format(scan, pathology, segmented_slice, np.argmax(slice_preds), np.max(slice_preds)))
 
        if N%10 == 0:
 
            result.to_csv(OUT + NAME + '/{}_result.csv'.format(name), index=False)
            
            roc_auc_test_final = roc_auc_score( [int(x) for x in result['y_true'].values],result['y_pred'].values)
            fpr_test, tpr_test, thresholds = roc_curve([int(x) for x in result['y_true'].values],result['y_pred'].values)
            malignants_test = result.loc[result['y_true'] == 1, 'y_pred']
            benigns_test = result.loc[result['y_true'] == 0, 'y_pred']         
            
            print('{} : AUC-ROC = {}'.format(name, roc_auc_test_final))
            
            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1)
            plt.plot(fpr_test, tpr_test, label='ROC curve (area = %0.3f)' % roc_auc_test_final)
            plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
            plt.xlim([-0.1, 1.1])
            plt.ylim([-0.1, 1.1])
            plt.xlabel('False Positive Rate or (1 - Specifity)')
            plt.ylabel('True Positive Rate or (Sensitivity)')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")   
            
            plt.subplot(1,3,2)                   
            plt.hist(malignants_test.values, color='r', alpha=1, bins=100)
            plt.hist(benigns_test.values, color='g', alpha=0.5, bins=100)
            plt.legend(['Malignants (N={})'.format(len(malignants_test)), 'Benigns  (N={})'.format(len(benigns_test))])
            plt.title('{} N = {}'.format(name, len(result)))
            
            plt.subplot(1,3,3)                   
            plt.hist(malignants_test.values, color='r', alpha=1, bins=100)
            plt.hist(benigns_test.values, color='g', alpha=0.5, bins=100)
            plt.yscale('log')
            plt.legend(['Malignants (N={})'.format(len(malignants_test)), 'Benigns  (N={})'.format(len(benigns_test))])
            plt.title('{} N = {}'.format(name, len(result)))
            plt.tight_layout()
            plt.savefig(OUT +  NAME + '/{}_result_ROC.png'.format(name), dpi=200)          
            plt.close()
    
    result.to_csv(OUT + NAME + '/{}_result.csv'.format(name), index=False)
    
    roc_auc_test_final = roc_auc_score( [int(x) for x in result['y_true'].values],result['y_pred'].values)
    fpr_test, tpr_test, thresholds = roc_curve([int(x) for x in result['y_true'].values],result['y_pred'].values)
    malignants_test = result.loc[result['y_true'] == 1, 'y_pred']
    benigns_test = result.loc[result['y_true'] == 0, 'y_pred']         
    
    print('{} : AUC-ROC = {}'.format(name, roc_auc_test_final))
    
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(fpr_test, tpr_test, label='ROC curve (area = %0.3f)' % roc_auc_test_final)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")   
    
    plt.subplot(1,3,2)                   
    plt.hist(malignants_test.values, color='r', alpha=1, bins=100)
    plt.hist(benigns_test.values, color='g', alpha=0.5, bins=100)
    plt.legend(['Malignants (N={})'.format(len(malignants_test)), 'Benigns  (N={})'.format(len(benigns_test))])
    plt.title('{} N = {}'.format(name, len(result)))
    
    plt.subplot(1,3,3)                   
    plt.hist(malignants_test.values, color='r', alpha=1, bins=100)
    plt.hist(benigns_test.values, color='g', alpha=0.5, bins=100)
    plt.yscale('log')
    plt.legend(['Malignants (N={})'.format(len(malignants_test)), 'Benigns  (N={})'.format(len(benigns_test))])
    plt.title('{} N = {}'.format(name, len(result)))
    plt.tight_layout()
    plt.savefig(OUT +  NAME + '/{}_result_ROC.png'.format(name), dpi=200)   
    
           

#%% Clinical Info functions  ---  Probably should make a data_utils own library
        
def add_age(df, clinical):
  ages = clinical['Unnamed: 1_level_0']
  ages['DE-ID']  = clinical['Unnamed: 0_level_0']['DE-ID']
  #ages['DE-ID'] = ages.index
  ages.reset_index(level=0, inplace=True)
  ages = ages[['DE-ID','DOB']]  
  df2 = df.copy()
  df2['ID_date'] = df2['scan_ID'].apply(lambda x : x[:-2])
  df2['DE-ID'] = df2['ID_date'].apply(lambda x : x[:-9]) 
  df3 = df2.merge(ages, on=['DE-ID'])
  df3.head()
  df3['Age'] = df3.apply(lambda row : int(row['ID_date'][-8:-4]) - int(row['DOB']), axis=1)
  df3 = df3[['scan_ID','Age']]
  df4 = df3.merge(df, on=['scan_ID'])
  df4 = df4.loc[df4['scan_ID'].isin(df['scan_ID'])]
  return df4

def add_ethnicity_oneHot(df, clinical):
 
  clinical_df = pd.concat([clinical['Unnamed: 0_level_0']['DE-ID'], clinical['Unnamed: 4_level_0']['ETHNICITY'], clinical['Unnamed: 3_level_0']['RACE']], axis=1)
  
  clinical_df = clinical_df.set_index('DE-ID')
  clinical_df.loc[clinical_df['ETHNICITY'] == 'NO VALUE ENTERED'] = 'UNKNOWN'  
  clinical_df.loc[clinical_df['RACE'] == 'OTHER'] = 'UNKNOWN'  
  clinical_df.loc[clinical_df['RACE'] == 'PT REFUSED TO ANSWER'] = 'UNKNOWN'  
  clinical_df.loc[clinical_df['RACE'] == 'NO VALUE ENTERED'] = 'UNKNOWN'  

  
  clinical_df = pd.get_dummies(clinical_df)

  df['DE-ID'] = df['scan_ID'].str[:20] 
  
  df2 =  pd.merge(df, clinical_df, on='DE-ID')

  return df2

def add_ethnicity(df, clinical):
  #clinical.columns
  ethn = clinical['Unnamed: 4_level_0']['ETHNICITY']
  race = clinical['Unnamed: 3_level_0']['RACE']
  DEIDs = clinical['Unnamed: 0_level_0']['DE-ID']
  feat = pd.DataFrame(columns=['ETHNICITY', 'RACE'])
  race[race == 'WHITE'] = 3
  race[race == 'BLACK OR AFRICAN AMERICAN'] = 2
  race[race == 'ASIAN-FAR EAST/INDIAN SUBCONT'] = 1
  race[~ race.isin([1,2,3])] = 0
  
  ethn[ethn == 'HISPANIC OR LATINO'] = 1
  ethn[ethn == 'NOT HISPANIC'] = -1
  ethn[~ ethn.isin([-1,1])] = 0
  
  feat['ETHNICITY'] = ethn.values
  feat['RACE'] = race.values
  feat['ETHNICITY'].value_counts()
  feat['DE-ID'] = DEIDs
  feat.reset_index(level=0, inplace=True)
  feat = feat[['DE-ID','ETHNICITY','RACE']]  
  df2 = df.copy()
  df2['ID_date'] = df2['scan_ID'].apply(lambda x : x[:-2])
  df2['DE-ID'] = df2['ID_date'].apply(lambda x : x[:-9]) 
  df3 = df2.merge(feat, on=['DE-ID'])
  df3.head()

  df3 = df3[['scan_ID','ETHNICITY','RACE']]
  df4 = df3.merge(df, on=['scan_ID'])
  df4 = df4.loc[df4['scan_ID'].isin(df['scan_ID'])]
  return df4

def add_family_hx(df, clinical):
  fam = pd.DataFrame(columns=['DE-ID','Family Hx'])
  fam['Family Hx'] = clinical['Family Hx']['Family Hx']
  fam['Family Hx'] = fam['Family Hx'].apply(lambda x : 1 if x == 'Yes' else 0)
  fam['DE-ID'] = clinical['Unnamed: 0_level_0']['DE-ID']
  fam.reset_index(level=0, inplace=True) 
  df2 = df.copy()
  df2['ID_date'] = df2['scan_ID'].apply(lambda x : x[:-2])
  df2['DE-ID'] = df2['ID_date'].apply(lambda x : x[:-9]) 
  df3 = df2.merge(fam, on=['DE-ID'])
  df3.head()
  df3 = df3[['scan_ID','Family Hx']]
  df4 = df3.merge(df, on=['scan_ID'])
  df4 = df4.loc[df4['scan_ID'].isin(df['scan_ID'])]
  return df4

def interleave_two_lists(lst1, lst2):
    result = []

    for pair in zip(lst1, lst2):
        result.extend(pair)

    if len(lst1) != len(lst2):
        lsts = [lst1, lst2]
        smallest = min(lsts, key = len)
        biggest = max(lsts, key = len)
        rest = biggest[len(smallest):]
        result.extend(rest)

    return result

#%% 

class DataGenerator_classifier(tf.keras.utils.Sequence): # inheriting from Sequence allows for multiprocessing functionalities
    'Generates data for Keras'
    def __init__(self, list_IDs,labels,clinical_info, data_path='', batch_size=4, dim=(512,512), n_channels=3,
                 n_classes=2, shuffledata=True, do_augmentation=True, use_clinical_info=False, 
                 use_contralateral=False, use_previous=False, data_description = ''):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.data_path = data_path
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffledata = shuffledata
        self.on_epoch_end()
        self.seed = 0
        self.do_augmentation = do_augmentation
        self.labels = labels
        self.clinical_info = clinical_info
        self.use_clinical_info = use_clinical_info
        self.use_contralateral = use_contralateral
        self.use_previous = use_previous
        self.data_description = data_description
        
        self.augmentor = tf.keras.preprocessing.image.ImageDataGenerator(
                    rotation_range=60,
                    shear_range=0.1,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest',
                )
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        
        # if self.do_augmentation:
        #     for i in range(self.batch_size):
        #         X[i] = X[i]*np.random.uniform(low=0.8, high=1.2, size=1)

        if self.use_clinical_info:
            clinic = np.zeros((self.batch_size,11))
            scanids = [ids.split('/')[-1][:31] for ids in list_IDs_temp]
            for i in range(len(scanids)):
                clinic[i] = self.clinical_info.loc[self.clinical_info['scan_ID'] == scanids[i],[u'Family Hx',u'Age',u'ETHNICITY_HISPANIC OR LATINO',u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',u'RACE_ASIAN-FAR EAST/INDIAN SUBCONT',u'RACE_BLACK OR AFRICAN AMERICAN',u'RACE_NATIVE AMERICAN-AM IND/ALASKA',u'RACE_NATIVE HAWAIIAN OR PACIFIC ISL',u'RACE_UNKNOWN',u'RACE_WHITE']].values

            return [X,clinic], tf.keras.utils.to_categorical(y, num_classes=self.n_classes) 
        
        else:
            return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes) 


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffledata == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.n_channels), dtype='float32')
        y = np.empty((self.batch_size, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):          
            # X[i,:,:,:3] = np.load(self.data_path + ID, allow_pickle=True)   # Here we add the path. ID can be the path
            X[i] = np.load(ID, allow_pickle=True)  

            y[i] = self.labels[ID.split('/')[-1][:31]]
            
                
#            if self.use_contralateral:
#                if self.data_description.loc[self.data_description['Scan_ID'] == ID[:31], 'Contralateral Available'].values[0] == 1:
#                    X[i,:,:,3:6] = np.load(self.data_path.replace('X','Contra') + ID, allow_pickle=True)   # Here we add the path. ID can be the path
#
#            if self.use_previous:
#                if self.data_description.loc[self.data_description['Scan_ID'] == ID[:31], 'Previous Available'].values[0] == 1:
#                    X[i,:,:,-3:] = np.load(self.data_path.replace('X','Previous') + ID, allow_pickle=True)   # Here we add the path. ID can be the path

            # if not np.isfinite(X[i]).all():  # remove after Ive checked all scans
            #     X[i] = np.zeros((X[i].shape))

#        if np.isnan(x[0]).any():
#            sys.exit(0)
        #assert not np.any(np.isnan(X)), 'NaNs found in data!!!'

        X[np.isnan(X)] = 0
        
        if self.do_augmentation:
            X_gen = self.augmentor.flow(X,y, batch_size=self.batch_size, shuffle=False, seed=self.seed)

            return next(X_gen)
        else:
            return X,y


#%%


class MyHistory(tf.keras.callbacks.Callback):
    def __init__(self, OUT, NAME, loss=[], acc=[], val_loss=[], val_acc=[]):
        self.OUT = OUT
        self.NAME = NAME     
        
        self.loss = loss
        self.acc = acc
        self.val_loss = val_loss
        self.val_acc = val_acc
        
#    def on_train_begin(self, logs={}):
#



    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        N = 2
        M = 1
        plt.figure(figsize=(10,12))
        plt.subplot(N,M,1); plt.title('Loss')
        plt.plot(self.loss); plt.grid()
        plt.plot(self.val_loss); plt.grid()
        plt.legend(['Train','Val'])
        plt.grid()
        plt.xlabel('Epochs')

        
        plt.subplot(N,M,2); plt.title('Accuracy')
        plt.plot(self.acc)        
        plt.plot(self.val_acc)
        plt.grid()
        plt.legend(['Train','Val'])
        plt.xlabel('Epochs')
        
        plt.tight_layout()
        
        plt.savefig(self.OUT + self.NAME + '/Training_curves.png')
        plt.close()

def freeze_layers(model):
    model_type = type(model) 

    for i in model.layers:
        i.trainable = False
        if type(i) == model_type:
            freeze_layers(i)
    return model


def save_model_and_weights(model, NAME, FOLDER):    
 
    weights = model.get_weights()
    np.save(FOLDER + NAME + '_weights.npy', weights)

class my_model_checkpoint(tf.keras.callbacks.Callback):
    
    def __init__(self, MODEL_PATH, MODEL_NAME):
        self.MODEL_PATH = MODEL_PATH
        self.MODEL_NAME = MODEL_NAME    
        self.val_loss = [999]               

    def on_epoch_end(self, epoch, logs={}):
        min_val_loss = min(self.val_loss)
        current_val_loss = logs.get('val_loss')
        self.val_loss.append(current_val_loss)
        print('Min loss so far: {}, new loss: {}'.format(min_val_loss, current_val_loss))
        if current_val_loss < min_val_loss :
            print('New best model! Epoch: {}'.format(epoch))
            save_model_and_weights(self.model, self.MODEL_NAME, self.MODEL_PATH)
        else:
            save_model_and_weights(self.model, '/last_model', self.MODEL_PATH)





def train_session(NAME, OUT, model, partition, DATA_PATH, training_generator, validation_generator, Custom_History, my_custom_checkpoint, EPOCHS = 100, BATCH_SIZE=4):

    # Some last check on data
    subjects_train = [x.split('/')[-1][:20] for x in training_generator.list_IDs]
    subjects_val = [x.split('/')[-1][:20] for x in validation_generator.list_IDs]
    assert len(set(subjects_train).intersection(set(subjects_val)))==0, 'subjects in both train and val!'
    
    if not os.path.exists(OUT+NAME):
        os.makedirs(OUT+NAME)
    
    
    np.save(OUT + NAME + '/Data.npy', partition)
    np.save(OUT + NAME + '/Labels.npy', training_generator.labels)
    tf.keras.utils.plot_model(model, to_file=OUT + NAME + '/DGNS_Model.png', show_shapes=True)
    training_generator.clinical_info.to_csv(OUT + NAME + '/Clinical_Data_Train_Val.csv', index=False)

    #print('Train malignants={}, Total={}'.format(np.sum([labels[x[:31]] for x in training_generator.list_IDs]), len(training_generator.list_IDs)))
    #print('Val malignants={}, Total={}'.format(np.sum([labels[x[:31]] for x in validation_generator.list_IDs]), len(validation_generator.list_IDs)))
    
    csv_logger = tf.keras.callbacks.CSVLogger(OUT+NAME + '/csvLogger.log', 
                                         separator=',', 
                                         append=True)
    

    myEarlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                    min_delta=0, 
                                                    patience=35, 
                                                    verbose=1, 
                                                    mode='min', 
                                                    baseline=None, 
                                                    restore_best_weights=False)

    # Train model on dataset
    history = model.fit_generator(generator=training_generator,
                                validation_data=validation_generator,
                                use_multiprocessing=True,
                                workers=12, 
                                verbose=1,
                                steps_per_epoch = len(training_generator.list_IDs) // BATCH_SIZE,
                                epochs = EPOCHS,
                                shuffle=True,
                                #class_weight = {0:1., 1:CLASS_WEIGHT},
                                callbacks=[Custom_History, csv_logger, my_custom_checkpoint, myEarlyStop])
    
    
    plt.figure(figsize=(5,15))
    plt.subplot(2,1,1); plt.title('Loss')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Train','Val'])
    
    plt.subplot(2,1,2); plt.title('Accuracy')
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.ylim([0,1])
    plt.legend(['Train','Val'])

   
    plt.tight_layout()
    plt.savefig(OUT + NAME + '/End_Training.png')
    plt.close()
    
    model.save(OUT + NAME +  '/last_model.h5')