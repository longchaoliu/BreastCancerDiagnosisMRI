#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:56:33 2022

Get ROIs from malignants in the training set, and compare with ground truth segmented locations (if available)

Get some metric for detection. 

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
import warnings
from sklearn.model_selection import train_test_split

from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.layers import concatenate


from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)

RISK = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Data_ExamHistory_Labels.csv')

MASTER = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/DATA/Axials_pathology_assigned_partition.csv')

OUTPUT_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/'

# SLICES PATH:
DATA_PATH = '/media/SD/Axial_Slices/X/'

# MRI PATH:
MRI_PATH = '/home/deeperthought/kirbyPRO/alignedNiiAxial-May2020-cropped-normed/'

# PARAMETERS
EPOCHS = 25
depth = 6
n_base_filters = 42
L2 = 1e-5
BATCH_SIZE = 8

USE_CLINICAL=True
USE_CONTRALATERAL = False
USE_PREVIOUS = False
DATA_AUGMENTATION = True

NAME = 'AXIAL_'

CNN = True
RESNET = False
IMAGENET_WEIGHTS = False
MODEL_NAME = 'AXIAL_'

LOAD_PRETRAINED_MODEL = False
PRETRAINED_MODEL_WEIGHTS = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/best_model_weights.npy'


LOAD_PRESAVED_DATA = True
PRESAVED_DATA = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/DATA/Axial_Data.npy'
PRESAVED_LABELS = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/DATA/Axial_Labels.npy'
PRESAVED_CLINICAL_INFO = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/DATA/Axial_Clinical_Data_Train_Val.csv'

PREDICTION_ONLY = False
SETS_FOR_PREDICTION = ['validation']#['validation']#['validation', 'test']

USE_SMALLER_DATASET = False
FRACTION = 0.5

# LOADING TEST RISK SET 1 year
df = pd.read_csv('/home/deeperthought/Projects/DGNS/Risk_Prediction/Sessions/FINAL/training_dropout0.5_classWeight1.0_Clinical/paper_selected/class_weight1/results_sheet_TEST.csv')
#SETS_FOR_PREDICTION = list(df['scanID'].values)

#%% METRICS AND LOSSES
        
    
def dice_loss(y_true, y_pred):
#  y_true = tf.cast(y_true, tf.float32)
#  y_pred = tf.math.sigmoid(y_pred)
  numerator = 2 * tf.math.reduce_sum(y_true * y_pred)
  denominator = tf.math.reduce_sum(y_true + y_pred)
  return 1 - numerator / denominator

def Generalised_dice_coef_multilabel2(y_true, y_pred, numLabels=2):
    """This is the loss function to MINIMIZE. A perfect overlap returns 0. Total disagreement returns numeLabels"""
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return numLabels + dice

def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f**2) + tf.reduce_sum(y_pred_f**2) + smooth)


def dice_coef_multilabel_bin0(y_true, y_pred):
    dice = dice_coef(y_true[:,:,:,0], tf.math.round(y_pred[:,:,:,0]))
    return dice

def dice_coef_multilabel_bin1(y_true, y_pred):
    dice = dice_coef(y_true[:,:,:,1], tf.math.round(y_pred[:,:,:,1]))
    return dice


def Generalised_dice_coef_multilabel2_numpy(y_true, y_pred, numLabels=2):
    """This is the loss function to MINIMIZE. A perfect overlap returns 0. Total disagreement returns numeLabels"""
    dice=0
    for index in range(numLabels):
        dice -= dice_coef_numpy(y_true[:,:,:,index], y_pred[:,:,:,index])
    return numLabels + dice

def dice_coef_multilabel_bin0_numpy(y_true, y_pred):
    dice = dice_coef_numpy(y_true[:,:,:,0], np.round(y_pred[:,:,:,0]))
    return dice
def dice_coef_multilabel_bin1_numpy(y_true, y_pred):
    dice = dice_coef_numpy(y_true[:,:,:,1], np.round(y_pred[:,:,:,1]))
    return dice

def dice_coef_numpy(y_true, y_pred):
    smooth = 1e-6
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f**2) + np.sum(y_pred_f**2) + smooth)


#%% 2D Unet




def double_conv_block(x, n_filters):
   # Conv2D then ReLU activation
   x = tf.keras.layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = tf.keras.layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   return x

def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = tf.keras.layers.MaxPool2D(2)(f)
   p = tf.keras.layers.Dropout(0)(p)
   return f, p

def upsample_block(x, conv_features, n_filters):
   # upsample
   x = tf.keras.layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # concatenate
   x = tf.keras.layers.concatenate([x, conv_features])
   # dropout
   x = tf.keras.layers.Dropout(0)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)
   return x

def make_unet(LR=1e-5):
   inputs = tf.keras.layers.Input(shape=(512,512,3))

   f01, p01 = downsample_block(inputs, 64)
   f02, p02 = downsample_block(p01, 64)


   # encoder: contracting path - downsample
   # 1 - downsample
   f1, p1 = downsample_block(p02, 64)
   # 2 - downsample
   f2, p2 = downsample_block(p1, 128)
   # 3 - downsample
   f3, p3 = downsample_block(p2, 256)
   # 4 - downsample
   f4, p4 = downsample_block(p3, 512)

   # 5 - bottleneck
   bottleneck = double_conv_block(p4, 1024)

   # decoder: expanding path - upsample
   # 6 - upsample
   u6 = upsample_block(bottleneck, f4, 512)
   # 7 - upsample
   u7 = upsample_block(u6, f3, 256)
   # 8 - upsample
   u8 = upsample_block(u7, f2, 128)
   # 9 - upsample
   u9 = upsample_block(u8, f1, 64)
   u10 = upsample_block(u9, f02, 64)
   u11 = upsample_block(u10, f01, 64)
   # outputs
   outputs = tf.keras.layers.Conv2D(2, 16, padding="same", activation = "softmax")(u11)

   # unet model with Keras Functional API
   unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

#   unet_model.compile(loss=dice_loss, optimizer=tf.keras.optimizers.Adam(lr=LR), metrics=['acc', dice_coef_multilabel_bin0])


   unet_model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=tf.keras.optimizers.Adam(lr=LR), 
                      metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])

   return unet_model

#model = make_unet()

#model.summary()

#%% MODEL 2



def create_convolution_block(input_layer, n_filters, kernel=(3, 3), padding='same', strides=(1, 1), L2=0):

    layer = Conv2D(n_filters, kernel, padding=padding, strides=strides, kernel_regularizer=regularizers.l2(L2))(input_layer)
    layer = BatchNormalization()(layer)

    return Activation('relu')(layer)


def get_up_convolution(n_filters, pool_size=(2,2), kernel_size=(2,2), strides=(2, 2),
                       deconvolution=True, bilinear_upsampling=False, L2=0):
    if deconvolution:
        if bilinear_upsampling:
            return Conv2DTranspose(filters=n_filters, kernel_size=(3,3),
                                   strides=strides, trainable=False)#, kernel_initializer=make_bilinear_filter_5D(shape=(3,3,3,n_filters,n_filters)), trainable=False)
        else:
            return Conv2DTranspose(filters=n_filters, kernel_size=(2,2),
                                   strides=strides, kernel_regularizer=regularizers.l2(L2))            
    else:
        return UpSampling2D(size=pool_size)

def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)



def unfreeze_layers(model):
    model_type = type(model) 
    for i in model.layers:
        i.trainable = True
        if type(i) == model_type:
            unfreeze_layers(i)
    return model


def load_model_frozen(PATH):
    #model.load_weights(FOLDER + NAME + '.h5')
    model_loaded = tf.keras.models.load_model(PATH, custom_objects={'FocalLoss':FocalLoss})
    model = unfreeze_layers(model_loaded)
    #adam = tf.keras.optimizers.Adam(learning_rate=5e-5)
    #model.compile(optimizer=adam, loss=FocalLoss,  metrics=['accuracy'])  
    return model


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
    
    t1post, slope1, slope2 = all_data[:,:,:,0], all_data[:,:,:,1], all_data[:,:,:,2]

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


def get_results_on_dataset(model, MRI_PATH, scans_list, labels, Data_description, NAME, OUT, USE_CLINICAL, USE_CONTRALATERAL, USE_PREVIOUS, clinical_info, name='VAL'):
    
    if os.path.exists(OUT + NAME + '/{}_result.csv'.format(name)):
        result = pd.read_csv(OUT + NAME + '/{}_result.csv'.format(name))
        print('Loading previous results.')
        print('{} scans already done.'.format(len(result)))
    else:
        result = pd.DataFrame(columns=['scan','y_pred','y_true','max_slice'])
    N = 0
    
    scans_list = [x for x in scans_list if x not in result['scan'].values]
    
    TOT = len(scans_list)
    
    for exam in scans_list:
        
        #scan = scan[:31]
        N += 1
        scanid = exam + '_l'

        print('{}/{}'.format(N,TOT))
        sides = []
        
        left, right, flag = load_axial_data(MRI_PATH, exam)
        
        if (flag == 1):
            continue

        clinic_info_exam = 0
        
        if USE_CLINICAL:
            if scanid in clinical_info['scanID'].values:
                clinic_info_exam = clinical_info.loc[clinical_info['scanID'] == scanid,[u'Family Hx',u'Age',u'ETHNICITY_HISPANIC OR LATINO',u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',u'RACE_ASIAN-FAR EAST/INDIAN SUBCONT',u'RACE_BLACK OR AFRICAN AMERICAN',u'RACE_NATIVE AMERICAN-AM IND/ALASKA',u'RACE_NATIVE HAWAIIAN OR PACIFIC ISL',u'RACE_UNKNOWN',u'RACE_WHITE']].values
            elif scanid.replace('_l','_r') in clinical_info['scanID'].values:
                clinic_info_exam = clinical_info.loc[clinical_info['scanID'] == scanid.replace('_l','_r') ,[u'Family Hx',u'Age',u'ETHNICITY_HISPANIC OR LATINO',u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',u'RACE_ASIAN-FAR EAST/INDIAN SUBCONT',u'RACE_BLACK OR AFRICAN AMERICAN',u'RACE_NATIVE AMERICAN-AM IND/ALASKA',u'RACE_NATIVE HAWAIIAN OR PACIFIC ISL',u'RACE_UNKNOWN',u'RACE_WHITE']].values

            else:
                clinic_info_exam = clinical_info[[u'Family Hx',u'Age',u'ETHNICITY_HISPANIC OR LATINO',u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',u'RACE_ASIAN-FAR EAST/INDIAN SUBCONT',u'RACE_BLACK OR AFRICAN AMERICAN',u'RACE_NATIVE AMERICAN-AM IND/ALASKA',u'RACE_NATIVE HAWAIIAN OR PACIFIC ISL',u'RACE_UNKNOWN',u'RACE_WHITE']].mode().values
                
        if (exam + '_l' in MASTER['scanID'].values):
            sides.append('left')
            pathology_left = MASTER.loc[MASTER['scanID'] == exam + '_l', 'pathology'].values[0]
            if pathology_left == 'Malignant':
                pathology_left = 1
            else:
                pathology_left = 0       
                
            slice_preds_left = make_prediction_whole_scan(model, left, clinic_info_exam, USE_CONTRALATERAL, USE_PREVIOUS)
        
        
            slice_preds_left = model.predict([left,np.repeat(clinic_info_exam, repeats=left.shape[0] ,axis=0) ])[:,1]

            slice_preds_left = slice_preds_left
        
            result = result.append({'scan':exam + '_l','y_pred':np.max(slice_preds_left), 'y_true':pathology_left, 
                                'max_slice':np.argmax(slice_preds_left)},ignore_index=True)

            print('{} - PATHOLOGY: {} max_slice: {} = {}:'.format(exam + '_l', pathology_left, np.argmax(slice_preds_left), np.max(slice_preds_left)))
   
        if (exam + '_r' in MASTER['scanID'].values):
            sides.append('right')
        
            pathology_right = MASTER.loc[MASTER['scanID'] == exam + '_r', 'pathology'].values[0]
            if pathology_right == 'Malignant':
                pathology_right = 1
            else:
                pathology_right = 0

            slice_preds_right = make_prediction_whole_scan(model, right, clinic_info_exam, USE_CONTRALATERAL, USE_PREVIOUS)
            
            result = result.append({'scan':exam + '_r','y_pred':np.max(slice_preds_right), 'y_true':pathology_right, 
                                    'max_slice':np.argmax(slice_preds_right)},ignore_index=True)                       
    
            print('{} - PATHOLOGY: {} max_slice: {} = {}:'.format(exam + '_r', pathology_right, np.argmax(slice_preds_right), np.max(slice_preds_right)))
 
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
    
    
def load_axial_data(MRI_PATH, exam):
   #load_data_prediction(scan,labels, Data_description, USE_CONTRALATERAL, USE_PREVIOUS) 
    
    hdr = nib.load(MRI_PATH + exam + '/T1_axial_02_01.nii.gz')
    
    
    t1post = hdr.get_data()
    slope1 = nib.load(MRI_PATH + exam + '/T1_axial_slope1.nii.gz').get_data()
    slope2 = nib.load(MRI_PATH + exam + '/T1_axial_slope2.nii.gz').get_data()

    t1post = t1post/float(40)
    slope1 = slope1/float(0.3)
    slope2 = slope2/float(0.12)           

#    plt.imshow(t1post[150])

    if not np.all(np.isfinite(t1post)):
        print('Nans! skip')
        return 0,0,1
    if not np.all(np.isfinite(slope1)):
        print('Nans! skip')
        return 0,0,1
    if not np.all(np.isfinite(slope2)):
        print('Nans! skip')
        return 0,0,1
    
    resolution = np.diag(hdr.affine)
    
    resolution_factor_X = resolution[1]/0.5
    resolution_factor_Y = resolution[2]/0.5
    
    #output_res =  (resolution[0], resolution[1]/resolution_factor_X, resolution[2]/resolution_factor_Y)  #'THIS SEEMS CORRECT THOUGH???'
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
    
    
    left_breast_stack = np.stack([left_breast,left_breast_slope1, left_breast_slope2], axis=-1)
    right_breast_stack = np.stack([right_breast,right_breast_slope1, right_breast_slope2], axis=-1)
    
    return left_breast_stack, right_breast_stack, 0

#%% MODELS

def FocalLoss(y_true, y_pred): 
    #y_true = tf.keras.backend.expand_dims(y_true,0)
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    term_0 = (1 - y_true[:,1]) * tf.keras.backend.pow(y_pred[:,1],5) * tf.keras.backend.log(1 - y_pred[:,1] + tf.keras.backend.epsilon())  
    term_1 = y_true[:,1] * tf.keras.backend.pow(1 - y_pred[:,1],5) * tf.keras.backend.log(y_pred[:,1] + tf.keras.backend.epsilon())   
    return -tf.keras.backend.mean(term_0 + term_1, axis=0)

def UNet_v0_2DTumorSegmenter(input_shape =  (512, 512,3), pool_size=(2, 2),initial_learning_rate=1e-5, deconvolution=True,
                      depth=4, n_base_filters=32, activation_name="softmax", L2=0):
        """ Simple version, padding 'same' on every layer, output size is equal to input size. Has border artifacts and checkerboard artifacts """
        inputs = Input(input_shape)
        levels = list()
        current_layer = Conv2D(n_base_filters, (1, 1))(inputs)
    
        # add levels with max pooling
        for layer_depth in range(depth):
            layer1 = create_convolution_block(input_layer=current_layer, kernel=(3,3), n_filters=n_base_filters*(layer_depth+1), padding='same', L2=L2)
            layer2 = create_convolution_block(input_layer=layer1, kernel=(3,3),  n_filters=n_base_filters*(layer_depth+1), padding='same', L2=L2)
            if layer_depth < depth - 1:
                current_layer = MaxPooling2D(pool_size=(2,2))(layer2)
                levels.append([layer1, layer2, current_layer])
            else:
                current_layer = layer2
                levels.append([layer1, layer2])

        for layer_depth in range(depth-2, -1, -1):
            
            up_convolution = get_up_convolution(pool_size=(2,2), deconvolution=deconvolution, n_filters=n_base_filters*(layer_depth+1), L2=L2)(current_layer)

            concat = concatenate([up_convolution, levels[layer_depth][1]] , axis=-1)
            current_layer = create_convolution_block(n_filters=n_base_filters*(layer_depth+1),kernel=(3,3), input_layer=concat, padding='same', L2=L2)
            current_layer = create_convolution_block(n_filters=n_base_filters*(layer_depth+1),kernel=(3,3), input_layer=current_layer, padding='same', L2=L2)

        current_layer = Conv2D(16, (1, 1), name='FEATURE_EXTRACTION_DGNS')(current_layer)   
        final_convolution = Conv2D(2, (1, 1))(current_layer)
              
        act = Activation(activation_name)(final_convolution)
        
        model = Model(inputs=[inputs], outputs=act)

        model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=initial_learning_rate), 
                      metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])

        return model
    
def UNet_v0_2D_Classifier(input_shape =  (512, 512,3), pool_size=(2, 2),initial_learning_rate=1e-5, deconvolution=True,
                      depth=4, n_base_filters=32, activation_name="softmax", L2=0, USE_CLINICAL=False):
        """ Simple version, padding 'same' on every layer, output size is equal to input size. Has border artifacts and checkerboard artifacts """
        inputs = Input(input_shape)
        levels = list()
        current_layer = Conv2D(n_base_filters, (1, 1))(inputs)
    
        # add levels with max pooling
        for layer_depth in range(depth):
            layer1 = create_convolution_block(input_layer=current_layer, kernel=(3,3), n_filters=n_base_filters*(layer_depth+1), padding='same', L2=L2)
            layer2 = create_convolution_block(input_layer=layer1, kernel=(3,3),  n_filters=n_base_filters*(layer_depth+1), padding='same', L2=L2)
            if layer_depth < depth - 1:
                current_layer = MaxPooling2D(pool_size=(2,2))(layer2)
                levels.append([layer1, layer2, current_layer])
            else:
                current_layer = layer2
                levels.append([layer1, layer2])
        
        current_layer = tf.keras.layers.Flatten()(current_layer)  
        
        image_features = tf.keras.layers.Dense(24, activation='relu')(current_layer)
        
        if USE_CLINICAL:
            clinical_inputs = Input((11))
            current_layer = tf.keras.layers.concatenate([image_features, clinical_inputs])
            
            current_layer = tf.keras.layers.Dense(16, activation='relu')(current_layer)
            act = tf.keras.layers.Dense(2, activation='softmax')(current_layer)
            
            model = Model(inputs=[inputs, clinical_inputs], outputs=act)

        
        else:
            act = tf.keras.layers.Dense(2, activation='softmax')(current_layer)
            model = Model(inputs=[inputs], outputs=act)

        model.compile(loss=FocalLoss, optimizer=Adam(lr=initial_learning_rate), metrics=['acc'])

        return model
                                        

#%% Clinical Info functions
        
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
#%% FROM DIRECTORY

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
        
        if self.do_augmentation:
            for i in range(self.batch_size):
                X[i] = X[i]*np.random.uniform(low=0.8, high=1.2, size=1)

        if self.use_clinical_info:
            clinic = np.zeros((self.batch_size,11))
            scanids = [ids[:31] for ids in list_IDs_temp]
            for i in range(len(scanids)):
                
                tmp = self.clinical_info.loc[self.clinical_info['scanID'] == scanids[i],[u'Family Hx',u'Age',u'ETHNICITY_HISPANIC OR LATINO',u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',u'RACE_ASIAN-FAR EAST/INDIAN SUBCONT',u'RACE_BLACK OR AFRICAN AMERICAN',u'RACE_NATIVE AMERICAN-AM IND/ALASKA',u'RACE_NATIVE HAWAIIAN OR PACIFIC ISL',u'RACE_UNKNOWN',u'RACE_WHITE']].values
                if tmp.shape[0] > 1:
                    tmp = tmp[0]
                clinic[i] = tmp
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
            asd = np.load(self.data_path + ID, allow_pickle=True) 
            if asd.shape[1] < 512:
                delta = 512 - asd.shape[1]
                asd = np.pad(asd, ((0,0),(0,delta),(0,0),(0,0)), 'minimum')

            X[i,:,:,:3] = asd#np.load(self.data_path + ID, allow_pickle=True)   # Here we add the path. ID can be the path
            y[i] = labels[ID[:31]]
            
        
        if self.do_augmentation:
            X_gen = self.augmentor.flow(X,y, batch_size=self.batch_size, shuffle=False, seed=self.seed)

            return next(X_gen)
        else:
            return X,y
    
#%% Test generator
#
#params_train = {'dim': (512,512),
#          'data_path': '/home/deeperthought/kirbyPRO/Saggittal_Full_Slices/2D_slices/X/',
#          'batch_size': 1,
#          'n_classes': 2,
#          'n_channels': 9,
#          'shuffledata': True,
#          'do_augmentation':True,
#          'clinical_info':clinical_info,
#          'use_clinical_info':True,
#          'use_contralateral':True,
#          'use_previous':True,
#          'data_description':MASTER}
#
## Generators
#training_generator = DataGenerator_classifier(partition['train'],labels, **params_train)
#
#x,y = training_generator.__getitem__(0)
#
#x[1].shape
#plt.imshow(x[0,:,:,8])
#
#plt.figure(figsize=(14,6))
#for seed in [0,1,2,3]:
#    training_generator.seed = seed
#    x,y = training_generator.__getitem__(9)
#    
#    x[1]
#    x[2]
#    print(y)
#    plt.subplot(1,4,seed+1)
#    plt.imshow(x[0,:,:,0]); plt.xticks([]); plt.yticks([])


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


#def save_model_and_weights(model, NAME, FOLDER):    
#    model_to_save = tf.keras.models.clone_model(model)
#    model_to_save.set_weights(model.get_weights())
#    model_to_save = freeze_layers(model_to_save)
##    model_to_save.save_weights(FOLDER + NAME + '_weights.h5')
#    model_to_save.save(FOLDER + NAME + '.h5')       
#    
    
def save_model_and_weights(model, NAME, FOLDER):    
#    model_to_save = tf.keras.models.clone_model(model)
#    model_to_save.set_weights(model.get_weights())
#    model_to_save = freeze_layers(model_to_save)
#    model_to_save.save_weights(FOLDER + NAME + '_weights.h5')
#    model_to_save.save(FOLDER + NAME + '.h5')       
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





def train_session(NAME, OUT, model, partition,labels, DATA_PATH, training_generator, validation_generator, Custom_History, my_model_checkpoint, EPOCHS = 100, BATCH_SIZE=4):

    # Some last check on data
    subjects_train = [x[:20] for x in training_generator.list_IDs]
    subjects_val = [x[:20] for x in validation_generator.list_IDs]
    #assert len(set(subjects_train).intersection(set(subjects_val)))==0, 'subjects in both train and val!'
    
    if not os.path.exists(OUT+NAME):
        os.makedirs(OUT+NAME)
    
    
    np.save(OUTPUT_PATH + NAME + '/Data.npy', partition)
    np.save(OUTPUT_PATH + NAME + '/Labels.npy', training_generator.labels)
    tf.keras.utils.plot_model(model, to_file=OUTPUT_PATH + NAME + '/DGNS_Model.png', show_shapes=True)
    training_generator.clinical_info.to_csv(OUTPUT_PATH + NAME + '/Clinical_Data_Train_Val.csv', index=False)

    print('Train malignants={}, Total={}'.format(np.sum([labels[x[:31]] for x in training_generator.list_IDs]), len(training_generator.list_IDs)))
    print('Val malignants={}, Total={}'.format(np.sum([labels[x[:31]] for x in validation_generator.list_IDs]), len(validation_generator.list_IDs)))
    
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
    
#%%
#%%
#%%
#%%
#%% LOAD PARTITIONS

if LOAD_PRESAVED_DATA:

    partition = np.load(PRESAVED_DATA, allow_pickle=True).item()
    labels = np.load(PRESAVED_LABELS, allow_pickle=True).item()
    clinical_features = pd.read_csv(PRESAVED_CLINICAL_INFO)
    
else:
    
    # Only use valid files (by generating in parallel, some are unfinished or faulty)
    #file_paths = [os.path.join(DATA_PATH, file_name) for file_name in os.listdir(DATA_PATH)]
    #file_sizes = [os.path.getsize(file_path) for file_path in file_paths]
    #df_file_sizes = pd.DataFrame({'file_path': file_paths, 'file_size': file_sizes}).sort_values('file_size', ascending = False)
    #
    #available_data = list(set(df_file_sizes.loc[df_file_sizes['file_size'] == 6291584, 'file_path'].values))
    #
    #bad_files = df_file_sizes.loc[df_file_sizes['file_size'] == 128, 'file_path'].values
    #
    #for badfile in bad_files:
    #    os.remove(badfile)
    
    available_data = os.listdir(DATA_PATH)
    
    available_data = [x.split('/')[-1] for x in available_data]
    available_data_scanIDs = list(set([x[:31] for x in available_data]))
    
    print('Removing available scans without pathology information..')
    available_data_scanIDs = list(set(available_data_scanIDs).intersection(set(MASTER['Scan_ID'].values) ))
    
    #-------- REMOVE CONVERTS -------------------------
    MASTER['Convert'] = MASTER['Convert'].fillna(0)
    
    converts = list(set(MASTER.loc[MASTER['Convert'] == 1, 'Scan_ID'].values))
    converts_subjects = list(set([x[:20] for x in converts]))
    
    MASTER = MASTER.loc[MASTER['Convert'] == 0]
    available_data_scanIDs = list(set(available_data_scanIDs).intersection(set(MASTER['Scan_ID'].values) ))
    
    available_data_scanIDs = [x for x in available_data_scanIDs if x[:20] not in converts_subjects]
    
    available_data = [ x for x in available_data if x[:31] in available_data_scanIDs]
    
    available_data_exams = list(set([x[:-2] for x in available_data_scanIDs]))
    available_data_breasts = list(set([x[:20] + '_' + x[-1] for x in available_data_scanIDs]))
    available_data_subjects = list(set([x[:-2] for x in available_data_breasts]))
    
    print('Found {} slices, from {} exams, of {} breasts, of {} subjects'.format(len(available_data),len(available_data_scanIDs),len(available_data_breasts),len(available_data_subjects)))
    
    #%% Split DATA
    
    '''
    No rules for splitting, I have free choices. No segmenter, no Risk (initially)
    
    Simplest partition: Just for diagnosis on the whole population. 
    
    I can leave aside undefined diagnosis scans: All converts (~1,000 exams). Everything else can stay inside
    
    Exclude:
        - Converts
        - Unsegmented malignants (shouldnt have any slices, but anyway)
    
    Partition: By subjects and breasts.
        
    '''
    ################ ASSIGN LABEL to Scans, Subjects, and Breasts ###########################
    
    RISK.loc[RISK['Scan_ID'].isin(available_data_scanIDs), 'Pathology'].value_counts()
    
    len(set(available_data_scanIDs) - set(RISK['Scan_ID'].values))
    
    len(set(available_data_scanIDs) - set(MASTER['Scan_ID'].values))
    
    MASTER = MASTER.loc[MASTER['Scan_ID'].isin(available_data_scanIDs)]
    
    
    #-------- REMOVE UNSEGMENTED MALIGNANTS ----------------
    MASTER['Segmented'] = MASTER['Segmented'].fillna(0)
    #MASTER.loc[MASTER['Pathology'] == 'Malignant', 'Segmented'].value_counts()
    assert len(MASTER.loc[(MASTER['Pathology'] == 'Malignant')*(MASTER['Segmented'] == 0)]) == 0, 'Unsegmented malignants still in!'
    
    
    #----- Label dictionary to Scans -----
    labels = pd.Series(np.array(MASTER.Pathology.values == 'Malignant', dtype=int), index=MASTER.Scan_ID).to_dict()
    
    np.sum(labels.values())
    
    #----- Label dictionary to subjects -----
    malignant_scanIDs = list(MASTER.loc[MASTER['Pathology'] == 'Malignant', 'Scan_ID'].values)
    malignant_subjects = list(set([x[:20] for x in malignant_scanIDs]))
    healthy_subjects = list(set(available_data_subjects) - set(malignant_subjects))
    assert len(malignant_subjects) + len(healthy_subjects) == len(available_data_subjects)
    Labels_subject = dict(zip(malignant_subjects, [1]*len(malignant_subjects)))
    Labels_subject.update(dict(zip(healthy_subjects, [0]*len(healthy_subjects))))
    
    #----- Label dictionary to breasts -----
    MASTER['BreastID'] = MASTER['Scan_ID'].str[:20] + '_' + MASTER['Scan_ID'].str[-1]
    malignant_breasts = list(set(MASTER.loc[MASTER['Pathology'] == 'Malignant', 'BreastID'].values))
    healthy_breasts = list(set(available_data_breasts) - set(malignant_breasts))
    assert len(malignant_breasts) + len(healthy_breasts) == len(available_data_breasts)
    Labels_breasts = dict(zip(malignant_breasts, [1]*len(malignant_breasts)))
    Labels_breasts.update(dict(zip(healthy_breasts, [0]*len(healthy_breasts))))
    
    ############## SPLIT BY SUBJECT ###########################
    
    X_train_subjects, X_test_subjects, y_train_subjects, y_test_subjects = train_test_split(Labels_subject.keys(), Labels_subject.values(), test_size=0.1, random_state=42, stratify=Labels_subject.values())
    
    len(X_train_subjects)
    len(X_test_subjects)
    np.sum(y_train_subjects)
    np.sum(y_test_subjects)
    
    X_train_subjects, X_val_subjects, y_train_subjects, y_val_subjects = train_test_split(X_train_subjects, y_train_subjects, test_size=0.1, random_state=42, stratify=y_train_subjects)
    
    len(X_train_subjects)
    len(X_val_subjects)
    np.sum(y_train_subjects)
    np.sum(y_val_subjects)
    
    ########### EXTEND FROM SUBJECT TO SCAN ####################
    
    X_train = [x for x in available_data if x[:20] in X_train_subjects]
    X_val = [x for x in available_data if x[:20] in X_val_subjects]
    X_test = [x for x in available_data if x[:20] in X_test_subjects]
    
    
    partition = {'train':[],'validation':[], 'test':[]}
    
    partition['train'] = X_train
    partition['validation'] = X_val
    partition['test'] = X_test
    
    clinical_df = pd.read_excel('/home/deeperthought/Projects/MSKCC/MSKCC/Data_spreadsheets/Diamond_and_Gold/CCNY_CLINICAL_4_17_2019.xlsx', header=[0,1])    
    X = list(available_data_scanIDs)
    
    scanID = [x.split('/')[-1][:31] for x in X] 
    clinical_features = pd.DataFrame(columns=['scan_ID','X'])
    clinical_features['scan_ID'] = scanID
    clinical_features['X'] = X
    clinical_features = add_age(clinical_features, clinical_df)
    clinical_features = add_ethnicity_oneHot(clinical_features, clinical_df)
    clinical_features = add_family_hx(clinical_features, clinical_df)
    clinical_features['Age'] = clinical_features['Age']/100.
    clinical_features = clinical_features.drop_duplicates()
    CLINICAL_FEATURE_NAMES = [u'Family Hx',u'Age',u'ETHNICITY_HISPANIC OR LATINO',u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',u'RACE_ASIAN-FAR EAST/INDIAN SUBCONT',u'RACE_BLACK OR AFRICAN AMERICAN',u'RACE_NATIVE AMERICAN-AM IND/ALASKA',u'RACE_NATIVE HAWAIIAN OR PACIFIC ISL',u'RACE_UNKNOWN',u'RACE_WHITE']
       
    clinical_features[clinical_features.isnull().any(axis=1)]
    
    np.sum(pd.isnull(clinical_features).any(axis=1))

print('Train: {}\nVal: {} \nTest: {}'.format(len(partition['train']), len(partition['validation']), len(partition['test'])))



#train_scans = partition['train']
#train_scanIDs = [x[:31] for x in train_scans]
#master_train = MASTER.loc[MASTER['Scan_ID'].isin(train_scanIDs)]
#master_train['BIRADS'].value_counts()
#train_birads5 = list(master_train.loc[master_train['BIRADS'] == '5', 'Scan_ID'].values)
#train_birads4 = list(master_train.loc[master_train['BIRADS'] == '4', 'Scan_ID'].values)
#train_birads3 = list(master_train.loc[master_train['BIRADS'] == '3', 'Scan_ID'].values)
#

### BALANCE CLASSES #########

#print('Balancing populations of segmented cancers and benigns:')
#train_cancers = [x for x in partition['train'] if labels[x[:31]] == 1]
#train_benigns = [x for x in partition['train'] if labels[x[:31]] == 0]
#
#partition['train'] = train_benigns
#partition['train'].extend( list(np.random.choice(train_cancers, size=len(train_benigns), replace=True)))
#random.Random(42).shuffle(partition['train'])
#
#train_cancers = [x for x in partition['train'] if labels[x[:31]] == 1]
#train_benigns = [x for x in partition['train'] if labels[x[:31]] == 0]
#
#
#print(len(train_cancers))
#print(len(train_benigns))

#np.sum([labels[x] for x in partitions_dictionary['DGNS_3D']['train']]), len(partitions_dictionary['DGNS_3D']['train'])
#np.sum([labels[x] for x in partitions_dictionary['DGNS_3D']['val']]), len(partitions_dictionary['DGNS_3D']['val'])
#np.sum([labels[x] for x in partitions_dictionary['DGNS_3D']['test']]), len(partitions_dictionary['DGNS_3D']['test'])
#
#np.sum([labels[x] for x in partitions_dictionary['SGMT_3D']['train']]), len(partitions_dictionary['SGMT_3D']['train'])
#np.sum([labels[x] for x in partitions_dictionary['SGMT_3D']['val']]), len(partitions_dictionary['SGMT_3D']['val'])
#
#np.sum([labels[x] for x in partitions_dictionary['SGMT_paper']['train']]), len(partitions_dictionary['SGMT_paper']['train'])
#np.sum([labels[x] for x in partitions_dictionary['SGMT_paper']['val']]), len(partitions_dictionary['SGMT_paper']['val'])
#np.sum([labels[x] for x in partitions_dictionary['SGMT_paper']['test']]), len(partitions_dictionary['SGMT_paper']['test'])



#%% TRAINING SESSION

# Make sure no NaNs!!
#partition.keys()
#df_nans = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Scan_IDs_with_NaNs_in_Images.csv')
#nan_images = list(set(df_nans.loc[df_nans['NaN'] == 1, 'Scan_ID'].values))
#partition['train'] = [x for x in partition['train'] if x[:31] not in nan_images]
#partition['validation'] = [x for x in partition['validation'] if x[:31] not in nan_images]
#partition['test'] = [x for x in partition['test'] if x[:31] not in nan_images]

CHANNELS = 3
if USE_CONTRALATERAL:
    CHANNELS += 3
    warnings.filterwarnings("ignore") # ImageDataGenerator expects only 3 channels. It raises a warning but doesnt affect.

if USE_PREVIOUS:
    CHANNELS += 3    
    warnings.filterwarnings("ignore") # ImageDataGenerator expects only 3 channels. It raises a warning but doesnt affect.

params_train = {'dim': (512,512),
          'data_path': DATA_PATH,
          'batch_size': BATCH_SIZE,
          'n_classes': 2,
          'n_channels': CHANNELS,
          'shuffledata': True,
          'do_augmentation':DATA_AUGMENTATION,
          'clinical_info':clinical_features,
          'use_clinical_info':USE_CLINICAL,
          'use_contralateral':USE_CONTRALATERAL,
          'use_previous':USE_PREVIOUS,
          'data_description':MASTER}

params_val = {'dim': (512,512),
          'data_path': DATA_PATH,
          'batch_size': BATCH_SIZE,
          'n_classes': 2,
          'n_channels': CHANNELS,
          'shuffledata': False,
          'do_augmentation':False,
          'clinical_info':clinical_features,
          'use_clinical_info':USE_CLINICAL,
          'use_contralateral':USE_CONTRALATERAL,
          'use_previous':USE_PREVIOUS,
          'data_description':MASTER}

if USE_SMALLER_DATASET:
    print('Using smaller dataset: fraction = {}'.format(FRACTION))
    #partition['train'] = np.random.choice(partition['train'][:1000], replace=False, size=len(partition['train'])*FRACTION)
    partition['train'] = np.random.choice(partition['train'], replace=False, size=int(len(partition['train'])*FRACTION))

#    partition['validation'] = np.random.choice(partition['validation'], replace=False, size=int(len(partition['train'])*0.1))

#    np.sum([labels[x[:31]] for x in partition['train']]), len(partition['train'])
#    np.sum([labels[x[:31]] for x in asd]), len(asd)

#------ GENERATORS -----------
training_generator = DataGenerator_classifier(partition['train'],labels, **params_train)
validation_generator = DataGenerator_classifier(partition['validation'],labels, **params_val)


#[x for x in partition['train'] if x[:31] == 'MSKCC_16-328_1_10774_20130807_r']
#clinical_features.loc[clinical_features['scan_ID'] == 'MSKCC_16-328_1_10774_20130807_r']
#labels['MSKCC_16-328_1_10774_20130807_r']

#x,y = training_generator.__getitem__(0)


#model.input
#model.output
#
#x[0].shape
#x[1].shape
#yhat = model.predict(x)
#model.evaluate(x,yhat)
#model.fit(x,y)

#
##----- MODEL -----------
#if LOAD_PRETRAINED_MODEL:
#    model = load_model_frozen(PRETRAINED_MODEL_PATH)
#    model.compile(loss=FocalLoss, optimizer=Adam(lr=1e-5), metrics=['acc'])
#
#    NAME = PRETRAINED_MODEL_PATH.split('/')[-2]
#    logger = pd.read_csv(OUTPUT_PATH + MODEL_NAME + '/csvLogger.log')
#    
#    my_custom_checkpoint = my_model_checkpoint(MODEL_PATH=OUTPUT_PATH + MODEL_NAME, MODEL_NAME='/best_model' )
#    Custom_History = MyHistory(OUTPUT_PATH, MODEL_NAME)
#
#    MIN_LOSS = np.min(logger['val_loss'])
#    my_custom_checkpoint.val_loss = [MIN_LOSS]
#    Custom_History.acc = list(logger['acc'].values)
#    Custom_History.loss = list(logger['loss'].values)
#    Custom_History.val_acc = list(logger['val_acc'].values)
#    Custom_History.val_loss = list(logger['val_loss'].values)
#else:
#    if not RESNET:
#        model = UNet_v0_2D_Classifier(input_shape =  (512,512,CHANNELS), pool_size=(2, 2),initial_learning_rate=1e-5, 
#                                         deconvolution=True, depth=depth, n_base_filters=n_base_filters,
#                                         activation_name="softmax", L2=L2, USE_CLINICAL=USE_CLINICAL)
#
#    
#
##        tf.keras.utils.plot_model(
##            model,
##            to_file="/home/deeperthought/Projects/DGNS/Detection_model/Sessions/2D_SGMT_model.png",
##            show_shapes=True,
##            show_layer_names=True,
##            rankdir="TB",
##        )
#
#    elif RESNET:
#        
#        if not IMAGENET_WEIGHTS:
#            model = tf.compat.v1.keras.applications.ResNet50(
#                include_top=False,
#                weights= None,
#                input_tensor=None,
#                input_shape=(512,512,3),
#                pooling=max,
#                classes=2)
#            
#        elif IMAGENET_WEIGHTS:
#            model = tf.compat.v1.keras.applications.ResNet50(
#                include_top=False,
#                weights= 'imagenet',
#                input_tensor=None,
#                input_shape=(512,512,3),
#                pooling=max,
#                classes=2)            
#        
#        regularizer = tf.keras.regularizers.l2(1e-4)
#
#        for layer in model.layers:
#            for attr in ['kernel_regularizer']:
#                if hasattr(layer, attr):
#                  setattr(layer, attr, regularizer)
#
#        if USE_CLINICAL:
#                clinical_inputs = Input((11))
#
#                final_layer = tf.keras.layers.MaxPooling2D(16)(model.layers[-1].output)                
#                final_layer = tf.keras.layers.Flatten()(final_layer)
#                
#                clinical_inputs = tf.keras.layers.Input((11))
#                final_layer = tf.keras.layers.concatenate([final_layer, clinical_inputs])
#                    
#                final_layer = tf.keras.layers.Dense(16, activation='relu')(final_layer)
#                act = tf.keras.layers.Dense(2, activation='softmax')(final_layer)
#            
#                model = tf.keras.Model(
#                                        inputs=[model.inputs, clinical_inputs],
#                                        outputs=act)
#
#        
#        model.compile(loss=FocalLoss, optimizer=Adam(lr=1e-5), metrics=['acc'])
#
#
#
#    NAME += '_classifier_train{}_val{}_'.format(len(list(set([x[:31] for x in training_generator.list_IDs]))), len(list(set([x[:31] for x in validation_generator.list_IDs]))))
#    if DATA_AUGMENTATION: NAME += 'DataAug_'
#    if USE_CLINICAL: NAME += 'Clinical_'
#    if USE_CONTRALATERAL: NAME += 'Contra_'
#    if USE_PREVIOUS: NAME += 'Previous_'
#    NAME += 'depth{}_filters{}_L2{}_batchsize{}'.format(depth, n_base_filters, L2, BATCH_SIZE)
#
#    #---- CHECKPOINTS ----------
#    my_custom_checkpoint = my_model_checkpoint(MODEL_PATH=OUTPUT_PATH+NAME, MODEL_NAME='/best_model' )
#    Custom_History = MyHistory(OUTPUT_PATH, NAME)
#


if CNN:
    model = UNet_v0_2D_Classifier(input_shape =  (512,512,CHANNELS), pool_size=(2, 2),initial_learning_rate=1e-5, 
                                     deconvolution=True, depth=depth, n_base_filters=n_base_filters,
                                     activation_name="softmax", L2=L2, USE_CLINICAL=USE_CLINICAL)

elif RESNET:
    
    if not IMAGENET_WEIGHTS:
        model = tf.compat.v1.keras.applications.ResNet50(
            include_top=False,
            weights= None,
            input_tensor=None,
            input_shape=(512,512,3),
            pooling=max,
            classes=2)
        
    elif IMAGENET_WEIGHTS:
        model = tf.compat.v1.keras.applications.ResNet50(
            include_top=False,
            weights= 'imagenet',
            input_tensor=None,
            input_shape=(224,224,3),
            pooling=max,
            classes=2)            
    
    regularizer = tf.keras.regularizers.l2(1e-3)

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
              setattr(layer, attr, regularizer)

    if USE_CLINICAL:
            clinical_inputs = Input((11))

            final_layer = tf.keras.layers.MaxPooling2D(7)(model.layers[-1].output)                
            final_layer = tf.keras.layers.Flatten()(final_layer)
            
            clinical_inputs = tf.keras.layers.Input((11))
            final_layer = tf.keras.layers.concatenate([final_layer, clinical_inputs])
                
            final_layer = tf.keras.layers.Dense(16, activation='relu')(final_layer)
            act = tf.keras.layers.Dense(2, activation='softmax')(final_layer)
        
            model = tf.keras.Model(
                                    inputs=[model.inputs, clinical_inputs],
                                    outputs=act)

if LOAD_PRETRAINED_MODEL:    

    weights = np.load(PRETRAINED_MODEL_WEIGHTS, allow_pickle=True)
    model.set_weights(weights)
    

    logger = pd.read_csv(OUTPUT_PATH + MODEL_NAME + '/csvLogger.log')
    
    my_custom_checkpoint = my_model_checkpoint(MODEL_PATH=OUTPUT_PATH + MODEL_NAME, MODEL_NAME='/best_model' )
    Custom_History = MyHistory(OUTPUT_PATH, MODEL_NAME)

    MIN_LOSS = np.min(logger['val_loss'])
    my_custom_checkpoint.val_loss = [MIN_LOSS]
    Custom_History.acc = list(logger['acc'].values)
    Custom_History.loss = list(logger['loss'].values)
    Custom_History.val_acc = list(logger['val_acc'].values)
    Custom_History.val_loss = list(logger['val_loss'].values)        
    
else:
        
    NAME += '_classifier_train{}_val{}_'.format(len(list(set([x[:31] for x in training_generator.list_IDs]))), len(list(set([x[:31] for x in validation_generator.list_IDs]))))
    if DATA_AUGMENTATION: NAME += 'DataAug_'
    if USE_CLINICAL: NAME += 'Clinical_'
    if USE_CONTRALATERAL: NAME += 'Contra_'
    if USE_PREVIOUS: NAME += 'Previous_'
    NAME += 'depth{}_filters{}_L2{}_batchsize{}'.format(depth, n_base_filters, L2, BATCH_SIZE)
    #---- CHECKPOINTS ----------
    my_custom_checkpoint = my_model_checkpoint(MODEL_PATH=OUTPUT_PATH+NAME, MODEL_NAME='/best_model' )
    Custom_History = MyHistory(OUTPUT_PATH, NAME)

    model.compile(loss=FocalLoss, optimizer=Adam(lr=1e-5), metrics=['acc'])
    
    print('transferring weights from sagittal model..')
    weights = np.load(PRETRAINED_MODEL_WEIGHTS, allow_pickle=True)
    model.set_weights(weights)

if PREDICTION_ONLY:
    
    for set_name in SETS_FOR_PREDICTION:
                
        scans_test = partition[set_name]
        
        class0 = list(set([x[:29] for x in scans_test if labels[x[:31]] == 0]))
        class1 = list(set([x[:29] for x in scans_test if labels[x[:31]] == 1]))
        scans_test = interleave_two_lists(class0,class1)
                
        if not os.path.exists(OUTPUT_PATH + NAME):
            os.mkdir(OUTPUT_PATH + NAME)
        
        get_results_on_dataset(model,MRI_PATH, scans_test, labels,  MASTER, NAME, OUTPUT_PATH, USE_CLINICAL, 
                               USE_CONTRALATERAL, USE_PREVIOUS, clinical_features, set_name)    

else:
    
    train_session(NAME, OUTPUT_PATH, model, partition, labels, DATA_PATH, training_generator, validation_generator, Custom_History, my_custom_checkpoint, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE)
    
    weights = np.load(OUTPUT_PATH + NAME + '/best_model_weights.npy', allow_pickle=True)
    model.set_weights(weights)
    
    scans_validation = list(set([x[:31] for x in partition['validation']]))
    class0 = list(set([x[:29] for x in scans_validation if labels[x[:31]] == 0]))
    class1 = list(set([x[:29] for x in scans_validation if labels[x[:31]] == 1]))
    scans_validation = interleave_two_lists(class0,class1)
    
    get_results_on_dataset(model,MRI_PATH, scans_validation, labels,  MASTER, NAME, OUTPUT_PATH, USE_CLINICAL, USE_CONTRALATERAL, USE_PREVIOUS, clinical_features)
    
    
    
    
#w1 = model.get_weights()
#
#np.save('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/test_weights.npy', w1)
#
#w2 = np.load('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/test_weights.npy', allow_pickle=True)
#
#
#X = np.random.random((1,512,512,3))
#clinical = np.random.random((1,11))
#
#yhat1 = model.predict([X, clinical])
#
#model.set_weights(w2)
#
#yhat2 = model.predict([X, clinical])
#
#
#
#for i in range(len(weights)):
#    print(np.std(w1[i] - w2[i]))
#    
