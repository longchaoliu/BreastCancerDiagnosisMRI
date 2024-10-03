#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 14:03:45 2022

@author: deeperthought
"""



import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="0"
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
import random


#%%
OUTPUT_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/'

# PARAMETERS
EPOCHS = 100
depth = 6
n_base_filters = 42
L2 = 1e-4
LR = 1e-5

USE_CLINICAL=False    # NEED TO FIX CLINICAL INPUT FIRST
USE_CONTRALATERAL = False
USE_PREVIOUS = False
DATA_AUGMENTATION = True
BATCH_SIZE = 8
BALANCE_PREVALENCE = False
USE_FOCAL_LOSS = True

LOAD_PRETRAINED_MODEL = False
MODEL_NAME = 'U-Net_classifier_train28154_val1620_DataAug_Clinical_depth6_filters42_L20_NaturalPrevalence'

PRETRAINED_MODEL_PATH = OUTPUT_PATH + '{}/best_model.h5'.format(MODEL_NAME)
LOAD_PRESAVED_DATA = False

PRESAVED_DATA = OUTPUT_PATH + '{}/Data.npy'.format(MODEL_NAME)
USE_SMALLER_DATASET = False

PREDICTION_ONLY = False

FINE_TUNE_WITH_HARD_CASES = False



#%% METRICS AND LOSSES
        
def FocalLoss(y_true, y_pred): 
    #y_true = tf.keras.backend.expand_dims(y_true,0)
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    term_0 = (1 - y_true[:,1]) * tf.keras.backend.pow(y_pred[:,1],5) * tf.keras.backend.log(1 - y_pred[:,1] + tf.keras.backend.epsilon())  
    term_1 = y_true[:,1] * tf.keras.backend.pow(1 - y_pred[:,1],5) * tf.keras.backend.log(y_pred[:,1] + tf.keras.backend.epsilon())   
    return -tf.keras.backend.mean(term_0 + term_1, axis=0)


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



#%% Clinical
    
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

#%% 2D Unet

from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.layers import concatenate



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
    model_loaded = tf.keras.models.load_model(PATH)
    model = unfreeze_layers(model_loaded)
    adam = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=adam, loss='binary_crossentropy',  metrics=['accuracy'])  
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


def make_prediction_whole_scan(model, all_data, USE_CLINICAL, clinic_info_exam, USE_CONTRALATERAL, USE_PREVIOUS):
    
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
        if USE_CLINICAL:
            if len(clinic_info_exam) == 1:
                yhat = model.predict([X,clinic_info_exam ])
            
        else:
            yhat = model.predict(X)
        slice_preds.append(yhat[0][-1])
    return slice_preds


def get_results_on_dataset(model, scans_list, labels, Data_description, NAME, OUT, USE_CLINICAL, USE_CONTRALATERAL, USE_PREVIOUS, clinical_info, name='VAL'):
    
    result = pd.DataFrame(columns=['scan','y_pred','y_true','max_slice','GT_slice'])
    N = 0
    TOT = len(scans_list)
    for scan in scans_list:
        scan = scan[:31]
        N += 1
        print('{}/{}'.format(N,TOT))
                
        all_data = load_data_prediction(scan,labels, Data_description, USE_CONTRALATERAL, USE_PREVIOUS) 
        
        pathology = all_data[0]
        segmented_slice = all_data[1]
        
        clinic_info_exam = 0
        if USE_CLINICAL:
            clinic_info_exam = clinical_info.loc[clinical_info['scan_ID'] == scan,['Family Hx', u'ETHNICITY', u'RACE', u'Age']].values

        slice_preds = make_prediction_whole_scan(model, all_data[2:], USE_CLINICAL, clinic_info_exam, USE_CONTRALATERAL, USE_PREVIOUS)
        
        result = result.append({'scan':scan,'y_pred':np.max(slice_preds), 'y_true':pathology, 
                                'max_slice':np.argmax(slice_preds), 'GT_slice':segmented_slice}, 
                ignore_index=True)

        print('{} - PATHOLOGY: {}\nsegmented_slice: {}, max_slice: {} = {}:'.format(scan, pathology, segmented_slice, np.argmax(slice_preds), np.max(slice_preds)))
 
        if N%100 == 0:
 
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
    
#%% MODELS

    
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

#        for layer_depth in range(depth-2, -1, -1):
#            
#            up_convolution = get_up_convolution(pool_size=(2,2), deconvolution=deconvolution, n_filters=n_base_filters*(layer_depth+1), L2=L2)(current_layer)
#
#            concat = concatenate([up_convolution, levels[layer_depth][1]] , axis=-1)
#            current_layer = create_convolution_block(n_filters=n_base_filters*(layer_depth+1),kernel=(3,3), input_layer=concat, padding='same', L2=L2)
#            current_layer = create_convolution_block(n_filters=n_base_filters*(layer_depth+1),kernel=(3,3), input_layer=current_layer, padding='same', L2=L2)

        
        current_layer = tf.keras.layers.Flatten()(current_layer)  
        
        image_features = tf.keras.layers.Dense(2)(current_layer)
        
        if USE_CLINICAL:
            clinical_inputs = Input((4))
            current_layer = tf.keras.layers.concatenate([image_features, clinical_inputs])
            image_features = tf.keras.layers.Dense(2)(current_layer)
            act = Activation(activation_name)(image_features)      
            model = Model(inputs=[inputs, clinical_inputs], outputs=act)

        
        else:
            act = Activation(activation_name)(image_features)
            model = Model(inputs=[inputs], outputs=act)

        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=initial_learning_rate), metrics=['acc'])

        return model
                                        

#%% FROM DIRECTORY


class DataGenerator_classifier(tf.keras.utils.Sequence): # inheriting from Sequence allows for multiprocessing functionalities
    'Generates data for Keras'
    def __init__(self, list_IDs,labels,clinical_info, batch_size=4, dim=(3,512,512), n_channels=3,
                 n_classes=1, shuffledata=True, do_augmentation=True, use_clinical_info=False, 
                 use_contralateral=False, use_previous=False, data_description = ''):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
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
            clinic = np.zeros((self.batch_size,4))
            scanids = [ids.split('/')[-1][:31] for ids in list_IDs_temp]
            for i in range(len(scanids)):
                clinic[i] = clinical_info.loc[clinical_info['scan_ID'] == scanids[i],['Family Hx', u'ETHNICITY', u'RACE', u'Age']].values

            clinic = np.array(clinic, dtype='float32')
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
            X[i] = np.load(ID, allow_pickle=True)
            #X[i] = tmp[1]
            y[i] = labels[ID.split('/')[-1][:31]]
                            
#            if self.use_contralateral:
#                if self.data_description.loc[self.data_description['Scan_ID'] == ID[:31], 'Contralateral Available'].values[0] == 1:
#                    X[i,:,:,:,3:6] = np.load(self.data_path.replace('X','Contra') + ID, allow_pickle=True)   # Here we add the path. ID can be the path
#
#            if self.use_previous:
#                if self.data_description.loc[self.data_description['Scan_ID'] == ID[:31], 'Previous Available'].values[0] == 1:
#                    X[i,:,:,-3:] = np.load(self.data_path.replace('X','Previous') + ID, allow_pickle=True)   # Here we add the path. ID can be the path

#            if not np.isfinite(X[i]).all():  # remove after Ive checked all scans
#                X[i] = np.zeros((X[i].shape))
        # make slices channels, so its a 2D image...
#        X = np.swapaxes(np.swapaxes(X,1,3),1,2)
#        X = np.reshape(X,(self.batch_size,512,512,9), order='F')
#            
        if self.do_augmentation:
            X_gen = self.augmentor.flow(X,y, batch_size=self.batch_size, shuffle=False, seed=self.seed)

            return next(X_gen)
        else:
            return X,y

class DataGenerator_classifier3D(tf.keras.utils.Sequence): # inheriting from Sequence allows for multiprocessing functionalities
    'Generates data for Keras'
    def __init__(self, list_IDs,labels,clinical_info, batch_size=4, dim=(3,512,512), n_channels=3,
                 n_classes=1, shuffledata=True, do_augmentation=True, use_clinical_info=False, 
                 use_contralateral=False, use_previous=False, data_description = ''):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
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
            clinic = np.zeros((self.batch_size,4))
            scanids = [ids.split('/')[-1][:31] for ids in list_IDs_temp]
            for i in range(len(scanids)):
                clinic[i] = clinical_info.loc[clinical_info['scan_ID'] == scanids[i],['Family Hx', u'ETHNICITY', u'RACE', u'Age']].values

            clinic = np.array(clinic, dtype='float32')
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
        X = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.dim[2], self.n_channels), dtype='float32')
        y = np.empty((self.batch_size, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):          
            X[i,:,:,:,:3] = np.load(ID, allow_pickle=True)
            y[i] = labels[ID.split('/')[-1][:31]]
                            
            if self.use_contralateral:
                if self.data_description.loc[self.data_description['Scan_ID'] == ID[:31], 'Contralateral Available'].values[0] == 1:
                    X[i,:,:,:,3:6] = np.load(self.data_path.replace('X','Contra') + ID, allow_pickle=True)   # Here we add the path. ID can be the path

            if self.use_previous:
                if self.data_description.loc[self.data_description['Scan_ID'] == ID[:31], 'Previous Available'].values[0] == 1:
                    X[i,:,:,-3:] = np.load(self.data_path.replace('X','Previous') + ID, allow_pickle=True)   # Here we add the path. ID can be the path

#            if not np.isfinite(X[i]).all():  # remove after Ive checked all scans
#                X[i] = np.zeros((X[i].shape))
        # make slices channels, so its a 2D image...
        X = np.swapaxes(np.swapaxes(X,1,3),1,2)
        X = np.reshape(X,(self.batch_size,512,512,9), order='F')
            
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


def save_model_and_weights(model, NAME, FOLDER):    
    model_to_save = tf.keras.models.clone_model(model)
    model_to_save.set_weights(model.get_weights())
    model_to_save = freeze_layers(model_to_save)
#    model_to_save.save_weights(FOLDER + NAME + '_weights.h5')
    model_to_save.save(FOLDER + NAME + '.h5')       

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





def train_session(NAME, OUTPUT_PATH, model, partition, training_generator, validation_generator, Custom_History, my_model_checkpoint, EPOCHS = 100, BATCH_SIZE=4):

    # Some last check on data
    subjects_train = [x.split('/')[-1][:20] for x in training_generator.list_IDs]
    subjects_val = [x.split('/')[-1][:20] for x in validation_generator.list_IDs]
    assert len(set(subjects_train).intersection(set(subjects_val)))==0, 'subjects in both train and val!'
    #scans_train = list(set([x.split('/')[-1][:31] for x in training_generator.list_IDs]))
    #TOTAL_TRAIN = len(scans_train)
    #MAL_TRAIN = sum([labels[x] for x in scans_train])
    #BEN_TRAIN = TOTAL_TRAIN - MAL_TRAIN
    #CLASS_WEIGHT = 1. #BEN_TRAIN/float(MAL_TRAIN)
    
    if not os.path.exists(OUTPUT_PATH+NAME):
        os.makedirs(OUTPUT_PATH+NAME)

    with open(OUTPUT_PATH+NAME + '/model_summary.txt', 'w') as f:
    
        model.summary(print_fn=lambda x: f.write(x + '\n'))    
    
    np.save(OUTPUT_PATH + NAME + '/Data.npy', partition)

    csv_logger = tf.keras.callbacks.CSVLogger(OUTPUT_PATH+NAME + '/csvLogger.log', 
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
                                shuffle=False,
                                #class_weight = {0:1., 1:CLASS_WEIGHT},  # Doesnt work with custom loss function.
                                callbacks=[Custom_History, csv_logger, my_custom_checkpoint, myEarlyStop])


#%%    
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
    plt.savefig(OUTPUT_PATH + NAME + '/End_Training.png')
    plt.close()
    
    model.save(OUTPUT_PATH + NAME +  '/last_model.h5')
    

#%% TRAINING SESSION CONFIGURATION




if LOAD_PRESAVED_DATA:
    partition_selected_slices = np.load('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/SelectedSlices_DGNS_train26324_val1339_DataAug_Clinical_depth6_filters42_L20/Data.npy', allow_pickle=True).item()
    partition_random_slices = np.load('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/DGNS_noConverts_float32_train27702_val1548_DataAug_Clinical_depth6_filters42_L20/Data.npy', allow_pickle=True).item()

    DATA_PATH_Original = '/home/deeperthought/kirbyPRO/Saggittal_Full_Slices/2D_slices/X/'
    
    partition = {'train':[],'validation':[]}
    
    partition['train'] = partition_selected_slices['train']
    partition['train'].extend([DATA_PATH_Original + x for x in partition_random_slices['train']])
    
    partition['validation'] = [DATA_PATH_Original + x for x in partition_random_slices['validation']]

    MASTER = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Data_Partitions.csv')

    labels = dict(zip(MASTER['Scan_ID'],(MASTER['Pathology']=='Malignant').astype(int)))

    BALANCE_PREVALENCE = False
    FINE_TUNE_WITH_HARD_CASES = False
#%% LOAD DATA

else:
    
    DATA_PATH_Original = '/home/deeperthought/kirbyPRO/Saggittal_Full_Slices/2D_slices/X/'
    DATA_PATH_selected_by_network = '/home/deeperthought/kirbyPRO/Saggittal_Full_Slices/U-Net_classifier_train28154_val1620_DataAug_Clinical_depth6_filters42_L20/2D_slices/X/'
    OUTPUT_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/' 
    partitions_dictionary = np.load('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Partitions_Pipelines.npy', allow_pickle=True).item()
    
    DATA_DGNS_VAL1 = np.load('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/U-Net_classifier_train28154_val1620_DataAug_Clinical_depth6_filters42_L20/Data.npy', allow_pickle=True).item()
    
    available_data_slices_selected_by_network = os.listdir(DATA_PATH_selected_by_network)
    available_data_original = os.listdir(DATA_PATH_Original)

    len(set(available_data_slices_selected_by_network))
    len(set(available_data_original))
    
    len(set(available_data_slices_selected_by_network).intersection(set(available_data_original)))
    
    
    MASTER = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Data_Partitions.csv')
    
    labels = dict(zip(MASTER['Scan_ID'],(MASTER['Pathology']=='Malignant').astype(int)))
    
    
    ## Keep segmented malignants from original data sampling (using segmented data)
    ## Keep benigns from second sampling (as selected by DGNS model)
    
    Segmented = MASTER.loc[MASTER['Segmented'] == 1, 'Scan_ID'].values
    Benigns = MASTER.loc[MASTER['Pathology'] == 'Benign', 'Scan_ID'].values
    
    segmented_slices = [DATA_PATH_Original + x for x in available_data_original if x[:31] in Segmented]
    benign_slices = [DATA_PATH_selected_by_network + x for x in available_data_slices_selected_by_network if x[:31] in Benigns]
    benign_slices.extend([DATA_PATH_Original + x for x in available_data_original if x[:31] in Benigns])

    
    available_data = segmented_slices +  benign_slices
    
    print('Found {} slices'.format(len(available_data)))
    
    partition = {'train':[],'validation':[]}
    
    partition['train'] = [x for x in available_data if x.split('/')[-1][:31] in partitions_dictionary['DGNS_3D']['train']]
    partition['validation'] = [DATA_PATH_Original + x for x in DATA_DGNS_VAL1['validation']]#[x for x in available_data if x.split('/')[-1][:31] in partitions_dictionary['DGNS_3D']['val']]

print('Train: {}\nVal: {}'.format(len(partition['train']), len(partition['validation'])))

#scanIDS = [x.split('/')[-1][:31] for x in partition['train']]
#len(scanIDS)
#len(set(scanIDS))
#np.sum([labels[x.split('/')[-1][:31]] for x in partition['train']])


if FINE_TUNE_WITH_HARD_CASES:
    df_model_results_train_set = pd.read_csv(OUTPUT_PATH + '{}/Train_Set_result.csv'.format(MODEL_NAME))
    HARD_BENIGNS = df_model_results_train_set.loc[(df_model_results_train_set['y_true'] == 0)*(df_model_results_train_set['y_pred'] > 0.5), 'scan'].values

    HARD_BENIGNS_SLICES_TRAIN = [x for x in partition['train'] if x.split('/')[-1][:31] in HARD_BENIGNS]
    MALIGNANT_SLICES_TRAIN = [x for x in partition['train'] if labels[x.split('/')[-1][:31]] == 1]
    
    partition['train'] = MALIGNANT_SLICES_TRAIN + HARD_BENIGNS_SLICES_TRAIN
    print('Selecting only hard benigns from model {}\n predictions > 0.5\n now I have {} scans in train set'.format(MODEL_NAME, len(partition['train'])))    

if BALANCE_PREVALENCE:

    print('Balancing populations of segmented cancers and benigns:')
    train_cancers = [x for x in partition['train'] if labels[x.split('/')[-1][:31]] == 1]
    train_benigns = [x for x in partition['train'] if labels[x.split('/')[-1][:31]] == 0]
    
    print('Natural prevalence: {}/{}'.format(len(train_cancers),len(train_benigns) ))  

    partition['train'] = train_benigns
    partition['train'].extend( list(np.random.choice(train_cancers, size=len(train_benigns), replace=True)))
    random.Random(42).shuffle(partition['train'])
    
    train_cancers = [x for x in partition['train'] if labels[x.split('/')[-1][:31]] == 1]
    train_benigns = [x for x in partition['train'] if labels[x.split('/')[-1][:31]] == 0]
    
    print('Balanced prevalence: {}/{}'.format(len(train_cancers),len(train_benigns) ))  


partition['train'] = [x for x in partition['train'] if not x.endswith('_0.npy')]


#%% QA

'MSKCC_16-328_1_09903_20060910_l' # no visible lesion...
'MSKCC_16-328_1_09071_20071108_r' # super noisy
'MSKCC_16-328_1_05846_20030226_r' # noisy, no visible lesion
'MSKCC_16-328_1_12064_20080513_r' # noisy. Should be able to compute SNR 
'MSKCC_16-328_1_03612_20030804_r' # bad quality image.. 2003


#%%  CLINICAL INFO

clinical_info = pd.read_csv('/home/deeperthought/kirbyPRO/Saccade_Features_SGMT/Clinical_features/Clinical_features_July2022.csv')

scanids_set = [x.split('/')[-1].split('.')[0][:31] for x in partition['train']]
SCANS_MISSING_CLINICAL_INFO = list(set(scanids_set) - set(clinical_info['scan_ID'].values))
print('Removing {} scans that have no clinical info:'.format(len(SCANS_MISSING_CLINICAL_INFO)))
for i in range(len(SCANS_MISSING_CLINICAL_INFO)):
    print(i)
    scan_slices = [x for x in partition['train'] if x.split('/')[-1].startswith(SCANS_MISSING_CLINICAL_INFO[i])]
    for scan_slice in scan_slices:
        partition['train'].remove(scan_slice)

scanids_set = [x.split('/')[-1].split('.')[0][:31] for x in partition['train']]
assert len(set(scanids_set) - set(clinical_info['scan_ID'].values)) == 0

#%%


CHANNELS = 3
warnings.filterwarnings("ignore") # ImageDataGenerator expects only 3 channels. It raises a warning but doesnt affect.

if USE_CONTRALATERAL:
    CHANNELS += 3
    warnings.filterwarnings("ignore") # ImageDataGenerator expects only 3 channels. It raises a warning but doesnt affect.

if USE_PREVIOUS:
    CHANNELS += 3    
    warnings.filterwarnings("ignore") # ImageDataGenerator expects only 3 channels. It raises a warning but doesnt affect.

if USE_SMALLER_DATASET:
    partition['train'] = partition['train'][:1000]
    partition['validation'] = partition['validation'][:1000]

params_train = {'dim': (512,512),
          'batch_size': BATCH_SIZE,
          'n_classes': 2,
          'n_channels': CHANNELS,
          'shuffledata': True,
          'do_augmentation': DATA_AUGMENTATION,
          'clinical_info':clinical_info,
          'use_clinical_info':USE_CLINICAL,
          'use_contralateral':USE_CONTRALATERAL,
          'use_previous':USE_PREVIOUS,
          'data_description':MASTER}

params_val = {'dim': (512,512),
          'batch_size': BATCH_SIZE,
          'n_classes': 2,
          'n_channels': CHANNELS,
          'shuffledata': False,
          'do_augmentation':False,
          'clinical_info':clinical_info,
          'use_clinical_info':USE_CLINICAL,
          'use_contralateral':USE_CONTRALATERAL,
          'use_previous':USE_PREVIOUS,
          'data_description':MASTER}

#------ GENERATORS -----------
training_generator = DataGenerator_classifier(partition['train'],labels, **params_train)
validation_generator = DataGenerator_classifier(partition['validation'],labels, **params_val)
    
NAME = 'SelectedSlices_and_Random_DGNS_train{}_val{}_'.format(len(training_generator.list_IDs), len(validation_generator.list_IDs))
if DATA_AUGMENTATION: NAME += 'DataAug_'
if USE_CLINICAL: NAME += 'Clinical_'
if USE_CONTRALATERAL: NAME += 'Contra_'
if USE_PREVIOUS: NAME += 'Previous_'
if USE_FOCAL_LOSS: NAME += 'FocalLoss_'
NAME += 'depth{}_filters{}_L2{}'.format(depth, n_base_filters, L2)

#---- CHECKPOINTS ----------
my_custom_checkpoint = my_model_checkpoint(MODEL_PATH=OUTPUT_PATH+NAME, MODEL_NAME='/best_model' )
Custom_History = MyHistory(OUTPUT_PATH, NAME)

#----- MODEL -----------
if LOAD_PRETRAINED_MODEL:
    model = load_model_frozen(PRETRAINED_MODEL_PATH)
    NAME = 'Fine_Tuning_' + PRETRAINED_MODEL_PATH.split('/')[-2]
    if not os.path.exists(OUTPUT_PATH + NAME):
        os.mkdir(OUTPUT_PATH + NAME)

    my_custom_checkpoint = my_model_checkpoint(MODEL_PATH=OUTPUT_PATH+NAME, MODEL_NAME='/best_model' )
    Custom_History = MyHistory(OUTPUT_PATH, NAME)

        
    if os.path.exists(OUTPUT_PATH + NAME + '/csvLogger.log'):
        logger = pd.read_csv(OUTPUT_PATH + NAME + '/csvLogger.log')
    
        MIN_LOSS = np.min(logger['val_loss'])
        my_custom_checkpoint.val_loss = [MIN_LOSS]
        Custom_History.acc = list(logger['acc'].values)
        Custom_History.loss = list(logger['loss'].values)
        Custom_History.val_acc = list(logger['val_acc'].values)
        Custom_History.val_loss = list(logger['val_loss'].values)
else:
    model = UNet_v0_2D_Classifier(input_shape =  (512,512,3), pool_size=(2, 2),initial_learning_rate=1e-5, 
                                     deconvolution=True, depth=depth, n_base_filters=n_base_filters,
                                     activation_name="sigmoid", L2=L2, USE_CLINICAL=USE_CLINICAL)
    
    #tf.keras.utils.plot_model(model, to_file=OUTPUT_PATH + '/DGNS_Model.png', show_shapes=True)
if USE_FOCAL_LOSS:
    model.compile(optimizer=Adam(lr=LR),#, amsgrad=True), 
      loss=FocalLoss, 
      metrics=['accuracy'])
    
if LOAD_PRESAVED_DATA:
    partition = np.load(PRESAVED_DATA, allow_pickle=True).item()
 
if PREDICTION_ONLY:
    scans_validation = list(set([x[:31] for x in partition['validation']]))
    #scans_validation = list(set([x[:31] for x in partition['train']]))
    #name='Train_Set'
    get_results_on_dataset(model, scans_validation, labels,  MASTER, NAME, OUTPUT_PATH, USE_CLINICAL, 
                           USE_CONTRALATERAL, USE_PREVIOUS, clinical_info)

else:
    train_session(NAME, OUTPUT_PATH, model, partition, training_generator, validation_generator, Custom_History, my_custom_checkpoint, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE)
    
    best_model = load_model_frozen(OUTPUT_PATH + NAME + '/best_model.h5')
    last_model = load_model_frozen(OUTPUT_PATH + NAME + '/last_model.h5')
    scans_validation = list(set([x.split('/')[-1][:31] for x in partition['validation']]))
    
    get_results_on_dataset(best_model, scans_validation, labels,  MASTER, NAME, OUTPUT_PATH, USE_CLINICAL, USE_CONTRALATERAL, USE_PREVIOUS, clinical_info)
    


#%%
#training_generator.list_IDs
#x, y = training_generator.__getitem__(0)
