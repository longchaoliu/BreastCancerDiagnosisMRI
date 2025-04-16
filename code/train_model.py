#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:56:33 2022

@author: deeperthought
"""


GPU = 0
import tensorflow as tf
if tf.__version__[0] == '1':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list="0"
    tf.keras.backend.set_session(tf.Session(config=config))

elif tf.__version__[0] == '2':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only use the first GPU
      try:
        tf.config.experimental.set_visible_devices(gpus[GPU], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[GPU], True)
      except RuntimeError as e:
        # Visible devices must be set at program startup
        print(e)

import os
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from numpy.random import seed
import random

from utils import FocalLoss, FocalLoss_5_0, UNet_v0_2D_Classifier, load_and_preprocess
from train_utils import DataGenerator_classifier, add_age, add_ethnicity_oneHot, add_family_hx, my_model_checkpoint, MyHistory, interleave_two_lists, get_results_on_dataset, train_session


# Set a seed for reproducibility
random.seed(42)
seed(42)
tf.random.set_seed(42)

RISK = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Data_ExamHistory_Labels.csv')
MASTER = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Data_Partitions.csv')
df = pd.read_csv('/home/deeperthought/Projects/DGNS/Risk_Prediction/Sessions/FINAL/training_dropout0.5_classWeight1.0_Clinical/paper_selected/class_weight1/results_sheet_TEST.csv')

clinical_df = pd.read_excel('/home/deeperthought/Projects/MSKCC/MSKCC/Data_spreadsheets/Diamond_and_Gold/CCNY_CLINICAL_4_17_2019.xlsx', header=[0,1])    

OUTPUT_PATH = '/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/sessions/'

# PARAMETERS
EPOCHS = 30
depth = 6
n_base_filters = 42
L2 = 1e-5
BATCH_SIZE = 8

USE_CLINICAL=False
USE_CONTRALATERAL = False
USE_PREVIOUS = False
DATA_AUGMENTATION = True

# DATASET = 'New'#'Paper'#'Paper'
# DATA_PATH = '/media/SD/X/'
DATA_PATH = '/media/HDD/Diagnosis_2D_slices/X/'
# DATA_PATH = '/home/deeperthought/kirbyPRO/Saggittal_Full_Slices/2D_slices/X/'

NAME = 'tfrecords_smallData_DataAug' 
# NAME = 'new_data_10%_add_axials_classifier_train16102_val5892_DataAug_depth6_filters42_L21e-05_batchsize8'

CNN = True
RESNET = False
IMAGENET_WEIGHTS = False
BINARY_CROSSENTROPY = False

# LOAD_PRESAVED_DATA = True
# USE_PRELOADED_DATA = False
# ADD_AXIALS = False
# ONLY_AXIALS = False 

LOAD_PRETRAINED_MODEL = True
PREDICTION_ONLY = True
SETS_FOR_PREDICTION = ['validation']

# PRESAVED_DATA = "/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/sessions/DATA/New_data/new_data.npy"
# PRESAVED_LABELS = "/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/sessions/DATA/New_data/new_data_labels.npy"
# PRESAVED_CLINICAL_INFO = "/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/sessions/DATA/New_data/new_data_clinical.csv"

# PRESAVED_DATA = "/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/sessions/DATA/New_data/Axial/new_data_10%_add_axials.npy"
# PRESAVED_LABELS = "/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/sessions/DATA/New_data/Axial/new_data_10%_add_axials_labels.npy"

# PRESAVED_DATA = "/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/sessions/DATA/New_data/Axial/Axial_data.npy"
# PRESAVED_LABELS = "/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/sessions/DATA/New_data/Axial/Axial_data_labels.npy"
# PRESAVED_CLINICAL_INFO = "/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/sessions/DATA/New_data/Axial/Axial_Clinical_Data_Train_Val.csv"

PRESAVED_DATA = "/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/sessions/DATA/New_data/10%/new_data_10%.npy"
PRESAVED_LABELS ="/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/sessions/DATA/New_data/10%/new_data_10%_labels.npy"
PRESAVED_CLINICAL_INFO ="/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/sessions/DATA/New_data/10%/new_data_10%_clinical.csv"

USE_SMALLER_DATASET = False
FRACTION = 0.1

#%%
MODEL_NAME = f'{NAME}'
PRETRAINED_MODEL_WEIGHTS = f'/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/sessions/{NAME}/last_model_weights.npy'

#%% LOAD PARTITIONS

partition = np.load(PRESAVED_DATA, allow_pickle=True).item()
labels = np.load(PRESAVED_LABELS, allow_pickle=True).item()
clinical_features = pd.read_csv(PRESAVED_CLINICAL_INFO)


print('Train: {}\nVal: {} \nTest: {}'.format(len(partition['train']), len(partition['validation']), len(partition['test'])))


assert set([x.split('/')[-1][:20] for x in partition['train']]).intersection(set([x.split('/')[-1][:20] for x in partition['validation']])) == set()
assert set([x.split('/')[-1][:20] for x in partition['train']]).intersection(set([x.split('/')[-1][:20] for x in partition['test']])) == set()
assert set([x.split('/')[-1][:20] for x in partition['validation']]).intersection(set([x.split('/')[-1][:20] for x in partition['test']])) == set()

#%% TRAINING SESSION

CHANNELS = 3
if USE_CONTRALATERAL:
    CHANNELS += 3
    warnings.filterwarnings("ignore") # ImageDataGenerator expects only 3 channels. It raises a warning but doesnt affect.

if USE_PREVIOUS:
    CHANNELS += 3    
    warnings.filterwarnings("ignore") # ImageDataGenerator expects only 3 channels. It raises a warning but doesnt affect.

params_train = {'dim': (512,512),
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
          'batch_size': BATCH_SIZE,
          'n_classes': 2,
          'n_channels': CHANNELS,
          'shuffledata': True,
          'do_augmentation':False,
          'clinical_info':clinical_features,
          'use_clinical_info':USE_CLINICAL,
          'use_contralateral':USE_CONTRALATERAL,
          'use_previous':USE_PREVIOUS,
          'data_description':MASTER}

if USE_SMALLER_DATASET:
    print('Using smaller dataset: fraction = {}'.format(FRACTION))
    partition['train'] = np.random.choice(partition['train'], replace=False, size=int(len(partition['train'])*FRACTION))

    #partition['validation'] = np.random.choice(partition['validation'], replace=False, size=int(len(partition['validation'])*FRACTION))


#------ GENERATORS -----------
training_generator = DataGenerator_classifier(partition['train'],labels, **params_train)
validation_generator = DataGenerator_classifier(partition['validation'],labels, **params_val)



if CNN:
    model = UNet_v0_2D_Classifier(input_shape =  (512,512,CHANNELS), pool_size=(2, 2), 
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
            clinical_inputs = tf.keras.layers.Input(shape=(11,))

            final_layer = tf.keras.layers.MaxPooling2D(7)(model.layers[-1].output)                
            final_layer = tf.keras.layers.Flatten()(final_layer)
            
            clinical_inputs = tf.keras.layers.Input(shape=(11,))
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
        
    NAME += '_classifier_train{}_val{}_'.format(len(list(set([x.split('/')[-1][:31] for x in training_generator.list_IDs]))), len(list(set([x.split('/')[-1][:31] for x in validation_generator.list_IDs]))))
    if DATA_AUGMENTATION: NAME += 'DataAug_'
    if USE_CLINICAL: NAME += 'Clinical_'
    if USE_CONTRALATERAL: NAME += 'Contra_'
    if USE_PREVIOUS: NAME += 'Previous_'
    NAME += 'depth{}_filters{}_L2{}_batchsize{}'.format(depth, n_base_filters, L2, BATCH_SIZE)
    #---- CHECKPOINTS ----------
    my_custom_checkpoint = my_model_checkpoint(MODEL_PATH=OUTPUT_PATH+NAME, MODEL_NAME='/best_model' )
    Custom_History = MyHistory(OUTPUT_PATH, NAME)


if BINARY_CROSSENTROPY:
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=Adam(lr=1e-5), metrics=['acc'])
    
else:
    model.compile(loss=FocalLoss, optimizer=Adam(lr=1e-5), metrics=['acc'])



if PREDICTION_ONLY:
    
    for set_name in SETS_FOR_PREDICTION:
                
        scans_test = partition[set_name]
        
        class0 = list(set([x.split('/')[-1][:31] for x in scans_test if labels[x.split('/')[-1][:31]] == 0]))
        class1 = list(set([x.split('/')[-1][:31] for x in scans_test if labels[x.split('/')[-1][:31]] == 1]))

        class0.sort()
        class1.sort()

        scans_test = interleave_two_lists(class0,class1)
                
        get_results_on_dataset(model, scans_test, labels,  MASTER, NAME, OUTPUT_PATH, USE_CLINICAL, 
                               USE_CONTRALATERAL, USE_PREVIOUS, clinical_features, set_name)    

else:
    
    train_session(NAME, OUTPUT_PATH, model, partition, DATA_PATH, training_generator, validation_generator, Custom_History, my_custom_checkpoint, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE)
    
    weights = np.load(OUTPUT_PATH + NAME + '/best_model_weights.npy', allow_pickle=True)
    model.set_weights(weights)
    
    class0 = list(set([x.split('/')[-1][:31] for x in partition['validation'] if labels[x.split('/')[-1][:31]] == 0]))
    class1 = list(set([x.split('/')[-1][:31] for x in partition['validation'] if labels[x.split('/')[-1][:31]] == 1]))
    
    class0.sort()
    class1.sort()
        
    scans_validation = interleave_two_lists(class0,class1)
        
    get_results_on_dataset(model, scans_validation, labels,  MASTER, NAME, OUTPUT_PATH, USE_CLINICAL, USE_CONTRALATERAL, USE_PREVIOUS, clinical_features, 'validation')
    
