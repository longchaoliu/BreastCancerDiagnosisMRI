#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:56:33 2022

@author: deeperthought
"""



GPU = 2
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
seed(42)

from utils import FocalLoss, UNet_v0_2D_Classifier, load_and_preprocess
from train_utils import DataGenerator_classifier, add_age, add_ethnicity_oneHot, add_family_hx, my_model_checkpoint, MyHistory, interleave_two_lists, get_results_on_dataset, train_session




####### NEED TO SIMPLIFY NEEDED METADATA! ##############


RISK = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Data_ExamHistory_Labels.csv')
MASTER = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Data_Partitions.csv')
df = pd.read_csv('/home/deeperthought/Projects/DGNS/Risk_Prediction/Sessions/FINAL/training_dropout0.5_classWeight1.0_Clinical/paper_selected/class_weight1/results_sheet_TEST.csv')

clinical_df = pd.read_excel('/home/deeperthought/Projects/MSKCC/MSKCC/Data_spreadsheets/Diamond_and_Gold/CCNY_CLINICAL_4_17_2019.xlsx', header=[0,1])    


# DATA_PATH = '/media/SD/X/'
DATA_PATH = '/media/HDD/Diagnosis_2D_slices/X/'
OUTPUT_PATH = '/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/sessions/'



# PARAMETERS
EPOCHS = 50
depth = 6
n_base_filters = 42
L2 = 1e-5
BATCH_SIZE = 8

USE_CLINICAL=True
USE_CONTRALATERAL = False
USE_PREVIOUS = False
DATA_AUGMENTATION = True

NAME = 'ReplicatePaperVal1_classifier_train1_val1_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8'

CNN = True
RESNET = False
IMAGENET_WEIGHTS = False
BINARY_CROSSENTROPY = False
MODEL_NAME = 'ReplicatePaperVal1_classifier_train1_val1_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8'

LOAD_PRETRAINED_MODEL = True
PRETRAINED_MODEL_WEIGHTS = '/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/sessions/ReplicatePaperVal1_classifier_train1_val1_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/best_model_weights.npy'

LOAD_PRESAVED_DATA = True
PRESAVED_DATA = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/DATA/Data.npy'
PRESAVED_LABELS = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/DATA/Labels.npy'
PRESAVED_CLINICAL_INFO = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/DATA/Clinical_Data_Train_Val.csv'

PREDICTION_ONLY = True
SETS_FOR_PREDICTION = ['validation']

USE_SMALLER_DATASET = False
FRACTION = 0.5

#%% LOAD PARTITIONS

if LOAD_PRESAVED_DATA:

    partition = np.load(PRESAVED_DATA, allow_pickle=True).item()
    labels = np.load(PRESAVED_LABELS, allow_pickle=True).item()
    clinical_features = pd.read_csv(PRESAVED_CLINICAL_INFO)
    
else:
    
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


scans_train = list(set([x[:31] for x in partition['train']]))
scans_val = list(set([x[:31] for x in partition['validation']]))

imgs_benign = os.listdir(DATA_PATH + 'BENIGN')
imgs_malign = os.listdir(DATA_PATH + 'MALIGNANT')

train_scans_ben = [DATA_PATH + 'BENIGN/' + x for x in imgs_benign if x[:31] in scans_train]
train_scans_mal = [DATA_PATH + 'MALIGNANT/' + x for x in imgs_malign if x[:31] in scans_train]



val_scans_ben = [DATA_PATH + 'BENIGN/' + x for x in imgs_benign if x[:31] in scans_val]
val_scans_mal = [DATA_PATH + 'MALIGNANT/' + x for x in imgs_malign if x[:31] in scans_val]

partition['train'] = train_scans_ben
partition['train'].extend(train_scans_mal)


partition['validation'] = val_scans_ben
partition['validation'].extend(val_scans_mal)

print('Train: {}\nVal: {} \nTest: {}'.format(len(partition['train']), len(partition['validation']), len(partition['test'])))


assert set([x.split('/')[-1][:20] for x in partition['train']]).intersection(set([x.split('/')[-1][:20] for x in partition['validation']])) == set()
assert set([x.split('/')[-1][:20] for x in partition['train']]).intersection(set([x.split('/')[-1][:20] for x in partition['test']])) == set()
assert set([x.split('/')[-1][:20] for x in partition['validation']]).intersection(set([x.split('/')[-1][:20] for x in partition['test']])) == set()

# assert set(subj_train_benign).intersection(set(subj_train_malignant)) == set()  I have breasts with both pathologies, same subject.


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
        
    NAME += '_classifier_train{}_val{}_'.format(len(list(set([x[:31] for x in training_generator.list_IDs]))), len(list(set([x[:31] for x in validation_generator.list_IDs]))))
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
        scans_test = interleave_two_lists(class0,class1)

                
        get_results_on_dataset(model, scans_test, labels,  MASTER, NAME, OUTPUT_PATH, USE_CLINICAL, 
                               USE_CONTRALATERAL, USE_PREVIOUS, clinical_features, set_name)    

else:
    
    train_session(NAME, OUTPUT_PATH, model, partition, DATA_PATH, training_generator, validation_generator, Custom_History, my_custom_checkpoint, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE)
    
    weights = np.load(OUTPUT_PATH + NAME + '/best_model_weights.npy', allow_pickle=True)
    model.set_weights(weights)
    
    scans_validation = list(set([x[:31] for x in partition['validation']]))
    class0 = list(set([x[:31] for x in scans_validation if labels[x[:31]] == 0]))
    class1 = list(set([x[:31] for x in scans_validation if labels[x[:31]] == 1]))
    scans_validation = interleave_two_lists(class0,class1)
    
    get_results_on_dataset(model, scans_validation, labels,  MASTER, NAME, OUTPUT_PATH, USE_CLINICAL, USE_CONTRALATERAL, USE_PREVIOUS, clinical_features)
    
    
    
    

