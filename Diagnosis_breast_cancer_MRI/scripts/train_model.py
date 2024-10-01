#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 2023

@author: Lukas Hirsch
"""

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="1"
tf.keras.backend.set_session(tf.Session(config=config))

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib
from skimage.transform import resize
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import warnings


from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)

os.chdir('/Diagnosis_breast_cancer_MRI/scripts/')
from model_utils import FocalLoss, UNet_v0_2D_Classifier

#%% USER INPUT

OUTPUT_PATH = '/Diagnosis_breast_cancer_MRI/sessions/' 

# Data description
DATA_DESCRIPTION_PATH = '/Diagnosis_breast_cancer_MRI/Training_data/Data_Description.csv'

# Pre-trained model 
PRETRAINED_MODEL_WEIGHTS_PATH = '/Diagnosis_breast_cancer_MRI/model/CNN_weights.npy' 

 # Pre-stored slices, normalized
DATA_PATH = '/Diagnosis_breast_cancer_MRI/Training_data/images/'

# Raw MRI data
MRI_PATH = '/Diagnosis_breast_cancer_MRI/MRI/Breast_MRI_001/' 

# Spatially aligned previous MRIs (optional)
MRI_ALIGNED_HISTORY_PATH = ''

# PARAMETERS
EPOCHS = 3
depth = 6
n_base_filters = 42
L2 = 1e-5
BATCH_SIZE = 2

USE_CLINICAL=True
USE_CONTRALATERAL = False
USE_PREVIOUS = False
DATA_AUGMENTATION = True

NAME = 'Test_session'

CNN = True
RESNET = False

LOAD_PRETRAINED_MODEL = True

USE_SMALLER_DATASET = False
FRACTION = 0.1

#%% Functions

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


def load_main_spreadsheet_and_create_sub_spreadsheets(DATA_DESCRIPTION_PATH):
    df = pd.read_csv(DATA_DESCRIPTION_PATH)
    df['Image'] = df['Image'].apply(lambda x : x + '.png')
    
    ###### CLINICAL 
    CLINICAL_DF_COLUMNS = ['DE-ID',	'Scan_ID',	'Image',	'Age',	'Family Hx',	
                           'ETHNICITY_HISPANIC OR LATINO', 'ETHNICITY_NOT HISPANIC', 	
                           'ETHNICITY_UNKNOWN',	'RACE_ASIAN-FAR EAST/INDIAN SUBCONT', 	
                           'RACE_BLACK OR AFRICAN AMERICAN', 	
                           'RACE_NATIVE AMERICAN-AM IND/ALASKA',	 
                           'RACE_NATIVE HAWAIIAN OR PACIFIC ISL',	
                           'RACE_UNKNOWN','RACE_WHITE']
    clinical_features = pd.DataFrame(df[CLINICAL_DF_COLUMNS])
    clinical_features['Age'] = clinical_features['Age']/100.   

    ###### CLINICAL 
    labels = pd.DataFrame(df[['Image', 'Pathology']])
    labels['Pathology'] = labels['Pathology'].apply(lambda x : 1 if x == 'Malignant' else 0)
    labels = dict(zip(labels['Image'], labels['Pathology']))
    
    ###### PATHOLOGY INFORMATION
    PATHOLOGY_COLUMNS = ['DE-ID', u'Exam', u'Scan_ID', u'BIRADS', u'Pathology']
    PATHOLOGY_SPREADSHEET = pd.DataFrame(df[PATHOLOGY_COLUMNS])
    
    ###### DATA INFORMATION
    DATA_COLUMNS = ['Image', 'Partition']
    df_partitions = pd.DataFrame(df[DATA_COLUMNS])
    partition = pd.DataFrame(columns=['train','validation'])
    TRAIN_DF = pd.Series(df_partitions.loc[df_partitions['Partition'] == 'train', 'Image'])
    TRAIN_DF.reset_index(inplace=True, drop=True)
    VAL_DF = pd.Series(df_partitions.loc[df_partitions['Partition'] == 'validation', 'Image'])
    VAL_DF.reset_index(inplace=True, drop=True)
    partition = pd.concat([TRAIN_DF, VAL_DF], ignore_index=True, axis=1)
    partition.columns = ['train', 'validation']
    partition = partition.to_dict(orient='list')  
    for key in partition.keys():
        partition[key] = [x for x in partition[key] if str(x) != 'nan']
    
    return partition, labels, PATHOLOGY_SPREADSHEET, clinical_features

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
            scanids = ['_'.join(ids.split('_')[:-1]) for ids in list_IDs_temp]
            for i in range(len(scanids)):
                clinic[i] = self.clinical_info.loc[self.clinical_info['Scan_ID'] == scanids[i],[u'Family Hx',u'Age',u'ETHNICITY_HISPANIC OR LATINO',u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',u'RACE_ASIAN-FAR EAST/INDIAN SUBCONT',u'RACE_BLACK OR AFRICAN AMERICAN',u'RACE_NATIVE AMERICAN-AM IND/ALASKA',u'RACE_NATIVE HAWAIIAN OR PACIFIC ISL',u'RACE_UNKNOWN',u'RACE_WHITE']].values

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
            
            X[i,:,:,:3] = np.array(Image.open(self.data_path + ID), dtype='float32')
            #X[i,:,:,:3] = np.load(self.data_path + ID, allow_pickle=True)   # Here we add the path. ID can be the path
            y[i] = self.labels[ID]
            
            if self.use_contralateral:
                if self.data_description.loc[self.data_description['Scan_ID'] == '_'.join(ID.split('_')[:-1]), 'Contralateral Available'].values[0] == 1:
                    
                    X[i,:,:,3:6] = np.array(Image.open(self.data_path.replace('X','Contra') + ID), dtype='float32')
                    #X[i,:,:,3:6] = np.load(self.data_path.replace('X','Contra') + ID, allow_pickle=True)   # Here we add the path. ID can be the path

            if self.use_previous:
                if self.data_description.loc[self.data_description['Scan_ID'] == '_'.join(ID.split('_')[:-1]), 'Previous Available'].values[0] == 1:
                    X[i,:,:,-3] = np.array(Image.open(self.data_path.replace('X','Previous') + ID), dtype='float32')
                    #X[i,:,:,-3:] = np.load(self.data_path.replace('X','Previous') + ID, allow_pickle=True)   # Here we add the path. ID can be the path

            if not np.isfinite(X[i]).all():  # remove after Ive checked all scans
                X[i] = np.zeros((X[i].shape))
        
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


def train_session(NAME, OUT, model, partition, labels, DATA_PATH, training_generator, validation_generator, Custom_History, my_model_checkpoint, EPOCHS = 100, BATCH_SIZE=4):

    # Some last check on data
    subjects_train = [x[:20] for x in training_generator.list_IDs]
    subjects_val = [x[:20] for x in validation_generator.list_IDs]
    assert len(set(subjects_train).intersection(set(subjects_val)))==0, 'subjects in both train and val!'
    
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
    

#%% LOAD PARTITIONS

partition, labels, PATHOLOGY_SPREADSHEET, clinical_features = load_main_spreadsheet_and_create_sub_spreadsheets(DATA_DESCRIPTION_PATH)

print('Train: {}\nVal: {}'.format(len(partition['train']), len(partition['validation'])))

#%% DEFINE TRAINING GENERATOR

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
          'data_description':PATHOLOGY_SPREADSHEET}

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
          'data_description':PATHOLOGY_SPREADSHEET}

if USE_SMALLER_DATASET:
    print('Using smaller dataset: fraction = {}'.format(FRACTION))
    partition['train'] = np.random.choice(partition['train'], replace=False, size=int(len(partition['train'])*FRACTION))

training_generator = DataGenerator_classifier(partition['train'],labels, **params_train)
validation_generator = DataGenerator_classifier(partition['validation'],labels, **params_val)


if CNN:
    model = UNet_v0_2D_Classifier(input_shape =  (512,512,CHANNELS), pool_size=(2, 2),initial_learning_rate=1e-5, 
                                     deconvolution=True, depth=depth, n_base_filters=n_base_filters,
                                     activation_name="softmax", L2=L2, USE_CLINICAL=USE_CLINICAL)
elif RESNET:
    model = tf.compat.v1.keras.applications.ResNet50(
        include_top=True,
        weights= None,
        input_tensor=None,
        input_shape=(512,512,3),
        pooling=max,
        classes=2)
    
    regularizer = tf.keras.regularizers.l2(1e-4)

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
              setattr(layer, attr, regularizer)
    
    model.compile(loss=FocalLoss, optimizer=tf.keras.optimizers.Adam(lr=1e-5), metrics=['acc'])

if LOAD_PRETRAINED_MODEL:
        
    model_weights = np.load(PRETRAINED_MODEL_WEIGHTS_PATH, allow_pickle=True)
   
    model.set_weights(model_weights)
    
    model.compile(loss=FocalLoss, optimizer=tf.keras.optimizers.Adam(lr=1e-5), metrics=['acc'])

    my_custom_checkpoint = my_model_checkpoint(MODEL_PATH=OUTPUT_PATH + NAME, MODEL_NAME='/best_model' )
    Custom_History = MyHistory(OUTPUT_PATH, NAME)

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
    
train_session(NAME, OUTPUT_PATH, model, partition, labels, DATA_PATH, training_generator, validation_generator, Custom_History, my_custom_checkpoint, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE)
    
