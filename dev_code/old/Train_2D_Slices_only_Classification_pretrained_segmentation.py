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


def load_data_prediction(scanID ,labels, MASTER):
    "Takes scanIDs (not paths) and loads raw MRIs from /home/deeperthought/kirby_MSK/alignedNii-Nov2019/, preprocesses and returns"
    exam = scanID[:-2]
    side = 'right'
    if scanID[-1] == 'l': side = 'left'
    
    pathology = labels[scanID]
    
    segmentation_GT = MASTER.loc[MASTER['ScanID'] == scanID, 'Segmentation_Path'].values[0]
        
    all_subject_channels = ['/home/deeperthought/kirby_MSK/alignedNii-Nov2019/' + exam + '/T1_{}_02_01.nii'.format(side),
                            '/home/deeperthought/kirby_MSK/alignedNii-Nov2019/' + exam + '/T1_{}_slope1.nii'.format(side),
                            '/home/deeperthought/kirby_MSK/alignedNii-Nov2019/' + exam + '/T1_{}_slope2.nii'.format(side)]
    T1_pre_nii_path = '/home/deeperthought/kirby_MSK/alignedNii-Nov2019/' + exam + '/T1_{}_01_01.nii'.format(side)
     
    t1post = nib.load(all_subject_channels[0]).get_data()
    slope1 = nib.load(all_subject_channels[1]).get_data()
    slope2 = nib.load(all_subject_channels[2]).get_data()    
    p95 = np.percentile(nib.load(T1_pre_nii_path).get_data(),95)
        
    if np.isfinite(segmentation_GT):
        groundtruth = nib.load(segmentation_GT).get_data()
        if np.sum(groundtruth) == 0:
            segmented_slice = 0            
        else:
            segmented_slice = list(set(np.where(groundtruth > 0)[0]))[0]
    else:
        groundtruth = 0
        segmented_slice = 0
        
    if t1post.shape[1] != 512:
        output_shape = (t1post.shape[0],512,512)
        t1post = resize(t1post, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
        slope1 = resize(slope1, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
        slope2 = resize(slope2, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
        if pathology == 1:   
            if segmented_slice > 0:
                groundtruth = resize(groundtruth, order=0, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
             
    t1post = t1post/p95    
    slope1 = slope1/p95    
    slope2 = slope2/p95    
    
    t1post = t1post/float(40)
    slope1 = slope1/float(0.3)
    slope2 = slope2/float(0.12)    

    return t1post, slope1, slope2, pathology, segmented_slice        


def make_prediction_whole_scan(model, t1post, slope1, slope2):
    
    slice_preds = []
    for i in range(t1post.shape[0]):
        X = np.expand_dims(np.stack([t1post[i],slope1[i],slope2[i]],-1),0)
        yhat = model.predict(X)
        slice_preds.append(yhat[0,1])
    return slice_preds


def get_results_on_dataset(model, scans_list, labels, Data_description, NAME, OUT):
    
    result = pd.DataFrame(columns=['scan','y_pred','y_true','max_slice','GT_slice'])
    name = 'VAL'
    N = 0
    TOT = len(scans_list)
    for scan in scans_list:
        scan = scan[:31]
        N += 1
        print('{}/{}'.format(N,TOT))
                
        t1post, slope1, slope2, pathology, segmented_slice  = load_data_prediction(scan,labels, Data_description) 
        
        slice_preds = make_prediction_whole_scan(model, t1post, slope1, slope2)
        
        result = result.append({'scan':scan,'y_pred':np.max(slice_preds), 'y_true':pathology, 
                                'max_slice':np.argmax(slice_preds), 'GT_slice':segmented_slice}, 
                ignore_index=True)

        print('{} - PATHOLOGY: {}\nsegmented_slice: {}, max_slice: {} = {}:'.format(scan, pathology, segmented_slice, np.argmax(slice_preds), np.max(slice_preds)))
 
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
    def __init__(self, list_IDs,labels,clinical_info, data_path='', batch_size=4, dim=(512,512), n_channels=3,
                 n_classes=2, shuffledata=True, do_augmentation=True, use_clinical_info=False):
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
        
#        if self.do_augmentation:
#            for i in range(self.batch_size):
#                X[i] = X[i]*np.random.uniform(low=0.8, high=1.2, size=1)

        if self.use_clinical_info:
            clinic = np.zeros((self.batch_size,4))
            scanids = [ids[:31] for ids in list_IDs_temp]
            for i in range(len(scanids)):
                clinic[i] = clinical_info.loc[clinical_info['scan_ID'] == scanids[i],['Family Hx', u'ETHNICITY', u'RACE', u'Age']].values

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
        X = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        y = np.empty((self.batch_size, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):          
            X[i] = np.load(self.data_path + ID)   # Here we add the path. ID can be the path
            y[i] = labels[ID[:31]]

#            if not np.isfinite(X[i]).all():  # remove after Ive checked all scans
#                X[i] = np.zeros((X[i].shape))
#            
            
        if self.do_augmentation:
            X_gen = self.augmentor.flow(X,y, batch_size=self.batch_size, shuffle=False, seed=self.seed)

            return next(X_gen)
        else:
            return X,y
    
#%% Test generator
#DATA0_PATH = '/home/deeperthought/kirbyPRO/Segmenter_v0_Full_Slice/Benigns_Train/X_slices/'
#DATA1_PATH = '/home/deeperthought/kirbyPRO/Segmenter_v0_Full_Slice/Malignants_Train/X_slices/'
#
#scans0 = [DATA0_PATH + x for x in os.listdir(DATA0_PATH)]
#scans1 = [DATA1_PATH + x for x in os.listdir(DATA1_PATH)]
#
#scans0
#
#labels = {}
#for scan in scans0:
#    labels[scan] = 0
#for scan in scans1:
#    labels[scan] = 1
#
#partition = {'train':scans0 + scans1}
#
#params_train = {'dim': (512,512),
#          'batch_size': 4,
#          'n_classes': 2,
#          'n_channels': 3,
#          'shuffledata': True,
#          'do_augmentation':True,
#          'clinical_info':clinical_info,
#          'use_clinical_info':True}
#
## Generators
#training_generator = DataGenerator_classifier(partition['train'],labels, **params_train)
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
    def __init__(self, OUT, NAME):
        self.OUT = OUT
        self.NAME = NAME        
    def on_train_begin(self, logs={}):

        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []

#    def on_batch_end(self, batch, logs={}):
#        self.loss.append(logs.get('loss'))
#        self.dice_coef_multilabel_bin1.append(logs.get('dice_coef_multilabel_bin1'))
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
            




def train_session(NAME, OUT, partition, DATA_PATH, model, training_generator, validation_generator, EPOCHS = 100, DATA_AUGMENTATION=True, BATCH_SIZE=4, CLASS_WEIGHT=1):
    
    if not os.path.exists(OUT+NAME):
        os.makedirs(OUT+NAME)
    
    Custom_History = MyHistory(OUT, NAME)
    
    csv_logger = tf.keras.callbacks.CSVLogger(OUT+NAME + '/csvLogger.log', 
                                         separator=',', 
                                         append=True)
    
    my_custom_checkpoint = my_model_checkpoint(MODEL_PATH=OUT+NAME, MODEL_NAME='/best_model')

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
                                steps_per_epoch = len(partition['train']) // BATCH_SIZE,
                                epochs = EPOCHS,
                                shuffle=True,
                                max_queue_size=64,
                                class_weight = {0:1., 1:CLASS_WEIGHT},
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
    

#%%  TRAINING SESSIONS

clinical_info = pd.read_csv('/home/deeperthought/kirbyPRO/Saccade_Features_SGMT/Clinical_features/Clinical_features_July2022.csv')

df = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Data_Partitions.csv')

labels = dict(zip(df['Scan_ID'],(df['Pathology']=='Malignant').astype(int)))

DATA_PATH = '/home/deeperthought/kirbyPRO/Saggittal_Full_Slices/2D_slices/X/'
OUTPUT_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/' 

partition = {'train':[],'validation':[]}
partition['train'] = os.listdir(DATA_PATH)

len(partition['train'])

#%% QA
nans = ['MSKCC_16-328_1_13656_20090914_l_19.npy',
        'MSKCC_16-328_1_12664_20081103_l_26.npy',
        'MSKCC_16-328_1_12664_20081103_l_32.npy']

for i in range(len(nans)):
    if nans[i] in partition['train']:
        partition['train'].remove(nans[i]) # Has NaNs in slope2 

#%%
#scanids_set = [x.split('/')[-1].split('.')[0][:31] for x in partition['train']]
#SCANS_MISSING_CLINICAL_INFO = list(set(scanids_set) - set(clinical_info['scan_ID'].values))
#print('Removing {} scans that have no clinical info:'.format(len(SCANS_MISSING_CLINICAL_INFO)))
#for i in range(len(SCANS_MISSING_CLINICAL_INFO)):
#    print(i)
#    scan_slices = [x for x in partition['train'] if x.startswith(SCANS_MISSING_CLINICAL_INFO[i])]
#    for scan_slice in scan_slices:
#        partition['train'].remove(scan_slice)
#
#scanids_set = [x.split('/')[-1].split('.')[0][:31] for x in partition['train']]
#assert len(set(scanids_set) - set(clinical_info['scan_ID'].values)) == 0


#%%


TEST_SET = pd.read_csv('/home/deeperthought/Projects/DGNS/Data_partitions_global/July2022/Test.csv')
VAL_FOLD = 1   
VAL_SET = TEST_SET.loc[TEST_SET['Fold'] == VAL_FOLD] 

assert len(set([ x[:31] for x in partition['train'] if x[:31] in VAL_SET['ScanID'].values])) == len(VAL_SET['ScanID']), 'Validation set incomplete!'

partition['validation'] = [ x for x in partition['train'] if x[:31] in VAL_SET['ScanID'].values]

partition['train'] = [x for x in partition['train'] if x not in partition['validation']]

subjects_train = [x[:20] for x in partition['train']]
subjects_val = [x[:20] for x in partition['validation']]

assert len(set(subjects_train).intersection(set(subjects_val)))==0, 'subjects in both train and val!'

##%%
scans_train = list(set([x[:31] for x in partition['train']]))

TOTAL_TRAIN = len(scans_train)
MAL_TRAIN = sum([labels[x] for x in scans_train])
CLASS_WEIGHT = TOTAL_TRAIN/float(MAL_TRAIN)


seg_paths = df[['Scan_ID','Segmentation_Path']]
seg_paths.columns = ['ScanID', 'Segmentation_Path']

VAL_SET = pd.merge(VAL_SET, seg_paths, on='ScanID')

#%%
#
#params_train = {'dim': (512,512),
#          'data_path': DATA_PATH,
#          'batch_size': 8,
#          'n_classes': 2,
#          'n_channels': 3,
#          'shuffledata': True,
#          'do_augmentation':True,
#          'clinical_info':clinical_info,
#          'use_clinical_info':False}
#
#params_val = {'dim': (512,512),
#          'data_path': DATA_PATH,
#          'batch_size': 8,
#          'n_classes': 2,
#          'n_channels': 3,
#          'shuffledata': False,
#          'do_augmentation':False,
#          'clinical_info':clinical_info,
#          'use_clinical_info':False}
#    
#NAME = 'U-Net_classifier_dataAug_train9151_val1646_classWeights'
##train_session(NAME, OUTPUT_PATH, partition, DATA_PATH, params_train, params_val, depth=6, n_base_filters=42, 
##              activation_name="softmax", L2=0, EPOCHS=100, DATA_AUGMENTATION=True, BATCH_SIZE=8)
#best_model = load_model_frozen(OUTPUT_PATH + NAME + '/best_model.h5')
#
#scans_validation = list(set([x[:31] for x in partition['validation']]))
#
#get_results_on_dataset(best_model, scans_validation, labels,  VAL_SET, NAME, OUTPUT_PATH)
#
#

#%%

params_train = {'dim': (512,512),
          'data_path': DATA_PATH,
          'batch_size': 8,
          'n_classes': 2,
          'n_channels': 3,
          'shuffledata': True,
          'do_augmentation':True,
          'clinical_info':clinical_info,
          'use_clinical_info':False}

params_val = {'dim': (512,512),
          'data_path': DATA_PATH,
          'batch_size': 8,
          'n_classes': 2,
          'n_channels': 3,
          'shuffledata': False,
          'do_augmentation':False,
          'clinical_info':clinical_info,
          'use_clinical_info':False}
# Generators
training_generator = DataGenerator_classifier(partition['train'],labels, **params_train)
validation_generator = DataGenerator_classifier(partition['validation'],labels, **params_val)
    
depth=6
n_base_filters=16
activation_name="softmax"
L2=0
model = UNet_v0_2D_Classifier(input_shape =  (512,512,3), pool_size=(2, 2),initial_learning_rate=1e-5, 
                                     deconvolution=True, depth=depth, n_base_filters=n_base_filters,
                                     activation_name="softmax", L2=L2, USE_CLINICAL=False)
model.summary()

PRETRAINED_MODEL = '/home/deeperthought/Projects/DGNS/Detection_model/Sessions/2D_Segmenter_original_partitions_afterQA/best_model.h5'
my_custom_objects = {'Generalised_dice_coef_multilabel2':Generalised_dice_coef_multilabel2,
                                 'dice_coef_multilabel_bin0':dice_coef_multilabel_bin0,
                                 'dice_coef_multilabel_bin1':dice_coef_multilabel_bin1}

segmenter_model = tf.keras.models.load_model(PRETRAINED_MODEL, custom_objects = my_custom_objects)


for i in range(len(model.layers[:-4])):
    print(model.layers[i], segmenter_model.layers[i])
    w = segmenter_model.layers[i].get_weights()
    model.layers[i].set_weights(w)
    model.layers[i].trainable = False   # SHOULD I DO THIS??
    
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-5), metrics=['acc'])

model.summary()

NAME = 'U-Net_classifier_pretrained_frozen'


train_session(NAME, OUTPUT_PATH, partition, DATA_PATH, model, training_generator, validation_generator, EPOCHS=100, DATA_AUGMENTATION=True, BATCH_SIZE=8, CLASS_WEIGHT=CLASS_WEIGHT)
best_model = load_model_frozen(OUTPUT_PATH + NAME + '/best_model.h5')
scans_validation = list(set([x[:31] for x in partition['validation']]))

get_results_on_dataset(best_model, scans_validation, labels,  VAL_SET, NAME, OUTPUT_PATH)



    