#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 18:24:04 2023

@author: deeperthought
"""


'''
1 - Script to create model architecture and load weights
(save weights first , to make tensroflow version agnostic script) DONE

2 - Remove kirby entries in scripts DONE

3 - add dummy versions of needed inputs (fake data, but names that match the script input)

'''


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
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.transform import resize

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

import pandas as pd
# from numpy.random import seed
# seed(42)
# from tensorflow import set_random_seed
# set_random_seed(42)

#%% USER INPUT

# MODEL_WEIGHTS_PATH = '/Path/to/Duke_data_Demo/weights.npy'

DATA_PATH = '/home/deeperthought/kirby_MSK/dukePublicData/alignedNii-normed/'      

# DATA_PATH = '/home/deeperthought/kirby_MSK/dukePublicData/alignedNii/'      


MODEL_WEIGHTS_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/best_model_weights.npy'
    

labels_df = pd.read_csv('/home/deeperthought/kirby_MSK/dukePublicData/25segDukePublic.csv')


labels_df = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Duke_predictions/Annotation_Boxes.csv')

#%% Functions

def create_convolution_block(input_layer, n_filters, kernel=(3, 3), padding='same', strides=(1, 1), L2=0):

    layer = Conv2D(n_filters, kernel, padding=padding, strides=strides, kernel_regularizer=regularizers.l2(L2))(input_layer)
    layer = BatchNormalization()(layer)

    return Activation('relu')(layer)

def FocalLoss(y_true, y_pred): 
    #y_true = tf.keras.backend.expand_dims(y_true,0)
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    term_0 = (1 - y_true[:,1]) * tf.keras.backend.pow(y_pred[:,1],5) * tf.keras.backend.log(1 - y_pred[:,1] + tf.keras.backend.epsilon())  
    term_1 = y_true[:,1] * tf.keras.backend.pow(1 - y_pred[:,1],5) * tf.keras.backend.log(y_pred[:,1] + tf.keras.backend.epsilon())   
    return -tf.keras.backend.mean(term_0 + term_1, axis=0)


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
    
#%% Create model and load pre-trained weights
                                  
model = UNet_v0_2D_Classifier(input_shape =  (512,512,3), pool_size=(2, 2),initial_learning_rate=1e-5, 
                                         deconvolution=True, depth=6, n_base_filters=42,
                                         activation_name="softmax", L2=1e-5, USE_CLINICAL=True)


loaded_weights = np.load(MODEL_WEIGHTS_PATH, allow_pickle=True, encoding='latin1')

model.set_weights(loaded_weights)

import sys
sys.path.append('/home/deeperthought/Projects/MultiPriors_MSKCC/scripts/')
from MultiPriors_Models_Collection import Generalised_dice_coef_multilabel2, dice_coef_multilabel_bin0,dice_coef_multilabel_bin1

my_custom_objects = {'Generalised_dice_coef_multilabel2':Generalised_dice_coef_multilabel2,
                                 'dice_coef_multilabel_bin0':dice_coef_multilabel_bin0,
                                 'dice_coef_multilabel_bin1':dice_coef_multilabel_bin1}

breastMask_model = tf.keras.models.load_model('/home/deeperthought/Projects/BreastMask_model/UNet_v0_BreastMask_breastMask_UNet_v0_BreastMask_2019-12-23_1658/models/breast_mask_model.h5', custom_objects = my_custom_objects)

breastMask_model.input

#%% breast mask functions:
    
    

def get_Coordinates_from_target_patch(img_shape,D1,D2,D3, output_dpatch) :

    x_ = range(D1-(output_dpatch[0]//2),D1+((output_dpatch[0]//2)+output_dpatch[0]%2))
    y_ = range(D2-(output_dpatch[1]//2),D2+((output_dpatch[1]//2)+output_dpatch[1]%2))
    z_ = range(D3-(output_dpatch[2]//2),D3+((output_dpatch[2]//2)+output_dpatch[2]%2))
    
    x_norm = np.array(x_)/float(img_shape[0])  
    y_norm = np.array(y_)/float(img_shape[1])  
    z_norm = np.array(z_)/float(img_shape[2])  
    
    x, y, z = np.meshgrid(x_norm, y_norm, z_norm, indexing='ij')    
    coords = np.stack([x,y,z], axis=-1)
    return coords
   
def extractCoordinates(shapes, voxelCoordinates, output_dpatch):
    """ Given a list of voxel coordinates, it returns the absolute location coordinates for a given patch size (output 1x9x9) """

    all_coordinates = []
    for i in np.arange(len(shapes)):
        #subject = str(subjects[i])[:-1]
        #img = nib.load(subject)
        img_shape = shapes[i]
        for j in np.arange(len(voxelCoordinates[i])):     
            D1,D2,D3 = voxelCoordinates[i][j]
            #all_coordinates.append(get_Coordinates_from_target_patch(img.shape,D1,D2,D3))                 
            all_coordinates.append(get_Coordinates_from_target_patch(img_shape,D1,D2,D3, output_dpatch))                    

        #img.uncache()
    return np.array(all_coordinates)    


def get_breast_mask(X, breastMask_model):

    X = X/float(np.percentile(X, 95))
    data = resize(X, output_shape=(X.shape[0],256,256), preserve_range=True, anti_aliasing=True, mode='reflect')   
    
    breaskMask = np.zeros(data.shape)
    for SLICE in range(1,data.shape[0]-1):   
        X_slice = data[SLICE-1:SLICE+2]
        voxelCoordinates = [[[SLICE,128,128]]]
        shapes = [data.shape]
        coords = extractCoordinates(shapes, voxelCoordinates, output_dpatch=[1,256,256])
        
        X_slice = np.expand_dims(X_slice,-1)
        X_slice = np.expand_dims(X_slice,0)
        
        y_pred = breastMask_model.predict([X_slice, coords])
        
        breaskMask[SLICE] = y_pred[0,0,:,:,1]
        
    return breaskMask
    


#%%


SUBJECTS = list(set(labels_df['Patient ID'].values))

SUBJECTS.sort()



for subj in SUBJECTS:
    print(f'\n ########### {subj} ############# \n')
    # subj = 'Breast_MRI_010'
    
    
    # SELECTED_SLICE = labels_df.loc[labels_df['ID'] == subj, 'sliceNumber'].values[0]
    
    DATA = [DATA_PATH + subj + '/T1_axial_01.nii.gz', 
            DATA_PATH + subj + '/T1_axial_02.nii.gz',
            DATA_PATH + subj + '/T1_axial_slope1.nii.gz',
            DATA_PATH + subj + '/T1_axial_slope2.nii.gz']
    
    all_subject_channels = DATA[1:]
    
    T1_pre_nii_path = DATA[0]
    
    if os.path.exists(T1_pre_nii_path):
        t1pre = nib.load(T1_pre_nii_path).get_data()
        p95 = np.percentile(t1pre,95)
    else:
        t1pre = 0
    
    hdr = nib.load(all_subject_channels[0])
    
    pixdim1 = hdr.header['pixdim'][1]
    pixdim2 = hdr.header['pixdim'][2]
 
    t1post_normp95 = nib.load(all_subject_channels[0]).get_fdata()
    slope1_normp95 = nib.load(all_subject_channels[1]).get_fdata()
    slope2_normp95 = nib.load(all_subject_channels[2]).get_fdata()    
    
    
    
    
    # t1post_normp95 = np.array(t1post)/p95    
    # slope1_normp95 =  np.array(slope1)/p95    
    # slope2_normp95 =  np.array(slope2)/p95    

    # t1post = np.array(t1post_normp95)/float(25.17)
    # slope1 = np.array(slope1_normp95)/float(0.18)
    # slope2 = np.array(slope2_normp95)/float(0.05)

    t1post = np.array(t1post_normp95)/float(40)
    slope1 = np.array(slope1_normp95)/float(0.3)
    slope2 = np.array(slope2_normp95)/float(0.12)
    
    # if 'axial_02' in subject:
    #     img_data = img_data/25.17
    # elif 'slope1' in subject:
    #     img_data = img_data/0.18
    # elif 'slope2' in subject:
    #     img_data = img_data/0.05
    dimensions = t1post.shape

    # original shape = 448, 448, 160. 
    # model needs sagittal slices with shape 512, 512. So target shape is: (N_slices, 512, 512)
    

    #resolution = hdr.header['pixdim'][:3]

    resolution = np.diag(hdr.affine)[:3]
    
    # projection_1d = np.sum(t1post, (0,2))
    # background_value = np.median(projection_1d[-20:])
    # breast_end = np.argwhere(projection_1d> background_value*5)[-1][0]
    # breast_end = breast_end + 10 # add some border
    # breast_end = np.max([breast_end, 256]) # if breast is small, just crop to 256

    # plt.imshow(t1post[:,:,100])
    
    
    # projection_1d =  np.sum(t1post[:,:,t1post.shape[-1]//2], 0)[::5]    
    # breast_end = np.argmin(np.diff(projection_1d[-len(projection_1d)//2:]))*5
    # breast_end = breast_end + 10 # pad a bit
    
    # projection_1d_chest = np.max(t1post[t1post.shape[0]//2,:,:],1)
    # breast_start = np.argmin(np.diff(projection_1d_chest[::5]))*5
    # breast_start = breast_start - 50 # pad a bit
    
    projection_1d_chest = np.max(t1post[t1post.shape[0]//2,:,:],1)
    background_values = np.median(projection_1d_chest[:-50])
    max_value_chest = np.max(projection_1d_chest)
    breast_start = np.argwhere(projection_1d_chest > background_values*2)[-1][0]
    breast_start = breast_start - 50 # pad a bit

    # plt.imshow(t1post[t1post.shape[0]//2,:,:])
    # plt.plot(projection_1d_chest)

    #assert breast_start < breast_end   
        
    t1post = t1post[:, breast_start:breast_start+256, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    slope1 = slope1[:, breast_start:breast_start+256, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    slope2 = slope2[:, breast_start:breast_start+256, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    
    
    breastmask = get_breast_mask(t1post, breastMask_model)
    breastmask = np.array(breastmask > 0.1, dtype='int8')


    breastmass = np.sum(breastmask, (1,2))
    slices_on_breast = breastmass > np.percentile(breastmass, 30)

    # plt.imshow(t1post[:,:,100])
    # plt.imshow(breastmask[:,:,100]>0.1)
    # plt.plot(breastmass)
    # plt.plot(slices_on_breast)
        
    # t1post = t1post[:, (breast_end-256):breast_end, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    # # t1pre = t1pre[:, (breast_end-256):breast_end, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    # slope1 = slope1[:, (breast_end-256):breast_end, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    # slope2 = slope2[:, (breast_end-256):breast_end, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    
    resolution = np.diag(hdr.affine)
    
    resolution_factor_X = resolution[1]/0.5
    resolution_factor_Y = resolution[2]/0.5
    
    output_res =  (resolution[0], resolution[1]/resolution_factor_X, resolution[2]/resolution_factor_Y)  #'THIS SEEMS CORRECT THOUGH???'
    output_shape = (t1post.shape[0], int(t1post.shape[1]*resolution_factor_X), int(t1post.shape[2]*resolution_factor_Y))

    t1post = resize(t1post, output_shape=output_shape, preserve_range=True, anti_aliasing=False)
    slope1 = resize(slope1, output_shape=output_shape, preserve_range=True, anti_aliasing=False)
    slope2 = resize(slope2, output_shape=output_shape, preserve_range=True, anti_aliasing=False)

    
    # factor_x = resolution[1]/0.33
    # factor_y = resolution[2]/0.33
    
    # # if resolution[1] > 0.4:
    # output_shape = (t1post.shape[0], int(t1post.shape[1]*factor_x), int(t1post.shape[2]*factor_y ))
    # output_resolution = (resolution[0], resolution[1]/factor_x, resolution[2]/factor_y)
        
    # # if resolution[1] > 0.5:
    # #     output_shape = (t1post.shape[0]*2, t1post.shape[1]*2, int(t1post.shape[2]* (resolution[2]/0.33) ))
    # #     output_resolution = (resolution[0]/2, resolution[1]/2, int(resolution[2]/ (resolution[2]*0.33) ))
        
        
        
        
    # #        output_shape = (512*2, 512*2, int(192*3.3))
    
    
    # # RESIZE to match resolutions.  I need final resolution: (whatever, 0.3, 0.3)
    # t1post = resize(t1post, output_shape=output_shape, preserve_range=True, anti_aliasing=True, order=2)
    # slope1 = resize(slope1, output_shape=output_shape, preserve_range=True, anti_aliasing=True, order=2)
    # slope2 = resize(slope2, output_shape=output_shape, preserve_range=True, anti_aliasing=True, order=2)
    # t1pre = resize(t1pre, output_shape=output_shape, preserve_range=True, anti_aliasing=False)
    

    if t1post.shape[1] < 512:
        border = 512 - t1post.shape[1]
        t1post = np.pad(t1post, ((0,0),(0,border),(0,0)), 'minimum')
        slope1 = np.pad(slope1, ((0,0),(0,border),(0,0)), 'constant')
        slope2 = np.pad(slope2, ((0,0),(0,border),(0,0)), 'constant')
        
    else:    
        t1post = t1post[:, 0:512, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
        slope1 = slope1[:, 0:512, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
        slope2 = slope2[:, 0:512, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast    
    
    
    if t1post.shape[2] < 512:
        border = 512 - t1post.shape[2]
        t1post = np.pad(t1post, ((0,0),(0,0),(0,border)), 'minimum')
        slope1 = np.pad(slope1, ((0,0),(0,0),(0,border)), 'constant')
        slope2 = np.pad(slope2, ((0,0),(0,0),(0,border)), 'constant')
        # t1pre = np.pad(t1pre, ((0,0),(0,0),(0,border)), 'minimum')
        
    else:    
        t1post = t1post[:, :, 0:512]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
        slope1 = slope1[:, :, 0:512]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
        slope2 = slope2[:, :, 0:512]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    
    
    X = np.stack([t1post, slope1, slope2], axis=-1)
        
    #For unavailable clinical information
    MODE_CLINICAL = np.array([[0.  , 0.51, 0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  ]])
    
    
    
    plt.imshow(X[70,:,:,0], cmap='gray')
    plt.imshow(X[:,:,70,0], cmap='gray')

    np.min(X[:,:,:,0])
    
    ################## BOTH #######################################################
    preds = []
    for i in range(1,X.shape[0]):
        #print(i)
        pred = model.predict([X[i-1:i], MODE_CLINICAL])
        preds.append(pred[0,1])
        
    preds_gated = preds*slices_on_breast[1:]
    global_prediction = np.max(preds_gated)
    max_slice = np.argwhere(preds == global_prediction)[0][0]
    
    # DISPLAY
    axial_projection_t1post = np.max(t1post,-1)
    axial_projection_slope1 = np.max(slope1,-1)
    
    t1post_original = nib.load(all_subject_channels[0]).get_fdata()
    # slope1_original = nib.load(all_subject_channels[1]).get_fdata()

    
    labels_df.loc[labels_df['Patient ID'] == subj, 'DIM X'] = dimensions[0]
    labels_df.loc[labels_df['Patient ID'] == subj, 'DIM Y'] = dimensions[1]
    labels_df.loc[labels_df['Patient ID'] == subj, 'DIM Z'] = dimensions[2]

    labels_df.loc[labels_df['Patient ID'] == subj, 'X1'] = dimensions[0] - labels_df.loc[labels_df['Patient ID'] == subj, 'End Column']
    labels_df.loc[labels_df['Patient ID'] == subj, 'X2'] = dimensions[0] - labels_df.loc[labels_df['Patient ID'] == subj, 'Start Column']

    labels_df.loc[labels_df['Patient ID'] == subj, 'Y1'] = dimensions[1] - labels_df.loc[labels_df['Patient ID'] == subj, 'End Row']
    labels_df.loc[labels_df['Patient ID'] == subj, 'Y2'] = dimensions[1] - labels_df.loc[labels_df['Patient ID'] == subj, 'Start Row']

    labels_df.loc[labels_df['Patient ID'] == subj, 'Z1'] = dimensions[2] - labels_df.loc[labels_df['Patient ID'] == subj, 'End Slice']
    labels_df.loc[labels_df['Patient ID'] == subj, 'Z2'] = dimensions[2] - labels_df.loc[labels_df['Patient ID'] == subj, 'Start Slice']
        
    MIDDLE_SLICE = int(0.5*(labels_df.loc[labels_df['Patient ID'] == subj, 'X2'].values[0] + labels_df.loc[labels_df['Patient ID'] == subj, 'X1'].values[0]))
    X1 = labels_df.loc[labels_df['Patient ID'] == subj, 'X1'].values[0] 
    X2 = labels_df.loc[labels_df['Patient ID'] == subj, 'X2'].values[0] 
    
    
    if (X2 > max_slice) and (max_slice > X1):
        OUTCOME = 'HIT'
        COLOR='r'
    else:
        OUTCOME = 'MISS'
        COLOR='y'
        
    plt.figure(1)
    fig, ax = plt.subplots(2,2, sharex=True, figsize=(15,7))    
    # plt.subplots_adjust(hspace=.0)
    
    
    ax[0][0].plot(preds)
    ax[0][0].plot(slices_on_breast*global_prediction)
    ax[0][0].set_aspect('auto')
    
    ax[1][0].imshow(np.rot90(axial_projection_t1post), cmap='gray' , vmax=np.percentile(axial_projection_t1post,99.9))
    ax[1][0].set_aspect('auto'); ax[1][0].set_xticks([]); ax[1][0].set_yticks([])
    ax[1][0].set_xlabel(OUTCOME)
    ax[1][0].vlines(X1,0,512, linestyle='--', color='dodgerblue')
    ax[1][0].vlines(X2,0,512, linestyle='--', color='dodgerblue')
    ax[1][0].vlines(max_slice,0,512,color=COLOR)
    #ax[2][0].imshow(np.rot90(axial_projection_slope1), cmap='gray' , vmax=np.percentile(axial_projection_slope1,99.9))
    #ax[2][0].set_aspect('auto'); ax[2].set_xticks([]); ax[2].set_yticks([])
   
    ax[0][1].imshow(np.rot90(t1post[max_slice]), cmap='gray' )
    ax[0][1].set_xlabel(f'Predicted slice: {max_slice}'); ax[0][1].set_xticks([]); ax[0][1].set_yticks([])
    ax[0][1].set_aspect('auto')

    
    ax[1][1].imshow(np.rot90(t1post_original[MIDDLE_SLICE,:,:]), cmap='gray', vmax=np.percentile(t1post_original, 99.9))
    ax[1][1].set_xlabel(f'GT Sagittal slices: {X1} - {X2}'); ax[1][1].set_xticks([]); ax[1][1].set_yticks([])
    ax[1][1].set_aspect('auto')

    plt.savefig('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Duke_predictions/predictions/' + subj + '.png', dpi=400)
    plt.close()
