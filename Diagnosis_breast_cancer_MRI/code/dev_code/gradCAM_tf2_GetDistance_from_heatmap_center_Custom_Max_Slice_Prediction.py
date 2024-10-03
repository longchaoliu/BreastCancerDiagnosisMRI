#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:14:03 2022

@author: deeperthought
"""


"""
Title: Grad-CAM class activation visualization
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/04/26
Last modified: 2021/03/07
Description: How to obtain a class activation heatmap for an image classification model.
"""
"""
Adapted from Deep Learning with Python (2017).
## Setup
"""

import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[3], True)
  except RuntimeError as e:
    # Visible devices must be set at program startup
    print(e)
    
    
from tensorflow import keras

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import nibabel as nib
from skimage.transform import resize


#%%
import pandas as pd
SEGMENTED_PATH = '/home/deeperthought/kirbyPRO/Saggittal_segmentations_clean/numpy/'


SESSION = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/'

RESULTS_SHEET = SESSION + 'VAL_result.csv'


# PRETRAINED_MODEL_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/last_model.h5'
# last_conv_layer_name = 'activation_11'


PRETRAINED_MODEL_PATH = SESSION + 'last_model.h5'
last_conv_layer_name = 'activation_11'

clinical_features = pd.read_csv(SESSION + 'Clinical_Data_Train_Val.csv')
clinical_features_val = clinical_features


df = pd.read_csv(RESULTS_SHEET)


OUTPUT_PATH = SESSION

#%% Functions


def FocalLoss(y_true, y_pred): 
    #y_true = tf.keras.backend.expand_dims(y_true,0)
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    term_0 = (1 - y_true[:,1]) * tf.keras.backend.pow(y_pred[:,1],5) * tf.keras.backend.log(1 - y_pred[:,1] + tf.keras.backend.epsilon())  
    term_1 = y_true[:,1] * tf.keras.backend.pow(1 - y_pred[:,1],5) * tf.keras.backend.log(y_pred[:,1] + tf.keras.backend.epsilon())   
    return -tf.keras.backend.mean(term_0 + term_1, axis=0)

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array



def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None, normalize=True, pooling='mean'):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:   # Using Tape to record operations, so then gradients can be computed between sources and targets (prediction_class , last_conv_layer_output)
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)  # `class_channel` will be differentiated against elements in `last_conv_layer_output`

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    if pooling == 'mean':
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)) # leave channel dimension (grads = (1,16,16,252))
    if pooling == 'max':
        pooled_grads = tf.reduce_max(grads, axis=(0, 1, 2)) # leave channel dimension (grads = (1,16,16,252))
        
    #pooled_grads = tf.reduce_mean(grads, axis=(0))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0] # remove batch dimension
#    heatmap = tf.linalg.matmul(last_conv_layer_output , pooled_grads[..., tf.newaxis])
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap) 

    if normalize:
        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def unfreeze_layers(model):
    model_type = type(model) 
    for i in model.layers:
        i.trainable = True
        if type(i) == model_type:
            unfreeze_layers(i)
    return model


def load_model_frozen(PATH, custom_objects=''):
    #model.load_weights(FOLDER + NAME + '.h5')
    model_loaded = tf.keras.models.load_model(PATH, custom_objects=custom_objects)
    model = unfreeze_layers(model_loaded)
    adam = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=adam, loss=FocalLoss,  metrics=['accuracy'])  
    return model




def make_prediction_whole_scan(model, all_data, clinic_info_exam=[], USE_CONTRALATERAL=False, USE_PREVIOUS=False):
    
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
              
        if len(clinic_info_exam) > 0:
            yhat = model.predict([X,clinic_info_exam ])
        
        else:
            yhat = model.predict(X)
        slice_preds.append(yhat[0])
    return slice_preds


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

def load_data_prediction(scanID ):
    "Takes scanIDs (not paths) and loads raw MRIs from /home/deeperthought/kirby_MSK/alignedNii-Nov2019/, preprocesses and returns"
    exam = scanID[:-2]
    patient = scanID[:20]
    side = 'right'
    contra_side = 'left'
    if scanID[-1] == 'l': 
        side = 'left'
        contra_side = 'right'
    #pathology = labels[scanID]
    
    MRI_PATH = '/home/deeperthought/kirby_MSK/alignedNii-Nov2019/'
    #MRI_ALIGNED_HISTORY_PATH = '/home/deeperthought/kirbyPRO/saggital_Nov2019_alignedHistory/'
    
    # segmentation_GT = MASTER.loc[MASTER['Scan_ID'] == scanID, 'Segmentation_Path'].values[0]
    # contralateral_available = MASTER.loc[MASTER['Scan_ID'] == scanID, 'Contralateral Available'].values[0]
    # previous_available = MASTER.loc[MASTER['Scan_ID'] == scanID, 'Previous Available'].values[0]
        
    
    all_subject_channels = [MRI_PATH + exam + '/T1_{}_02_01.nii'.format(side),
                           MRI_PATH + exam + '/T1_{}_slope1.nii'.format(side),
                           MRI_PATH + exam + '/T1_{}_slope2.nii'.format(side)]

    T1_pre_nii_path = MRI_PATH + exam + '/T1_{}_01_01.nii'.format(side)

    t1post, slope1, slope2 = load_and_preprocess(all_subject_channels, T1_pre_nii_path)


       
    # if USE_CONTRALATERAL:
    #     t1post_contra, slope1_contra, slope2_contra = np.zeros(t1post.shape),np.zeros(t1post.shape)  ,np.zeros(t1post.shape)      

    #     if contralateral_available:
    #         all_contralateral_channels = [MRI_PATH + exam + '/T1_{}_02_01_contralateral_aligned.nii'.format(contra_side),
    #                                       MRI_PATH + exam + '/T1_{}_slope1_contralateral_aligned.nii'.format(contra_side),
    #                                       MRI_PATH + exam + '/T1_{}_slope2_contralateral_aligned.nii'.format(contra_side)]
            
    #         T1_pre_nii_path_contralateral = MRI_PATH + exam + '/T1_{}_01_01.nii'.format(contra_side)
    
    #         if not os.path.exists(all_contralateral_channels[0]):
    #             print('previous exam not aligned yet.. skip')
    #         else:
    #             t1post_contra, slope1_contra, slope2_contra = load_and_preprocess(all_contralateral_channels, T1_pre_nii_path_contralateral)


    # if USE_PREVIOUS:
    #     t1post_previous, slope1_previous, slope2_previous = np.zeros(t1post.shape),np.zeros(t1post.shape)  ,np.zeros(t1post.shape)      

    #     if previous_available:
    
    #         breast_history = MASTER.loc[(MASTER['DE-ID'] == patient) & (MASTER['Scan_ID'].str[-1] == scanID[-1]), 'Scan_ID'].values
    #         previous_exam = breast_history[np.argwhere(breast_history == scanID)[0][0] - 1][21:-2]        
            
    #         MRI_ALIGNED_HISTORY_PATH + patient + '/' + previous_exam
    #         all_previous_channels = [MRI_ALIGNED_HISTORY_PATH + patient + '/' + previous_exam+ '/T1_{}_02_01_TimeAlignment_to_{}.nii.gz'.format(side, exam[21:]),
    #                                  MRI_ALIGNED_HISTORY_PATH + patient + '/' + previous_exam+ '/T1_{}_slope1_TimeAlignment_to_{}.nii.gz'.format(side, exam[21:]),
    #                                  MRI_ALIGNED_HISTORY_PATH + patient + '/' + previous_exam+ '/T1_{}_slope2_TimeAlignment_to_{}.nii.gz'.format(side, exam[21:])]            
    
    #         T1_pre_nii_path_previous = MRI_PATH + patient + '_' + previous_exam + '/T1_{}_01_01.nii'.format(side)
        
    #         if not os.path.exists(all_previous_channels[0]):
    #             print('previous exam not aligned yet.. skip')
    #         else:
    #             t1post_previous, slope1_previous, slope2_previous = load_and_preprocess(all_previous_channels, T1_pre_nii_path_previous)

        
    # if not pd.isnull(segmentation_GT):
    #     groundtruth = nib.load(segmentation_GT).get_data()
    #     if np.sum(groundtruth) == 0:
    #         segmented_slice = 0            
    #     else:
    #         segmented_slice = list(set(np.where(groundtruth > 0)[0]))[0]
    # else:
    #     groundtruth = 0
    #     segmented_slice = 0
        
    # if not USE_CONTRALATERAL and not USE_PREVIOUS:
        
    return t1post, slope1, slope2

    # if not USE_CONTRALATERAL and USE_PREVIOUS:
        
    #     return t1post, slope1, slope2, t1post_previous, slope1_previous, slope2_previous    

    # if USE_CONTRALATERAL and not USE_PREVIOUS:

    #     return t1post, slope1, slope2, t1post_contra, slope1_contra, slope2_contra

    # if USE_CONTRALATERAL and USE_PREVIOUS:

    #     return t1post, slope1, slope2, t1post_contra, slope1_contra, slope2_contra, t1post_previous, slope1_previous, slope2_previous    


#%%
import os
PATH = '/home/deeperthought/kirbyPRO/Saggittal_Full_Slices/2D_slices/X/'


model = load_model_frozen(PRETRAINED_MODEL_PATH, custom_objects={'FocalLoss':FocalLoss})

#%%

SEGMENTED_PATH = '/home/deeperthought/kirbyPRO/Saggittal_segmentations_clean/2D/'

available_segmentations = os.listdir(SEGMENTED_PATH)

df = df.loc[df['y_true'] == 1]

scanID_malignants = df['scan'].values

Minimum_distance = dict(zip(scanID_malignants, [999]*len(scanID_malignants)))

for scanID in scanID_malignants:
    print(scanID)
    
        
    segmented_slice = [x for x in available_segmentations if x.startswith(scanID)]
    if len(segmented_slice) > 0:
        segmented_slice = segmented_slice[0]
    else:
        print('Scan not segmented!')
        
    segmentation = np.load(SEGMENTED_PATH + segmented_slice, allow_pickle=True)
    SEGMENTED_SLICE = int(segmented_slice.split('_')[-1].split('.npy')[0])
    all_data = load_data_prediction(scanID) 
    
            
    
    # clinic_info_exam = 0
    # if USE_CLINICAL:
    #     clinic_info_exam = clinical_info.loc[clinical_info['scan_ID'] == scan,['Family Hx', u'ETHNICITY', u'RACE', u'Age']].values
    
    
    clinical = clinical_features_val.loc[clinical_features_val['scan_ID'] == scanID,[u'Family Hx',u'Age',u'ETHNICITY_HISPANIC OR LATINO',u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',u'RACE_ASIAN-FAR EAST/INDIAN SUBCONT',u'RACE_BLACK OR AFRICAN AMERICAN',u'RACE_NATIVE AMERICAN-AM IND/ALASKA',u'RACE_NATIVE HAWAIIAN OR PACIFIC ISL',u'RACE_UNKNOWN',u'RACE_WHITE']].values
    #clinical = np.array([[0.,-1.,2.,0.4]])
    slice_preds = make_prediction_whole_scan(model, all_data, clinical)
        
    
    slice_preds_class1 = [x[1] for x in slice_preds]
    
    
    SLICE_NR = np.argmax(slice_preds_class1)
    
    # SLICE_NR = 0#15
    
    img_array = np.expand_dims(np.array([x[SLICE_NR] for x in all_data]), -1)
    
    img_array = np.swapaxes(img_array, 0,-1)
    
    
    try:
        yhat = model.predict(img_array)
    except:
        yhat = model.predict([img_array, clinical])
    
    img_tensor = tf.convert_to_tensor(img_array)
    
    clin_tensor = tf.convert_to_tensor(clinical)
    
    #model([img_tensor, clin_tensor])
    
    # Remove last layer's softmax
    #model.layers[-1].activation = None
    
    #last_conv_layer_name = 'activation_8'
    
    
    # Generate class activation heatmap
    
    
    #heatmap = make_gradcam_heatmap(img_tensor, model, last_conv_layer_name, pred_index=None, normalize=True, pooling='max')
    heatmap = make_gradcam_heatmap([img_tensor, clin_tensor], model, last_conv_layer_name, pred_index=1, normalize=True, pooling='max')
    
     # Remove boundary artifacts
    heatmap = np.pad(heatmap[2:-2,2:-2], pad_width=(4,4), mode='minimum') 
    
    # Shift to remove effect of unpadded maximum pooling in model
    heatmap = heatmap[3:,3:] 
    
    """
    ## Create a superimposed visualization
    
    """
    
    
    
    img = img_array[0]
    img = np.uint8(255 * img)
    img[:,:,1] = img[:,:,0]
    img[:,:,2] = img[:,:,0]
    
    
    heatmap += abs(np.min(heatmap))
    
    heatmap /= np.max(heatmap)
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    MAX_CENTER = np.argwhere(heatmap == heatmap.max())
    MAX_CENTER = np.array(MAX_CENTER*512./17, dtype=int)[0]
    
    # #########################################################
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")
    
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    jet_heatmap = resize(jet_heatmap, output_shape=(img.shape[1], img.shape[0]), anti_aliasing=True)
    jet_heatmap = jet_heatmap*255.
    
    # Create an image with RGB colorized heatmap
    # jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    # jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    # jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    
    #Save the superimposed image
    alpha = 0.05
    superimposed_array = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_array)
    
    fig, axes = plt.subplots(3,2, figsize=(8,14))
    axes[0][0].imshow(img_array[0,:,:,0], cmap='gray', aspect="auto"), axes[0][0].set_xticks([]) , axes[0][0].set_yticks([])
    axes[0][1].imshow(img_array[0,:,:,1], cmap='gray', aspect="auto"), axes[0][1].set_xticks([]) , axes[0][1].set_yticks([])
    #axes[0][2].imshow(img_array[0,:,:,2], cmap='gray', aspect="auto"), axes[0][2].set_xticks([]) , axes[0][2].set_yticks([])
    axes[1][0].plot(slice_preds_class1); axes[1][0].set_xlabel('Slice'); axes[1][0].set_ylabel('Pred');
    axes[1][0].vlines(SLICE_NR,np.min(slice_preds_class1),np.max(slice_preds_class1), color='r', alpha=0.5)
    axes[1][0].grid()
    axes[1][1].imshow(superimposed_img, aspect="auto"); axes[1][1].set_title('T1post + heatmap'); axes[1][1].set_xticks([]) , axes[1][1].set_yticks([])
    #axes[1][2].axis('off')
    plt.suptitle('{}\nSlice: {}; Model pred = {:.2f}'.format(scanID,SLICE_NR, yhat[0,-1]))
    
    
    DUMMY = np.zeros(img_array[0,:,:,0].shape)
    DUMMY[MAX_CENTER[0]-10:MAX_CENTER[0]+10, MAX_CENTER[1]-10:MAX_CENTER[1]+10] = 1
    #plt.imshow(img_array[0,:,:,0] + DUMMY, cmap='gray')
    
    axes[2][0].imshow(img_array[0,:,:,0] + DUMMY*0.08, cmap='gray')
    axes[2][0].set_title(SLICE_NR)
    
    axes[2][1].imshow(segmentation )
    axes[2][1].set_title(SEGMENTED_SLICE)
    
    
    #########################################################
    
    segmented_pixels = np.argwhere(segmentation > 0)
    
    distances_pixels = []
    for coord in segmented_pixels:
        distances_pixels.append(np.sqrt(3.*(SEGMENTED_SLICE - SLICE_NR)**2 + 0.3*(coord[0] - MAX_CENTER[0])**2 + 0.3*(coord[1] - MAX_CENTER[1])**2))
    
    MINIMUM_DISTANCE_FROM_CENTER_TO_LESION_MILLIMETERS= np.min(distances_pixels)

    Minimum_distance[scanID] = MINIMUM_DISTANCE_FROM_CENTER_TO_LESION_MILLIMETERS
    print(Minimum_distance[scanID])
    
    plt.savefig(OUTPUT_PATH + '/gradCAM_images/' + scanID + '_' + str(np.round(MINIMUM_DISTANCE_FROM_CENTER_TO_LESION_MILLIMETERS,3)) + '.png', dpi=400)
    plt.close()
    
MEDIAN = np.median(list(Minimum_distance.values()))

plt.figure(figsize=(4,4))
plt.hist(Minimum_distance.values(), bins=50)
plt.xlabel('Distance [mm]')
plt.title('Center of heatmap to nearest segmented pixel')

plt.vlines(MEDIAN, 0,30, color='k', linestyle='--')
plt.savefig(OUTPUT_PATH + 'Heatmap_distance_nearest_segmentation.png', dpi=400)


dfdf = pd.DataFrame.from_dict(Minimum_distance,orient='index', columns=['min_d'])

dfdf.to_csv(OUTPUT_PATH + 'Heatmap_distance_nearest_segmentation.csv')
