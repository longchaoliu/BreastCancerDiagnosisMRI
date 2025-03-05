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
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[1], True)
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
import pandas as pd
import os


#%% Functions


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


def load_model_frozen(PATH):
    #model.load_weights(FOLDER + NAME + '.h5')
    model_loaded = tf.keras.models.load_model(PATH)
    model = unfreeze_layers(model_loaded)
    adam = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=adam, loss='binary_crossentropy',  metrics=['accuracy'])  
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

#%%

PATH = '/home/deeperthought/kirbyPRO/Saggittal_Full_Slices/2D_slices/X/'

PRETRAINED_MODEL_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/RandomSlices_DataAug__classifier_train38530_val4103_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/best_model.h5'
last_conv_layer_name = 'activation_11'


model = load_model_frozen(PRETRAINED_MODEL_PATH)

#%% First load whole MRI, make prediction for all slices, pick max.

MASTER = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Data_Partitions.csv')

clinical_features = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/RandomSlices_DataAug__classifier_train38530_val4103_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Clinical_Data_Train.csv')
clinical_features_val = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/RandomSlices_DataAug__classifier_train38530_val4103_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Clinical_Data_Val.csv')
clinical_df = pd.read_excel('/home/deeperthought/Projects/MSKCC/MSKCC/Data_spreadsheets/Diamond_and_Gold/CCNY_CLINICAL_4_17_2019.xlsx', header=[0,1])    

converts = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Converts_1_year_follow_ups_and_segmentations_BIRADS123.csv')

scanID_list = list(converts['now'].values)

#%% ADD CLINICAL

X = list(scanID_list)

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

assert np.sum(pd.isnull(clinical_features).any(axis=1)) == 0, 'Clinical features have NaNs, some have missing clinical info!'


# Path to aligned converts:
DATA_PATH = '/home/deeperthought/kirbyPRO/Alignment_in_time/aligned_193_1yr_converts/'

#%%

for scanID in scanID_list:
    
    #MASTER.loc[MASTER['Scan_ID'] == scanID, ['Pathology', 'BIRADS']]
    segmentation_path = converts.loc[converts['now'] == scanID, 'segmentation'].values[0]
    
    reference = converts.loc[converts['now'] == scanID, 'next'].values[0]
    
    exam = scanID[:-2]
    side = 'right'
    if scanID[-1] == 'l':
        side = 'left'
    
    if not os.path.exists(DATA_PATH + exam ):
        print('Path invalid, skip')
        continue

    all_subject_channels = [DATA_PATH + exam + '/T1_{}_02_01_WarpAlignment_to_{}.nii.gz'.format(side, reference),
                            DATA_PATH + exam + '/T1_{}_slope1_WarpAlignment_to_{}.nii.gz'.format(side, reference),
                            DATA_PATH + exam + '/T1_{}_slope2_WarpAlignment_to_{}.nii.gz'.format(side, reference)]

    T1_pre_nii_path = '/home/deeperthought/kirby_MSK/alignedNii-Nov2019/' + exam + '/T1_{}_01_01.nii'.format(side)

    all_data = load_and_preprocess(all_subject_channels, T1_pre_nii_path)
            
    clinical = clinical_features.loc[clinical_features['scan_ID'] == scanID,[u'Family Hx',u'Age',u'ETHNICITY_HISPANIC OR LATINO',u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',u'RACE_ASIAN-FAR EAST/INDIAN SUBCONT',u'RACE_BLACK OR AFRICAN AMERICAN',u'RACE_NATIVE AMERICAN-AM IND/ALASKA',u'RACE_NATIVE HAWAIIAN OR PACIFIC ISL',u'RACE_UNKNOWN',u'RACE_WHITE']].values
    
    slice_preds = make_prediction_whole_scan(model, all_data, clinical)
        
    slice_preds_class1 = [x[1] for x in slice_preds]
    
    
    SLICE_NR = np.argmax(slice_preds_class1)
    
    img_array = np.expand_dims(np.array([x[SLICE_NR] for x in all_data]), -1)
    img_array = np.swapaxes(img_array, 0,-1)
    yhat = model.predict([img_array, clinical])
    img_tensor = tf.convert_to_tensor(img_array)
    clin_tensor = tf.convert_to_tensor(clinical)
    heatmap = make_gradcam_heatmap([img_tensor, clin_tensor], model, last_conv_layer_name, pred_index=1, normalize=True, pooling='max')
    img = img_array[0]
    img = np.uint8(255 * img)
    img[:,:,1] = img[:,:,0]
    img[:,:,2] = img[:,:,0]
    heatmap += abs(np.min(heatmap))
    heatmap /= np.max(heatmap)
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    # Save the superimposed image
    alpha = 0.4
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    
 #%%   
    segmentation_reference = nib.load(segmentation_path).get_data()
    segmented_slice = list(set(np.argwhere(segmentation_reference > 0)[:,0]))[0]

    T1_REF = nib.load('/home/deeperthought/kirby_MSK/alignedNii-Nov2019/' + reference[:-2] + '/T1_{}_02_01.nii'.format(side)).get_data()
       
#%%

    fig, axes = plt.subplots(3,3, figsize=(12,12))
    axes[0][0].imshow(img_array[0,:,:,0], cmap='gray', aspect="auto"), axes[0][0].set_xticks([]) , axes[0][0].set_yticks([])
    axes[0][1].imshow(img_array[0,:,:,1], cmap='gray', aspect="auto"), axes[0][1].set_xticks([]) , axes[0][1].set_yticks([])
    axes[0][2].imshow(img_array[0,:,:,2], cmap='gray', aspect="auto"), axes[0][2].set_xticks([]) , axes[0][2].set_yticks([])
    axes[1][0].plot(slice_preds_class1); axes[1][0].set_xlabel('Slice'); axes[1][0].set_ylabel('Pred');
    axes[1][0].vlines(SLICE_NR,np.min(slice_preds_class1),np.max(slice_preds_class1), color='r', alpha=0.5)
    axes[1][0].grid()
    axes[1][1].imshow(superimposed_img, aspect="auto"); axes[1][1].set_title('T1post + heatmap'); axes[1][1].set_xticks([]) , axes[1][1].set_yticks([])
    axes[1][1].set_title('Predicted slice: {}'.format(SLICE_NR))
    axes[1][2].axis('off')
    plt.suptitle('{}\nSlice: {}; Model pred = {:.2f}'.format(scanID,SLICE_NR, yhat[0,-1]))
    
    axes[2][0].imshow(T1_REF[segmented_slice], cmap='gray', aspect="auto"), axes[2][0].set_xticks([]) , axes[2][0].set_yticks([]); axes[2][0].set_title('Convert, slice:{}'.format(segmented_slice))
    axes[2][1].imshow(segmentation_reference[segmented_slice], cmap='gray', aspect="auto"), axes[2][1].set_xticks([]) , axes[2][1].set_yticks([]); axes[2][1].set_title('Convert, Slice: {}'.format(segmented_slice))
    axes[2][2].imshow(all_data[0][segmented_slice], cmap='gray', aspect="auto"), axes[2][2].set_xticks([]) , axes[2][2].set_yticks([]); axes[2][2].set_title('Current exam, slice:{}'.format(segmented_slice))

    plt.savefig('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/RandomSlices_DataAug__classifier_train38530_val4103_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Converts/{}.png'.format(scanID))
    plt.close()