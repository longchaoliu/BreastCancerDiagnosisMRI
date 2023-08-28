#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 18:24:04 2023

@author: deeperthought
"""

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="0"
tf.keras.backend.set_session(tf.Session(config=config))

from model_utils import FocalLoss, UNet_v0_2D_Classifier

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.transform import resize

from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)

#%% USER INPUT

PRETRAINED_MODEL_WEIGHTS_PATH = '/../model/CNN_weights.npy' 

T1_PRE_PATH = '/../T1_axial_01.nii.gz'
T1_POST_PATH =  '/../T1_axial_01.nii.gz'
DCE_IN_PATH = '/../T1_axial_02.nii.gz'
DCE_OUT_PATH = '/../T1_axial_slope1.nii.gz'

#%% Create model with standard architecture and load weights.
 
model = UNet_v0_2D_Classifier(input_shape =  (512,512,3), pool_size=(2, 2),initial_learning_rate=1e-5, 
                                 deconvolution=True, depth=6, n_base_filters=42,
                                 activation_name="softmax", L2=1e-5, USE_CLINICAL=True)

model_weights = np.load(PRETRAINED_MODEL_WEIGHTS_PATH, allow_pickle=True)   
model.set_weights(model_weights)
model.compile(loss=FocalLoss, optimizer=tf.keras.optimizers.Adam(lr=1e-5), metrics=['acc'])

#%% Load axial data and reshape for model

DATA = [T1_PRE_PATH, T1_POST_PATH, DCE_IN_PATH, DCE_OUT_PATH]

all_subject_channels = DATA[1:]
T1_pre_nii_path = DATA[0]

hdr = nib.load(all_subject_channels[0])
t1pre = nib.load(T1_pre_nii_path).get_data()
t1post = nib.load(all_subject_channels[0]).get_data()
slope1 = nib.load(all_subject_channels[1]).get_data()
slope2 = nib.load(all_subject_channels[2]).get_data()    

if t1pre.shape[0] != 512: # input of network is 512,512
    output_shape = (512,512,t1pre.shape[-1])
    t1post = resize(t1post, output_shape=output_shape, preserve_range=True, anti_aliasing=False)
    slope1 = resize(slope1, output_shape=output_shape, preserve_range=True, anti_aliasing=False)
    slope2 = resize(slope2, output_shape=output_shape, preserve_range=True, anti_aliasing=False)
    t1pre = resize(t1pre, output_shape=output_shape, preserve_range=True, anti_aliasing=False)

resolution = np.diag(hdr.affine)

projection_1d = np.max(np.max(t1post, 0),1)

breast_end = np.argmin(np.diff(projection_1d[np.arange(0,len(projection_1d),5)]))*5
breast_end = breast_end + 10 # add some border
breast_end = np.max([breast_end, 256]) # if breast is small, just crop to 256
    
t1post = t1post[:, (breast_end-256):breast_end, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
t1pre = t1pre[:, (breast_end-256):breast_end, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
slope1 = slope1[:, (breast_end-256):breast_end, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
slope2 = slope2[:, (breast_end-256):breast_end, :]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast


if resolution[0] > 0.5:
    output_shape = (t1post.shape[0], t1post.shape[1]*2, int(t1post.shape[2]* (resolution[2]/0.33) ))
#        output_shape = (512*2, 512*2, int(192*3.3))


# RESIZE to match resolutions.  I need final resolution: (whatever, 0.3, 0.3)
t1post = resize(t1post, output_shape=output_shape, preserve_range=True, anti_aliasing=False)
slope1 = resize(slope1, output_shape=output_shape, preserve_range=True, anti_aliasing=False)
slope2 = resize(slope2, output_shape=output_shape, preserve_range=True, anti_aliasing=False)
t1pre = resize(t1pre, output_shape=output_shape, preserve_range=True, anti_aliasing=False)


if t1post.shape[2] < 512:
    border = 512 - t1post.shape[2]
    t1post = np.pad(t1post, ((0,0),(0,0),(0,border)), 'minimum')
    slope1 = np.pad(slope1, ((0,0),(0,0),(0,border)), 'minimum')
    slope2 = np.pad(slope2, ((0,0),(0,0),(0,border)), 'minimum')
    t1pre = np.pad(t1pre, ((0,0),(0,0),(0,border)), 'minimum')
    
else:    
    t1post = t1post[:, :, 0:512]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    slope1 = slope1[:, :, 0:512]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    slope2 = slope2[:, :, 0:512]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast
    t1pre = t1pre[:, :, 0:512]  # Dont cut across sagittal. Cut on coronal to remove some chest and most blank space, cut on axial to remove blank space above breast

p95 = np.percentile(t1pre,95)
    
t1post = t1post/p95    
slope1 = slope1/p95    
slope2 = slope2/p95    

t1post = t1post/float(40)
slope1 = slope1/float(0.3)
slope2 = slope2/float(0.12)             

X = np.stack([t1post, slope1, slope2], axis=-1)

X.shape

number_slices = t1post.shape[0]
left_breast = X[:number_slices/2]
right_breast = X[number_slices/2:]

#For unavailable clinical information
MODE_CLINICAL = np.array([[0.  , 0.51, 0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  ]])

################## LEFT ########################################################
preds = []
for i in range(1,left_breast.shape[0]):
    #print(i)
    pred = model.predict([left_breast[i-1:i], MODE_CLINICAL])
    preds.append(pred[0,1])
    
global_prediction = np.max(preds)
max_slice = np.argmax(preds)

# DISPLAY
    
axial_projection_t1post = np.max(t1post,-1)
axial_projection_slope1 = np.max(slope1,-1)

plt.figure(1)
fig, ax = plt.subplots(4,1, sharex=True, figsize=(5,11))    
plt.subplots_adjust(hspace=.0)
ax[0].plot(preds)
ax[1].imshow(np.rot90(axial_projection_t1post), cmap='gray' , vmax=np.percentile(axial_projection_t1post,99.9))
ax[1].set_aspect('auto')
ax[2].imshow(np.rot90(axial_projection_slope1), cmap='gray' , vmax=np.percentile(axial_projection_slope1,99.9))
ax[2].set_aspect('auto')
ax[3].imshow(np.rot90(t1post[max_slice]), cmap='gray' )


################## RIGHT ########################################################
preds = []
for i in range(1,right_breast.shape[0]):
    #print(i)
    pred = model.predict([right_breast[i-1:i], MODE_CLINICAL])
    preds.append(pred[0,1])
    
global_prediction = np.max(preds)
max_slice = np.argmax(preds)

# DISPLAY
    
axial_projection_t1post = np.max(t1post,-1)
axial_projection_slope1 = np.max(slope1,-1)

plt.figure(2)

fig, ax = plt.subplots(4,1, sharex=True, figsize=(5,11))    
plt.subplots_adjust(hspace=.0)
ax[0].plot(([0]*int(number_slices/2)) + list(preds))
ax[1].imshow(np.rot90(axial_projection_t1post), cmap='gray' , vmax=np.percentile(axial_projection_t1post,99.9))
ax[1].set_aspect('auto')
ax[2].imshow(np.rot90(axial_projection_slope1), cmap='gray' , vmax=np.percentile(axial_projection_slope1,99.9))
ax[2].set_aspect('auto')
ax[3].imshow(np.rot90(t1post[max_slice]), cmap='gray' )

