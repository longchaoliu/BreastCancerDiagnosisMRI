#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 15:39:18 2024

@author: deeperthought
"""


GPU = 1

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
        
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.transform import resize

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

import pandas as pd
from numpy.random import seed
seed(42)

import scipy
from skimage import morphology
import matplotlib.cm as cm

#%%

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

    
def add_age(clinical_features, clinical_df):
  ages = clinical_df['Unnamed: 1_level_0']
  ages['DE-ID']  = clinical_df['Unnamed: 0_level_0']['DE-ID']
  #ages['DE-ID'] = ages.index
  ages.reset_index(level=0, inplace=True)
  ages = ages[['DE-ID','DOB']]  
  clinical_features2 = clinical_features.copy()
  clinical_features2['DE-ID'] = clinical_features2['exam'].apply(lambda x : x[:20]) 
  clinical_features2 = clinical_features2.merge(ages, on=['DE-ID'])
  clinical_features2['Age'] = clinical_features2.apply(lambda row : int(row['exam'][-8:-4]) - int(row['DOB']), axis=1)
  clinical_features2 = clinical_features2[['exam','Age']]
  return clinical_features2


def add_ethnicity_oneHot(df, clinical):
 
  clinical_df = pd.concat([clinical['Unnamed: 0_level_0']['DE-ID'], clinical['Unnamed: 4_level_0']['ETHNICITY'], clinical['Unnamed: 3_level_0']['RACE']], axis=1)
  
  clinical_df = clinical_df.set_index('DE-ID')
  clinical_df.loc[clinical_df['ETHNICITY'] == 'NO VALUE ENTERED'] = 'UNKNOWN'  
  clinical_df.loc[clinical_df['RACE'] == 'OTHER'] = 'UNKNOWN'  
  clinical_df.loc[clinical_df['RACE'] == 'PT REFUSED TO ANSWER'] = 'UNKNOWN'  
  clinical_df.loc[clinical_df['RACE'] == 'NO VALUE ENTERED'] = 'UNKNOWN'  

  
  clinical_df = pd.get_dummies(clinical_df)

  df['DE-ID'] = df['exam'].str[:20] 
  
  df2 =  pd.merge(df, clinical_df, on='DE-ID')

  return df2

def add_family_hx(df, clinical):
  fam = pd.DataFrame(columns=['DE-ID','Family Hx'])
  fam['Family Hx'] = clinical['Family Hx']['Family Hx']
  fam['Family Hx'] = fam['Family Hx'].apply(lambda x : 1 if x == 'Yes' else 0)
  fam['DE-ID'] = clinical['Unnamed: 0_level_0']['DE-ID']
  fam.reset_index(level=0, inplace=True) 
  df2 = df.copy()
  df2['DE-ID'] = df2['exam'].apply(lambda x : x[:20]) 
  df3 = df2.merge(fam, on=['DE-ID'])
  df3.head()
  df3 = df3[['exam','Family Hx']]
  df4 = df3.merge(df, on=['exam'])
  df4 = df4.loc[df4['exam'].isin(df['exam'])]
  return df4



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
    
    
#%%

def load_and_preprocess_V0(all_subject_channels, T1_pre_nii_path):
  
    t1post = nib.load(all_subject_channels[0]).get_fdata()
   
    slope1 = nib.load(all_subject_channels[1]).get_fdata()
   
    slope2 = nib.load(all_subject_channels[2]).get_fdata()    
  
    if (t1post.shape[1] != 512) or ((t1post.shape[2] != 512)):
        output_shape = (t1post.shape[0],512,512)
        t1post = resize(t1post, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
        slope1 = resize(slope1, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
        slope2 = resize(slope2, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')

    p95 = np.percentile(nib.load(T1_pre_nii_path).get_fdata(),95)
        
    t1post = t1post/p95    
    slope1 = slope1/p95    
    slope2 = slope2/p95    

    t1post = t1post/float(40)
    slope1 = slope1/float(0.3)
    slope2 = slope2/float(0.12)     

    return t1post, slope1, slope2



def load_and_preprocess(all_subject_channels, T1_pre_nii_path='',side='left', imaging_protocol='axial', breast_center=0, debug=False, order=1):

    ''' Make it work for both axial and sagittal'''    

    hdr = nib.load(all_subject_channels[0])
    t1post = hdr.get_fdata()
    t1post_shape = hdr.shape
    slope1 = nib.load(all_subject_channels[1]).get_fdata()
    slope2 = nib.load(all_subject_channels[2]).get_fdata()    
    shape = t1post.shape    
    resolution = np.diag(hdr.affine)
    new_res = np.array([resolution[0], 0.4, 0.4])
    target_shape = [shape[0], int(resolution[1]/new_res[1]*shape[1]), int(resolution[2]/new_res[2]*shape[2])]
    
    # Target resolution in sagittal is ~ 0.4 x 0.4
    print('resampling to a resolution of 0.4 x 0.4mm..')
    t1post = resize(t1post, output_shape=target_shape, preserve_range=True, anti_aliasing=True, mode='reflect', order=1)
    slope1 = resize(slope1, output_shape=target_shape, preserve_range=True, anti_aliasing=True, mode='reflect', order=1)
    slope2 = resize(slope2, output_shape=target_shape, preserve_range=True, anti_aliasing=True, mode='reflect', order=order)    
       
    
    print('cropping/padding to size 512 x 512 pixels..')    
    if t1post.shape[1] < 512:
        pad = 512 - t1post.shape[1]
        t1post = np.pad(t1post, ((0,0),(0,pad),(0,0)), 'constant')
        slope1 = np.pad(slope1, ((0,0),(0,pad),(0,0)), 'constant')
        slope2 = np.pad(slope2, ((0,0),(0,pad),(0,0)), 'constant')
    
    if t1post.shape[2] < 512:
        pad = 512 - t1post.shape[2]
        t1post = np.pad(t1post, ((0,0),(0,0),(pad//2,pad//2+pad%2)), 'constant')
        slope1 = np.pad(slope1, ((0,0),(0,0),(pad//2,pad//2+pad%2)), 'constant')
        slope2 = np.pad(slope2, ((0,0),(0,0),(pad//2,pad//2+pad%2)), 'constant')
        
    
    if breast_center == 0:
        background = np.percentile(t1post, 75)
        # Find chest:
        if imaging_protocol == 'axial':
            chest_slice = t1post[t1post.shape[0]//2]
        elif imaging_protocol == 'sagittal':
            if side == 'right':
                chest_slice = t1post[0]
            elif side == 'left':
                chest_slice = t1post[-1]

        # Find chest:
        chest_slice = scipy.ndimage.gaussian_filter(chest_slice, sigma=2)
        chest_slice_bin = np.array(morphology.opening(chest_slice > background, np.ones((4,4))), 'int')
        chest_vector = np.max(chest_slice_bin, 1)
        breast_start = np.argwhere(chest_vector > 0)[-1][0]
        breast_start = breast_start - 125  # I see the chest middle is always further than the sides...
    
        # Find end of breast
        axial_max_projection = np.max(t1post,-1)
        axial_max_projection = scipy.ndimage.gaussian_filter(axial_max_projection, sigma=2)
        axial_max_projection_bin = np.array(morphology.opening(axial_max_projection > background, np.ones((10,10))), 'int')
        breast_vector = np.max(axial_max_projection_bin, 0)
        breast_end = np.argwhere(breast_vector > 0)[-1][0]
        breast_center = (breast_start + breast_end) //2
                        
    
    if breast_center+256 > t1post.shape[1]:
        end = t1post.shape[1]
        start = end-512
    elif breast_center-256 < 0:
        start = 0
        end = 512
    else:
        start = breast_center-256 
        end = breast_center + 256 

    if debug:
        r,c = 4,2
        plt.figure(0, figsize=(9,9))
        plt.suptitle(all_subject_channels[0].split('/')[-2])
        plt.subplot(r,c,1)
        plt.imshow(np.rot90(chest_slice)); plt.title('Chest slice')
        plt.subplot(r,c,3)
        plt.imshow(np.rot90(chest_slice_bin)); plt.title('chest_slice > background')
        plt.subplot(r,c,2)
        plt.imshow(axial_max_projection, aspect='auto'); plt.title('axial_max_projection')
        plt.subplot(r,c,4)
        plt.imshow(axial_max_projection_bin, aspect='auto'); plt.title('axial_max_projection > background')
        plt.subplot(r,c,5)
        plt.title(f'x1:{breast_start}, x:{breast_end}. --> [{start} : {end}]')
        plt.imshow(np.rot90(t1post[t1post.shape[0]//2,start:end]), vmax=np.percentile(t1post, 97), vmin=-100, cmap='gray')
        plt.xticks([]); plt.yticks([])
        plt.xlabel('Cropped image at resolution 0.4mm')
        plt.subplot(r,c,6)
        plt.plot(chest_vector, label='chest vector')
        plt.plot(breast_vector, label='breast vector')
        plt.legend()

        plt.tight_layout()
        plt.show()
            
    t1post = t1post[:,start:end,:512]
    slope1 = slope1[:,start:end,:512]
    slope2 = slope2[:,start:end,:512]
    
    if len(T1_pre_nii_path) > 0:
        print('normalizing by p95 from T1 pre..')
        p95 = np.percentile(nib.load(T1_pre_nii_path).get_fdata(), 95)
        
        t1post = t1post/p95
        slope1 = slope1/p95
        slope2 = slope2/p95
    
    t1post = t1post/float(40)
    slope1 = slope1/float(0.3)
    slope2 = slope2/float(0.12)     

    return  np.stack([t1post, slope1, slope2], axis=-1), t1post_shape #t1post, slope1, slope2

    

def color_map(value):
  """Maps a value between 0 and 1 to a color between green and red."""
  # Ensure value is within 0-1 range
  value = max(0, min(value, 1))
  
  # Calculate RGB values
  red = value
  green = 1 - value
  blue = 0

  return (red, green, blue)



def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = tf.keras.preprocessing.image.img_to_array(img)
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


def generate_gradCAM_image(model, img_array, clinical_info ,alpha = 0.35):
    layer_of_interest = [x for x in model.layers if x.name.startswith('activation')][-2]
    last_conv_layer_name = layer_of_interest.name #'max_pooling2d_29'# 'activation_70'
    
    img_tensor = tf.convert_to_tensor(img_array)
    clin_tensor = tf.convert_to_tensor(clinical_info)
    heatmap = make_gradcam_heatmap([img_tensor, clin_tensor], model, last_conv_layer_name, pred_index=1, normalize=True, pooling='max')
    
     # Remove boundary artifacts
    # heatmap = np.pad(heatmap[2:-2,2:-2], pad_width=(4,4), mode='minimum') 
    # Shift to remove effect of unpadded maximum pooling in model
    # heatmap = heatmap[3:,3:] 
    
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
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = resize(jet_heatmap, output_shape=(img.shape[1], img.shape[0]), anti_aliasing=True)
    jet_heatmap = jet_heatmap*255.

    # Save the superimposed image
   
    superimposed_array = jet_heatmap * alpha  + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_array)
    
    return heatmap, img, superimposed_img