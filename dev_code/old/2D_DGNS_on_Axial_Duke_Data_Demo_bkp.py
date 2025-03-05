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
import scipy
from skimage import morphology
#%% USER INPUT

# MODEL_WEIGHTS_PATH = '/Path/to/Duke_data_Demo/weights.npy'

DATA_PATH = '/home/deeperthought/kirby_MSK/dukePublicData/alignedNii-normed/'      

# DATA_PATH = '/home/deeperthought/kirby_MSK/dukePublicData/alignedNii/'      


MODEL_WEIGHTS_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/best_model_weights.npy'
    

# labels_df = pd.read_csv('/home/deeperthought/kirby_MSK/dukePublicData/25segDukePublic.csv')


labels_df = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Duke_predictions/Annotation_Boxes.csv')


Duke_orientation = pd.read_csv('/home/deeperthought/kirby/homes/lukas/Duke_Data/manifest-1607053360376/dicom_ImageOrientation.csv')

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

MODEL_WEIGHTS_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/best_model_weights.npy'

# MODEL_WEIGHTS_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/best_model_weights.npy'

loaded_weights = np.load(MODEL_WEIGHTS_PATH, allow_pickle=True, encoding='latin1')

model.set_weights(loaded_weights)

# import sys
# sys.path.append('/home/deeperthought/Projects/MultiPriors_MSKCC/scripts/')
# from MultiPriors_Models_Collection import Generalised_dice_coef_multilabel2, dice_coef_multilabel_bin0,dice_coef_multilabel_bin1

# my_custom_objects = {'Generalised_dice_coef_multilabel2':Generalised_dice_coef_multilabel2,
#                                  'dice_coef_multilabel_bin0':dice_coef_multilabel_bin0,
#                                  'dice_coef_multilabel_bin1':dice_coef_multilabel_bin1}

# breastMask_model = tf.keras.models.load_model('/home/deeperthought/Projects/BreastMask_model/UNet_v0_BreastMask_breastMask_UNet_v0_BreastMask_2019-12-23_1658/models/breast_mask_model.h5', custom_objects = my_custom_objects)

# breastMask_model.input

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
    


def load_and_preprocess_v0(all_subject_channels, T1_pre_nii_path=''): 
  
    t1post = nib.load(all_subject_channels[0]).get_fdata()
   
    slope1 = nib.load(all_subject_channels[1]).get_fdata()
   
    slope2 = nib.load(all_subject_channels[2]).get_fdata()    
  
    
    if (t1post.shape[1] != 512) or ((t1post.shape[2] != 512)):
        output_shape = (t1post.shape[0],512,512)
        t1post = resize(t1post, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
        slope1 = resize(slope1, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
        slope2 = resize(slope2, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')

    # p95 = np.percentile(nib.load(T1_pre_nii_path).get_data(),95)
        
    # t1post = t1post/p95    
    # slope1 = slope1/p95    
    # slope2 = slope2/p95    

    t1post = t1post/float(40)
    slope1 = slope1/float(0.3)
    slope2 = slope2/float(0.12)     

    return np.stack([t1post, slope1, slope2], axis=-1), t1post.shape



def load_and_preprocess(all_subject_channels, T1_pre_nii_path='', breast_center=0, debug=False):
  
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
    t1post = resize(t1post, output_shape=target_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
    slope1 = resize(slope1, output_shape=target_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
    slope2 = resize(slope2, output_shape=target_shape, preserve_range=True, anti_aliasing=True, mode='reflect')    
       
    
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
    
        middle_sagittal_slice = t1post[t1post.shape[0]//2]
        middle_sagittal_slice = scipy.ndimage.gaussian_filter(middle_sagittal_slice, sigma=2)
        middle_sagittal_slice_bin = np.array(morphology.opening(middle_sagittal_slice > background, np.ones((4,4))), 'int')
        chest_vector = np.max(middle_sagittal_slice_bin, 1)
    
    
        
        
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
        plt.figure(i, figsize=(9,9))
        
        plt.suptitle(all_subject_channels[0].split('/')[-2])
        
        plt.subplot(r,c,1)
        plt.imshow(np.rot90(middle_sagittal_slice))
        plt.subplot(r,c,3)
        plt.imshow(np.rot90(middle_sagittal_slice_bin > background ))
        plt.subplot(r,c,5)
        plt.plot(chest_vector)

        plt.subplot(r,c,2)
         
        plt.imshow(axial_max_projection, aspect='auto')
           
        plt.subplot(r,c,4)
         
        plt.imshow(axial_max_projection_bin, aspect='auto')
           
        plt.subplot(r,c,6)
         
        plt.plot(breast_vector)
        plt.subplot(r,c,7)
    
        plt.title(f'x1:{breast_start}, x:{breast_end}. --> [{breast_center-256} : {breast_center+256}]')
        plt.imshow(np.rot90(t1post[256//2,start:end]), cmap='gray')
        
        plt.tight_layout()
        plt.show()
            
    t1post = t1post[:,start:end,:512]
    slope1 = slope1[:,start:end,:512]
    slope2 = slope2[:,start:end,:512]
    


    t1post = t1post/float(40)
    slope1 = slope1/float(0.3)
    slope2 = slope2/float(0.12)     

    return  np.stack([t1post, slope1, slope2], axis=-1), t1post_shape #t1post, slope1, slope2



#%%

OUTPUT_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Duke_predictions/predictions/'

# import matplotlib.backends.backend_pdf
# pdf = matplotlib.backends.backend_pdf.PdfPages(OUTPUT_PATH + 'Summary_results_Duke.pdf')


RESULTS_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Duke_predictions/predictions/results_duke_smartCropChest_demographics_noRace.csv'

if os.path.exists(RESULTS_PATH):
    RESULTS = pd.read_csv(RESULTS_PATH)
else:    
    RESULTS = pd.DataFrame(columns=['Patient ID','X1','X2','max_slice','Hit','max_pred','left_breast_pathology','left_breast_global_pred','right_breast_pathology','right_breast_global_pred','slice_preds'])



SUBJECTS = list(set(os.listdir(DATA_PATH)))

SUBJECTS.sort()

SUBJECTS = [x for x in SUBJECTS if x not in RESULTS['Patient ID'].values]

# seg_path = '/home/deeperthought/kirby_MSK/dukePublicData/manSegRes/Segmentation_Results-Lars50cases (1)/Segmentation_Results/segmentation/'
# segmented_subjects = list(set([x[:14] for x in os.listdir(seg_path)]))

t1pre = 0


manual_crop = {}

manual_crop['Breast_MRI_881'] = 550
manual_crop['Breast_MRI_356'] = 500
manual_crop['Breast_MRI_523'] = 310
manual_crop['Breast_MRI_580'] = 590
manual_crop['Breast_MRI_512'] = 510
manual_crop['Breast_MRI_156'] = 500
manual_crop['Breast_MRI_195'] = 500
manual_crop['Breast_MRI_671'] = 500
manual_crop['Breast_MRI_522'] = 400
manual_crop['Breast_MRI_713'] = 600

manual_crop['Breast_MRI_779'] = 500
manual_crop['Breast_MRI_493'] = 550
manual_crop['Breast_MRI_691'] = 400
manual_crop['Breast_MRI_384'] = 400
manual_crop['Breast_MRI_030'] = 450
manual_crop['Breast_MRI_143'] = 560
manual_crop['Breast_MRI_096'] = 450
manual_crop['Breast_MRI_922'] = 500
manual_crop['Breast_MRI_564'] = 600
manual_crop['Breast_MRI_396'] = 500

manual_crop['Breast_MRI_210'] = 500





duke_demo = pd.read_excel('/home/deeperthought/kirby_MSK/dukePublicData/Clinical_and_Other_Features.xlsx', header=[0,1,2])

duke_demo['Demographics'].columns

df_duke_demographics = pd.DataFrame(columns=['Patient ID','Age (days)', 'Age','Race'])


df_duke_demographics['Patient ID'] = duke_demo['Patient Information']['Patient ID']
df_duke_demographics['Age (days)'] = duke_demo['Demographics']['Date of Birth (Days)']
df_duke_demographics['Race'] = duke_demo['Demographics']['Race and Ethnicity']


df_duke_demographics['Age'] = abs(df_duke_demographics['Age (days)'].astype(int))//365/100.


# 0 = N/A
# 1 = white,
# 2 = black,
# 3 = asian,
# 4 = native,
# 5 = hispanic,
# 6 = multi,
# 7 = hawa 
# 8 = amer indian


# 5


'''
Family Hx	= 0
Age         = AGE
'''

'''
ETHNICITY_HISPANIC OR LATINO	 = 5          [1.  , 0.  , 0. ]
ETHNICITY_NOT HISPANIC          != 5          [0.  , 1.  , 0. ]
ETHNICITY_UNKNOWN                = 0          [0.  , 0.  , 1. ]
'''

'''
RACE_ASIAN-FAR EAST/INDIAN SUBCONT  = 3         [1.  , 0.  , 0.  , 0.  , 0.  , 0. ]
RACE_BLACK OR AFRICAN AMERICAN	    = 2         [0.  , 1.  , 0.  , 0.  , 0.  , 0. ]
RACE_NATIVE AMERICAN-AM IND/ALASKA  = 4, 8      [0.  , 0.  , 1.  , 0.  , 0.  , 0. ]
RACE_NATIVE HAWAIIAN OR PACIFIC ISL = 7         [0.  , 0.  , 0.  , 1.  , 0.  , 0. ]
RACE_UNKNOWN	                    = 0, 6, 5   [0.  , 0.  , 0.  , 0.  , 1.  , 0. ]
RACE_WHITE                          = 1         [0.  , 0.  , 0.  , 0.  , 0.  , 1. ]
'''

MODE_CLINICAL = np.array([[0.  , 0.51,      0.  , 1.  , 0.  ,        0.  , 0.  , 0.  , 0.  , 0.  , 1.  ]])


SUBJECTS = ['Breast_MRI_363', 'Breast_MRI_237', 'Breast_MRI_117',
       'Breast_MRI_082', 'Breast_MRI_552', 'Breast_MRI_877',
       'Breast_MRI_708', 'Breast_MRI_835', 'Breast_MRI_175',
       'Breast_MRI_038']

i = 1
# segmented_subjects.sort()
for subj in SUBJECTS:
    
# for subj in manual_crop.keys():
    
    # subj = SUBJECTS[13]
    
    print(f'\n ########### {subj} ############# \n')
    # subj = 'Breast_MRI_003' # 060
    
    subj_age = df_duke_demographics.loc[df_duke_demographics['Patient ID'] == subj, 'Age'].values[0]
    subj_race = df_duke_demographics.loc[df_duke_demographics['Patient ID'] == subj, 'Race'].values[0]
    
    race_entry = [0.  , 0.  , 0.  , 0.  , 0.  , 1. ]
    ethnicity_entry = [0.  , 1.  , 0. ]

    # if subj_race in [0,5,6]:
    #     race_entry = [0.  , 0.  , 0.  , 0.  , 1.  , 0. ]
    # elif subj_race == 1:
    #     race_entry = [0.  , 0.  , 0.  , 0.  , 0.  , 1. ]
    # elif subj_race == 2:
    #     race_entry = [0.  , 1.  , 0.  , 0.  , 0.  , 0. ]
    # elif subj_race == 3:
    #     race_entry = [1.  , 0.  , 0.  , 0.  , 0.  , 0. ]
    # elif subj_race in [4,8]:
    #     race_entry = [0.  , 0.  , 1.  , 0.  , 0.  , 0. ]
    # elif subj_race == 7:
    #     race_entry = [0.  , 0.  , 0.  , 1.  , 0.  , 0. ]     

    # if subj_race == 5:
    #     ethnicity_entry = [1.  , 0.  , 0. ]
    # elif subj_race == 0:
    #     ethnicity_entry = [0.  , 0.  , 1. ]
    
    CLINICAL_INFO = np.expand_dims(np.array( [0.  , subj_age] +  ethnicity_entry +  race_entry),0)
    
    DATA = [DATA_PATH + subj + '/T1_axial_01.nii.gz', 
            DATA_PATH + subj + '/T1_axial_02.nii.gz',
            DATA_PATH + subj + '/T1_axial_slope1.nii.gz',
            DATA_PATH + subj + '/T1_axial_slope2.nii.gz']
        
    centre = 0
    if subj in manual_crop.keys():
        centre = manual_crop[subj]
        
    X, dimensions = load_and_preprocess(DATA[1:], breast_center=centre, debug=False)
    
    # X, dimensions = load_and_preprocess_v0(DATA[1:])

    preds = model.predict([X,np.tile(CLINICAL_INFO, (dimensions[0], 1))], batch_size=1, use_multiprocessing=True, workers=10, verbose=0)[:,-1]
    
    del X

    
#     X.shape
#     850-512
#     res = {}
    
#     i = 1
#     plt.figure(figsize=(15,3))
#     for border_begin in np.arange(0,350,50):
#         plt.subplot(1,7,i)
#         plt.title(border_begin)
#         plt.imshow(np.rot90(X[72,border_begin:border_begin+512,:,0])   ,cmap='gray' )
#         plt.xticks([]); plt.yticks([])
#         i += 1
    
#         preds = model.predict([X[:,border_begin:border_begin+512,:],np.tile(MODE_CLINICAL, (dimensions[0], 1))], batch_size=1, use_multiprocessing=True, workers=10, verbose=0)[:,-1]

#         MAX = np.max(preds)
#         SLICE = np.argmax(preds)

#         res[border_begin] = (MAX, SLICE)
#         print('-------')
#         print(border_begin)
#         print(MAX,SLICE)
    
# p = []
# s = []
# shifts=[]
# for k in res.keys():
#     p.append(res[k][0])
#     s.append(res[k][1])
#     shifts.append(k)
    
# res_df = pd.DataFrame(zip(shifts,p))

# res_df = res_df.sort_values(0)    

# plt.plot(res_df[0], res_df[1], '.-')
    

    global_prediction = np.max(preds)
    max_slice = np.argwhere(preds == global_prediction)[0][0]
    
    
    labels_df.loc[labels_df['Patient ID'] == subj, 'DIM X'] = dimensions[0]
    labels_df.loc[labels_df['Patient ID'] == subj, 'DIM Y'] = dimensions[1]
    labels_df.loc[labels_df['Patient ID'] == subj, 'DIM Z'] = dimensions[2]

    labels_df.loc[labels_df['Patient ID'] == subj, 'X1'] = dimensions[0] - labels_df.loc[labels_df['Patient ID'] == subj, 'End Column']
    labels_df.loc[labels_df['Patient ID'] == subj, 'X2'] = dimensions[0] - labels_df.loc[labels_df['Patient ID'] == subj, 'Start Column']

    labels_df.loc[labels_df['Patient ID'] == subj, 'Y1'] = dimensions[1] - labels_df.loc[labels_df['Patient ID'] == subj, 'End Row']
    labels_df.loc[labels_df['Patient ID'] == subj, 'Y2'] = dimensions[1] - labels_df.loc[labels_df['Patient ID'] == subj, 'Start Row']

    labels_df.loc[labels_df['Patient ID'] == subj, 'Z1'] = dimensions[2] - labels_df.loc[labels_df['Patient ID'] == subj, 'End Slice']
    labels_df.loc[labels_df['Patient ID'] == subj, 'Z2'] = dimensions[2] - labels_df.loc[labels_df['Patient ID'] == subj, 'Start Slice']
        
    # MIDDLE_SLICE = int(0.5*(labels_df.loc[labels_df['Patient ID'] == subj, 'X2'].values[0] + labels_df.loc[labels_df['Patient ID'] == subj, 'X1'].values[0]))
   
    FLIP_FLAG = Duke_orientation.loc[Duke_orientation['subj'] == subj, 'orientation'].values[0]


  
    
    if FLIP_FLAG == 1:
        
        X1 = labels_df.loc[labels_df['Patient ID'] == subj, 'X1'].values[0] 
        X2 = labels_df.loc[labels_df['Patient ID'] == subj, 'X2'].values[0] 
  
    elif FLIP_FLAG == -1:
        
        X1 = labels_df.loc[labels_df['Patient ID'] == subj, 'Start Column'].values[0]
        X2 = labels_df.loc[labels_df['Patient ID'] == subj, 'End Column'].values[0]

  
    MIDDLE_SLICE = (X1+X2)//2
    
    if (X2 >= max_slice) and (max_slice >= X1):
        OUTCOME = 'HIT'
        COLOR='r'
    else:
        OUTCOME = 'MISS'
        COLOR='y'

    HIT = 0
    if OUTCOME == 'HIT':
        HIT=1
        
        
    image_half = preds.shape[0]//2
    left_breast_predictions = preds[:image_half]
    right_breast_predictions = preds[image_half:]
        
    left_breast_global_pred = np.max(left_breast_predictions)
    right_breast_global_pred = np.max(right_breast_predictions)
    
    
    left_breast_pathology = int(MIDDLE_SLICE <= image_half)
    right_breast_pathology = int(MIDDLE_SLICE > image_half)
    

    RESULTS.drop(RESULTS.loc[RESULTS['Patient ID']==subj].index, inplace=True)


    RESULTS = RESULTS.append({'Patient ID':subj,'X1':X1,'X2':X2,'max_slice':max_slice,'Hit':HIT,'max_pred':global_prediction,
     'left_breast_pathology':left_breast_pathology,'left_breast_global_pred':left_breast_global_pred,
     'right_breast_pathology':right_breast_pathology,'right_breast_global_pred':right_breast_global_pred,'slice_preds':preds}, ignore_index=True)
   
    RESULTS.to_csv(RESULTS_PATH, index=False)    

    del preds

   # # # DISPLAY
   #  t1post = X[:,:,:,0]
   #  axial_projection_t1post = np.max(t1post,-1)
 
   #  t1post_original = nib.load(DATA[1]).get_fdata()
   #  slope1_original = nib.load(DATA[2]).get_fdata()    
   #  X1 = int(X1)
   #  X2 = int(X2)
   #  MIDDLE_SLICE = int(MIDDLE_SLICE)
  
   #  plt.figure(1)
   #  fig, ax = plt.subplots(2,2, sharex=True, figsize=(15,7))    
   #  # plt.subplots_adjust(hspace=.0)
  
   #  ax[0][0].set_title(subj)
   #  ax[0][0].plot(preds)
   #  # ax[0][0].plot(slices_on_breast*global_prediction)
   #  ax[0][0].set_aspect('auto')
  
   #  ax[1][0].imshow(np.rot90(axial_projection_t1post), cmap='gray' , vmax=np.percentile(axial_projection_t1post,99.9))   
   #  ax[1][0].set_aspect('auto'); ax[1][0].set_xticks([]); ax[1][0].set_yticks([])
   #  ax[1][0].set_xlabel(OUTCOME)
   #  ax[1][0].vlines(X1,0,512, linestyle='--', color='dodgerblue')
   #  ax[1][0].vlines(X2,0,512, linestyle='--', color='dodgerblue')
   #  ax[1][0].vlines(max_slice,0,512,color=COLOR)
   #  #ax[2][0].imshow(np.rot90(axial_projection_slope1), cmap='gray' , vmax=np.percentile(axial_projection_slope1,99.9))
   #  ax[1][0].set_aspect('auto');
 
   #  ax[0][1].imshow(np.rot90(t1post[max_slice]), cmap='gray' )
   #  ax[0][1].set_xlabel(f'Predicted slice: {max_slice}'); ax[0][1].set_xticks([]); ax[0][1].set_yticks([])
   #  ax[0][1].set_aspect('auto')

  
   #  ax[1][1].imshow(np.rot90(t1post_original[MIDDLE_SLICE,:,:]), cmap='gray', vmax=np.percentile(t1post_original, 99.9))
   #  ax[1][1].set_xlabel(f'GT Sagittal slices: {X1} - {X2}'); ax[1][1].set_xticks([]); ax[1][1].set_yticks([])
   #  ax[1][1].set_aspect('auto')
  
  
  
   #  ax[2][0].imshow(np.rot90(axial_projection_slope1))
   #  ax[2][0].set_aspect('auto'); ax[2][0].set_xticks([]); ax[2][0].set_yticks([])
   #        # ax[2][1].imshow(np.flip(np.rot90(np.max(segmentation_img, 0)),1))
   #  ax[2][1].set_aspect('auto'); ax[2][0].set_xticks([]); ax[2][0].set_yticks([])
     


    # plt.savefig('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Duke_predictions/predictions/' + subj + '_segmentation.png', dpi=400)
    
    # pdf.savefig(fig)
    # plt.close()

# pdf.close()
