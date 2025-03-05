#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:47:06 2023

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




#%%
# def load_and_preprocess(all_subject_channels, T1_pre_nii_path):
#     t1post = nib.load(all_subject_channels[0]).get_data()
#     slope1 = nib.load(all_subject_channels[1]).get_data()
#     slope2 = nib.load(all_subject_channels[2]).get_data()    
    
#     if t1post.shape[1] != 512:
#         output_shape = (t1post.shape[0],512,512)
#         t1post = resize(t1post, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
#         slope1 = resize(slope1, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
#         slope2 = resize(slope2, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')

#     p95 = np.percentile(nib.load(T1_pre_nii_path).get_data(),95)
        
#     t1post = t1post/p95    
#     slope1 = slope1/p95    
#     slope2 = slope2/p95    

#     t1post = t1post/float(40)
#     slope1 = slope1/float(0.3)
#     slope2 = slope2/float(0.12)     

#     return t1post, slope1, slope2


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

def load_and_preprocess(all_subject_channels, T1_pre_nii_path):
  
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
    
#%%
model = UNet_v0_2D_Classifier(input_shape =  (512,512,3), pool_size=(2, 2),initial_learning_rate=1e-5, 
                                         deconvolution=True, depth=6, n_base_filters=42,
                                         activation_name="softmax", L2=1e-5, USE_CLINICAL=True)

MODEL_WEIGHTS_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/best_model_weights.npy'

# MODEL_WEIGHTS_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/best_model_weights.npy'

loaded_weights = np.load(MODEL_WEIGHTS_PATH, allow_pickle=True, encoding='latin1')

model.set_weights(loaded_weights)



clinical = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Clinical_Data_Train.csv')

REDCAP = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/REDCAP/2023/REDCAP_EZ.csv') 


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#%%

labels = np.load('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Labels.npy', allow_pickle=True).item()

#%%

# DATA_PATH = '/home/deeperthought/kirby_MSK/alignedNiiAxial-Nov2019/'

DATA_PATH = '/home/deeperthought/kirbyPRO/alignedNiiAxial-May2020-cropped-normed/'

exams = os.listdir(DATA_PATH)


REDCAP = REDCAP.loc[REDCAP['Exam'].isin(exams)]

REDCAP['bi_rads_assessment_for_stu'].value_counts()

REDCAP.loc[REDCAP['bi_rads_assessment_for_stu'] < '4', 'overall_study_assessment'].value_counts()
REDCAP.loc[REDCAP['bi_rads_assessment_for_stu'] >= '4', 'overall_study_assessment'].value_counts()

REDCAP.loc[REDCAP['bi_rads_assessment_for_stu'] < '4', 'right_breast_tumor_status'].value_counts()
REDCAP.loc[REDCAP['bi_rads_assessment_for_stu'] < '4', 'left_breast_tumor_status'].value_counts()

REDCAP.loc[REDCAP['bi_rads_assessment_for_stu'] >= '4', 'right_breast_tumor_status'].value_counts()
REDCAP.loc[REDCAP['bi_rads_assessment_for_stu'] >= '4', 'left_breast_tumor_status'].value_counts()

# GATHER BENIGNS
REDCAP_B123 = REDCAP.loc[REDCAP['bi_rads_assessment_for_stu'] < '4']
REDCAP_B123 = REDCAP_B123.loc[(REDCAP_B123['right_breast_tumor_status'] != 'Malignant') * (REDCAP_B123['left_breast_tumor_status'] != 'Malignant')]
REDCAP_benigns = REDCAP_B123.loc[REDCAP_B123['true_negative_mri_no_cance'] == 'Yes']

# GATHER MALIGNANTS
REDCAP_B45 = REDCAP.loc[REDCAP['bi_rads_assessment_for_stu'] > '3']
REDCAP_malignants = REDCAP_B45.loc[(REDCAP_B45['right_breast_tumor_status'] == 'Malignant') + (REDCAP_B45['left_breast_tumor_status'] == 'Malignant')]


REDCAP_malignants.index = np.arange(0,len(REDCAP_malignants)*2, 2)
REDCAP_benigns.index = np.arange(1,len(REDCAP_benigns)*2, 2)


AXIAL_SCANS = pd.concat([REDCAP_malignants,REDCAP_benigns])
AXIAL_SCANS.sort_index(inplace=True)


#%%

clinical_df = pd.read_excel('/home/deeperthought/Projects/MSKCC/MSKCC/Data_spreadsheets/Diamond_and_Gold/CCNY_CLINICAL_4_17_2019.xlsx', header=[0,1])    

clinical_df.columns

X = list(AXIAL_SCANS['Exam'])

clinical_features = pd.DataFrame(columns=['exam'])
clinical_features['exam'] = X
clinical_features = add_age(clinical_features, clinical_df)
clinical_features = add_ethnicity_oneHot(clinical_features, clinical_df)
clinical_features = add_family_hx(clinical_features, clinical_df)
clinical_features['Age'] = clinical_features['Age']/100.
clinical_features = clinical_features.drop_duplicates()
CLINICAL_FEATURE_NAMES = [u'Family Hx',u'Age',u'ETHNICITY_HISPANIC OR LATINO',u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',u'RACE_ASIAN-FAR EAST/INDIAN SUBCONT',u'RACE_BLACK OR AFRICAN AMERICAN',u'RACE_NATIVE AMERICAN-AM IND/ALASKA',u'RACE_NATIVE HAWAIIAN OR PACIFIC ISL',u'RACE_UNKNOWN',u'RACE_WHITE']
   
clinical_features[clinical_features.isnull().any(axis=1)]

np.sum(pd.isnull(clinical_features).any(axis=1))


#%%
OUTPUT_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/axial_results/'

#EXAM = 'MSKCC_16-328_1_00001_20150710' # benign
#
#EXAM = 'MSKCC_16-328_1_02914_20140516' # malignant right

RESOLUTIONS = {}
SHAPES ={}

if os.path.exists(OUTPUT_PATH + 'results_noFineTune.csv'):
    results = pd.read_csv(OUTPUT_PATH + 'results_noFineTune.csv')
else:
    results = pd.DataFrame(columns=['Exam','max_slice','max_pred','left_breast_pathology','left_breast_global_pred','right_breast_pathology','right_breast_global_pred','BIRADS', 'TrueNegative', 'overall_study_assessment', 'interval_cancer_cancer_dia','slice_preds'])


# import matplotlib.backends.backend_pdf
# pdf = matplotlib.backends.backend_pdf.PdfPages(OUTPUT_PATH + 'Summary_results.pdf')


test_res = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/test_result_QA.csv')

AXIAL_SCANS = AXIAL_SCANS.loc[AXIAL_SCANS['Exam'].isin(test_res['Exam'])]

for row in AXIAL_SCANS.iterrows():
    
    EXAM = row[1]['Exam']
    
    # if     EXAM == 'MSKCC_16-328_1_00548_20150123':
    #     break

    
    print(EXAM)
    
    if EXAM in results['Exam'].values:
        print('Already done, continue..')
        continue
    
    all_subject_channels = [DATA_PATH + EXAM + '/T1_axial_02_01.nii.gz',
                            DATA_PATH + EXAM + '/T1_axial_slope1.nii.gz',
                            DATA_PATH + EXAM + '/T1_axial_slope2.nii.gz']
    
    # T1_pre_nii_path = DATA_PATH + EXAM + '/T1_axial_01_01.nii'  
    
    try:
        hdr = nib.load(all_subject_channels[0])
        # t1pre = nib.load(T1_pre_nii_path).get_fdata()
        t1post = nib.load(all_subject_channels[0]).get_fdata()
        slope1 = nib.load(all_subject_channels[1]).get_fdata()
        slope2 = nib.load(all_subject_channels[2]).get_fdata()    
    except:
        continue

    
    # if not np.all(np.isfinite(t1post)):
    #     print('Nans! skip')
    #     continue
    if not np.all(np.isfinite(slope1)):
        print('Nans! skip')
        continue
    # if not np.all(np.isfinite(slope2)):
    #     print('Nans! skip')
    #     continue
    
    resolution = np.diag(hdr.affine)
    RESOLUTIONS[EXAM] = resolution
    SHAPES[EXAM] = t1post.shape
    
    print(resolution)
       
    if (t1post.shape[1] != 512) or ((t1post.shape[2] != 512)):
          output_shape = (t1post.shape[0],512,512)
          t1post = resize(t1post, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
          slope1 = resize(slope1, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
          slope2 = resize(slope2, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')


    # p95 = np.percentile(t1pre,95)
        
    # t1post = t1post/p95    
    # slope1 = slope1/p95    
    # slope2 = slope2/p95    

    t1post = t1post/float(40)
    slope1 = slope1/float(0.3)
    slope2 = slope2/float(0.12)     

    # t1post = t1post/float(26) #25
    # slope1 = slope1/float(0.6) # 0.2
    # slope2 = slope2/float(0.24)      # 0.08
    
    X = np.stack([t1post, slope1, slope2], axis=-1)
        
    #For unavailable clinical information
    MODE_CLINICAL = np.array([[0.  , 0.51, 0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  ]])
        
    
    print('Data preprocessed.. model inference')
    
    ################## BOTH #######################################################
    preds = []
    for i in range(1,X.shape[0]):
        #print(i)
        pred = model.predict([X[i-1:i], MODE_CLINICAL])
        preds.append(pred[0,1])
        
    print('prediction done..')
    # preds_gated = preds*slices_on_breast[1:]
    global_prediction = np.max(preds)
    max_slice = np.argwhere(preds == global_prediction)[0][0]

    
    left_breast_pathology = np.nan
    right_breast_pathology = np.nan
    BIRADS = row[1]['bi_rads_assessment_for_stu'] 
    
    # SAVE RESULT
    if row[1]['left_breast_tumor_status'] == 'Malignant' and row[1]['right_breast_tumor_status'] != 'Malignant':
        left_breast_pathology = 1

    if row[1]['right_breast_tumor_status'] == 'Malignant' and row[1]['left_breast_tumor_status'] != 'Malignant':
        right_breast_pathology = 1
        
    if row[1]['right_breast_tumor_status'] != 'Malignant' and row[1]['left_breast_tumor_status'] != 'Malignant' and BIRADS < '4':
        left_breast_pathology = 0
        right_breast_pathology = 0
        
    BIRADS = row[1]['bi_rads_assessment_for_stu']
    TrueNegative = row[1]['true_negative_mri_no_cance']
    overall_study_assessment = row[1]['overall_study_assessment']
    interval_cancer_cancer_dia = row[1]['interval_cancer_cancer_dia']
    
   
    print(f'{len(preds)}')
    print(f'{t1post.shape}')
    image_half = t1post.shape[0]//2
    left_breast_predictions = preds[:image_half]
    right_breast_predictions = preds[image_half:]
    print(f'{image_half}')
    print(f'{len(left_breast_predictions)}, {len(right_breast_predictions)}')
        
    left_breast_global_pred = np.max(left_breast_predictions)
    right_breast_global_pred = np.max(right_breast_predictions)
    
    
    

    results = results.append({'Exam':EXAM,'max_slice':max_slice,'max_pred':global_prediction,
      'left_breast_pathology':left_breast_pathology,'left_breast_global_pred':left_breast_global_pred,
      'right_breast_pathology':right_breast_pathology,'right_breast_global_pred':right_breast_global_pred,
      'BIRADS':BIRADS, 'TrueNegative':TrueNegative, 'overall_study_assessment':overall_study_assessment, 'interval_cancer_cancer_dia':interval_cancer_cancer_dia,
      'slice_preds':preds}, ignore_index=True)


    
    results.to_csv(OUTPUT_PATH + 'results_noFineTune.csv', index=False)

    

    # plt.imshow(X[370,:,:,0])


#     axial_projection_t1post = np.max(t1post,-1)
#     plt.figure(1)
#     fig, ax = plt.subplots(3,1, sharex=True, figsize=(10,10))    
#     # plt.subplots_adjust(hspace=.0)
    
#     ax[0].set_title(EXAM)
#     ax[0].plot(preds)
#     # ax[0][0].plot(slices_on_breast*global_prediction)
#     ax[0].set_aspect('auto')
    
#     ax[1].imshow(np.rot90(axial_projection_t1post), cmap='gray' , vmax=np.percentile(axial_projection_t1post,99))   
#     ax[1].set_aspect('auto'); ax[1].set_xticks([]); ax[1].set_yticks([])
#     ax[1].vlines(max_slice,0,512,'r')
#     #ax[2][0].imshow(np.rot90(axial_projection_slope1), cmap='gray' , vmax=np.percentile(axial_projection_slope1,99.9))
#     ax[1].set_aspect('auto');
   
#     ax[2].imshow(np.rot90(t1post[max_slice]), cmap='gray' ,vmax=np.percentile(t1post[max_slice],97))
#     ax[2].set_xlabel(f'Predicted slice: {max_slice}'); ax[2].set_xticks([]); ax[2].set_yticks([])
#     ax[2].set_aspect('auto')


#     pdf.savefig(fig)
#     plt.close()

# pdf.close()

        

