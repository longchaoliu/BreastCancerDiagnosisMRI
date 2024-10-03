#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:16:09 2024

@author: deeperthought
"""


import os
os.chdir('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/')
from utils import UNet_v0_2D_Classifier, load_and_preprocess, generate_gradCAM_image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%

# partition_sagittal = np.load("/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/DATA/Data.npy",allow_pickle=True).item()
# labels_sagittal = np.load("/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/DATA/Labels.npy",allow_pickle=True).item()
# clinical_features = pd.read_csv("/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/DATA/Clinical_Data_Train_Val.csv")

# SEG_PATH = '/home/deeperthought/kirbyPRO/Saggittal_segmentations_clean/nifti/'
# segmentations = os.listdir(SEG_PATH)




result = pd.read_csv('/home/deeperthought/Documents/Papers_and_grants/Diagnosis_paper/data/Axial_test_result.csv')

result['y_true'].value_counts()



SEG_PATH1 = '/home/deeperthought/kirby_MSK/segExtd-Mar2021/'
SEG_PATH2 = "/home/deeperthought/kirby_MSK/250caseSegExtd-May2020-cropped/"

segmented_scans = [SEG_PATH1 + x for x in  os.listdir(SEG_PATH1)]
segmented_scans.extend( [SEG_PATH2 + x for x in  os.listdir(SEG_PATH2)])

segmented_exams = list(set([x[:29] for x in segmented_scans]))

axial_annotations = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/DATA/Axial_segmented_annotations_lesions.csv')

'RIA_19-093_000_08151_20160202' in segmented_exams

'''
'RIA_19-093_000_01507_20170209' IS NOT VOLUMETRIC SEGMENTATION!!!! MUST REMOVE FROM ALL LOCALIZATION RESULTS!!
RIA_19-093_000_01561_20170206

'''

result_segmented = result.loc[result['Exam'].isin(axial_annotations['Exam'])]

result_segmented['scanID'] = result_segmented['Exam']
result_segmented['Exam'] = result_segmented['scanID'].str[:-2]


len(set(segmented_exams).intersection(set(result_segmented['Exam'].values)))

cancers_test = result_segmented['scanID'].values

#%%
MODEL_WEIGHTS_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/best_model_weights.npy'

model = UNet_v0_2D_Classifier(input_shape =  (512,512,3), pool_size=(2, 2),initial_learning_rate=1e-5, 
                                         deconvolution=True, depth=6, n_base_filters=42,
                                         activation_name="softmax", L2=1e-5, USE_CLINICAL=True)

loaded_weights = np.load(MODEL_WEIGHTS_PATH, allow_pickle=True, encoding='latin1')

model.set_weights(loaded_weights)


#%%
from matplotlib.patches import Rectangle
from skimage.transform import resize


import nibabel as nib

def euclidian(x,y):
    
    return(np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[2]-y[2])**2))


df_gradcam = pd.DataFrame(columns=['exam', 'distance'])

# df_gradcam = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/gradCAM_images/MSK_Axial/gradCAM_summary.csv')

N = len(cancers_test)
iii = 0
for scanID in cancers_test:
    print(f'{iii}/{N}')
    iii += 1
    EXAM = scanID[:-2]
    SIDE = 'right'
    if scanID[-1] == 'l':
        SIDE = 'left'
        
    if EXAM in df_gradcam['exam'].values:
        print('done..skip')
        continue
        
    #---- MSKCC Sagittal Exam -----------------------
    # t1post_path = f"/home/deeperthought/kirby_MSK/alignedNii-Nov2019/{EXAM}/T1_{SIDE}_02_01.nii"
    # slope1_path = f"/home/deeperthought/kirby_MSK/alignedNii-Nov2019/{EXAM}/T1_{SIDE}_slope1.nii"
    # slope2_path = f"/home/deeperthought/kirby_MSK/alignedNii-Nov2019/{EXAM}/T1_{SIDE}_slope2.nii"
    # T1_pre_nii_path = f'/home/deeperthought/kirby_MSK/alignedNii-Nov2019/{EXAM}/T1_{SIDE}_01_01.nii'
    # MODALITY = 'sagittal' # 'axial'
    
    
    # #---- MASKCC Axial Exam -----------------------
    t1post_path = f"/home/deeperthought/kirbyPRO/alignedNiiAxial-May2020-cropped-normed/{EXAM}/T1_axial_02_01.nii.gz"
    slope1_path = f"/home/deeperthought/kirbyPRO/alignedNiiAxial-May2020-cropped-normed/{EXAM}/T1_axial_slope1.nii.gz"
    slope2_path = f"/home/deeperthought/kirbyPRO/alignedNiiAxial-May2020-cropped-normed/{EXAM}/T1_axial_slope2.nii.gz"
    T1_pre_nii_path = ''
    MODALITY = 'axial' # 'axial'
    SIDE = ''
    
    
    # #---- JHU Exam -----------------------
    # t1post_path = "/home/deeperthought/kirbyPRO/jhuData/alignedNii-normed/ML00925838_20140504/T1_axial_02.nii"
    # slope1_path = "/home/deeperthought/kirbyPRO/jhuData/alignedNii-normed/ML00925838_20140504/T1_axial_slope1.nii"
    # slope2_path = "/home/deeperthought/kirbyPRO/jhuData/alignedNii-normed/ML00925838_20140504/T1_axial_slope2.nii"
    # T1_pre_nii_path = ''
    # MODALITY = 'axial' # 'axial'
    # SIDE = ''
    
    
    all_subject_channels = [t1post_path, slope1_path, slope2_path]

    SEGMENTATION_PATH = [x for x in segmented_scans if x.split('/')[-1].startswith(EXAM)][0]
    segmentation_img = nib.load(SEGMENTATION_PATH).get_fdata()
    # t1post_raw = nib.load(t1post_path).get_fdata()

    coords = np.argwhere(segmentation_img > 0)
    if len(set(coords[:,0])) < 2 or len(set(coords[:,1])) < 2 or len(set(coords[:,2])) < 2:
        print('segmentation not 3D!:')
        print(EXAM)
        continue

     
    X, shape = load_and_preprocess(all_subject_channels, T1_pre_nii_path=T1_pre_nii_path, side=SIDE, imaging_protocol=MODALITY, debug=False)
    
    print('Data preprocessed.. model inference')
    
    MODE_CLINICAL = np.array([[0.  , 0.51, 0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  ]])
        
    preds = model.predict([X, np.tile(MODE_CLINICAL, (shape[0], 1))], batch_size=1, use_multiprocessing=True, workers=10, verbose=0)[:,-1]
        
    print('prediction done..')
    # preds_gated = preds*slices_on_breast[1:]
    global_prediction = np.max(preds)
    max_slice = np.argwhere(preds == global_prediction)[0][0]
    
    # if MODALITY == 'axial':
    #     image_half = preds.shape[0]//2
    #     left_breast_predictions = preds[:image_half]
    #     right_breast_predictions = preds[image_half:]
    #     print(f'{image_half}')
    #     print(f'{len(left_breast_predictions)}, {len(right_breast_predictions)}')
            
    #     left_breast_global_pred = np.max(left_breast_predictions)
    #     right_breast_global_pred = np.max(right_breast_predictions)
    
    
    axial_projection_t1post = np.max(X[:,:,:,0],2)
    axial_projection_slope1 = np.max(X[:,:,:,1],2)
    axial_projection_slope2 = np.max(X[:,:,:,2],2)
    
    print('Generating gradCAM heatmap..')
    heatmap, img, superimposed_img = generate_gradCAM_image(model, X[max_slice:max_slice+1], MODE_CLINICAL ,alpha = 0.30)
    
    myslice = X[max_slice,:,:,0]

    # Load segmentation

    
    all_subject_channels[-1] = SEGMENTATION_PATH
    X_seg, shape = load_and_preprocess(all_subject_channels, T1_pre_nii_path=T1_pre_nii_path, side=SIDE, imaging_protocol=MODALITY, debug=False, order=0)

    segmentation_reshaped = X_seg[:,:,:,-1]
    

    coords = np.argwhere(segmentation_reshaped > 0)
    if len(set(coords[:,0])) < 2 or len(set(coords[:,1])) < 2 or len(set(coords[:,2])) < 2:
        print('segmentation not 3D!:')
        print(EXAM)
        continue
    
    # plt.imshow(np.rot90(myslice))
    # plt.imshow(np.rot90(segmentation_reshaped[max_slice]*0.01 + myslice))
    
    # before reshaping: original t1post and original segmentation:
    # plt.imshow(np.rot90(t1post_raw[max_slice]))    
    # plt.imshow(np.rot90(t1post_raw[max_slice]*0.1 + segmentation_img[max_slice]))

    # crop border artifacts:
    heatmap[:2] = np.min(heatmap)
    heatmap[:,-2:] = np.min(heatmap)
    heatmap[:,0] = np.min(heatmap)
    
    heatmap_resized = resize(heatmap, output_shape=(512,512), anti_aliasing=True)
    center = tuple(np.argwhere(heatmap_resized == np.max(heatmap_resized))[0])
    segmented_area = [tuple(x) for x in coords]
    center_3d = tuple((max_slice, center[0], center[1]))
    
    distance = np.min([euclidian(center_3d, x) for x in segmented_area])     
        
    
    df_gradcam = df_gradcam.append({'exam':EXAM, 'distance':distance}, ignore_index=True)

    df_gradcam.to_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/gradCAM_images/MSK_Axial/gradCAM_summary_distances.csv', index=False)

    plt.figure(figsize=(8,8))
    plt.subplot(2,2,1)
    plt.imshow(myslice); plt.xticks([]); plt.yticks([])
    plt.subplot(2,2,2)
    plt.imshow(superimposed_img); plt.xticks([]); plt.yticks([])

    plt.subplot(2,2,3)
    plt.imshow(heatmap_resized)
    plt.imshow((myslice))
    plt.plot(center[1], center[0], 'o', color='r'); plt.xticks([]); plt.yticks([])

    plt.subplot(2,2,4)
    plt.imshow((segmentation_reshaped[max_slice]))
    plt.plot(center[1], center[0], 'o', color='r'); plt.xticks([]); plt.yticks([])

    plt.tight_layout()
    
    plt.savefig(f'/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/gradCAM_images/MSK_Axial/{distance}_{scanID}.png')    
    plt.close()