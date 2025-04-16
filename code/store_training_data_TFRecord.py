#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:53:31 2025

@author: deeperthought
"""


import sys
sys.path.append('/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/code/')
from utils import load_and_preprocess
import numpy as np

#%%

GPU = 3
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
        

def _bytes_feature(value):
  """Returns a bytes_list from a string / bytes."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_tf_example(image_data, label):
  """Creates a tf.train.Example proto."""
  feature = {
      'image_raw': _bytes_feature(image_data.tobytes()),
      'label': _int64_feature(label)
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))

def write_to_tfrecord(mri_data, labels, output_path):
  """Writes MRI data to a TFRecord file.

  Args:
    mri_data: A list of numpy arrays, where each array represents an MRI volume.
    labels: A list of corresponding labels for each MRI volume.
    output_path: The path to the output TFRecord file.
  """
  with tf.io.TFRecordWriter(output_path) as writer:
    for mri_volume, label in zip(mri_data, labels):
      for i in range(mri_volume.shape[0]):  # Iterate through slices of each volume
        slice_data = mri_volume[i]
        slice_data = slice_data.astype(np.float16)
        # Check the shape of the slice data
        example = create_tf_example(slice_data, label) 
        writer.write(example.SerializeToString())

def load_tfrecord_dataset(tfrecord_path, batch_size):
  """Loads a TFRecord dataset."""
  def _parse_function(example_proto):
    features = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    image = tf.io.decode_raw(parsed_features['image_raw'], tf.float16) 
    image = tf.reshape(image, (512, 512,3)) 
    label = parsed_features['label']
    return image, label

  dataset = tf.data.TFRecordDataset(tfrecord_path)
  dataset = dataset.map(_parse_function)
  dataset = dataset.shuffle(buffer_size=1000)
  dataset = dataset.batch(batch_size)
  return dataset



#%%

partition = np.load("/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/sessions/DATA/New_data/10%/new_data_10%.npy", allow_pickle=True).item()
labels = np.load("/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/sessions/DATA/New_data/10%/new_data_10%_labels.npy", allow_pickle=True).item()
MRI_PATH = ''
DTYPE = 'float16'
PARTITION = 'validation'

SCANIDS = [x.split('/')[-1][:31] for x in partition[PARTITION]]
healthy_scanIDs = list(set([x for x in SCANIDS if labels[x] == 0]))
cancer_scanIDs = [x for x in SCANIDS if labels[x] == 1]

cancer_slices = {x.split('/')[-1].split('.')[0][:31] : x.split('/')[-1].split('.')[0].split('_')[-1] for x in partition[PARTITION] if x.split('/')[-1][:31] in cancer_scanIDs}

from collections import defaultdict
healthy_slices = defaultdict(list)

[healthy_slices[x.split('/')[-1].split('.')[0][:31]].append(x.split('/')[-1].split('.')[0].split('_')[-1]) for x in partition[PARTITION] if x.split('/')[-1][:31] in healthy_scanIDs]


tfrecord_path = '/media/HDD/example_data_MSKCC_16-328/Mixed_shuffled_10%data.tfrecord'

cancer_images_for_tfrecord = np.empty((len(cancer_scanIDs) + 5639*3,1,512,512,3), dtype='float16')
# Make TFRecord for cancer slices, top50
for SUBJECT_INDEX in range(len(cancer_scanIDs)): #TOT
    print(SUBJECT_INDEX, len(cancer_scanIDs))
    scanID = cancer_scanIDs[SUBJECT_INDEX]
    patient = scanID[:20]
    exam = scanID[:29]
    side = 'right'
    contra_side = 'left'
    if scanID[-1] == 'l': 
        side = 'left'
        contra_side = 'right'
    #-------------------- Get paths of DCE-MRI. T1 if not normalized before  ---------------------------------
    print("####### IPSILATERAL ##############")
    all_subject_channels = [MRI_PATH + exam + '/T1_{}_02_01.nii'.format(side),
                           MRI_PATH + exam + '/T1_{}_slope1.nii'.format(side),
                           MRI_PATH + exam + '/T1_{}_slope2.nii'.format(side)]
    T1_pre_nii_path = MRI_PATH + exam + '/T1_{}_01_01.nii'.format(side)
    #-------------------- Load and preprocess DCE MRI  ---------------------------------
    X, shape = load_and_preprocess(all_subject_channels, T1_pre_nii_path=T1_pre_nii_path, side=side, imaging_protocol='sagittal', debug=False)
    if DTYPE == 'float16':
        X = X.astype(np.float16)
    #-------------------- Get pathology. If cancer, get slice number, else random slice ---------------------------------
    if scanID in cancer_slices.keys():
        PATHOLOGY = 'MALIGNANT'
        selected_slice = int(cancer_slices[scanID])
    #-------------------- Store Slices  ---------------------------------
    print("####### STORE ##############")

    cancer_images_for_tfrecord[SUBJECT_INDEX] = X[selected_slice:selected_slice+1]


cancer_labels = np.ones(shape=(len(cancer_scanIDs),), dtype=int)

write_to_tfrecord(cancer_images_for_tfrecord, cancer_labels, tfrecord_path) 






subjects_per_shard = 100
PATH = '/media/HDD/example_data_MSKCC_16-328/validation'

images = []
labels = []
SHARD_NR = 0
for SUBJECT_INDEX in range(len(healthy_scanIDs)): #TOT

    print(SUBJECT_INDEX)
    scanID = healthy_scanIDs[SUBJECT_INDEX]
    patient = scanID[:20]
    exam = scanID[:29]
    side = 'right'
    contra_side = 'left'
    if scanID[-1] == 'l': 
        side = 'left'
        contra_side = 'right'

    print("####### IPSILATERAL ##############")
    all_subject_channels = [MRI_PATH + exam + '/T1_{}_02_01.nii'.format(side),
                           MRI_PATH + exam + '/T1_{}_slope1.nii'.format(side),
                           MRI_PATH + exam + '/T1_{}_slope2.nii'.format(side)]
    T1_pre_nii_path = MRI_PATH + exam + '/T1_{}_01_01.nii'.format(side)

    X, shape = load_and_preprocess(all_subject_channels, T1_pre_nii_path=T1_pre_nii_path, side=side, imaging_protocol='sagittal', debug=False)
    if DTYPE == 'float16':
        X = X.astype(np.float16)

    if scanID in healthy_slices.keys():
        selected_slice = healthy_slices[scanID]
    #-------------------- Store Slices  ---------------------------------
    print("####### STORE ##############")
    # images.append(X)  Full Image storing.

    for sl in selected_slice:
        sl = int(sl)
        images.append(X[sl:sl+1])    
        labels.append(0)  

    if (SUBJECT_INDEX + 1) % subjects_per_shard == 0:
        tfrecord_path = f"{PATH}/validation_healthy_shard_{SHARD_NR}.tfrecord"
        write_to_tfrecord(images, labels, tfrecord_path)
        SHARD_NR += 1
        images = []
        labels = []

# Handle any remaining subjects
if images:
    tfrecord_path = f"{PATH}/validation_healthy_shard_{SHARD_NR}.tfrecord"
    write_to_tfrecord(images, labels, tfrecord_path) 


#%% BIG CHUNKS

from collections import defaultdict
import random

all_data_dict = defaultdict(list)


[all_data_dict[x.split('/')[-1].split('.')[0][:31]].append(x.split('/')[-1].split('.')[0].split('_')[-1]) for x in partition[PARTITION] if x.split('/')[-1][:31] in cancer_scanIDs]

[all_data_dict[x.split('/')[-1].split('.')[0][:31]].append(x.split('/')[-1].split('.')[0].split('_')[-1]) for x in partition[PARTITION] if x.split('/')[-1][:31] in healthy_scanIDs]

len(all_data_dict.keys())

len(all_data_dict.values())


all_scanIDs = cancer_scanIDs + healthy_scanIDs

random.shuffle(all_scanIDs)



subjects_per_shard = 1000
PATH = '/media/HDD/example_data_MSKCC_16-328/mixed_10%data'

import os
import pandas as pd
if not os.path.exists(PATH):
    os.mkdir(PATH)

shard_scanIDs = pd.DataFrame(columns=['scanID','slice','ytrue','shard'])
images = []
labels = []
SHARD_NR = 0
for SUBJECT_INDEX in range(len(all_scanIDs)): #TOT

    print(SUBJECT_INDEX)
    scanID = all_scanIDs[SUBJECT_INDEX]
    patient = scanID[:20]
    exam = scanID[:29]
    side = 'right'
    contra_side = 'left'
    if scanID[-1] == 'l': 
        side = 'left'
        contra_side = 'right'


    print("####### IPSILATERAL ##############")
    all_subject_channels = [MRI_PATH + exam + '/T1_{}_02_01.nii'.format(side),
                           MRI_PATH + exam + '/T1_{}_slope1.nii'.format(side),
                           MRI_PATH + exam + '/T1_{}_slope2.nii'.format(side)]
    T1_pre_nii_path = MRI_PATH + exam + '/T1_{}_01_01.nii'.format(side)

    X, shape = load_and_preprocess(all_subject_channels, T1_pre_nii_path=T1_pre_nii_path, side=side, imaging_protocol='sagittal', debug=False)
    if DTYPE == 'float16':
        X = X.astype(np.float16)

    #-------------------- Store Slices  ---------------------------------
    print("####### STORE ##############")

    for sl in all_data_dict[scanID]:
        sl = int(sl)
        images.append(X[sl:sl+1])    
        
        GT = int(scanID in cancer_scanIDs)
        
        labels.append(GT)  
        shard_scanIDs = shard_scanIDs.append({'scanID':scanID,'slice':sl,'ytrue':GT, 'shard':SHARD_NR}, ignore_index=True)
        
        

    if (SUBJECT_INDEX + 1) % subjects_per_shard == 0:
        
        c = list(zip(images, labels))
        
        random.shuffle(c)
        
        images, labels = zip(*c)

        
        tfrecord_path = f"{PATH}/mixed_10%data_shard_{SHARD_NR}.tfrecord"
        write_to_tfrecord(images, labels, tfrecord_path)
        SHARD_NR += 1
        images = []
        labels = []
        shard_scanIDs.to_csv(PATH + '/shard_information.csv', index=False)


# Handle any remaining subjects
if images:
    tfrecord_path = f"{PATH}/validation_healthy_shard_{SHARD_NR}.tfrecord"
    write_to_tfrecord(images, labels, tfrecord_path) 


#%% lOAD MULTIPLE FILES TOGETHER

import matplotlib.pyplot as plt
import time
import pandas as pd

tfrecord_healthy_0 = '/media/HDD/example_data_MSKCC_16-328/healthy_shard_0.tfrecord'
tfrecord_healthy_1 = '/media/HDD/example_data_MSKCC_16-328/healthy_shard_1.tfrecord'

tfrecord_cancers = '/media/HDD/example_data_MSKCC_16-328/cancers_269.tfrecord'

all_data = [tfrecord_healthy_0.replace('_0',f'_{i}') for i in range(191)]
all_data.append(tfrecord_cancers)



all_data = '/media/HDD/example_data_MSKCC_16-328/MSKCC_16-328_1_05260_20110926_r_20_debug.tfrecord'

dataset = load_tfrecord_dataset(all_data, batch_size=1)
# dataset = load_tfrecord_dataset([tfrecord_healthy_0, tfrecord_healthy_1, tfrecord_cancers], batch_size=8)
# dataset = load_tfrecord_dataset(tfrecord_cancers, batch_size=8)

# dataset = dataset.shuffle(buffer_size=1000) # No need. Dataset already has a shuffle on loading.

dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

class0 = []
class1 = []

for batch in dataset:
    images, slice_label = batch 
    slice_label = slice_label.numpy()
    n = len(slice_label)
    c = sum(slice_label)
    class0.append(n-c)
    class1.append(c)



img = images.numpy()

plt.imshow(img[0,:,:,0])

#%%

asd = np.load('/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/sessions/DATA/New_data/new_data.npy', allow_pickle=True).item()

[x for x in asd['validation'] if 'MSKCC_16-328_1_05260_20110926_r' in x]

scan = '/media/HDD/Diagnosis_2D_slices/X/MALIGNANT/MSKCC_16-328_1_05260_20110926_r_20.npy'

x = np.load(scan, allow_pickle=True)

plt.imshow(x[0,:,:,0])

np.std(img - x)

plt.imshow(x[0,:,:,2] - img[0,:,:,2])


##%

plt.plot(class1)
plt.plot(class0)

def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for batch in dataset:
            images, slice_label = batch 
            break
            time.sleep(0.02) # The longer the step inside, the better prefetch helps
    print("Execution time:", time.perf_counter() - start_time)
    

def inference_dataset(dataset, model, num_epochs=2):
    start_time = time.perf_counter()
    result_ytrue = []
    result_ypred = []
    for batch in dataset.take(2):
        
        images, slice_label = batch 
        ypred = model.predict(images)
        ypred = ypred[:,1]
        ytrue = slice_label.numpy()
        
        result_ytrue.extend(list(ytrue))
        result_ypred.extend(list(ypred))
        
    print("Execution time:", time.perf_counter() - start_time)
    return pd.DataFrame(zip(result_ytrue, result_ypred), columns=['ytrue','ypred'])
    

benchmark(dataset)

model = tf.compat.v1.keras.applications.ResNet50(
    include_top=True,
    weights= None,
    input_tensor=None,
    input_shape=(512,512,3),
    pooling=max,
    classes=2)

from utils import FocalLoss, FocalLoss_5_0, UNet_v0_2D_Classifier, load_and_preprocess
model = UNet_v0_2D_Classifier(input_shape =  (512,512,3), pool_size=(2, 2), 
                                 deconvolution=True, depth=6, n_base_filters=42,
                                 activation_name="softmax", L2=1e-5, USE_CLINICAL=False)

model.compile(loss=FocalLoss, optimizer=tf.keras.optimizers.Adam(lr=1e-5), metrics=['acc']) # FOcal loss has some problem with TFRecord
# I think it has to do with that TFRecord labels are just 0-1, not categorical... binary_crossentropy is probably coded to parse those variations


PRETRAINED_MODEL_WEIGHTS = '/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/sessions/10%_NewData_noClinical_classifier_train9833_val5892_DataAug_depth6_filters42_L21e-05_batchsize8/last_model_weights.npy'
weights = np.load(PRETRAINED_MODEL_WEIGHTS, allow_pickle=True)
model.set_weights(weights)


df_res = inference_dataset(dataset, model)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(lr=1e-5), metrics=['acc'])

history = model.fit(dataset, epochs=2, verbose=True)

plt.plot(history.history['loss'])
plt.plot(history.history['acc'])


model.weights

def show_batch_predictions(batch):
    plt.figure(figsize=(10, 10))
    for n in range(16):
        ax = plt.subplot(4,4, n + 1)
        img = batch[0][n].numpy()
        pred = model.predict(batch[0][n:n+1])
        plt.imshow(img[:,:,0])
        GT = batch[1][n].numpy()
        if GT == 0:
            color='green'
        else:
            color='red'
        PRED = np.round(pred[0,1],2)
        plt.title(PRED, color=color)
        plt.axis("off")

dataset = load_tfrecord_dataset([tfrecord_healthy_0, tfrecord_healthy_1, tfrecord_cancers], batch_size=25)

batch = next(iter(dataset))

ypred = model.predict(batch)

batch = next(iter(dataset))

show_batch_predictions(batch)

model.weights
