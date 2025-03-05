#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:55:01 2025

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
        
import sys
sys.path.append('/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/code/')
from utils import load_and_preprocess
import numpy as np

from utils import FocalLoss, FocalLoss_5_0, UNet_v0_2D_Classifier, load_and_preprocess

import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import os


#%%

LOAD_PRETRAINED_MODEL = False

BATCH_SIZE = 8

TRAIN_DATA_PATH = '/media/HDD/example_data_MSKCC_16-328/training_10%_data/'

all_data = [TRAIN_DATA_PATH + x for x in os.listdir(TRAIN_DATA_PATH)]


#%% FUNCTIONS

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
    label = tf.one_hot(label, depth=2)  # Added to see if this works with my custom focal loss
    return image, label

  dataset = tf.data.TFRecordDataset(tfrecord_path)
  dataset = dataset.map(_parse_function)
  dataset = dataset.shuffle(buffer_size=1000)
  dataset = dataset.batch(batch_size)
  return dataset



def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for batch in dataset:
            images, slice_label = batch 
            break
            time.sleep(0.02) # The longer the step inside, the better prefetch helps
    print("Execution time:", time.perf_counter() - start_time)
    
def inference_dataset(dataset, model):
    start_time = time.perf_counter()
    result_ytrue = []
    result_ypred = []
    for batch in dataset:
        
        images, slice_label = batch 
        ypred = model.predict(images)
        ypred = ypred[:,1]
        ytrue = slice_label.numpy()[:,1]
        
        result_ytrue.extend(list(ytrue))
        result_ypred.extend(list(ypred))
        
    print("Execution time:", time.perf_counter() - start_time)
    return pd.DataFrame(zip(result_ytrue, result_ypred), columns=['ytrue','ypred'])
    

def show_batch_predictions(batch):
    plt.figure(figsize=(10, 10))
    for n in range(batch[0].shape[0]):
        ax = plt.subplot(4,4, n + 1)
        img = batch[0][n].numpy()
        pred = model.predict(batch[0][n:n+1])
        plt.imshow(img[:,:,0])
        GT = batch[1][n][1].numpy()
        if GT == 0:
            color='green'
        else:
            color='red'
        PRED = np.round(pred[0,1],2)
        plt.title(PRED, color=color)
        plt.axis("off")

#%% LOAD DATASET

dataset = load_tfrecord_dataset(all_data, batch_size=BATCH_SIZE)

dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)


#%% INSPECT DATASET

# class0 = []
# class1 = []

# for batch in dataset:
#     images, slice_label = batch 
#     slice_label = slice_label.numpy()
#     n = len(slice_label)
#     c = sum(slice_label)
#     class0.append(n-c)
#     class1.append(c)

# plt.plot(class1)
# plt.plot(class0)

# benchmark(dataset)

#%% INFERENCE 

model = UNet_v0_2D_Classifier(input_shape =  (512,512,3), pool_size=(2, 2), 
                                  deconvolution=True, depth=6, n_base_filters=42,
                                  activation_name="softmax", L2=1e-5, USE_CLINICAL=False)

if LOAD_PRETRAINED_MODEL:
    
    PRETRAINED_MODEL_WEIGHTS = '/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/sessions/10%_NewData_noClinical_classifier_train9833_val5892_DataAug_depth6_filters42_L21e-05_batchsize8/last_model_weights.npy'
    weights = np.load(PRETRAINED_MODEL_WEIGHTS, allow_pickle=True)
    model.set_weights(weights)



df_res = inference_dataset(dataset, model)

auc = roc_auc_score(df_res['ytrue'], df_res['ypred'])

plt.figure(1)
plt.title('performance on train data before training')
plt.hist(df_res.loc[df_res['ytrue']==0,'ypred'], color='green', alpha=0.6, bins=70)
plt.hist(df_res.loc[df_res['ytrue']==1,'ypred'], color='red', alpha=0.6, bins=70)
plt.xlabel(f'AUC={round(auc,3)}')
plt.yscale('log')


# fpr, tpr, thr = roc_curve(df_res['ytrue'], df_res['ypred'])



#%% TRAIN


model.compile(loss=FocalLoss, optimizer=tf.keras.optimizers.Adam(lr=1e-5), metrics=['acc']) 

'''
Still missing:
    - validation data
    - data augmentation
    - checkpoints
    
Only then I can properly benchmark with previous method (data generator)

'''

history = model.fit(dataset, epochs=5, verbose=True)
 

plt.figure(2)
plt.subplot(121); plt.title('loss')
plt.plot(history.history['loss'])
plt.subplot(122); plt.title('acc')
plt.plot(history.history['acc'])


val_tfrecord_cancers = "/media/HDD/example_data_MSKCC_16-328/validation/validation_cancers.tfrecord"
val_tfrecord_healthy_0 = "/media/HDD/example_data_MSKCC_16-328/validation/validation_healthy_shard_0.tfrecord"
dataset = load_tfrecord_dataset([val_tfrecord_healthy_0, val_tfrecord_cancers], batch_size=16)

batch = next(iter(dataset))

show_batch_predictions(batch)



df_res = inference_dataset(dataset, model)

auc = roc_auc_score(df_res['ytrue'], df_res['ypred'])

plt.figure(1)
plt.title('performance on train data before training')
plt.hist(df_res.loc[df_res['ytrue']==0,'ypred'], color='green', alpha=0.6, bins=70)
plt.hist(df_res.loc[df_res['ytrue']==1,'ypred'], color='red', alpha=0.6, bins=70)
plt.xlabel(f'AUC={round(auc,3)}')
plt.yscale('log')

