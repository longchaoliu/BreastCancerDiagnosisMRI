#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:55:01 2025

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
        
import sys
sys.path.append('/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/code/')

import numpy as np
from utils import UNet_v0_2D_Classifier, FocalLoss #, FocalLoss_5_0, load_and_preprocess
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# from functools import partial
import os

from train_utils import my_model_checkpoint, MyHistory


#%%


BATCH_SIZE = 8
EPOCHS = 50

TRAIN_DATA_PATH = '/media/HDD/example_data_MSKCC_16-328/training_10%_data/'

VAL_DATA_PATH = '/media/HDD/example_data_MSKCC_16-328/validation/'

OUTPUT_PATH = '/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/sessions/'

NAME = 'tfrecords_10%data_normalPrevalenc_shuffledData'

LOAD_PRETRAINED_MODEL = False

DEBUG_DATA_LOADING = False

DATA_AUGMENTATION = True

SMALL_DATA = False

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

# def load_tfrecord_dataset(tfrecord_path, batch_size=8):
#   """Loads a TFRecord dataset."""
#   def _parse_function(example_proto):
#     features = {
#         'image_raw': tf.io.FixedLenFeature([], tf.string),
#         'label': tf.io.FixedLenFeature([], tf.int64)
#     }
#     parsed_features = tf.io.parse_single_example(example_proto, features)
#     image = tf.io.decode_raw(parsed_features['image_raw'], tf.float16) 
#     image = tf.reshape(image, (512, 512,3))     
#     label = parsed_features['label']
#     label = tf.one_hot(label, depth=2)  # Added to see if this works with my custom focal loss
#     return image, label

#   dataset = tf.data.TFRecordDataset(tfrecord_path)
#   dataset = dataset.map(_parse_function)
#   dataset = dataset.shuffle(buffer_size=1000)
#   dataset = dataset.batch(batch_size)
#   return dataset



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
        plt.subplot(4,4, n + 1)
        img = batch[0][n].numpy()
        pred = model.predict(batch[0][n:n+1])
        plt.imshow(np.rot90(img[:,:,0]), cmap='gray')
        GT = batch[1][n][1].numpy()
        if GT == 0:
            color='green'
        else:
            color='red'
        PRED = np.round(pred[0,1],2)
        plt.title(PRED, color=color)
        plt.axis("off")


def _parse_function(example_proto, augment=False):
  features = {
      'image_raw': tf.io.FixedLenFeature([], tf.string),
      'label': tf.io.FixedLenFeature([], tf.int64)
  }
  parsed_features = tf.io.parse_single_example(example_proto, features)  # why parse_single_example??
  image = tf.io.decode_raw(parsed_features['image_raw'], tf.float16) 
  image = tf.reshape(image, (512, 512,3)) 
  
  if augment:
    image = tf.image.random_brightness(image, max_delta=0.2)   # DID THIS RUIN TRAINING?
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # DID THIS TOO?  
    # image = tf.image.rot90(image, tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32))
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_flip_up_down(image)
    # image = tf.image.shear_ranges(image, shear_factor=0.1)
  
  label = parsed_features['label']
  label = tf.one_hot(label, depth=2)  # Added to see if this works with my custom focal loss
  return image, label

def load_dataset(filenames, augment=True):
  ignore_order = tf.data.Options()
  ignore_order.experimental_deterministic = False 
  dataset = tf.data.TFRecordDataset(filenames)
  

  dataset = dataset.with_options(ignore_order)
  
  dataset = dataset.map(lambda x: _parse_function(x, augment), num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
  return dataset


def get_train_dataset(filenames, BATCH_SIZE=8):
  dataset = load_dataset(filenames, augment=True)    # False: 23.39 s. True: 30.5s
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  dataset = dataset.batch(BATCH_SIZE)
  dataset = dataset.shuffle(2048)  # shuffling before batching mixes element-wise

  # dataset = dataset.repeat() # turn ON after tests

  return dataset

def get_validation_dataset(filenames, BATCH_SIZE=16):
  dataset = load_dataset(filenames, augment=False) 
 
    
  # dataset = dataset.shuffle(256) #  16s
  # dataset = dataset.shuffle(2048) # 19 s
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # 19s

  dataset = dataset.batch(BATCH_SIZE)
  
  dataset = dataset.shuffle(1024) #  21s better mix, although mix is of batches.

  # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # 51.637s
  # dataset = dataset.repeat()

  return dataset

def focal_loss_one_hot(gamma=2.0, alpha=0.25):
  """
  Focal Loss for binary classification with one-hot encoded labels.

  Args:
    gamma: A scalar for focusing parameter.
    alpha: A scalar for weight balancing between positive and negative examples.

  Returns:
    A callable loss function.
  """
  def focal_loss_fixed(y_true, y_pred):
    """
    Calculates the Focal Loss.

    Args:
      y_true: Ground truth labels (one-hot encoded), shape: (batch_size, 2).
      y_pred: Predicted probabilities, shape: (batch_size, 2).

    Returns:
      The computed Focal Loss.
    """
    # Ensure y_pred is in the valid range (0, 1)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    # Calculate cross-entropy
    cross_entropy = -y_true * tf.math.log(y_pred) 

    # Calculate modulating factor
    p_t = tf.reduce_sum(y_true * y_pred, axis=-1)  # Extract probabilities for the true class
    modulating_factor = (1.0 - p_t) ** gamma

    # Calculate weighted loss
    weighted_loss = alpha * y_true * modulating_factor * cross_entropy 

    return tf.reduce_mean(weighted_loss)

  return focal_loss_fixed
# # This was working before I added data augmnetation
# def load_dataset(filenames):
#     ignore_order = tf.data.Options()
#     ignore_order.experimental_deterministic = False  # disable order, increase speed
#     dataset = tf.data.TFRecordDataset(filenames)  # automatically interleaves reads from multiple files
#     dataset = dataset.with_options(ignore_order)  # uses data as soon as it streams in, rather than in its original order
#     dataset = dataset.map(partial(_parse_function), num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
#     return dataset


# def get_dataset(filenames):
#     dataset = load_dataset(filenames)
#     dataset = dataset.shuffle(2048)
#     dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#     dataset = dataset.batch(8)
#     return dataset

#%% LOAD DATASET

TRAINING_FILENAMES = tf.io.gfile.glob(TRAIN_DATA_PATH + "*.tfrecord")
VALIDATION_FILENAMES = tf.io.gfile.glob(VAL_DATA_PATH + "*.tfrecord")

print("Train TFRecord Files:", len(TRAINING_FILENAMES))
print("Validation TFRecord Files:", len(VALIDATION_FILENAMES))


# VALIDATION_FILENAMES.extend(['/media/HDD/example_data_MSKCC_16-328/validation/validation_cancers.tfrecord']*20)
# TRAINING_FILENAMES.extend(['/media/HDD/example_data_MSKCC_16-328/training_10%_data/cancers_269.tfrecord']*20)

# import random
# random.shuffle(VALIDATION_FILENAMES)
# random.shuffle(TRAINING_FILENAMES)



if DATA_AUGMENTATION:
    train_dataset = get_train_dataset(TRAINING_FILENAMES, BATCH_SIZE=BATCH_SIZE) 

else:
    train_dataset = get_validation_dataset(TRAINING_FILENAMES, BATCH_SIZE=BATCH_SIZE) # REMOVE DATA AUG TO CHECK IF THAT RUINED TRAINING



train_dataset = get_train_dataset(TRAINING_FILENAMES, BATCH_SIZE=BATCH_SIZE) 

validation_dataset = get_validation_dataset(VALIDATION_FILENAMES, BATCH_SIZE=16)


if DEBUG_DATA_LOADING:
    images, label_batch = next(iter(train_dataset))
    plt.figure(figsize=(10, 10))
    for n in range(images.shape[0]):
        plt.subplot(4,4, n + 1)
        img = images[n].numpy()
        plt.imshow(np.rot90(img[:,:,0]), cmap='gray')

    images, label_batch = next(iter(validation_dataset))
    plt.figure(figsize=(10, 10))
    for n in range(images.shape[0]):
        plt.subplot(4,4, n + 1)
        img = images[n].numpy()
        plt.imshow(np.rot90(img[:,:,0]), cmap='gray')


if SMALL_DATA:
    train_dataset = train_dataset.take(1500)
    validation_dataset = validation_dataset.take(1000)



#%% INSPECT DATASET





# validation_dataset = get_validation_dataset(VALIDATION_FILENAMES, BATCH_SIZE=16)

# train_dataset = get_train_dataset(TRAINING_FILENAMES, BATCH_SIZE=BATCH_SIZE) 

# class0 = [0]
# class1 = [0]


# start_time = time.perf_counter()


# for batch in train_dataset:
#     images, slice_label = batch 
#     slice_label = slice_label.numpy()
#     slice_label = [x[1] for x in slice_label]
#     n = len(slice_label)
#     c = sum(slice_label)
#     class0.append(class0[-1] + n-c)
#     class1.append(class1[-1] + c)

# print("Execution time:", time.perf_counter() - start_time)


# plt.plot(class1, label='class 1')
# plt.legend()

# benchmark(dataset)


# TRAINING_FILENAMES = '/media/HDD/example_data_MSKCC_16-328/MSKCC_16-328_1_05260_20110926_r_20_debug.tfrecord'
# train_dataset = get_validation_dataset(TRAINING_FILENAMES, BATCH_SIZE=BATCH_SIZE) 
# images, label_batch = next(iter(train_dataset))

# img = images.numpy()
# plt.imshow(img[0,:,:,0])

# label_batch.numpy()


# asd = np.load('/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/sessions/DATA/New_data/new_data.npy', allow_pickle=True).item()

# [x for x in asd['validation'] if 'MSKCC_16-328_1_05260_20110926_r' in x]

# scan = '/media/HDD/Diagnosis_2D_slices/X/MALIGNANT/MSKCC_16-328_1_05260_20110926_r_20.npy'

# x = np.load(scan, allow_pickle=True)

# plt.imshow(x[0,:,:,0])

# np.std(img - x)

# np.max(x)
# np.max(img)

# plt.imshow(x[0,:,:,0] - img[0,:,:,0])


#%% INFERENCE 

# Enable mixed precision
# policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
# tf.keras.mixed_precision.experimental.set_policy(policy)

model = UNet_v0_2D_Classifier(input_shape =  (512,512,3), pool_size=(2, 2), 
                                  deconvolution=True, depth=6, n_base_filters=42,
                                  activation_name="softmax", L2=1e-5, USE_CLINICAL=False)

if LOAD_PRETRAINED_MODEL:
    
    PRETRAINED_MODEL_WEIGHTS = '/home/deeperthought/Projects/Diagnosis_breast_cancer_MRI_github/develop/sessions/tfrecords_10%data_NODataAug_repeat/best_model_weights.npy'
    weights = np.load(PRETRAINED_MODEL_WEIGHTS, allow_pickle=True)
    model.set_weights(weights)


# df_res = inference_dataset(validation_dataset.take(100), model)

# auc = roc_auc_score(df_res['ytrue'], df_res['ypred'])

# plt.figure(1)
# plt.title('performance on val data before training')
# plt.hist(df_res.loc[df_res['ytrue']==0,'ypred'], color='green', alpha=0.6, bins=70)
# plt.hist(df_res.loc[df_res['ytrue']==1,'ypred'], color='red', alpha=0.6, bins=70)
# plt.xlabel(f'AUC={round(auc,3)}')
# plt.yscale('log')


# fpr, tpr, thr = roc_curve(df_res['ytrue'], df_res['ypred'])



#%% TRAIN

'''
Still missing:
    - validation data    DONE
    - data augmentation  DONE
    - checkpoints
    
Only then I can properly benchmark with previous method (data generator)

'''

model.compile(loss=FocalLoss, optimizer=tf.keras.optimizers.Adam(lr=1e-5), metrics=['acc']) 

my_custom_checkpoint = my_model_checkpoint(MODEL_PATH=OUTPUT_PATH+NAME, MODEL_NAME='/best_model' )
Custom_History = MyHistory(OUTPUT_PATH, NAME)

csv_logger = tf.keras.callbacks.CSVLogger(OUTPUT_PATH+NAME + '/csvLogger.log', 
                                     separator=',', 
                                     append=True)


myEarlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                min_delta=0, 
                                                patience=35, 
                                                verbose=1, 
                                                mode='min', 
                                                baseline=None, 
                                                restore_best_weights=False)
if not os.path.exists(OUTPUT_PATH+NAME):
    os.mkdir(OUTPUT_PATH+NAME)


history = model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, verbose=True, 
                    steps_per_epoch= 3616, 
                    validation_steps= 1074, # actually should be 1074 ...
                    callbacks=[Custom_History, csv_logger, my_custom_checkpoint, myEarlyStop])
 

plt.figure(2)
plt.subplot(121); plt.title('loss')
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()

plt.subplot(122); plt.title('acc')
plt.plot(history.history['acc'], label='train acc')
plt.plot(history.history['val_acc'], label='val acc')
plt.legend()



# model.evaluate(validation_dataset.take(537))

df_res = inference_dataset(validation_dataset, model)
auc = roc_auc_score(df_res['ytrue'], df_res['ypred'])
plt.figure(3)
plt.title('performance on val data after training')
plt.hist(df_res.loc[df_res['ytrue']==0,'ypred'], label=f"benigns={len(df_res.loc[df_res['ytrue']==0,'ypred'])}", color='green', alpha=0.6, bins=70)
plt.hist(df_res.loc[df_res['ytrue']==1,'ypred'], label=f"cancer={len(df_res.loc[df_res['ytrue']==1,'ypred'])}", color='red', alpha=0.6, bins=70)
plt.xlabel(f'AUC={round(auc,3)}')
plt.yscale('log')
plt.legend()

plt.savefig(OUTPUT_PATH+NAME+'/val_result.png')



df_res = inference_dataset(train_dataset.take(1000), model) # I need to use take i think when adding .repeat() to dataset
auc = roc_auc_score(df_res['ytrue'], df_res['ypred'])
plt.figure(4)
plt.title('performance on train data after training')
plt.hist(df_res.loc[df_res['ytrue']==0,'ypred'], label=f"benigns={len(df_res.loc[df_res['ytrue']==0,'ypred'])}", color='green', alpha=0.6, bins=70)
plt.hist(df_res.loc[df_res['ytrue']==1,'ypred'], label=f"cancer={len(df_res.loc[df_res['ytrue']==1,'ypred'])}", color='red', alpha=0.6, bins=70)
plt.xlabel(f'AUC={round(auc,3)}')
plt.yscale('log')
plt.legend()

plt.savefig(OUTPUT_PATH+NAME+'/train_result_1000batches.png')



batch = next(iter(validation_dataset.take(1)))
show_batch_predictions(batch)