#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 17:00:40 2024

@author: deeperthought
"""


import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#%%

model = UNet_v0_2D_Classifier(input_shape =  (512,512,3), pool_size=(2, 2),initial_learning_rate=1e-5, 
                                         deconvolution=True, depth=6, n_base_filters=42,
                                         activation_name="softmax", L2=1e-5, USE_CLINICAL=True)

MODEL_WEIGHTS_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/best_model_weights.npy'

# MODEL_WEIGHTS_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/best_model_weights.npy'

loaded_weights = np.load(MODEL_WEIGHTS_PATH, allow_pickle=True, encoding='latin1')

model.set_weights(loaded_weights)


# Example usage (replace with your model, layer index, image, and hyperparameters)
learning_rate = 0.01
iterations = 200



# Convert image to a tensor
input_tensor = np.random.random((1,512,512,3))

MODE_CLINICAL = np.array([[0.  , 0.51, 0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  ]])


#%%

# The dimensions of our input image
img_width = 512
img_height = 512
# Our target layer: we will visualize the filters from this layer.
# See `model.summary()` for list of layer names, if you want to change this.
layer_name = model.layers[-3].name


layer = model.get_layer(name=layer_name)
feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=layer.output)


def compute_loss(input_image, filter_index):
    activation = feature_extractor([input_image, MODE_CLINICAL])
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    

    return tf.reduce_mean(filter_activation)


@tf.function
def gradient_ascent_step(img, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img


def initialize_image():
    # We start from a gray image with some random noise
    img = tf.random.uniform((1, img_width, img_height, 3))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return img #(img - 0.5) * 0.25


def visualize_filter(filter_index):
    # We run gradient ascent for 20 steps
    iterations = 20
    learning_rate = 10.0
    img = initialize_image()
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate)

    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    return loss, img


def deprocess_image(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img

#%%

filter_number = 0

loss, img = visualize_filter(filter_number)

plt.suptitle(f'Layer: {layer_name}, Filter: {filter_number}')
plt.subplot(221); plt.imshow(img[:,:,0],cmap='gray')
plt.subplot(222); plt.imshow(img[:,:,1],cmap='gray')
plt.subplot(223); plt.imshow(img[:,:,2],cmap='gray')
plt.subplot(224); plt.imshow(img)

# keras.utils.save_img("0.png", img)