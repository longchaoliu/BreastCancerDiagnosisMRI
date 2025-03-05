#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 12:49:55 2023

@author: deeperthought
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:27:12 2023

From: https://github.com/emla2805/vision-transformer/blob/master/model.py

https://stackoverflow.com/questions/67342988/verifying-the-implementation-of-multihead-attention-in-transformer


@author: deeperthought
"""



import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Add, Dense, Dropout, Embedding, GlobalAveragePooling1D, Input, Layer, LayerNormalization

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[3], True)
  except RuntimeError as e:
    # Visible devices must be set at program startup
    print(e)
    

#%% MULTIHEAD ATTENTION IMPLEMENTATION 2

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        # add the mask to the scaled tensor.
        if mask is not None: scaled_attention_logits += (mask * -1e9)
        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
        return output, attention_weights

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights


#%%
        

####################################################################################

# X = np.load('/home/deeperthought/kirbyPRO/Saggittal_Full_Slices/2D_slices/X/MSKCC_16-328_1_02842_20090925_l_29.npy', allow_pickle=True)
# X = X[0]

# CHANNELS = 3
# INPUT_SIZE = 64
# PATCH_SIZE = 16
# NUM_PATCHES = int((INPUT_SIZE*INPUT_SIZE)/(PATCH_SIZE*PATCH_SIZE))
# PROJECTION_DIM = int(PATCH_SIZE*PATCH_SIZE*CHANNELS)

# image = tf.image.resize(tf.convert_to_tensor(X), size=(INPUT_SIZE, INPUT_SIZE))
# plt.imshow(image[:,:,0])

# patches = PatchExtractor(patch_size=PATCH_SIZE)(np.expand_dims(image,0))
# patches.shape

# patches = patches.numpy()
# n = int(np.sqrt(patches.shape[1]))
# for i, patch in enumerate(patches[0]):
#     ax = plt.subplot(n, n, i + 1)
#     patch_img = np.reshape(patch, (PATCH_SIZE, PATCH_SIZE, 3))
#     plt.imshow(patch_img[:,:,0])
#     ax.axis("off")
    


# # 196 == (224*224)/(16*16)  # 196 patches if we tile a 224x224 image into 16x16

# # 768 == 16*16*3  # each tile will be flattened into an array of size 768 (16x16 for all 3 channels)

# embeddings = PatchEncoder(num_patches=NUM_PATCHES, projection_dim=PROJECTION_DIM)(patches)
# embeddings.shape



#%%


# class PatchExtractor(Layer):
#     def __init__(self, patch_size=16):
#         super(PatchExtractor, self).__init__()
#         self.patch_size = patch_size

#     def call(self, images):
#         batch_size = tf.shape(images)[0]
#         patches = tf.image.extract_patches(
#             images=images,
#             #sizes=[1, 16, 16, 1],
#             #strides=[1, 16, 16, 1],
            
#             sizes=[1, self.patch_size, self.patch_size, 1],
#             strides=[1, self.patch_size, self.patch_size, 1],
            
#             rates=[1, 1, 1, 1],
#             padding="VALID",
#         )
#         patch_dims = patches.shape[-1]
#         patches = tf.reshape(patches, [batch_size, -1, patch_dims])
#         return patches

# class PatchEncoder(Layer):
#     def __init__(self, num_patches=196, projection_dim=768): #196, 768
#         super(PatchEncoder, self).__init__()
#         self.num_patches = num_patches
#         self.projection_dim = projection_dim
#         w_init = tf.random_normal_initializer()
#         class_token = w_init(shape=(1, projection_dim), dtype="float32")
#         self.class_token = tf.Variable(initial_value=class_token, trainable=True)
#         self.projection = Dense(units=projection_dim)
#         self.position_embedding = Embedding(input_dim=num_patches+1, output_dim=projection_dim)

#     def call(self, patch):
#         batch = tf.shape(patch)[0]
#         # reshape the class token embedins
#         class_token = tf.tile(self.class_token, multiples = [batch, 1])
#         class_token = tf.reshape(class_token, (batch, 1, self.projection_dim))
#         # calculate patches embeddings
#         patches_embed = self.projection(patch)
#         patches_embed = tf.concat([patches_embed, class_token], 1)
#         # calcualte positional embeddings
#         positions = tf.range(start=0, limit=self.num_patches+1, delta=1)
#         positions_embed = self.position_embedding(positions)
#         # add both embeddings
#         encoded = patches_embed + positions_embed
#         return encoded


# class MLP(Layer):
#     def __init__(self, hidden_features, out_features, dropout_rate=0.1):
#         super(MLP, self).__init__()
#         self.dense1 = Dense(hidden_features, activation=tf.nn.relu)
#         self.dense2 = Dense(out_features)
#         self.dropout = Dropout(dropout_rate)

#     def call(self, x):
#         x = self.dense1(x)
#         x = self.dropout(x)
#         x = self.dense2(x)
#         y = self.dropout(x)
#         return y



# class Block(Layer):
#     def __init__(self, projection_dim, num_heads=4, dropout_rate=0.1):
#         super(Block, self).__init__()
#         self.norm1 = LayerNormalization(epsilon=1e-6)
#         self.attn = MultiHeadAttention(num_heads=num_heads, d_model=projection_dim)
#         self.norm2 = LayerNormalization(epsilon=1e-6)
#         self.mlp = MLP(projection_dim * 2, projection_dim, dropout_rate)

#     def call(self, x):
#         # Layer normalization 1.
#         x1 = self.norm1(x) # encoded_patches
#         # Create a multi-head attention layer.
#         attention_output, _ = self.attn(x1,x1,x1)
#         # Skip connection 1.
#         x2 = Add()([attention_output, x]) #encoded_patches
#         # Layer normalization 2.
#         x3 = self.norm2(x2)
#         # MLP.
#         x3 = self.mlp(x3)
#         # Skip connection 2.
#         y = Add()([x3, x2])
#         return y


# class TransformerEncoder(Layer):
#     def __init__(self, projection_dim, num_heads=4, num_blocks=12, dropout_rate=0.1):
#         super(TransformerEncoder, self).__init__()
#         self.blocks = [Block(projection_dim, num_heads, dropout_rate) for _ in range(num_blocks)]
#         self.norm = LayerNormalization(epsilon=1e-6)
#         self.dropout = Dropout(0.5)

#     def call(self, x):
#         # Create a [batch_size, projection_dim] tensor.
#         for block in self.blocks:
#             x = block(x)
#         x = self.norm(x)
#         y = self.dropout(x)
#         return y




# def create_VisionTransformer(num_classes, num_patches=196, projection_dim=768, input_shape=(224, 224, 3)):
#     inputs = Input(shape=input_shape)
#     # Patch extractor
#     patches = PatchExtractor()(inputs)
#     # Patch encoder
#     patches_embed = PatchEncoder(num_patches, projection_dim)(patches)
#     # Transformer encoder
#     representation = TransformerEncoder(projection_dim)(patches_embed)
#     representation = GlobalAveragePooling1D()(representation)
#     # MLP to classify outputs
#     logits = MLP(projection_dim, num_classes, 0.5)(representation)
#     # Create model
#     model = Model(inputs=inputs, outputs=logits)
#     return model

#%%

class PatchExtractor(Layer):
    def __init__(self, patch_size=16):
        super(PatchExtractor, self).__init__()
        self.patch_size = patch_size

    # NotImplementedError: Layers with arguments in `__init__` must override `get_config`.
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size
        })
        return config

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            #sizes=[1, 16, 16, 1],
            #strides=[1, 16, 16, 1],
            
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(Layer):
    def __init__(self, num_patches=196, projection_dim=768, d_model=32): #196, 768
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.d_model = d_model
        w_init = tf.random_normal_initializer()
        class_token = w_init(shape=(1, d_model), dtype="float32")
        self.class_token = tf.Variable(initial_value=class_token, trainable=True)
        self.projection = Dense(units=d_model)
        self.position_embedding = Embedding(input_dim=num_patches+1, output_dim=d_model)

    # NotImplementedError: Layers with arguments in `__init__` must override `get_config`.
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim, 
            'd_model': self.d_model,
            'class_token' : tf.Variable(initial_value=self.class_token, trainable=True),
            'projection': Dense(units=self.d_model),
            'position_embedding' : Embedding(input_dim=self.num_patches+1, output_dim=self.d_model)
        })
        return config

    def call(self, patch):
        batch = tf.shape(patch)[0]
        # reshape the class token embedins
        class_token = tf.tile(self.class_token, multiples = [batch, 1])
        class_token = tf.reshape(class_token, (batch, 1, self.d_model))
        # calculate patches embeddings
        patches_embed = self.projection(patch)
        patches_embed = tf.concat([patches_embed, class_token], 1)
        # calcualte positional embeddings
        positions = tf.range(start=0, limit=self.num_patches+1, delta=1)
        positions_embed = self.position_embedding(positions)
        # add both embeddings
        encoded = patches_embed + positions_embed
        return encoded


class MLP(Layer):
    def __init__(self, hidden_features, out_features, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate
        
        self.dense1 = Dense(hidden_features, activation=tf.nn.relu)
        self.dense2 = Dense(out_features)
        self.dropout = Dropout(dropout_rate)

    # NotImplementedError: Layers with arguments in `__init__` must override `get_config`.
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'hidden_features':self.hidden_features,
            'out_features':self.out_features,
            'dropout_rate':self.dropout_rate,
            
            'dense1': Dense(self.hidden_features, activation=tf.nn.relu),            
            'dense2': Dense(self.out_features),
            'dropout': Dropout(self.dropout_rate)
        })
        return config

    def call(self, x):
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        y = self.dropout(x)
        return y

class Block(Layer):
    def __init__(self, d_model, num_heads=4, dropout_rate=0.1):
        super(Block, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.attn = MultiHeadAttention(num_heads=num_heads, d_model=d_model)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(d_model * 2, d_model, dropout_rate)

    # NotImplementedError: Layers with arguments in `__init__` must override `get_config`.
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'd_model':self.d_model,
            'num_heads':self.num_heads,
            'dropout_rate':self.dropout_rate,
            'norm1': LayerNormalization(epsilon=1e-6),            
            'attn': MultiHeadAttention(num_heads=self.num_heads, d_model=self.d_model),
            'norm2': LayerNormalization(epsilon=1e-6),
            'mlp': MLP(self.d_model * 2, self.d_model, self.dropout_rate)
        })
        return config

    def call(self, x):
        # Layer normalization 1.
        x1 = self.norm1(x) # encoded_patches
        # Create a multi-head attention layer.
        attention_output, _ = self.attn(x1,x1,x1)
        # Skip connection 1.
        x2 = Add()([attention_output, x]) #encoded_patches
        # Layer normalization 2.
        x3 = self.norm2(x2)
        # MLP.
        x3 = self.mlp(x3)
        # Skip connection 2.
        y = Add()([x3, x2])
        return y

class TransformerEncoder(Layer):
    def __init__(self, d_model, num_heads=4, num_blocks=12, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
       
        self.blocks = [Block(d_model, num_heads, dropout_rate) for _ in range(num_blocks)]
        self.norm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(0.5)

    # NotImplementedError: Layers with arguments in `__init__` must override `get_config`.
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'd_model':self.d_model,
            'num_heads':self.num_heads,
            'dropout_rate':self.dropout_rate,
            'num_blocks': self.num_blocks,
            'blocks': [Block(self.d_model, self.num_heads, self.dropout_rate) for _ in range(self.num_blocks)],
            'norm': LayerNormalization(epsilon=1e-6),
            'dropout': Dropout(0.5)
        })
        return config


    def call(self, x):
        # Create a [batch_size, projection_dim] tensor.
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        y = self.dropout(x)
        return y

def create_VisionTransformer(num_classes, num_patches=196, projection_dim=768, d_model=32, patch_size=16, input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)
    # Patch extractor
    patches = PatchExtractor(patch_size)(inputs)
    # Patch encoder
    patches_embed = PatchEncoder(num_patches, projection_dim, d_model)(patches)
    # Transformer encoder
    representation = TransformerEncoder(d_model)(patches_embed)
    representation = GlobalAveragePooling1D()(representation)
    # MLP to classify outputs
    logits = MLP(projection_dim, num_classes, 0.5)(representation)
    # Create model
    model = Model(inputs=inputs, outputs=logits)#logits)
    return model

#%%

CHANNELS = 3
INPUT_SIZE = 128
PATCH_SIZE = 16
NUM_PATCHES = int((INPUT_SIZE*INPUT_SIZE)/(PATCH_SIZE*PATCH_SIZE))
PROJECTION_DIM = int(PATCH_SIZE*PATCH_SIZE*CHANNELS)
D_MODEL = 4

ViT = create_VisionTransformer(num_classes=2, input_shape=(INPUT_SIZE,INPUT_SIZE,CHANNELS), projection_dim=PROJECTION_DIM, d_model=D_MODEL, patch_size=PATCH_SIZE, num_patches=NUM_PATCHES)

ViT.input, ViT.output, ViT.summary()

X = np.load('/home/deeperthought/kirbyPRO/Saggittal_Full_Slices/2D_slices/X/MSKCC_16-328_1_02842_20090925_l_29.npy', allow_pickle=True)

X = tf.image.resize(tf.convert_to_tensor(X), size=(INPUT_SIZE, INPUT_SIZE))

yhat = ViT.predict(X)

ViT.save('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/ViT_Patch64_SmallerData1_classifier_train1022_val102_Clinical_depth6_filters42_L21e-05_batchsize8/vit_model.h5')


#%% Check every submodel works
    
CHANNELS = 3
INPUT_SIZE = 128
PATCH_SIZE = 16
NUM_PATCHES = int((INPUT_SIZE*INPUT_SIZE)/(PATCH_SIZE*PATCH_SIZE))
PROJECTION_DIM = int(PATCH_SIZE*PATCH_SIZE*CHANNELS)
D_MODEL = 4

X = np.load('/home/deeperthought/kirbyPRO/Saggittal_Full_Slices/2D_slices/X/MSKCC_16-328_1_02842_20090925_l_29.npy', allow_pickle=True)

X = X[0]

image = tf.image.resize(tf.convert_to_tensor(X), size=(INPUT_SIZE, INPUT_SIZE))

plt.imshow(image[:,:,0])

patches = PatchExtractor(patch_size=PATCH_SIZE)(np.expand_dims(image,0))
patches.shape



patches = patches.numpy()
n = int(np.sqrt(patches.shape[1]))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = np.reshape(patch, (PATCH_SIZE, PATCH_SIZE, 3))
    plt.imshow(patch_img[:,:,0])
    ax.axis("off")
    

embeddings = PatchEncoder(num_patches=NUM_PATCHES, projection_dim=PROJECTION_DIM, d_model=D_MODEL)(patches)
embeddings.shape

mlp = MLP(D_MODEL * 2, D_MODEL)
y = mlp(tf.zeros((1, NUM_PATCHES, D_MODEL)))
y.shape


block = Block(D_MODEL)
y = block(tf.zeros((1, NUM_PATCHES, D_MODEL)))
y.shape

transformer = TransformerEncoder(D_MODEL)
y = transformer(embeddings)
y.shape

#%%

CHANNELS = 3
INPUT_SIZE = 512
PATCH_SIZE = 64
NUM_PATCHES = int((INPUT_SIZE*INPUT_SIZE)/(PATCH_SIZE*PATCH_SIZE))
PROJECTION_DIM = int(PATCH_SIZE*PATCH_SIZE*CHANNELS)
D_MODEL = 4

# Visualize patches
X = np.load('/home/deeperthought/kirbyPRO/Saggittal_Full_Slices/2D_slices/X/MSKCC_16-328_1_02842_20090925_l_29.npy', allow_pickle=True)[0]
image = tf.image.resize(tf.convert_to_tensor(X), size=(INPUT_SIZE, INPUT_SIZE))
patches = PatchExtractor(patch_size=PATCH_SIZE)(np.expand_dims(image,0))
patches = patches.numpy()
n = int(np.sqrt(patches.shape[1]))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = np.reshape(patch, (PATCH_SIZE, PATCH_SIZE, 3))
    plt.imshow(patch_img[:,:,0], cmap='gray')
    ax.axis("off")
    
# Make model

ViT = create_VisionTransformer(num_classes=2, input_shape=(INPUT_SIZE,INPUT_SIZE,CHANNELS), projection_dim=PROJECTION_DIM,d_model=D_MODEL, num_patches=NUM_PATCHES)

ViT.input, ViT.output, ViT.summary()


#%%

X = np.load('/home/deeperthought/kirbyPRO/Saggittal_Full_Slices/2D_slices/X/MSKCC_16-328_1_02842_20090925_l_29.npy', allow_pickle=True)

X = tf.image.resize(tf.convert_to_tensor(X), size=(INPUT_SIZE, INPUT_SIZE))

yhat = ViT.predict(X)

ViT.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])  

yhat = ViT.predict(X)

ViT.evaluate(X,yhat)

ViT.fit(X,yhat, epochs=2)
