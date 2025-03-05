#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:54:19 2023

@author: deeperthought
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

MASTER = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/DATA/Axials_pathology_assigned_partition.csv')
scans = os.listdir('/media/SD/Axial_Slices/X/')

scanids = [x[:31] for x in scans]

MASTER = MASTER.loc[MASTER['scanID'].isin(scanids)]

cancer = MASTER.loc[MASTER['pathology'] == 'Malignant', 'scanID'].values
healthy = MASTER.loc[MASTER['pathology'] == 'Benign', 'scanID'].values

img_cancer = [x for x in scans if x[:31] in cancer]


img_cancer_random = np.random.choice(img_cancer, replace=False, size=50)

for img in img_cancer_random:
    x = np.load('/media/SD/Axial_Slices/X/' + img, allow_pickle=True)
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.imshow(x[0,:,:,0], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(x[0,:,:,1], cmap='gray')
    
    
    plt.xlabel('CANCER')
    plt.savefig('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/DATA/axial_img/{}.png'.format(img.split('.npy')[0]))
    plt.close()
    
    
img_healthy = [x for x in scans if x[:31] in healthy]

img_healthy_random = np.random.choice(img_healthy, replace=False, size=50)

for img in img_healthy_random:
    x = np.load('/media/SD/Axial_Slices/X/' + img, allow_pickle=True)
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.imshow(x[0,:,:,0], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(x[0,:,:,1], cmap='gray')
    plt.xlabel('Healthy')
    plt.savefig('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/DATA/axial_img/{}.png'.format(img.split('.npy')[0]))
    plt.close()
    