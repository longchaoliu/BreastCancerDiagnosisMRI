#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:51:45 2023

@author: deeperthought
"""


import numpy as np
import pandas as pd


df = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/Converts_1_year_follow_ups_and_segmentations.csv')

data_2d_dgns = np.load('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/SelectedSlices_and_Random_DGNS_train110464_val3150_DataAug_FocalLoss_depth6_filters42_L20.0001/Data.npy', allow_pickle=True).item()

global_partition = np.load('/home/deeperthought/Projects/DGNS/Risk_Prediction/Sessions/AddedSGMTDataTrain_OrderedBatch_PartitionBreasts_PartitionBySpreadSheet_Clinical2layersBehind_ClinicalOneHot_SmallNet2_Softmax_PrevalenceNatural_LR0.0001_FocalLoss_1000Epochs_SharedWeights_OnlyWithContralateral/dropout0.5_L10.0001_L20.0001_classWeight1.0/Global_Partition.npy', allow_pickle=True).item()


#%%

converts1yr = list(df['now'].values)

global_partition.keys()
data_2d_dgns.keys()

len(data_2d_dgns['train'])
len(data_2d_dgns['validation'])

dgns_train_scans = [x.split('/')[-1][:31] for x in data_2d_dgns['train']]

set(dgns_train_scans).intersection(set(converts1yr))
