#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:24:11 2023

@author: deeperthought
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

Axial_slices = os.listdir('/media/SD/Axial_Slices/X/')

PATHOLOGY_AXIAL = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/REDCAP/2023/Axials_pathology_assigned_partition.csv')
blacklisted = pd.read_csv('/home/deeperthought/kirbyPRO/Blacklisted_Axial_Segmented_Images_alignedNiiAxial-May2020-cropped-normed.csv')

print('{} images in total'.format(len(Axial_slices)))

axial_scanID_slices = list(set([x[:31] for x in Axial_slices]))

axial_scanID_slices = [ x for x in axial_scanID_slices if x not in blacklisted['scanID'].values]

print('Slices from {} Axial scanIDs\n'.format(len(axial_scanID_slices)))

print(PATHOLOGY_AXIAL.loc[PATHOLOGY_AXIAL['scanID'].isin(axial_scanID_slices), 'partition'].value_counts())

print(PATHOLOGY_AXIAL.loc[PATHOLOGY_AXIAL['scanID'].isin(axial_scanID_slices), 'pathology'].value_counts())




extracted_cancer_slices = PATHOLOGY_AXIAL.loc[(PATHOLOGY_AXIAL['scanID'].isin(axial_scanID_slices))*(PATHOLOGY_AXIAL['pathology'] == 'Malignant')]

cancer_scanIDs = list(set(extracted_cancer_slices['scanID'].values))

cancer_images = [x for x in Axial_slices if x[:31] in cancer_scanIDs]
healthy_images = [x for x in Axial_slices if x[:31] not in cancer_scanIDs]

cancer_images.sort()
healthy_images.sort()


sampled_cancer_images = []
for scanid in cancer_scanIDs:
    scanid_images = [x for x in cancer_images if x[:31] == scanid]
    scanid_images.sort()
    slice_numbers = [int(x.split('_')[-1].split('.')[0]) for x in scanid_images]
    slice_numbers.sort()
    everyth_nth_slice = scanid_images[::10]
    sampled_cancer_images.extend(everyth_nth_slice)

' REMOVE BLACKLISTED !!'

'KEEP ONLY EVERY 10th SLICE !!!'


cancer_images = sampled_cancer_images

#%%  Make partitions

len(healthy_images), len(cancer_images)

healthy_patients = list(set([x[:20] for x in healthy_images]))
cancer_patients = list(set([x[:20] for x in cancer_images]))

len(healthy_patients), len(cancer_patients)

N_val_healthy = int(len(healthy_patients)*0.1)
N_val_cancer = int(len(cancer_patients)*0.1)

val_patients_benigns = np.random.choice(healthy_patients, size=N_val_healthy, replace=False)
val_patients_cancer = np.random.choice(cancer_patients, size=N_val_cancer, replace=False)

train_patients_benigns = [x for x in healthy_patients if x not in val_patients_benigns]
train_patients_cancer = [x for x in cancer_patients if x not in val_patients_cancer]

val_healthy_images = [x for x in healthy_images if x[:20] in val_patients_benigns]
val_cancer_images = [x for x in cancer_images if x[:20] in val_patients_cancer]


train_healthy_images = [x for x in healthy_images if x[:20] in train_patients_benigns]
train_cancer_images = [x for x in cancer_images if x[:20] in train_patients_cancer]

len(val_healthy_images), len(val_cancer_images)
len(train_healthy_images), len(train_cancer_images)

set(train_healthy_images).intersection(set(val_healthy_images))
set(train_cancer_images).intersection(set(val_cancer_images))

train_images = train_cancer_images + train_healthy_images
train_scanIDs = [x[:31] for x in train_images]

val_images = val_cancer_images + val_healthy_images
val_scanIDs = [x[:31] for x in val_images]

PATHOLOGY_AXIAL.loc[PATHOLOGY_AXIAL['scanID'].isin(train_scanIDs), 'pathology'].value_counts()
PATHOLOGY_AXIAL.loc[PATHOLOGY_AXIAL['scanID'].isin(val_scanIDs), 'pathology'].value_counts()

partition_axial = {'train':train_images, 'validation':val_images, 'test':[]}

'''
Just missing some test exams. Just gather from PATHOLOGY_AXIAL the marked in TEST partition.

Gather like 1000 exams and thats it

'''

PATHOLOGY_AXIAL.loc[PATHOLOGY_AXIAL['partition'] == 'Test', 'pathology'].value_counts()

all_test_cancer = PATHOLOGY_AXIAL.loc[(PATHOLOGY_AXIAL['partition'] == 'Test')*(PATHOLOGY_AXIAL['pathology'] == 'Malignant'), 'scanID'].values
all_test_healthy = PATHOLOGY_AXIAL.loc[(PATHOLOGY_AXIAL['partition'] == 'Test')*(PATHOLOGY_AXIAL['pathology'] == 'Benign'), 'scanID'].values

len(all_test_cancer), len(all_test_healthy)


test_cancer_subjects = list(set([x[:20] for x in all_test_cancer]))
test_healthy_subjects = list(set([x[:20] for x in all_test_healthy]))

len(test_cancer_subjects), len(test_healthy_subjects)

selected_test_benigns = np.random.choice(all_test_healthy, size=1000, replace=False)
selected_test_cancer = np.random.choice(all_test_cancer, size=250, replace=False)

len(selected_test_cancer), len(selected_test_benigns)

test_selected = list(selected_test_cancer) + list(selected_test_benigns)

partition_axial['test'] = test_selected


axial_label = PATHOLOGY_AXIAL[['scanID','pathology']]

axial_label['pathology'] = (axial_label['pathology'] == 'Malignant').astype(int)

axial_label_dict = dict(zip(axial_label.scanID.values,axial_label.pathology.values))


np.save('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/DATA/Axial_Data.npy', partition_axial)
np.save('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/DATA/Axial_Labels.npy', axial_label_dict)


#partition_axial['train'].remove('MSKCC_16-328_1_01051_20140412_r_80.npy')

[x for x in partition_axial['train'] if x[:31] not in axial_label_dict.keys()]
[x for x in partition_axial['validation'] if x[:31] not in axial_label_dict.keys()]
[x for x in partition_axial['test'] if x[:31] not in axial_label_dict.keys()]

#[PATHOLOGY_AXIAL.loc[PATHOLOGY_AXIAL['scanID'] == x[:31]] for x in partition_axial['train']]
#
#PATHOLOGY_AXIAL.loc[PATHOLOGY_AXIAL['scanID'] == 'MSKCC_16-328_1_01051_20140412_r']
#                    
#PATHOLOGY_AXIAL
#%%


def add_age(df, clinical):
  ages = clinical['Unnamed: 1_level_0']
  ages['DE-ID']  = clinical['Unnamed: 0_level_0']['DE-ID']
  #ages['DE-ID'] = ages.index
  ages.reset_index(level=0, inplace=True)
  ages = ages[['DE-ID','DOB']]  
  df2 = df.copy()
  df2['ID_date'] = df2['scan_ID'].apply(lambda x : x[:-2])
  df2['DE-ID'] = df2['ID_date'].apply(lambda x : x[:-9]) 
  df3 = df2.merge(ages, on=['DE-ID'])
  df3.head()
  df3['Age'] = df3.apply(lambda row : int(row['ID_date'][-8:-4]) - int(row['DOB']), axis=1)
  df3 = df3[['scan_ID','Age']]
  df4 = df3.merge(df, on=['scan_ID'])
  df4 = df4.loc[df4['scan_ID'].isin(df['scan_ID'])]
  return df4

def add_ethnicity_oneHot(df, clinical):
 
  clinical_df = pd.concat([clinical['Unnamed: 0_level_0']['DE-ID'], clinical['Unnamed: 4_level_0']['ETHNICITY'], clinical['Unnamed: 3_level_0']['RACE']], axis=1)
  
  clinical_df = clinical_df.set_index('DE-ID')
  clinical_df.loc[clinical_df['ETHNICITY'] == 'NO VALUE ENTERED'] = 'UNKNOWN'  
  clinical_df.loc[clinical_df['RACE'] == 'OTHER'] = 'UNKNOWN'  
  clinical_df.loc[clinical_df['RACE'] == 'PT REFUSED TO ANSWER'] = 'UNKNOWN'  
  clinical_df.loc[clinical_df['RACE'] == 'NO VALUE ENTERED'] = 'UNKNOWN'  

  
  clinical_df = pd.get_dummies(clinical_df)

  df['DE-ID'] = df['scan_ID'].str[:20] 
  
  df2 =  pd.merge(df, clinical_df, on='DE-ID')

  return df2


def add_family_hx(df, clinical):
  fam = pd.DataFrame(columns=['DE-ID','Family Hx'])
  fam['Family Hx'] = clinical['Family Hx']['Family Hx']
  fam['Family Hx'] = fam['Family Hx'].apply(lambda x : 1 if x == 'Yes' else 0)
  fam['DE-ID'] = clinical['Unnamed: 0_level_0']['DE-ID']
  fam.reset_index(level=0, inplace=True) 
  df2 = df.copy()
  df2['ID_date'] = df2['scan_ID'].apply(lambda x : x[:-2])
  df2['DE-ID'] = df2['ID_date'].apply(lambda x : x[:-9]) 
  df3 = df2.merge(fam, on=['DE-ID'])
  df3.head()
  df3 = df3[['scan_ID','Family Hx']]
  df4 = df3.merge(df, on=['scan_ID'])
  df4 = df4.loc[df4['scan_ID'].isin(df['scan_ID'])]
  return df4


#%% add clinical ifno

clnical_sag = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/DATA/Clinical_Data_Train_Val.csv')

clnical_sag.columns

REDCAP_dem = pd.read_csv('/home/deeperthought/Projects/MSKCC_Data_Organization/data/REDCAP/2023/demographics/20425MachineLearning_DATA_2023-05-11_1234.csv')

REDCAP_dem.columns

REDCAP_dem['ethnicity_demog'].value_counts()
REDCAP_dem['race_demog'].value_counts()
REDCAP_dem['date_of_birth_demog'].value_counts()
REDCAP_dem['family_history'].value_counts()

#%%

clinical_df = pd.read_excel('/home/deeperthought/Projects/MSKCC/MSKCC/Data_spreadsheets/Diamond_and_Gold/CCNY_CLINICAL_4_17_2019.xlsx', header=[0,1])    
X = list(set(PATHOLOGY_AXIAL['scanID'].values))

scanID = [x.split('/')[-1][:31] for x in X] 
clinical_features = pd.DataFrame(columns=['scan_ID','X'])
clinical_features['scan_ID'] = scanID
clinical_features['X'] = X
clinical_features = add_age(clinical_features, clinical_df)
clinical_features = add_ethnicity_oneHot(clinical_features, clinical_df)
clinical_features = add_family_hx(clinical_features, clinical_df)
clinical_features['Age'] = clinical_features['Age']/100.
clinical_features = clinical_features.drop_duplicates()
CLINICAL_FEATURE_NAMES = [u'Family Hx',u'Age',u'ETHNICITY_HISPANIC OR LATINO',u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',u'RACE_ASIAN-FAR EAST/INDIAN SUBCONT',u'RACE_BLACK OR AFRICAN AMERICAN',u'RACE_NATIVE AMERICAN-AM IND/ALASKA',u'RACE_NATIVE HAWAIIAN OR PACIFIC ISL',u'RACE_UNKNOWN',u'RACE_WHITE']
   
clinical_features[clinical_features.isnull().any(axis=1)]

np.sum(pd.isnull(clinical_features).any(axis=1))


len(list(set(clinical_features['scan_ID'].values)))


len(set(train_scanIDs).intersection(set(clinical_features['scan_ID'].values))), len(train_scanIDs)
val_scanIDs
partition_axial['test']
#%%


REDCAP_dem['ethnicity_demog'].value_counts()
REDCAP_dem['race_demog'].value_counts()
REDCAP_dem['date_of_birth_demog'].value_counts()
REDCAP_dem['family_history'].value_counts()

all_axial_scanids = list(set(train_scanIDs)) + list(set(val_scanIDs)) + list(set(partition_axial['test']))


[u'Family Hx',u'Age',
 'ETHNICITY_HISPANIC OR LATINO',u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',
 'RACE_ASIAN-FAR EAST/INDIAN SUBCONT','RACE_BLACK OR AFRICAN AMERICAN','RACE_NATIVE AMERICAN-AM IND/ALASKA',
 'RACE_NATIVE HAWAIIAN OR PACIFIC ISL','RACE_UNKNOWN','RACE_WHITE']

'RACE_UNKNOWN'

REDCAP_dem.loc[REDCAP_dem['race_demog'] == 'PT REFUSED TO ANSWER', 'race_demog'] = 'UNKNOWN'
REDCAP_dem.loc[REDCAP_dem['race_demog'] == 'OTHER', 'race_demog'] = 'UNKNOWN'
REDCAP_dem.loc[REDCAP_dem['race_demog'] == 'NO VALUE ENTERED', 'race_demog'] = 'UNKNOWN'

REDCAP_dem.loc[REDCAP_dem['ethnicity_demog'] == 'NO VALUE ENTERED', 'ethnicity_demog'] = 'UNKNOWN'

REDCAP_dem.index = REDCAP_dem['mrn_id']

clinical_eth_race = REDCAP_dem[['mrn_id','ethnicity_demog', 'race_demog']]
clinical_eth_race = clinical_eth_race.dropna()

clinical_eth_race = pd.get_dummies(clinical_eth_race[['ethnicity_demog', 'race_demog']])

clinical_eth_race['mrn_id'] = clinical_eth_race.index
clinical_eth_race.reset_index(inplace=True, drop=True)
#%%


REDCAP_dem['DOB'] = pd.to_datetime(REDCAP_dem['date_of_birth_demog'], errors='coerce')

REDCAP_dem['DOB'] = REDCAP_dem['DOB'].apply(lambda x : x.year)

ages = REDCAP_dem[['mrn_id','DOB']].drop_duplicates()
ages = ages.dropna()
ages.reset_index(inplace=True, drop=True)

ages = pd.merge(PATHOLOGY_AXIAL, ages, on='mrn_id')

ages['Age'] = ages.apply(lambda row : int(row['scanID'][21:25]) - int(row['DOB']), axis=1)

ages = ages[['mrn_id','scanID','Age']]

clinical_eth_race = clinical_eth_race.drop_duplicates()

dems = pd.merge(ages,clinical_eth_race, on='mrn_id' )

dems.drop_duplicates()

dems['Family Hx'] = 0

dems.columns = [u'mrn_id', u'scanID', u'Age', u'ETHNICITY_HISPANIC OR LATINO',
       u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',
       u'RACE_ASIAN-FAR EAST/INDIAN SUBCONT',
       u'RACE_BLACK OR AFRICAN AMERICAN',
       u'RACE_NATIVE AMERICAN-AM IND/ALASKA',
       u'RACE_NATIVE HAWAIIAN OR PACIFIC ISL', 
       'RACE_UNKNOWN',
       u'RACE_WHITE', u'Family Hx']




CLINICAL_AXIAL = pd.DataFrame(columns=['scanID','Family Hx',u'Age',
 'ETHNICITY_HISPANIC OR LATINO',u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',
 'RACE_ASIAN-FAR EAST/INDIAN SUBCONT','RACE_BLACK OR AFRICAN AMERICAN','RACE_NATIVE AMERICAN-AM IND/ALASKA',
 'RACE_NATIVE HAWAIIAN OR PACIFIC ISL','RACE_UNKNOWN','RACE_WHITE'])

dems[[u'Family Hx',u'Age',
 'ETHNICITY_HISPANIC OR LATINO',u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',
 'RACE_ASIAN-FAR EAST/INDIAN SUBCONT','RACE_BLACK OR AFRICAN AMERICAN','RACE_NATIVE AMERICAN-AM IND/ALASKA',
 'RACE_NATIVE HAWAIIAN OR PACIFIC ISL','RACE_UNKNOWN','RACE_WHITE']].mode(axis=0).values

mode = [0,53,0,1,0,0,0,0,0,0,1]

CLINICAL_AXIAL['scanID'] = all_axial_scanids

CLINICAL_AXIAL[[u'Family Hx',u'Age',
 'ETHNICITY_HISPANIC OR LATINO',u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',
 'RACE_ASIAN-FAR EAST/INDIAN SUBCONT','RACE_BLACK OR AFRICAN AMERICAN','RACE_NATIVE AMERICAN-AM IND/ALASKA',
 'RACE_NATIVE HAWAIIAN OR PACIFIC ISL','RACE_UNKNOWN','RACE_WHITE']] = mode
                
for row in CLINICAL_AXIAL.iterrows():
    
    scanid = row[1]['scanID']
    if scanid in dems['scanID'].values:
#        print('yes')
        CLINICAL_AXIAL.loc[CLINICAL_AXIAL['scanID'] == scanid, [u'Family Hx',u'Age',
     'ETHNICITY_HISPANIC OR LATINO',u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',
     'RACE_ASIAN-FAR EAST/INDIAN SUBCONT','RACE_BLACK OR AFRICAN AMERICAN','RACE_NATIVE AMERICAN-AM IND/ALASKA',
     'RACE_NATIVE HAWAIIAN OR PACIFIC ISL','RACE_UNKNOWN','RACE_WHITE']] = dems.loc[dems['scanID'] == scanid, ['Family Hx',u'Age',
     'ETHNICITY_HISPANIC OR LATINO',u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',
     'RACE_ASIAN-FAR EAST/INDIAN SUBCONT','RACE_BLACK OR AFRICAN AMERICAN','RACE_NATIVE AMERICAN-AM IND/ALASKA',
     'RACE_NATIVE HAWAIIAN OR PACIFIC ISL','RACE_UNKNOWN','RACE_WHITE']].values   
#    else:
#        print('no')
        
                           
CLINICAL_AXIAL.to_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/DATA/Axial_Clinical_Data_Train_Val.csv', index=False)


