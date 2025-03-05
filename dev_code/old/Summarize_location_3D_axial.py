# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:01:03 2024

@author: lukas
"""

import pandas as pd
import matplotlib.pyplot as plt


localization_results = pd.read_csv("Z:/Documents/Papers_and_grants/Diagnosis_paper/data/Axial_localization_width.csv")

localization_results['exam'] = localization_results['Exam'].str[:-2]

df = pd.read_csv(r"Z:\Projects\DGNS\2D_Diagnosis_model\Sessions\FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8\gradCAM_images\MSK_Axial\gradCAM_summary_distances.csv")

df = pd.merge(df, localization_results[['exam','Hit']], on='exam')

# df = df.loc[df['Hit'] == 1]

df['distance_mm'] = df['distance']*0.4
df['distance_cm'] = df['distance']*0.4*0.1

mean_mm = df.loc[df['distance_mm'] >0, 'distance_mm'].mean()
sd_mm = df.loc[df['distance_mm'] >0, 'distance_mm'].std()

n1 = len(df.loc[df['distance'] == 0])
n = len(df)


'RIA_19-093_000_06374_20140507_r  segmentation looks wrong!!'



print(f'Hits: {n1}/{n} = {round(n1*100./n, 2)}%')

print(f'Miss {n-n1}/{n}: Mean={round(mean_mm,2)} mm (SD={round(sd_mm,2)}) ')

plt.figure(1, figsize=(5,5))
plt.hist(df.loc[df['distance_cm'] == 0, 'distance_cm'], bins = 20, color='royalblue', label=f'hit = {n1}/{n}')
plt.hist(df.loc[df['distance_cm'] > 0, 'distance_cm'], bins = 200, color='darkorange', label=f'miss = {n-n1}/{n}')
plt.xlabel('Centimeters from segmented lesion ')
plt.xlim([0,2])
plt.legend()

nclose = len(df.loc[df['distance_mm'] < 1])

print(f'VERY CLOSE (under 1 mm, or 2 pixels): {nclose}/{n}, {round(nclose*100./n,2)}%' )
#%%
plt.figure(2, figsize=(5,5))
plt.hist(df['distance_mm'], bins=100)
plt.xlabel('Millimeters from segmented lesion ')
plt.xlim([0,20])
# plt.legend()