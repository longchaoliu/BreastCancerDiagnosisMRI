# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:34:15 2024

@author: lukas
"""


import pandas as pd
import numpy as np

# df = pd.read_csv(r"Z:\Documents\Papers_and_grants\Diagnosis_paper\data\sagittal_volumetric_localization_results.csv")

df = pd.read_csv('/home/deeperthought/Documents/Papers_and_grants/Diagnosis_paper/data/sagittal_volumetric_localization_results.csv')



pre = df['previous_HIT'].value_counts()

print(pre[1]/(pre[1]+pre[0]))


now = df['Hit'].value_counts()

print(now[1]/(now[1]+now[0]))


df.columns

df['width'] = 0

for row in df.iterrows():
    slices = row[1]['sagittal_slices_GT']
    slices = slices.replace('[','').replace(']','').split(', ')
    len(slices)
    df.loc[df['scan'] == row[1]['scan'], 'width'] = len(slices)


df['width'].mean()*3
df['width'].std()*3
df['width'].median()*3
df['width'].min()*3
df['width'].max()*3
df['width'].mode()*3

df['width'].value_counts()

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.hist(df['width'], bins=42)
plt.xticks(np.arange(0,42))



#%%
# PATH = r"Z:\Documents\Papers_and_grants\Diagnosis_paper\data\sagittal_test_results.csv"
PATH = '/home/deeperthought/Documents/Papers_and_grants/Diagnosis_paper/data/sagittal_test_results.csv'
result = pd.read_csv(PATH)
result_M = result.loc[result['y_true'] == 1]
result = pd.merge(result_M, df[['scan','sagittal_slices_GT','Hit','width']], on='scan')

#%% OLD FIGURE

# plt.figure(figsize=(10,5))

# result

# for row in result.iterrows():
#     string = row[1]['sagittal_slices_GT']
#     pred = row[1]['max_slice']
#     hit = row[1]['Hit']
    
#     GT_slice = row[1]['GT_slice']
    
#     width = string.replace('[','').replace(']','').split(', ')
#     width = [int(x) for x in width]    
#     width.sort()
    
#     color = 'steelblue'
#     if hit == 0:
#         color = 'darkorange'
        
#     pred += np.random.normal(loc=0, scale=0.25, size=1)
#     plt.subplot(1,2,1)
#     plt.plot(width, [pred]*len(width), color=color, alpha=0.25)
#     plt.scatter(GT_slice,pred, alpha=0.4, color=color)#, c=result_M['y_pred'], cmap='inferno'); plt.colorbar()
    
    

# plt.xlim([5,51])
# plt.ylim([5,51])
# plt.plot([5,51],[5,51],'k--', alpha=0.3)

# plt.xlabel('Index slice', fontsize=15)
# plt.ylabel('Predicted slice', fontsize=15)



# hits = [0.885, 0.928, 0.877]
# misses = [1-x for x in hits] 

# fontsize = 14

# plt.subplot(1,2,2)
# plt.bar([0,1,2], hits, alpha=0.8, label='Hit')
# plt.bar([0,1,2], misses, bottom=hits, alpha=0.8, label='Miss')

# plt.xticks([0,1,2], ['Sagittal data\nprimary site', 'Axial data\nprimary site', 'Axial data\nsecondary site'], fontsize=11, rotation=0)

# plt.text(x=0-0.15,y=0.4,s='232', fontsize=fontsize)
# plt.text(x=1-0.15,y=0.4,s='272', fontsize=fontsize)
# plt.text(x=2-0.15,y=0.4,s='807', fontsize=fontsize)

# plt.text(x=0-0.15,y=0.95,s=f'{-232+262}', fontsize=fontsize)
# plt.text(x=1-0.15,y=0.95,s=f'{-272+293}', fontsize=fontsize)
# plt.text(x=2-0.15,y=0.95,s=f'{-807+920}', fontsize=fontsize)

# plt.ylabel('Percent of annotated cancers', fontsize=fontsize)

# plt.legend(bbox_to_anchor=(0.94, 0.65),
#           ncol=1, fancybox=True, shadow=True)


#%% SAGITTAL

plt.figure(figsize=(5,5))

result = result.sort_values(['Hit','width'])

subjnr = 0
# plt.subplot(1,2,1)

for row in result.iterrows():
    string = row[1]['sagittal_slices_GT']
    pred = row[1]['max_slice']
    hit = row[1]['Hit']
    GT_slice = row[1]['GT_slice']

    
    width = string.replace('[','').replace(']','').split(', ')
    width = [int(x) for x in width]    
    width.sort()
    
    start = np.min(width)
    end = np.max(width)
    GT_slice = (start+end)/2
    
    color = 'steelblue'
    if hit == 0:
        color = 'darkorange'
        
    # pred += np.random.normal(loc=0, scale=0.25, size=1)
    distance = (GT_slice - pred)
    
  
    if hit:
        plt.plot(distance, subjnr, '.', color=color, alpha=0.8, markersize=5)
    else:
        plt.plot(distance, subjnr, '.', color=color, alpha=0.7, markersize=5)
    
    plt.plot( [x-GT_slice for x in width], [subjnr]*len(width), color=color, alpha=0.25)
    
    subjnr += 1
    
plt.xticks([-30,-20,-10,0,10,20],[-9,-6,-3,0,3,6])
plt.xlabel('Distance from center slice [cm]', fontsize=15)
plt.ylabel('Malignant Breast', fontsize=15)

# plt.xlim([0,20])
plt.ylim([0,263])




#%% AXIAL

# axial_annotations = pd.read_csv(r"Z:\Projects\DGNS\2D_Diagnosis_model\DATA\Axial_segmented_annotations_lesions.csv")
# res = pd.read_csv(r"Z:\Documents\Papers_and_grants\Diagnosis_paper\data\Axial_test_result.csv")
# resolutions = pd.read_csv(r"Z:\Projects\DGNS\2D_Diagnosis_model\DATA\Axial_Resolutions.csv")


axial_annotations = pd.read_csv("/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/DATA/Axial_segmented_annotations_lesions.csv")
res = pd.read_csv("/home/deeperthought/Documents/Papers_and_grants/Diagnosis_paper/data/Axial_test_result.csv")
resolutions = pd.read_csv("/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/DATA/Axial_Resolutions.csv")





interslice = [x.replace('[','').replace(']','').split(' ')[0] for x in resolutions['res_in'].values]

interslice = [float(x) for x in interslice if len(x) > 0]

np.mean(interslice)

result_segmented = res.loc[res['Exam'].isin(axial_annotations['Exam'])]

result_segmented.columns

localization_results = pd.merge(axial_annotations, result_segmented, left_on='Exam', right_on='Exam')

localization_results[['exam_max_slice','z1','z2','shape0']]

localization_results['Hit'] = 0

localization_results['Hit'] = (localization_results['exam_max_slice'] >= localization_results['z1']) * (localization_results['exam_max_slice'] <= localization_results['z2'])


localization_results['Hit']  = localization_results['Hit'].astype(int) 

localization_results['Hit'].value_counts()

localization_results['width'] = localization_results['z2'] - localization_results['z1']



plt.figure(figsize=(5,5))


localization_results = localization_results.sort_values(['Hit','width'])



axial_resolutions = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Duke_predictions/predictions/MSK_axial_resolutions.csv')

localization_results['width_cm'] = 0

for row in localization_results.iterrows():
    exam = row[1]['Exam']
    
    resolution = axial_resolutions.loc[axial_resolutions['Exam'] == exam[:-2], 'resolution_x'].values[0]
    
    localization_results.loc[localization_results['Exam'] == exam, 'width_cm'] = row[1]['width']*resolution*0.1


localization_results['width'].mean()*0.64
localization_results['width'].std()*0.64


localization_results['width_cm'].mean()
localization_results['width_cm'].std()

localization_results = localization_results.sort_values(['Hit','width_cm'])

#%% DUKE

# localization_results = pd.read_csv(r"Z:\Projects\DGNS\2D_Diagnosis_model\Sessions\AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8\Duke_predictions\predictions\results_duke_smartCropChest.csv")

duke_localization_results = pd.read_csv("/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/AXIAL__classifier_train4908_val521_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Duke_predictions/predictions/results_duke_smartCropChest.csv")

resolution_interslice = 0.64

duke_localization_results['Hit'].value_counts()

duke_localization_results['width'] = duke_localization_results['X2'] - duke_localization_results['X1']

duke_localization_results['width'].mean()*resolution_interslice
duke_localization_results['width'].std()*resolution_interslice


duke_localization_results = duke_localization_results.sort_values(['Hit','width'])




#%%  SUMMARY BAR PLOT

hits = [0.885, 0.928, 0.877]
misses = [1-x for x in hits] 

fontsize = 14

plt.figure(figsize=(5,5))

plt.subplot(1,2,2)
plt.bar([0,1,2], hits, alpha=0.8, label='Hit')
plt.bar([0,1,2], misses, bottom=hits, alpha=0.8, label='Miss')

plt.xticks([0,1,2], ['Sagittal data\nprimary site', 'Axial data\nprimary site', 'Axial data\nsecondary site'], fontsize=11, rotation=0)

plt.text(x=0-0.15,y=0.4,s='232', fontsize=fontsize)
plt.text(x=1-0.15,y=0.4,s='272', fontsize=fontsize)
plt.text(x=2-0.15,y=0.4,s='807', fontsize=fontsize)

plt.text(x=0-0.15,y=0.95,s=f'{-232+262}', fontsize=fontsize)
plt.text(x=1-0.15,y=0.95,s=f'{-272+293}', fontsize=fontsize)
plt.text(x=2-0.15,y=0.95,s=f'{-807+920}', fontsize=fontsize)

plt.ylabel('Percent of annotated cancers', fontsize=fontsize)

# plt.legend(bbox_to_anchor=(0.94, 0.65),
#           ncol=1, fancybox=True, shadow=True)


plt.legend(bbox_to_anchor=(1.94, 0.65),
          ncol=1, fancybox=True, shadow=True)

plt.tight_layout()

plt.savefig('/home/deeperthought/Projects/Others/2D_brain_segmenter/Sessions/summaryfig_loc.png', dpi=400)

#%% SUBPLOTS

# Create the figure and subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Plot the first subplot
result = result.sort_values(['Hit','width'])
subjnr = 0
# plt.subplot(1,2,1)

for row in result.iterrows():
    string = row[1]['sagittal_slices_GT']
    pred = row[1]['max_slice']
    hit = row[1]['Hit']
    GT_slice = row[1]['GT_slice']

    
    width = string.replace('[','').replace(']','').split(', ')
    width = [int(x) for x in width]    
    width.sort()
    
    start = np.min(width)
    end = np.max(width)
    GT_slice = (start+end)/2
    
    color = 'steelblue'
    if hit == 0:
        color = 'darkorange'
        
    # pred += np.random.normal(loc=0, scale=0.25, size=1)
    distance = (GT_slice - pred)
    
  
    if hit:
        axs[0,0].plot(distance, subjnr, '.', color=color, alpha=0.8, markersize=5)
    else:
        axs[0,0].plot(distance, subjnr, '.', color=color, alpha=0.7, markersize=5)
    
    axs[0,0].plot( [x-GT_slice for x in width], [subjnr]*len(width), color=color, alpha=0.25)
    
    subjnr += 1
    
axs[0,0].set_title('Sagittal Data Primary Site', fontsize=15)
axs[0,0].set_xticks([-30,-20,-10,0,10,20],[-9,-6,-3,0,3,6])
axs[0,0].set_xlabel('Distance from center slice [cm]', fontsize=15)
axs[0,0].set_ylabel('Malignant Breast', fontsize=15)

axs[0,0].set_ylim([0,263])
axs[0,0].set_xlim([-20,20])



# Axial, Primary


subjnr=0
for row in localization_results.iterrows():
    start = row[1]['z1']
    end = row[1]['z2']
    pred = row[1]['exam_max_slice']
    hit = row[1]['Hit']
    GT_slice = (start+end)/2
    
    exam = row[1]['Exam']

    resolution = axial_resolutions.loc[axial_resolutions['Exam'] == exam[:-2], 'resolution_x'].values[0]

    
    color = 'steelblue'
    if hit == 0:
        color = 'darkorange'
        
    # pred += np.random.normal(loc=0, scale=0.25, size=1)
    distance = (GT_slice - pred)*resolution*0.1
    
    # distance = distance*resolution_interslice*0.1
    
    if hit:
        axs[1, 0].plot(distance, subjnr, '.', color=color, alpha=0.80, markersize=5)
    else:
        axs[1, 0].plot(distance, subjnr, '.', color=color, alpha=0.7, markersize=5)
    
    axs[1, 0].plot( [(start-GT_slice)*resolution*0.1, (end-GT_slice)*resolution*0.1], [subjnr, subjnr], color=color, alpha=0.25)
    subjnr += 1
    
axs[1, 0].set_title('Axial Data Primary Site', fontsize=15)
# axs[1, 0].set_xticks(np.arange(-300,200,25),[0.1*x*resolution_interslice for x in np.arange(-300,200,25)])
axs[1, 0].set_xlabel('Distance from lesion center [cm]', fontsize=15)
axs[1, 0].set_ylabel('Malignant Breast', fontsize=15)
axs[1, 0].set_ylim([0,294])
axs[1, 0].set_xlim([-6,6])




# Plot the third subplot, spanning two rows
axs[0, 1].remove()  # Remove the first subplot in the second row
axs[1, 1].remove()  # Remove the second subplot in the second row



ax3 = fig.add_subplot(1, 2, 2)
subjnr = 0
for row in duke_localization_results.iterrows():
    start = row[1]['X1']
    end = row[1]['X2']
    pred = row[1]['max_slice']
    hit = row[1]['Hit']
    GT_slice = (start+end)/2
    
    
    color = 'steelblue'
    if hit == 0:
        color = 'darkorange'
        
    # pred += np.random.normal(loc=0, scale=0.25, size=1)
    distance = (GT_slice - pred)*0.64*0.1
    
    if hit:
        ax3.plot(distance, subjnr, '.', color=color, alpha=0.65, markersize=5)
    else:
        ax3.plot(distance, subjnr, '.', color=color, alpha=0.45, markersize=5)
    
    ax3.plot( [(start-GT_slice)*0.64*0.1, (end-GT_slice)*0.64*0.1], [subjnr, subjnr], color=color, alpha=0.25)
    
    subjnr += 1
    
    
ax3.set_title('Axial Data Secondary Site', fontsize=15)
# ax3.set_xticks(np.arange(-300,200,25),[0.1*x*resolution_interslice for x in np.arange(-300,200,25)])
ax3.set_xlabel('Distance from lesion center [cm]', fontsize=15)
ax3.set_ylabel('Malignant Breast', fontsize=15)
ax3.set_ylim([0,921])
ax3.set_xlim([-6,6])



# Adjust spacing between subplots
plt.tight_layout()

plt.savefig('/home/deeperthought/Documents/Papers_and_grants/Diagnosis_paper/data/Figure4_localization.png', dpi=400)