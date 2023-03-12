import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage
import pickle
import operator
from scipy.ndimage import zoom, morphology
import os
import fnmatch
import scipy.special as sc
from scipy import ndimage
from skimage.filters import gaussian
import pydicom as py
from scipy.spatial.distance import directed_hausdorff
import seaborn as sns
from scipy.stats import ttest_ind, kstest, describe, ks_2samp, wilcoxon

import cv2


main_dir = '/data/PBruck/GKA/Patients/'

preproc_file = [
    '111-120/2022-11-09_Pats_111-120___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCropBlur_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps',
    '121-130/2022-11-09_Pats_121-130___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCropBlur_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps'
        ]

# ========================================================================
# ===================== 20x20x20 Stride Pred =============================
# ========================================================================
# # ===== SMALL TARGET TRAINING =====
# # ----- UNET -----
# gpu_0_files = ['Prediction2022-12-29_DosePred_Pats111-120_SmallTarg_Stride20-20-20_Mdl-UNet_Inp-Targ___5fold_22-12-10',
#                'Prediction2022-12-29_DosePred_Pats121-130_SmallTarg_Stride20-20-20_Mdl-UNet_Inp-Targ___5fold_22-12-10']
# gpu_1_files = ['Prediction2022-12-29_DosePred_Pats111-120_SmallTarg_Stride20-20-20_Mdl-UNet_Inp-InnerOuterEDT___5fold_22-12-10',
#                'Prediction2022-12-29_DosePred_Pats121-130_SmallTarg_Stride20-20-20_Mdl-UNet_Inp-InnerOuterEDT___5fold_22-12-10']

# # ----- HD-UNET -----
# gpu_0_files = ['Prediction2022-12-29_DosePred_Pats111-120_SmallTarg_Stride20-20-20_Mdl-HD-UNet_Inp-Targ___5fold_22-12-10',
#                'Prediction2022-12-29_DosePred_Pats121-130_SmallTarg_Stride20-20-20_Mdl-HD-UNet_Inp-Targ___5fold_22-12-10']
# gpu_1_files = ['Prediction2022-12-29_DosePred_Pats111-120_SmallTarg_Stride20-20-20_Mdl-HD-UNet_Inp-InnerOuterEDT___5fold_22-12-10',
#                'Prediction2022-12-29_DosePred_Pats121-130_SmallTarg_Stride20-20-20_Mdl-HD-UNet_Inp-InnerOuterEDT___5fold_22-12-10']

# # ===== Large TARGET TRAINING =====
# # ----- UNET -----
# gpu_0_files = ['Prediction2022-12-31_DosePred_Pats111-120_LargeTarg_Stride20-20-20_Mdl-UNet_Inp-Targ___5fold_22-12-10',
#                'Prediction2022-12-31_DosePred_Pats121-130_LargeTarg_Stride20-20-20_Mdl-UNet_Inp-Targ___5fold_22-12-10']
# gpu_1_files = ['Prediction2022-12-31_DosePred_Pats111-120_LargeTarg_Stride20-20-20_Mdl-UNet_Inp-InnerOuterEDT___5fold_22-12-10',
#                'Prediction2022-12-31_DosePred_Pats121-130_LargeTarg_Stride20-20-20_Mdl-UNet_Inp-InnerOuterEDT___5fold_22-12-10']

# # ----- HD-UNET -----
gpu_0_files = ['Prediction2022-12-31_DosePred_Pats111-120_LargeTarg_Stride20-20-20_Mdl-HD-UNet_Inp-Targ___5fold_22-12-10',
               'Prediction2022-12-31_DosePred_Pats121-130_LargeTarg_Stride20-20-20_Mdl-HD-UNet_Inp-Targ___5fold_22-12-10']
gpu_1_files = ['Prediction2022-12-31_DosePred_Pats111-120_LargeTarg_Stride20-20-20_Mdl-HD-UNet_Inp-InnerOuterEDT___5fold_22-12-10',
               'Prediction2022-12-31_DosePred_Pats121-130_LargeTarg_Stride20-20-20_Mdl-HD-UNet_Inp-InnerOuterEDT___5fold_22-12-10']


gpu_0_dmaps = []
for i in range(len(gpu_0_files)):
    dmaps_file = open(gpu_0_files[i], 'rb')
    dmaps = pickle.load(dmaps_file)
    gpu_0_dmaps.append(dmaps)

gpu_1_dmaps = []
for i in range(len(gpu_1_files)):
    dmaps_file = open(gpu_1_files[i], 'rb')
    dmaps = pickle.load(dmaps_file)
    gpu_1_dmaps.append(dmaps)

preproc_dmaps = []
for i in range(len(preproc_file)):
    dmaps_file = open(main_dir + preproc_file[i], 'rb')
    dmaps = pickle.load(dmaps_file)
    preproc_dmaps.append(dmaps)

def lst_avg(lst):
    return sum(lst)/len(lst)

def lst_avg_std(lst):
    avg = sum(lst)/len(lst)
    variance = sum([((x - avg) ** 2) for x in lst]) / len(lst) 
    std = variance ** 0.5
    return avg, std


# Prediction structure:
# [0]  - OG dmap
# [1]  - Target Contour
# [2]  - Input 1 
# [3]  - Input 2 
# [-1] - Prediction


print('Volume Cutoff (Voxel Count): 4893')
# Need to get Rx info from preproc file for each target cohort
global_info = []
large_info = []
small_info = []
# Need OG cov, sel, gri to compare against
# Divided into [0] global, [1] large, [2] small
og_cov = [[],[],[]]
og_sel = [[],[],[]]
og_gri = [[],[],[]]

vol_ero_lg = []
vol_dil_lg = []
vol_ero_sm = []
vol_dil_sm = []
# Loop through preproc files which contain this info and can easily calc the og values
all_targ_track = 1
for i_file in range(len(preproc_dmaps)):
    for pat_num in range(len(preproc_dmaps[i_file])):
    # for pat_num in range(8):
        print('===============================================================================')
        print('Starting new patient [', pat_num+1, '/', len(preproc_dmaps[i_file]), ']:  (', preproc_dmaps[i_file][pat_num][0], ')')
        print('===============================================================================')
        for tar_num in range(len(preproc_dmaps[i_file][pat_num][3])):
            print('     -------------------------------------------------------------------------------')
            print('     Starting new target calculations [', tar_num+1, '/', len(preproc_dmaps[i_file][pat_num][3]), ']:    ', all_targ_track)
            targ_info = []
            targ = np.array(preproc_dmaps[i_file][pat_num][9][tar_num])
            vol = np.sum(targ)
            
            ero_targ = morphology.binary_erosion(targ, iterations=1)
            vol_ero = np.sum(ero_targ)
            dil_targ = morphology.binary_dilation(targ, iterations=1)
            vol_dil = np.sum(dil_targ)
            
            # 5 is the Rx iso threshold
            # 9 is the target contour
            # 10 is the 50% V of the Rx
            
            cov = np.divide(np.sum(preproc_dmaps[i_file][pat_num][5][tar_num][preproc_dmaps[i_file][pat_num][9][tar_num]==1]),
                          np.sum(preproc_dmaps[i_file][pat_num][9][tar_num]))
            og_cov[0].append(cov)
            sel = np.divide(np.sum(preproc_dmaps[i_file][pat_num][5][tar_num][preproc_dmaps[i_file][pat_num][9][tar_num]==1]),
                          np.sum(preproc_dmaps[i_file][pat_num][5][tar_num]))
            og_sel[0].append(sel)
            gri = np.divide(np.sum(preproc_dmaps[i_file][pat_num][10][tar_num]),
                         np.sum(preproc_dmaps[i_file][pat_num][5][tar_num]))
            og_gri[0].append(gri)
            
            # Information used for evaluation of predictions
            # Rx info for thresholding dose maps
            targ_info.append(preproc_dmaps[i_file][pat_num][15][tar_num])
            
            global_info.append(targ_info)
            if vol >= 4893.0:
                large_info.append(targ_info)
                print('        Large: [V, C, S, G]', vol, cov, sel, gri)
                og_cov[1].append(cov)
                og_sel[1].append(sel)
                og_gri[1].append(gri)
                vol_ero_lg.append(abs(vol-vol_ero)/vol*100)
                vol_dil_lg.append(abs(vol-vol_dil)/vol*100)
            if vol < 4893.0:
                small_info.append(targ_info)
                print('        Small [V, C, S, G]:', vol, cov, sel, gri)
                og_cov[2].append(cov)
                og_sel[2].append(sel)
                og_gri[2].append(gri)
                vol_ero_sm.append(abs(vol-vol_ero)/vol*100)
                vol_dil_sm.append(abs(vol-vol_dil)/vol*100)
                
            all_targ_track += 1

print(lst_avg_std(vol_ero_lg))
print(lst_avg_std(vol_dil_lg))
print(lst_avg_std(vol_ero_sm))
print(lst_avg_std(vol_dil_sm))


gpu_0_cov = [[],[]]
gpu_0_sel = [[],[]]
gpu_0_gri = [[],[]]

# erode_cov = []
# erode_sel = []
# erode_vol = []
# pre_vol = []
# targ_names = []
targ_track = 0
for i_file in range(len(gpu_0_files)):
    print('Starting new fileset')
    for tar_num in range(len(gpu_0_dmaps[i_file])):
        print('     ========================')
        print('     Starting new target [', tar_num+1, '/', len(gpu_0_dmaps[i_file]), ']:')
        dose = np.array(gpu_0_dmaps[i_file][tar_num][0]).astype(float)  # TMR10 dose
        targ = np.array(gpu_0_dmaps[i_file][tar_num][1]).astype(float)  # Targ contour
        pred = np.array(gpu_0_dmaps[i_file][tar_num][-1]).astype(float) # Prediction dose
        
        info = global_info[tar_num]
#         info = large_info[tar_num]
#         info = small_info[tar_num]

        tar_D = info[0][0]           # Dose at Rx% line [Gy]
        tar_Rx = info[0][1]          # Rx% line (decimal, example: 0.5)
        gri_thresh = tar_Rx/2
        pred_gri = np.array(pred)  
            
        pred = pred/np.max(pred)
        pred[pred <  tar_Rx] = 0
        pred[pred >= tar_Rx] = 1
        
        pred_gri[pred_gri <  gri_thresh] = 0
        pred_gri[pred_gri >= gri_thresh] = 1
        
            
        
        cov = np.divide(np.sum(pred[targ==1]),
                        np.sum(targ))
        gpu_0_cov[i_file].append(cov)
        print('        Cov: ', cov)

        sel = np.divide(np.sum(pred[targ==1]),
                          np.sum(pred))
        gpu_0_sel[i_file].append(sel)
        print('        Sel: ', sel)

        gri = np.divide(np.sum(pred_gri),
                         np.sum(pred))
        gpu_0_gri[i_file].append(gri)
        print('        GrI: ', gri)
        
        targ_track += 1

print()
print('FINISHED')
print('Number of targets processed: ', (len(gpu_0_cov[0])+len(gpu_0_cov[1])))

gpu_1_cov = [[],[]]
gpu_1_sel = [[],[]]
gpu_1_gri = [[],[]]

# targ_names = []
targ_track = 0
for i_file in range(len(gpu_1_files)):
    print('Starting new fileset')
    for tar_num in range(len(gpu_1_dmaps[i_file])):
        print('     ========================')
        print('     Starting new target [', tar_num+1, '/', len(gpu_1_dmaps[i_file]), ']:')
        dose = np.array(gpu_1_dmaps[i_file][tar_num][0]).astype(float)  # TMR10 dose
        targ = np.array(gpu_1_dmaps[i_file][tar_num][1]).astype(float)  # Targ contour
        pred = np.array(gpu_1_dmaps[i_file][tar_num][-1]).astype(float) # Prediction dose
        
        info = global_info[tar_num]
#         info = large_info[tar_num]
#         info = small_info[tar_num]

        tar_D = info[0][0]           # Dose at Rx% line [Gy]
        tar_Rx = info[0][1]          # Rx% line (decimal, example: 0.5)
        gri_thresh = tar_Rx/2
        pred_gri = np.array(pred)
        
        pred = pred/np.max(pred)
        pred[pred <  tar_Rx] = 0
        pred[pred >= tar_Rx] = 1
        
        pred_gri[pred_gri <  gri_thresh] = 0
        pred_gri[pred_gri >= gri_thresh] = 1
        
        cov = np.divide(np.sum(pred[targ==1]),
                        np.sum(targ))
        gpu_1_cov[i_file].append(cov)
        print('        Cov: ', cov)

        sel = np.divide(np.sum(pred[targ==1]),
                          np.sum(pred))
        gpu_1_sel[i_file].append(sel)
        print('        Sel: ', sel)

        gri = np.divide(np.sum(pred_gri),
                         np.sum(pred))
        gpu_1_gri[i_file].append(gri)
        print('        GrI: ', gri)

print()
print('FINISHED')
print('Number of targets processed: ', (len(gpu_1_cov[0])+len(gpu_1_cov[1])))

print(np.shape(og_cov[0]), np.shape(og_cov[1]), np.shape(og_cov[2]))

gpu_0_cov_all = gpu_0_cov[0] + gpu_0_cov[1]
gpu_0_sel_all = gpu_0_sel[0] + gpu_0_sel[1]
gpu_0_gri_all = gpu_0_gri[0] + gpu_0_gri[1]
print(len(gpu_0_cov_all))

gpu_1_cov_all = gpu_1_cov[0] + gpu_1_cov[1]
gpu_1_sel_all = gpu_1_sel[0] + gpu_1_sel[1]
gpu_1_gri_all = gpu_1_gri[0] + gpu_1_gri[1]
print(len(gpu_1_cov_all))



num_proc = len(gpu_0_cov_all)
for i in range(len(og_cov)):
    if len(gpu_0_cov_all) == len(og_cov[i]):
        ind = i

def percdiff(val1, val2):
    diff = abs(val1 - val2) / ((val1 + val2)/2) * 100
    return diff

def percerr(true, est):
    err = (true - est) / true * 100
    return err
        
# UNET Differences
cov_diff_unet = []
cov_zip = zip(og_cov[ind], gpu_0_cov_all)
for og_i, rc_i in cov_zip:
    cov_diff_unet.append(percdiff(og_i,rc_i))
#     cov_diff_unet.append(percerr(og_i,rc_i))
#     cov_diff_unet.append(og_i - rc_i)

sel_diff_unet = []
sel_zip = zip(og_sel[ind], gpu_0_sel_all)
for og_i, rc_i in sel_zip:
    sel_diff_unet.append(percdiff(og_i,rc_i))
#     sel_diff_unet.append(percerr(og_i,rc_i))
#     sel_diff_unet.append(og_i - rc_i)
    
gri_diff_unet = []
gri_zip = zip(og_gri[ind], gpu_0_gri_all)
for og_i, rc_i in gri_zip:
    gri_diff_unet.append(percdiff(og_i,rc_i))
#     gri_diff_unet.append(percerr(og_i,rc_i))
#     gri_diff_unet.append(og_i - rc_i)
    
# HD-UNET Differences
cov_diff_hdunet = []
cov_zip = zip(og_cov[ind], gpu_1_cov_all)
for og_i, rc_i in cov_zip:
    cov_diff_hdunet.append(percdiff(og_i,rc_i))
#     cov_diff_hdunet.append(percerr(og_i,rc_i))
#     cov_diff_hdunet.append(og_i - rc_i)

sel_diff_hdunet = []
sel_zip = zip(og_sel[ind], gpu_1_sel_all)
for og_i, rc_i in sel_zip:
    sel_diff_hdunet.append(percdiff(og_i,rc_i))
#     sel_diff_hdunet.append(percerr(og_i,rc_i))
#     sel_diff_hdunet.append(og_i - rc_i)
    
gri_diff_hdunet = []
gri_zip = zip(og_gri[ind], gpu_1_gri_all)
for og_i, rc_i in gri_zip:
    gri_diff_hdunet.append(percdiff(og_i,rc_i))
#     gri_diff_hdunet.append(percerr(og_i,rc_i))
#     gri_diff_hdunet.append(og_i - rc_i)



OG_cov_avg, OG_cov_sd = lst_avg_std(og_cov[ind])
gpu0_cov_avg, gpu0_cov_sd = lst_avg_std(gpu_0_cov_all)

OG_sel_avg, OG_sel_sd = lst_avg_std(og_sel[ind])
gpu0_sel_avg, gpu0_sel_sd = lst_avg_std(gpu_0_sel_all)

OG_gri_avg, OG_gri_sd = lst_avg_std(og_gri[ind])
gpu0_gri_avg, gpu0_gri_sd = lst_avg_std(gpu_0_gri_all)

unet_cov_diff_avg, unet_cov_diff_sd = lst_avg_std(cov_diff_unet)
unet_sel_diff_avg, unet_sel_diff_sd = lst_avg_std(sel_diff_unet)
unet_gri_diff_avg, unet_gri_diff_sd = lst_avg_std(gri_diff_unet)



gpu1_cov_avg, gpu1_cov_sd = lst_avg_std(gpu_1_cov_all)

gpu1_sel_avg, gpu1_sel_sd = lst_avg_std(gpu_1_sel_all)

gpu1_gri_avg, gpu1_gri_sd = lst_avg_std(gpu_1_gri_all)

hdunet_cov_diff_avg, hdunet_cov_diff_sd = lst_avg_std(cov_diff_hdunet)
hdunet_sel_diff_avg, hdunet_sel_diff_sd = lst_avg_std(sel_diff_hdunet)
hdunet_gri_diff_avg, hdunet_gri_diff_sd = lst_avg_std(gri_diff_hdunet)

print('OG Cov: ', OG_cov_avg, OG_cov_sd)
print('gpu0 Cov: ', gpu0_cov_avg, gpu0_cov_sd)
print('gpu1 Cov: ', gpu1_cov_avg, gpu1_cov_sd)
print('OG Sel: ', OG_sel_avg, OG_sel_sd)
print('gpu0 Sel: ', gpu0_sel_avg, gpu0_sel_sd)
print('gpu1 Sel: ', gpu1_sel_avg, gpu1_sel_sd)
print('OG GrI: ', OG_gri_avg, OG_gri_sd)
print('gpu0 GrI: ', gpu0_gri_avg, gpu0_gri_sd)
print('gpu1 GrI: ', gpu1_gri_avg, gpu1_gri_sd)
# print(OG_sel_avg, RC_sel_avg)
# print(OG_gi_avg, RC_gi_avg)

print()
print('UNET Diffs')
print('Cov Diff: ', unet_cov_diff_avg, unet_cov_diff_sd)
print('Sel Diff: ', unet_sel_diff_avg, unet_sel_diff_sd)
print('GrI Diff: ', unet_gri_diff_avg, unet_gri_diff_sd)

print()
print('HD-UNET Diffs')
print('Cov Diff: ', hdunet_cov_diff_avg, hdunet_cov_diff_sd)
print('Sel Diff: ', hdunet_sel_diff_avg, hdunet_sel_diff_sd)
print('GrI Diff: ', hdunet_gri_diff_avg, hdunet_gri_diff_sd)

# all_deliv_data = [og_cov[ind], gpu_0_cov_all, gpu_1_cov_all, og_sel[ind], gpu_0_sel_all, gpu_1_sel_all, og_gri[ind], gpu_0_gri_all, gpu_1_gri_all]
# plt.figure(facecolor='white')
# plt.boxplot(all_deliv_data, labels=['TMR10', 'Target', 'InOut', 'TMR10', 'Target', 'InOut', 'TMR10', 'Target', 'InOut'], positions=[0.4,0.8,1.2, 1.8,2.2,2.6, 3.2,3.6,4.0], notch=False)
# # plt.boxplot(cov_data, labels=['TMR10', 'UNET', 'HD-UNET'], notch=False)
# # plt.title('Global Coverage Results\n(69 Targets)')
# # plt.title('Large Coverage Results\n(20 Targets)')
# plt.title('Deliverability Metrics, Small Targets\n(49 Targets)')

plt.tight_layout()

cov_data = [og_cov[ind], gpu_0_cov_all, gpu_1_cov_all]
plt.figure(facecolor='white')
plt.boxplot(cov_data, labels=['TMR10', 'Target', 'InOut'], notch=False, showmeans=True, meanline=True)
# plt.boxplot(cov_data, labels=['TMR10', 'UNET', 'HD-UNET'], notch=False)
# plt.title('Global Coverage Results\n(69 Targets)')
# plt.title('Large Coverage Results\n(20 Targets)')
plt.title('Small Coverage Results\n(49 Targets)')
ax = plt.gca()
ax.set_ylim([-0.05,1.05])


sel_data = [og_sel[ind], gpu_0_sel_all, gpu_1_sel_all]
plt.figure(facecolor='white')
plt.boxplot(sel_data, labels=['TMR10', 'Target', 'InOut'], notch=False, showmeans=True, meanline=True)
# plt.boxplot(sel_data, labels=['TMR10', 'UNET', 'HD-UNET'], notch=False)
# plt.title('Global Selectivity Results\n(69 Targets)')
# plt.title('Large Selectivity Results\n(20 Targets)')
plt.title('Small Selectivity Results\n(49 Targets)')
ax = plt.gca()
ax.set_ylim([-0.05,1.05])

gri_data = [og_gri[ind], gpu_0_gri_all, gpu_1_gri_all]
plt.figure(facecolor='white')
plt.boxplot(gri_data, labels=['TMR10', 'Target', 'InOut'], notch=False, showmeans=True, meanline=True)
# plt.boxplot(gri_data, labels=['TMR10', 'UNET', 'HD-UNET'], notch=False)
# plt.title('Global Gradient Index Results\n(69 Targets)')
# plt.title('Large Gradient Index Results\n(20 Targets)')
plt.title('Small Gradient Index Results\n(49 Targets)')
ax = plt.gca()
ax.set_ylim([0,9])

cov_diff_data = [cov_diff_unet, cov_diff_hdunet]
plt.figure(facecolor='white')
plt.boxplot(cov_diff_data, widths=0.2, labels=['Target', 'InOut'], notch=False, showmeans=True, meanline=True)
# plt.boxplot(cov_diff_data, labels=['UNET', 'HD-UNET'], notch=False)
# plt.title('Global Coverage Results\n(69 Targets)')
# plt.title('Large Coverage %Difference Results\n(20 Targets)')
plt.title('Small Coverage %Difference Results\n(49 Targets)')
plt.ylabel('Percent Difference')
# plt.title('Large Coverage %Error Results\n(20 Targets)')
# plt.ylabel('Percent Error')
# plt.title('Small Coverage Results\n(49 Targets)')
ax = plt.gca()
ax.set_ylim([-5,205])
# ax.set_ylim([-5,140])
# ax.set_ylim([-100,100])

sel_diff_data = [sel_diff_unet, sel_diff_hdunet]
plt.figure(facecolor='white')
plt.boxplot(sel_diff_data, labels=['Target', 'InOut'], notch=False, showmeans=True, meanline=True)
# plt.boxplot(sel_diff_data, labels=['UNET', 'HD-UNET'], notch=False)
# plt.title('Global Selectivity Results\n(69 Targets)')
# plt.title('Large Selectivity %Difference Results\n(20 Targets)')
plt.title('Small Selectivity %Difference Results\n(49 Targets)')
plt.ylabel('Percent Difference')
# plt.title('Large Selectivity %Error Results\n(20 Targets)')
# plt.ylabel('Percent Error')
# plt.title('Small Selectivity Results\n(49 Targets)')
ax = plt.gca()
ax.set_ylim([-5,205])
# ax.set_ylim([-5,140])
# ax.set_ylim([-100,100])

gri_diff_data = [gri_diff_unet, gri_diff_hdunet]
plt.figure(facecolor='white')
plt.boxplot(gri_diff_data, labels=['Target', 'InOut'], notch=False, showmeans=True, meanline=True)
# plt.boxplot(gri_diff_data, labels=['UNET', 'HD-UNET'], notch=False)
# plt.title('Global Gradient Index Results\n(69 Targets)')
# plt.title('Large Gradient Index %Difference Results\n(20 Targets)')
plt.title('Small Gradient Index %Difference Results\n(49 Targets)')
plt.ylabel('Percent Difference')
# plt.title('Large Gradient Index %Error Results\n(20 Targets)')
# plt.ylabel('Percent Error')
# plt.title('Small Gradient Index Results\n(49 Targets)')
ax = plt.gca()
ax.set_ylim([-5,205])
# ax.set_ylim([-5,140])
# ax.set_ylim([-100,100])

all_diff_data = [cov_diff_unet, cov_diff_hdunet, sel_diff_unet, sel_diff_hdunet, gri_diff_unet, gri_diff_hdunet]
plt.figure(facecolor='white')
plt.boxplot(all_diff_data, widths=0.2, labels=['Target', 'InOut', 'Target', 'InOut', 'Target', 'InOut'], positions=[.8,1.2, 1.8,2.2, 2.8,3.2], notch=False, showmeans=True, meanline=True)
# plt.boxplot(cov_diff_data, labels=['UNET', 'HD-UNET'], notch=False)
# plt.title('Global Coverage Results\n(69 Targets)')
plt.title('Large Target Deliverability Metrics\n%Difference Results (20 Targets)')
# plt.title('Small Target Deliverability Metrics\n%Difference Results (49 Targets)')
plt.ylabel('Percent Difference')
# plt.title('Large Coverage %Error Results\n(20 Targets)')
# plt.ylabel('Percent Error')
# plt.title('Small Coverage Results\n(49 Targets)')
ax = plt.gca()
ax.set_ylim([-5,205])
# ax.set_ylim([-5,140])
# ax.set_ylim([-100,100])
