# Initialization and Declaration
import tensorflow as tf
from tensorflow.python.keras.layers import Add, Conv3D, Input, Dense, Flatten, Dropout, concatenate, Concatenate, Lambda, MaxPooling1D, BatchNormalization
from tensorflow.python.keras.layers import ThresholdedReLU, Activation, PReLU, LeakyReLU, ReLU, Cropping3D, ZeroPadding3D, MaxPooling3D, UpSampling3D, SeparableConv2D, Average, AveragePooling3D, average, MaxPool3D, GlobalMaxPooling3D, add, TimeDistributed, Reshape, LSTM, GRU, Conv2D, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, UpSampling2D, Cropping2D, Conv1D
from tensorflow.python.keras import models, optimizers
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.utils import multi_gpu_model
import h5py
import matplotlib.pyplot as plt
import pydicom
import glob
import cv2
import fnmatch
import pickle
import re
from collections import Counter
import scipy
import pydicom.uid
from skimage.transform import warp
from skimage.draw import polygon
from skimage import exposure
from scipy import ndimage
import numpy as np
import deepdish as dd
import os, sys
from timeit import default_timer as timer
#from EPorter
import data_augment_dosepred_6 as da
import assist_functions_DS_DosePred as af
import os
import time
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.measurements import center_of_mass
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
gpu_num = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num # "0, 1" for multiple

##---USER INPUTS---#
# # main_directory_folder = '/mnt/sdb/home/dsolis/PycharmProjects/KerasTestingFiles/DosePred_Data/'
#------------------
main_dir = '/data/PBruck/GKA/Patients/'

files = [
    '111-120/2022-11-09_Pats_111-120___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCropBlur_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps',
    '121-130/2022-11-09_Pats_121-130___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCropBlur_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps'
        ]

file_name0 = main_dir + files[-1]
# file_name0 = main_dir + files[0]

# ---------------------------------------------------------
# ----- 2023.01.27 Targ-InOut KFold Results (001-110) -----
# ---------------------------------------------------------
#
# ----- Small ------
# UNET Target
# save_name1 = [
#     '2023_01_27_Titan_gpu_0_UNET_DosePred_Target_KFold_Small/Kfold-1_Pats_001-110_Target_gpu_0_ch_1_xyz_96x96x96_stepsz_20_ep_250_numpts_0_date_run_2023_01_27_mdl_UNet_snglPat_True',
#     '2023_01_27_Titan_gpu_0_UNET_DosePred_Target_KFold_Small/Kfold-2_Pats_001-110_Target_gpu_0_ch_1_xyz_96x96x96_stepsz_20_ep_250_numpts_0_date_run_2023_01_27_mdl_UNet_snglPat_True',
#     '2023_01_27_Titan_gpu_0_UNET_DosePred_Target_KFold_Small/Kfold-3_Pats_001-110_Target_gpu_0_ch_1_xyz_96x96x96_stepsz_20_ep_250_numpts_0_date_run_2023_01_27_mdl_UNet_snglPat_True',
#     '2023_01_27_Titan_gpu_0_UNET_DosePred_Target_KFold_Small/Kfold-4_Pats_001-110_Target_gpu_0_ch_1_xyz_96x96x96_stepsz_20_ep_250_numpts_0_date_run_2023_01_27_mdl_UNet_snglPat_True',
#     '2023_01_27_Titan_gpu_0_UNET_DosePred_Target_KFold_Small/Kfold-5_Pats_001-110_Target_gpu_0_ch_1_xyz_96x96x96_stepsz_20_ep_250_numpts_0_date_run_2023_01_27_mdl_UNet_snglPat_True'
#             ]

# UNET InnerOuterEDT
# save_name1 = [
#     '2023_01_27_Titan_gpu_0_UNET_DosePred_InOut_KFold_Small/Kfold-1_Pats_001-110_InOut_gpu_0_ch_2_xyz_96x96x96_stepsz_20_ep_250_numpts_0_date_run_2023_01_27_mdl_UNet_snglPat_True',
#     '2023_01_27_Titan_gpu_0_UNET_DosePred_InOut_KFold_Small/Kfold-2_Pats_001-110_InOut_gpu_0_ch_2_xyz_96x96x96_stepsz_20_ep_250_numpts_0_date_run_2023_01_27_mdl_UNet_snglPat_True',
#     '2023_01_27_Titan_gpu_0_UNET_DosePred_InOut_KFold_Small/Kfold-3_Pats_001-110_InOut_gpu_0_ch_2_xyz_96x96x96_stepsz_20_ep_250_numpts_0_date_run_2023_01_27_mdl_UNet_snglPat_True',
#     '2023_01_27_Titan_gpu_0_UNET_DosePred_InOut_KFold_Small/Kfold-4_Pats_001-110_InOut_gpu_0_ch_2_xyz_96x96x96_stepsz_20_ep_250_numpts_0_date_run_2023_01_27_mdl_UNet_snglPat_True',
#     '2023_01_27_Titan_gpu_0_UNET_DosePred_InOut_KFold_Small/Kfold-5_Pats_001-110_InOut_gpu_0_ch_2_xyz_96x96x96_stepsz_20_ep_250_numpts_0_date_run_2023_01_27_mdl_UNet_snglPat_True'
#             ]

# ----- Large -----
# UNET Target
# save_name1 = [
#     '2023_01_27_Titan_gpu_0_UNET_DosePred_Target_KFold_Large/Kfold-1_Pats_001-110_Target_gpu_0_ch_1_xyz_96x96x96_stepsz_20_ep_250_numpts_0_date_run_2023_01_27_mdl_UNet_snglPat_True',
#     '2023_01_27_Titan_gpu_0_UNET_DosePred_Target_KFold_Large/Kfold-2_Pats_001-110_Target_gpu_0_ch_1_xyz_96x96x96_stepsz_20_ep_250_numpts_0_date_run_2023_01_27_mdl_UNet_snglPat_True',
#     '2023_01_27_Titan_gpu_0_UNET_DosePred_Target_KFold_Large/Kfold-3_Pats_001-110_Target_gpu_0_ch_1_xyz_96x96x96_stepsz_20_ep_250_numpts_0_date_run_2023_01_27_mdl_UNet_snglPat_True',
#     '2023_01_27_Titan_gpu_0_UNET_DosePred_Target_KFold_Large/Kfold-4_Pats_001-110_Target_gpu_0_ch_1_xyz_96x96x96_stepsz_20_ep_250_numpts_0_date_run_2023_01_27_mdl_UNet_snglPat_True',
#     '2023_01_27_Titan_gpu_0_UNET_DosePred_Target_KFold_Large/Kfold-5_Pats_001-110_Target_gpu_0_ch_1_xyz_96x96x96_stepsz_20_ep_250_numpts_0_date_run_2023_01_27_mdl_UNet_snglPat_True'
#             ]

# UNET InnerOuterEDT
save_name1 = [
    '2023_01_27_Titan_gpu_0_UNET_DosePred_InOut_KFold_Large/Kfold-1_Pats_001-110_InOut_gpu_0_ch_2_xyz_96x96x96_stepsz_20_ep_250_numpts_0_date_run_2023_01_27_mdl_UNet_snglPat_True',
    '2023_01_27_Titan_gpu_0_UNET_DosePred_InOut_KFold_Large/Kfold-2_Pats_001-110_InOut_gpu_0_ch_2_xyz_96x96x96_stepsz_20_ep_250_numpts_0_date_run_2023_01_27_mdl_UNet_snglPat_True',
    '2023_01_27_Titan_gpu_0_UNET_DosePred_InOut_KFold_Large/Kfold-3_Pats_001-110_InOut_gpu_0_ch_2_xyz_96x96x96_stepsz_20_ep_250_numpts_0_date_run_2023_01_27_mdl_UNet_snglPat_True',
    '2023_01_27_Titan_gpu_0_UNET_DosePred_InOut_KFold_Large/Kfold-4_Pats_001-110_InOut_gpu_0_ch_2_xyz_96x96x96_stepsz_20_ep_250_numpts_0_date_run_2023_01_27_mdl_UNet_snglPat_True',
    '2023_01_27_Titan_gpu_0_UNET_DosePred_InOut_KFold_Large/Kfold-5_Pats_001-110_InOut_gpu_0_ch_2_xyz_96x96x96_stepsz_20_ep_250_numpts_0_date_run_2023_01_27_mdl_UNet_snglPat_True'
            ]

# use_best_model = True
use_best_model = False
# use_2D = True
use_2D = False

# model_channels = [0]
model_channels = [0,1]
# model_channels = [0,1,2]

# l_mno = np.array([70,70,10])
# l_mno = np.array([50,50,20])
# l_mno = np.array([64,64,16])
l_mno = np.array([96,96,96])
# l_mno = np.array([120, 120, 25])
stride_mno = np.array(np.ceil(l_mno/30), dtype = int)
# stride_mno = np.array([70,70,70])
stride_mno = np.array([20,20,20])
print(stride_mno)

work_directory_folder = '/home/pbruck/Scripts/DicomProgramsFromDavid/trainingRelatedPrograms_DS/'

# print('Loading file: ' + main_dir + files[3])
print('Loading file: ' + file_name0)
# data = np.load(file_name + '.npz')
# with open(main_dir + files[3], 'rb') as f:
with open(file_name0, 'rb') as f:
    DATA = pickle.load(f)
print('File Loaded')
all_files = []
all_files.append(DATA)



print(np.shape(DATA[0]))
train_img = []
train_lbl = []
targ_contours = []
        
# [9] = target contour, [3] = OG crop
# for i_file in range(len(all_files)):
#     for i_pat in range(len(all_files[i_file])):
# #         if i_pat == (len(all_files[i_file])-1):
#         for i_tar in range(len(all_files[i_file][i_pat][3])):
# #             if i_tar == (len(all_files[i_file][i_pat][3])-1):
#             this_targ = all_files[i_file][i_pat][9][i_tar]
#             if np.sum(this_targ) >= 4893.0:
#                 train_img.append([(all_files[i_file][i_pat][9][i_tar]/np.max(all_files[i_file][i_pat][9][i_tar]))])
#                 train_lbl.append([(all_files[i_file][i_pat][3][i_tar]/np.max(all_files[i_file][i_pat][3][i_tar]))])
#                 targ_contours.append(all_files[i_file][i_pat][9][i_tar])

# # [13] is for target contour edt, [14] is for outer edt, [3] is for OG crop
for i_file in range(len(all_files)):
    for i_pat in range(len(all_files[i_file])):
#         if i_pat == (len(all_files[i_file])-1):
        for i_tar in range(len(all_files[i_file][i_pat][3])):
#             if i_tar == (len(all_files[i_file][i_pat][3])-1):
            this_targ = all_files[i_file][i_pat][9][i_tar]
            if np.sum(this_targ) >= 4893.0:
                train_img.append([(all_files[i_file][i_pat][13][i_tar]/np.max(all_files[i_file][i_pat][13][i_tar])),
                                  (all_files[i_file][i_pat][14][i_tar]/np.max(all_files[i_file][i_pat][14][i_tar]))])
                train_lbl.append([(all_files[i_file][i_pat][3][i_tar]/np.max(all_files[i_file][i_pat][3][i_tar]))])
                targ_contours.append(all_files[i_file][i_pat][9][i_tar])
                                     
test_img = train_img
test_lbl = train_lbl


img_data = np.array(test_img)[:,:,:,:,:]

img_lbl = np.array(test_lbl)[:,:,:,:,:]
# img_lbl = np.array(test_lbl)[:,:,:,:]

print(len(train_img), len(train_img[0]), np.shape(train_img[0][0]))
print(len(train_lbl), len(train_lbl[0]), np.shape(train_lbl[0][0]))

print(len(test_img), len(test_img[0]), np.shape(test_img[0][0]))
# print(len(test_img_OAR), len(test_img_OAR[0]), np.shape(test_img_OAR[0][0]))
print(len(test_lbl), len(test_lbl[0]), np.shape(test_lbl[0][0]))

print('File Loaded')

#-------------------------------------------------------------------------------
##To take in input and stack the image appropriately 
##[pts][ch][xdim, ydim, zdim] --> [pts][xdim, ydim, zdim, ch] 
#-------------------------------------------------------------------------------
def OGin_2_ImgStk_lblStk_multipatient(data_img, data_lbl): 
    num_pts = len(data_img)
    num_ch = len(data_img[0])
    img_stk_multipt = []
    lbl_stk_multipt = []
    for i_pts in range(num_pts):
        for i_ch in range(num_ch):
            X_img = np.zeros((np.shape(data_img[i_pts][i_ch])[0],
                              np.shape(data_img[i_pts][i_ch])[1],
                              np.shape(data_img[i_pts][i_ch])[2],   
                              num_ch))
            X_img[:,:,:,i_ch] = data_img[i_pts][i_ch]
        img_stk_multipt.append(X_img)

        X_lbl = np.zeros((np.shape(data_lbl[i_pts][0])[0],
                          np.shape(data_lbl[i_pts][0])[1],         
                          np.shape(data_lbl[i_pts][0])[2],
                          1))
        X_lbl[:,:,:,0] = data_lbl[i_pts][0]
        lbl_stk_multipt.append(X_lbl)
    
    return img_stk_multipt, lbl_stk_multipt

#-------------------------------------------------------------------------------
##To construct a 4D chunked data stk for 1 patient 
##(can work for either img_stk or lbl_stk)
#-------------------------------------------------------------------------------
def ImgStk_2_4DChunkStk(X, l_mno, stride_mno):      
    OG_size = np.shape(X)

    l_m = l_mno[0]
    l_n = l_mno[1]
    l_o = l_mno[2]
    
    stride_m = stride_mno[0]
    stride_n = stride_mno[1]
    stride_o = stride_mno[2]
    
    max_i_size = OG_size[0] - l_m
    max_j_size = OG_size[1] - l_n
    max_k_size = OG_size[2] - l_o
    num_ch = OG_size[3]
    
    num_chunks = 0
    for k in range(0, max_k_size + stride_o, stride_o):
        for i in range(0, max_i_size + stride_m, stride_m):
            for j in range(0, max_j_size + stride_n, stride_n):
                num_chunks += 1
    
    chunk_stk = np.zeros((num_chunks, l_m, l_n, l_o, num_ch))
    i_chunk = -1
    for k in range(0, max_k_size + stride_o, stride_o):
        for i in range(0, max_i_size + stride_m, stride_m):
            for j in range(0, max_j_size + stride_n, stride_n):
                i_chunk += 1
                if i >=  max_i_size:
                    i = max_i_size   
                if j >=  max_j_size:
                    j = max_j_size   
                if k >=  max_k_size:
                    k = max_k_size   
                chunk_stk[i_chunk, :, :, :, :] = X[i:i+l_m, j:j+l_n, k:k+l_o, :]
    #     return chunk_stk_0, chunk_stk_1, chunk_stk_2, chunk_stk_L0
    return chunk_stk
#-------------------------------------------------------------------------------
##To construct a 3D chunked image stk from a 3D image
#-------------------------------------------------------------------------------
def ImgStk_2_3DChunkStk(X, l_mno, stride_mno): 
    OG_size = np.shape(X)

    l_m = l_mno[0]
    l_n = l_mno[1]
    l_o = l_mno[2]
    
    stride_m = stride_mno[0]
    stride_n = stride_mno[1]
    stride_o = stride_mno[2]
    
    max_i_size = OG_size[0] - l_m
    max_j_size = OG_size[1] - l_n
    max_k_size = OG_size[2] - l_o

    num_chunks = 0
    for k in range(0, max_k_size + stride_o, stride_o):
        for i in range(0, max_i_size + stride_m, stride_m):
            for j in range(0, max_j_size + stride_n, stride_n):
                num_chunks += 1
    
    chunk_stk_val = np.zeros((num_chunks, l_m, l_n, l_o))
    i_chunk = -1
    for k in range(0, max_k_size + stride_o, stride_o):
        for i in range(0, max_i_size + stride_m, stride_m):
            for j in range(0, max_j_size + stride_n, stride_n):
                i_chunk += 1
                if i >=  max_i_size:
                    i = max_i_size   
                if j >=  max_j_size:
                    j = max_j_size   
                if k >=  max_k_size:
                    k = max_k_size   
                chunk_stk_val[i_chunk, :, :, :] = X[i:i+l_m, j:j+l_n, k:k+l_o]
    return chunk_stk_val

#-------------------------------------------------------------------------------
## To reconstruct the original 3D imgstk from a 3Dchunked image stk
#-------------------------------------------------------------------------------
def Chunk3DStk_2_ImgStk(chunk_stk_val, OG_size, stride_mno): 
    print(np.shape(chunk_stk_val))
    l_m = np.shape(chunk_stk_val)[1]
    l_n = np.shape(chunk_stk_val)[2]
    l_o = np.shape(chunk_stk_val)[3]
    
    stride_m = stride_mno[0]
    stride_n = stride_mno[1]
    stride_o = stride_mno[2]
    
    max_i_size = OG_size[0] - l_m
    max_j_size = OG_size[1] - l_n
    max_k_size = OG_size[2] - l_o

    i_chunk = -1
    new_X_val = np.zeros(OG_size)
    new_X_track = np.zeros(OG_size)
    for k in range(0, max_k_size + stride_o, stride_o):
        for i in range(0, max_i_size + stride_m, stride_m):
            for j in range(0, max_j_size + stride_n, stride_n):
                i_chunk += 1
                if i >=  max_i_size:
                    i = max_i_size   
                if j >=  max_j_size:
                    j = max_j_size   
                if k >=  max_k_size:
                    k = max_k_size   
                new_X_val[i:i+l_m, j:j+l_n, k:k+l_o] = (new_X_val[i:i+l_m, 
                                                                  j:j+l_n, 
                                                                  k:k+l_o] 
                                                        + np.squeeze(chunk_stk_val[i_chunk, 
                                                                                   :, 
                                                                                   :, 
                                                                                   :], axis = 3))
                new_X_track[i:i+l_m, j:j+l_n, k:k+l_o]=(new_X_track[i:i+l_m, 
                                                                    j:j+l_n, 
                                                                    k:k+l_o] 
                                                        + 1)
    new_X_val = np.divide(new_X_val, new_X_track, out=np.zeros_like(new_X_val), where = new_X_track!=0)

    return new_X_val, new_X_track



folds = []

for i_fold in range(len(save_name1)):
    # -------------- load the saved model --------------
    from tensorflow.python.keras.models import model_from_json

    model_directory_folder = '/home/pbruck/Scripts/DicomProgramsFromDavid/trainingRelatedPrograms_DS/with_generator/'

    # Specify the fold number
    # kfold_num = 0
    # kfold_num = 1
    # kfold_num = 2
    # kfold_num = 3
#     kfold_num = 4
    file_name1 = save_name1[i_fold][:] #model name
    file_name2 = save_name1[i_fold][:] + '_best' #weights name

    # load json and create model
    json_file = open(model_directory_folder + 'DosePred/' + file_name1 + '_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model

    if use_best_model == False:
        loaded_model.load_weights(model_directory_folder + 'DosePred/' + file_name2 + '_weights.h5')
    else:
        loaded_model.load_weights(model_directory_folder + 'DosePred/' + file_name2 + '_best_weights' + '.h5')
    # model_multi = multi_gpu_model(loaded_model, gpus=4)
    print("Loaded model from disk")

    print(loaded_model.layers[0].output_shape)
    print()



    Z_img = (np.array(np.rollaxis(img_data,1,5)))
    # Z_lbl = (np.array(np.rollaxis(img_lbl,1,5)))
    Z_lbl = (np.array(np.rollaxis(img_lbl,1,5)))

    test_img_stk_multipt = Z_img
    test_lbl_stk_multipt = Z_lbl
    print(len(test_img_stk_multipt), np.shape(test_img_stk_multipt[0]))        
    print(len(test_lbl_stk_multipt), np.shape(test_lbl_stk_multipt[0]))       

    # test_img_stk_multipt, test_lbl_stk_multipt =  OGin_2_ImgStk_lblStk_multipatient(test_img, test_lbl)
    # print(len(test_img_stk_multipt), np.shape(test_img_stk_multipt[0]))        
    # print(len(test_lbl_stk_multipt), np.shape(test_lbl_stk_multipt[0]))        

    test_img_chunk_pts = []
    test_lbl_chunk_pts = []
    for i_pts in range(len(test_img_stk_multipt)):
        if i_pts%5 == 0:
            print(i_pts+1, ' : ', len(test_img_stk_multipt))
        test_img_chunk_pts.append([])
        X = test_img_stk_multipt[i_pts,:,:,:,:]
        test_img_chunk_pts[i_pts] = ImgStk_2_4DChunkStk(X, l_mno, stride_mno)

        test_lbl_chunk_pts.append([])
        Y = test_lbl_stk_multipt[i_pts,:,:,:,:]
        test_lbl_chunk_pts[i_pts] = ImgStk_2_4DChunkStk(Y, l_mno, stride_mno)
        if i_pts%5 == 0:
            print('      ', 
                  np.shape(test_img_chunk_pts[i_pts]),
                  np.shape(test_lbl_chunk_pts[i_pts]))
    print('Chunking test_img_stk complete: ', 
          np.shape(test_img_chunk_pts[0]),
          np.shape(test_lbl_chunk_pts[0]))

    #FOR 3D#
    if use_2D == False:
        print('starting prediction')
        dataX_img = test_img_chunk_pts
        dataX_lbl = test_lbl_chunk_pts

        chunk_test_pred = np.zeros((np.shape(dataX_lbl)))
        print('chunk_test_pred size = ', np.shape(chunk_test_pred))
        X_chunk = np.zeros((1,
                            np.shape(dataX_img)[2],
                            np.shape(dataX_img)[3],
                            np.shape(dataX_img)[4],
                            np.shape(dataX_img)[5]))

        for i_pred_pt in range(np.shape(dataX_img)[0]):
            for i_pred_chunk in range(np.shape(dataX_img)[1]):
                if i_pred_chunk%50 == 0:
                    print('Patient: ', i_pred_pt+1, ' : ', np.shape(dataX_img)[0],
                          ';   Chunk: ', i_pred_chunk + 1, ' : ', np.shape(dataX_img)[1])
    #             X_chunk[0, :,:,:,:] = np.squeeze(np.array(dataX_img[i_pred_pt][i_pred_chunk, :,:,:,:]))
                X_chunk[0, :,:,:,:] = np.array(dataX_img[i_pred_pt][i_pred_chunk, :,:,:,:])
                chunk_test_pred[i_pred_pt, i_pred_chunk, :,:,:,:] = loaded_model.predict(X_chunk)
        print(np.shape(chunk_test_pred))
        print(np.min(chunk_test_pred), np.max(chunk_test_pred), len(np.unique(chunk_test_pred)))



    test_pred = []
    test_track = []
    for i_targ in range(len(chunk_test_pred)):
        test_pred2, test_track2 = Chunk3DStk_2_ImgStk(chunk_test_pred[i_targ,:,:,:,:,:], np.array(np.shape(img_data[0][0])), stride_mno)
        test_pred.append(test_pred2)
        test_track.append(test_track2)
    folds.append(test_pred)
    print('     -----')
    print('     FOLDS:', np.shape(folds))
    print('     -----')

    # test_ind = 9
    # test_pred2, test_track2 = Chunk3DStk_2_ImgStk(chunk_test_pred[test_ind,:,:,:,:,:], np.array(np.shape(img_data[0][0])), stride_mno)
    print(np.shape(test_pred2))
    # size = np.shape(test)[0]
    # test_prediction = model.predict(test[0:20,:,:,:])
    # test_label2 = model.predict(test_label[0:20,:,:,:])
    # # test_loss, test_acc = model.evaluate(test[0:size,:,:,:], test_label[0:size,:])
    print(np.shape(test_pred2), np.shape(test_lbl) )
    print(np.min(test_pred2), np.max(test_pred2), len(np.unique(test_pred2)))
    # print(np.min(Z_lbl), np.max(Z_lbl), len(np.unique(Z_lbl)))

print()
print('done')

num_inp = len(test_img[0])
print(num_inp)

# Ground Truth Image
img2 = np.array(np.squeeze(Z_lbl))
img2 = img2/100
img2 = img2/np.max(img2)*1

# Input Channel 1
img3 = np.array(np.squeeze(img_data[:,0,:,:,:]))
img3 = img3
img3 = img3/np.max(img3)*1

# Input Channel 2 
if num_inp >= 2:
    inp_ch2 = np.array(np.squeeze(img_data[:,1,:,:,:]))
    inp_ch2 = inp_ch2/np.max(inp_ch2)*1

# Input Channel 3 
if num_inp >= 3:
    inp_ch3 = np.array(np.squeeze(img_data[:,2,:,:,:]))
    inp_ch3 = inp_ch3/np.max(inp_ch3)*1

all_pred = []
all_mse = []
for i_targ in range(len(chunk_test_pred)):
# for i_targ in range(8):
    targ_list = []
    # Ground Truth Image
    targ_list.append(img2[i_targ])
    
    # Target Contour
    targ_list.append(targ_contours[i_targ])
    
    # Input Channel 1
    targ_list.append(img3[i_targ])
    
    # Input Channel 2
    if num_inp >= 2:
        targ_list.append(inp_ch2[i_targ])
    
    # Input Channel 3
    if num_inp >= 3:
        targ_list.append(inp_ch3[i_targ])
    
    # Prediction Image, fold 1
    fold1 = np.squeeze(folds[0][i_targ])
    fold1[fold1<0]=0
    fold1 = fold1/100
    fold1 = (fold1 - np.min(fold1))/(np.max(fold1)-np.min(fold1))*1
    targ_list.append(np.squeeze(fold1)) 
    
    # Prediction Image, fold 2
    fold2 = np.squeeze(folds[1][i_targ])
#     fold2[fold2<0]=0
#     fold2 = fold2/100
#     fold2 = (fold2 - np.min(fold2))/(np.max(fold2)-np.min(fold2))*1
    targ_list.append(np.squeeze(fold2)) 
    
    # Prediction Image, fold 3
    fold3 = np.squeeze(folds[2][i_targ])
#     fold3[fold3<0]=0
#     fold3 = fold3/100
#     fold3 = (fold3 - np.min(fold3))/(np.max(fold3)-np.min(fold3))*1
    targ_list.append(np.squeeze(fold3)) 
    
    # Prediction Image, fold 4
    fold4 = np.squeeze(folds[3][i_targ])
#     fold4[fold4<0]=0
#     fold4 = fold4/100
#     fold4 = (fold4 - np.min(fold4))/(np.max(fold4)-np.min(fold4))*1
    targ_list.append(np.squeeze(fold4)) 
    
    # Prediction Image, fold 5
    fold5 = np.squeeze(folds[4][i_targ])
#     fold5[fold5<0]=0
#     fold5 = fold5/100
#     fold5 = (fold5 - np.min(fold5))/(np.max(fold5)-np.min(fold5))*1
    targ_list.append(np.squeeze(fold5)) 
    
    # Average prediction
    avg_pred = (fold1 + fold2 + fold3 + fold4 + fold5)/5
    targ_list.append(avg_pred)
    
    # Calc MSE
#     mse = ((img2[i_targ] - img1)**2).mean(axis=2)
#     all_mse.append(mse)
    
    # Put them all together
    all_pred.append(targ_list)

# print(np.shape(all_mse[0]))
# plt.figure()
# plt.imshow(all_mse[0], cmap='nipy_spectral')
# plt.colorbar()

colsize = int(len(all_pred[0])*10)
rowsize = int(len(test_lbl)*10)
# fig, axs = plt.subplots(nrows=len(all_pred), ncols=len(all_pred[0]), 
#                         figsize=(60,312))
fig, axs = plt.subplots(nrows=len(all_pred), ncols=len(all_pred[0]), 
                        figsize=(colsize,rowsize))
fig.subplots_adjust(hspace=0.1, wspace=0.1)
fig.patch.set_facecolor('white')
# axs = axs.ravel()
# print(range(len(axs)))

ind = 70
for j in range(len(all_pred)):
    for k in range(len(all_pred[0])):
        axs[j,k].imshow(all_pred[j][k][:,:,ind].astype(float), cmap='nipy_spectral')



# 2023.01.29 Targ & InOut Preds
# Small ----------
# predsave = 'Prediction2023-01-29_DosePred_Pats121-130_SmallTarg_Stride20-20-20_Mdl-UNet_Inp-InnerOuterEDT_NoNorm___5fold_23-01-27'
# predsave = 'Prediction2023-01-29_DosePred_Pats121-130_SmallTarg_Stride20-20-20_Mdl-UNet_Inp-Targ_NoNorm___5fold_23-01-27'

# predsave = 'Prediction2023-01-29_DosePred_Pats111-120_SmallTarg_Stride20-20-20_Mdl-UNet_Inp-InnerOuterEDT_NoNorm___5fold_23-01-27'
# predsave = 'Prediction2023-01-29_DosePred_Pats111-120_SmallTarg_Stride20-20-20_Mdl-UNet_Inp-Targ_NoNorm___5fold_23-01-27'

# Large ----------
predsave = 'Prediction2023-01-29_DosePred_Pats121-130_LargeTarg_Stride20-20-20_Mdl-UNet_Inp-InnerOuterEDT_NoNorm___5fold_23-01-27'
# predsave = 'Prediction2023-01-29_DosePred_Pats121-130_LargeTarg_Stride20-20-20_Mdl-UNet_Inp-Targ_NoNorm___5fold_23-01-27'

# predsave = 'Prediction2023-01-29_DosePred_Pats111-120_LargeTarg_Stride20-20-20_Mdl-UNet_Inp-InnerOuterEDT_NoNorm___5fold_23-01-27'  
# predsave = 'Prediction2023-01-29_DosePred_Pats111-120_LargeTarg_Stride20-20-20_Mdl-UNet_Inp-Targ_NoNorm___5fold_23-01-27'

with open(predsave, 'wb') as f:
    pickle.dump(all_pred, f)
print('Saved: ', predsave)
