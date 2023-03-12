# Initialization and Declaration
import tensorflow as tf
from tensorflow.python.keras.layers import Add, Conv2D, Conv3D, Input, Dense, Flatten, Dropout, concatenate, Concatenate, Lambda, MaxPooling1D, BatchNormalization
from tensorflow.python.keras.layers import ThresholdedReLU, Activation, PReLU, LeakyReLU, ReLU, Cropping3D, ZeroPadding3D, MaxPooling3D, UpSampling3D, SeparableConv2D, Average, AveragePooling3D, average, MaxPool3D, GlobalMaxPooling3D, add, TimeDistributed, Reshape, LSTM, GRU, Conv2D, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, UpSampling2D, Cropping2D, Conv1D
from tensorflow.python.keras import models, optimizers
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.initializers import glorot_uniform

from sklearn.model_selection import KFold

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
from skimage.measure import regionprops
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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
# gpu_num = "0,1"
gpu_num = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num # "0, 1" for multiple


def OGData_2_gendata(X):
    X_new = []
    for i_O2g in range(np.shape(X)[0]):
        X_new.append([])
        for i_O2g_ch in range(np.shape(X)[4]):
            X_new[i_O2g].append(np.squeeze(X[i_O2g, :, :, :, i_O2g_ch]))
    return X_new

work_directory_folder = '/home/pbruck/Scripts/DicomProgramsFromDavid/trainingRelatedPrograms_DS/'
#===USER INPUTS===#
# Produces Batches
#allowed sizes for ReLU: [96, 96, 40][512,512,3][224,224,16][512x512x1]
date_run = '2023_01_27'

# Where the .dmaps files are located
main_dir = '/data/PBruck/GKA/Patients/'

files = [
    'Temp/2022-11-09_Pats_001-010___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCropBlur_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps',
    '011-020/2022-11-09_Pats_011-020___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCropBlur_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps',
    '021-030/2022-11-09_Pats_021-030___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCropBlur_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps',
    '031-040/2022-11-09_Pats_031-040___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCropBlur_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps',
    '041-050/2022-11-09_Pats_041-050___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCropBlur_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps',
    '051-060/2022-11-09_Pats_051-060___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCropBlur_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps',
    '061-070/2022-11-09_Pats_061-070___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCropBlur_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps',
    '071-080/2022-11-09_Pats_071-080___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCropBlur_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps',
    '081-090/2022-11-09_Pats_081-090___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCropBlur_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps',
    '091-100/2022-11-09_Pats_091-100___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCropBlur_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps',
    '101-110/2022-11-09_Pats_101-110___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCropBlur_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps'
        ]

# files = [
#     'Temp/2022-11-09_Pats_001-010___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCropBlur_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps',
#     '011-020/2022-11-09_Pats_011-020___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCropBlur_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps',
#     '021-030/2022-11-09_Pats_021-030___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCropBlur_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps',
#     '031-040/2022-11-09_Pats_031-040___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCropBlur_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps',
#     '041-050/2022-11-09_Pats_041-050___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCropBlur_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps'
#         ]

# files = [
#     'Temp/2022-10-11_Pats_001-010___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCrop_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps',
#     '011-020/2022-10-11_Pats_011-020___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCrop_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps',
#     '021-030/2022-10-11_Pats_021-030___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCrop_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps',
#     '031-040/2022-10-11_Pats_031-040___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCrop_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps',  
#     '041-050/2022-10-11_Pats_041-050___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCrop_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps'
#         ]

########################3 Networks from Steve Jiang##########################################
## "Three-Dimensional Radiotherapy Dose Prediction on 
##  Head and Neck Cancer Patients with a Hierarchically 
##  Densely Connected U-net Deep Learning Architecture"


all_files = []
print('Loading files:')
for i in range(len(files)):
    with open(main_dir + files[i], 'rb') as f:
        DATA = pickle.load(f)
        all_files.append(DATA)
    print('---')
    print('Loaded file: ' + main_dir + files[i])
print()
print('All files loaded')



# HD-UNET: Inputs = target
x_size = 96
y_size = 96
z_size = 96        # 1 for 2D

# model_channels = [0,1] # 0 -> Target EDT; 1 -> Outer EDT
model_channels = [0] # target
ch_size = len(model_channels)

num_pts = 0 #use num_pts = 0 to execute for all patients

batch_size = 1     # Pull # set of x,y,z size. Must be even to run on both gpu
step_size = 20     # Train on that step_size times before updating weights. Completes one epoch
epochs = 250
#=================#
# file_name0 = 'Patients_1-10_Target-Contours-DMaps_w_Uniform140Crop_3D-Depth_30-30-30_RxBinary_GIBinary_edtTarg_2021-06-08'
fold_0 = work_directory_folder + 'with_generator/' + 'DosePred/'
fold_1 = date_run + '_Titan_gpu_' + gpu_num + '_HD-UNET_DosePred_Target_KFold_Small' + '/'

save_name1 = ('Pats_001-110' + '_' + 'Target'
# save_name1 = (fold_1 + files[0][5:28] + '_' + 'TargOuterEDT'
# save_name1 = (fold_1 + files[1][7:28] 
              + '_' + 'gpu_' + gpu_num
              + '_' + 'ch_' + str(ch_size)
              + '_' + 'xyz_' + str(x_size) + 'x' + str(y_size) + 'x' + str(z_size)
              + '_' + 'stepsz_' + str(step_size)
              + '_' + 'ep_' + str(epochs)
              + '_' + 'numpts_' + str(num_pts)
              + '_' + 'date_run_' + date_run
              + '_' + 'mdl_HD-UNet'
              + '_' + 'snglPat_True')

os.makedirs(fold_0 + fold_1, exist_ok=True)

print('Start time @:', time.strftime("%Y-%m-%d %H:%M"))
print("files[0] = " + "'" + files[0] + "'")
print("save_name1 = " + "'" + save_name1 + "'")

# # Steve Jiang --> HD U-Net
conv_sq_size = (3,3,3)
pooling_size = (2,2,2)
pooling_size2= (2,2,2)
striding_size = (2,2,2)
striding_size2 = (2,2,2)
def BatchNorm_Activation(x):#relu before BN is recommended even though original paper uses BN followed by Relu
    out = Activation("relu")(x)
    out = BatchNormalization()(out)
    return out

def denseConv_unit(x, nfiltx, conv_size):
    out1 = Conv3D(nfiltx, conv_size, padding='same')(x)
    out1 = BatchNorm_Activation(out1)
    out1 = concatenate([x, out1], axis = 4)
    
    out2 = Conv3D(nfiltx, conv_size, padding='same')(out1)
    out2 = BatchNorm_Activation(out2)
    out = concatenate([out1, out2], axis = 4)
    return out

def denseDownSample_unit(x, pool_size, nfiltx, conv_size, stride_size):
#     out1 = AveragePooling3D(pool_size, padding='same')(x)
    out1 = MaxPooling3D(pool_size, padding='same')(x)
    out2 = Conv3D(nfiltx, conv_size, strides=stride_size, padding='same')(x)
    out = concatenate([out1, out2], axis = 4)
    return out

def denseUpsample_deConv_unit(x, xskip, nfiltx, conv_size, pool_size):
    out = UpSampling3D(pool_size)(x)
    nfiltx = nfiltx * 4 #4 is the number of levels downsampled used in model
    out = Conv3D(nfiltx, conv_size, padding='same')(out)
    out = BatchNorm_Activation(out)
    out = concatenate([out, xskip], axis = 4)
    return out

inputs = Input(shape=(x_size, y_size, z_size, ch_size)) #96, 96, 64, 3

nfilt = 16
C0 = denseConv_unit(inputs, nfilt, conv_sq_size)

nfilt = nfilt
C1 = denseDownSample_unit(C0, pooling_size, nfilt, conv_sq_size, striding_size)
C1 = denseConv_unit(C1, nfilt, conv_sq_size)

nfilt = nfilt
C2 = denseDownSample_unit(C1, pooling_size, nfilt, conv_sq_size, striding_size)
C2 = denseConv_unit(C2, nfilt, conv_sq_size)

nfilt = nfilt
C3 = denseDownSample_unit(C2, pooling_size, nfilt, conv_sq_size, striding_size)
C3 = denseConv_unit(C3, nfilt, conv_sq_size)

nfilt = nfilt
C4 = denseDownSample_unit(C3, pooling_size2, nfilt, conv_sq_size, striding_size2)
C4 = denseConv_unit(C4, nfilt, conv_sq_size)
C4 = denseConv_unit(C4, nfilt, conv_sq_size)

nfilt = nfilt
U3 = denseUpsample_deConv_unit(C4, C3, nfilt, conv_sq_size, pooling_size2)
U3 = denseConv_unit(U3, nfilt, conv_sq_size)

nfilt = nfilt
U2 = denseUpsample_deConv_unit(U3, C2, nfilt, conv_sq_size, pooling_size)
U2 = denseConv_unit(U2, nfilt, conv_sq_size)

nfilt = nfilt
U1 = denseUpsample_deConv_unit(U2, C1, nfilt, conv_sq_size, pooling_size)
U1 = denseConv_unit(U1, nfilt, conv_sq_size)

nfilt = nfilt
U0 = denseUpsample_deConv_unit(U1, C0, nfilt, conv_sq_size, pooling_size)
U0 = denseConv_unit(U0, nfilt, conv_sq_size)

final = Conv3D(1, (1,1,1), padding='same')(U0)
final = Activation("relu")(final)

model = Model(inputs=inputs, outputs=final)

model.summary()

# vol cutoff: 0.611625, based on 50% threshold of 8mm shot
# Also corresponds to 4893.0 voxels (0.5^3 * 0.001)

# 2022-10-11_Pats_021-030___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_
# 5-RxOG_6-RxRC_7-DistCrop_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_
# 13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo_16-FullTarg___Crops140_Dist-Beachball-phi15-100step-n^r.dmaps

print(np.shape(all_files[0]))
train_img = []
train_lbl = []

# # [13] is for target contour edt, [14] is for outer edt, [3] is for OG crop
# for i_pat in range(len(DATA)):
#     for i_tar in range(len(DATA[i_pat][3])):
#         train_img.append([(DATA[i_pat][13][i_tar]/np.max(DATA[i_pat][13][i_tar])),
#                            DATA[i_pat][14][i_tar]/np.max(DATA[i_pat][14][i_tar])])
#         train_lbl.append([(DATA[i_pat][3][i_tar]/np.max(DATA[i_pat][3][i_tar]))])


# [9] = target contour, [3] = OG crop
for i_file in range(len(all_files)):
    for i_pat in range(len(all_files[i_file])):
        for i_tar in range(len(all_files[i_file][i_pat][3])):
            this_targ = all_files[i_file][i_pat][9][i_tar]
            if np.sum(this_targ) < 4893.0:
                train_img.append([(all_files[i_file][i_pat][9][i_tar]/np.max(all_files[i_file][i_pat][9][i_tar]))])
                train_lbl.append([(all_files[i_file][i_pat][3][i_tar]/np.max(all_files[i_file][i_pat][3][i_tar]))])


# train_img = train_img[-1]
# train_lbl = train_lbl[-1]

print(np.shape(train_img), np.shape(train_lbl))
print()


print(len(train_img), len(train_img[0]), np.shape(train_img[0][0]))
print(len(train_lbl), len(train_lbl[0]), np.shape(train_lbl[0][0]))


print('File Loaded')

if num_pts == 0 or num_pts >=  len(train_img):
    img_data = np.array(train_img)[:,:,:,:,:]
    lbl_data = np.array(train_lbl)[:,:,:,:,:]
else:
    img_data = np.array(train_img)[:num_pts,:,:,:,:]
    lbl_data = np.array(train_lbl)[:num_pts,0:,:,:,:]

# Commented out adding noise when input is binary
# img_data[:, 0, :, :, :] += np.random.randint(-10, 10, size= np.shape(img_data[:, 0, :, :, :]) )*10**-6

# img_data[:, 1, :, :, :] += np.random.randint(-10, 10, size= np.shape(img_data[:, 1, :, :, :]) )*10**-6
# img_data[:, 2, :, :, :] += np.random.randint(-10, 10, size= np.shape(img_data[:, 2, :, :, :]) )*10**-6

img_data = np.array(img_data)[:,model_channels,:,:,:]
lbl_data = np.array(lbl_data)

print(len(img_data), len(img_data[0]), np.shape(img_data[0][0]))
print(len(lbl_data), len(lbl_data[0]), np.shape(lbl_data[0][0]), [np.min(lbl_data), np.max(lbl_data)])
print('')

for i_print in range(len(model_channels)):
    print(str(model_channels[i_print]) + ': ', 
          np.min(img_data[:, i_print, :, :, :]),
          np.max(img_data[:, i_print, :, :, :]))

# print('0: ', np.min(img_data[:, 0, :, :, :]),np.max(img_data[:, 0, :, :, :]))
# print('1: ', np.min(img_data[:, 1, :, :, :]),np.max(img_data[:, 1, :, :, :]))
# print('2: ', np.min(img_data[:, 2, :, :, :]),np.max(img_data[:, 2, :, :, :]))

print(np.shape(img_data))
print(np.shape(lbl_data))

kf = KFold(n_splits = 5, random_state=None, shuffle=False)
kf.get_n_splits(img_data)
fold_num = 1
for train_index, test_index in kf.split(img_data):
    # Save each fold's results separately
    save_pre = 'Kfold-' + str(fold_num) + '_'
    kfold_savename = fold_1 + save_pre + save_name1
    chk_pt_filepath = fold_0 + kfold_savename + '_best_weights' + '.h5' #save best as different file
    chk_pt_filepath_rec = fold_0 + kfold_savename + '_mostrecent_weights' + '.h5' #save best as different file
    csv_filename = fold_0 + kfold_savename + '_csvLog' + '.csv'
    
    print('TRAIN:', train_index, 'TEST:', test_index)
    X_train, X_test = img_data[train_index], img_data[test_index]
    y_train, y_test = lbl_data[train_index], lbl_data[test_index]
    
    fit_params = {'patient_data': X_train,
                  'labels': y_train,
                  'dimensions': (x_size, y_size, z_size),
    #               'dimensions': (100, 100, 20),
                  'normalized' : False,
                  'channels_last' : True,
                  'single_patient_batch' : True,
                  'single_slice_batch' : False,
                  'Dim2_option' : False,
                  'n_sets': batch_size}

    val_params = {'patient_data': X_test,
                  'labels': y_test,
                  'dimensions': (x_size, y_size, z_size),
    #               'dimensions': (100, 100, 20),
                  'normalized' : False,
                  'channels_last' : True,
                  'single_patient_batch' : False,
                  'single_slice_batch' : False,
                  'Dim2_option' : False,
                  'n_sets': batch_size}

    fit_generator = da.crop_dataGenerator(**fit_params).return_generator(calls = (step_size) * (epochs*2))
    # fit_generator = da.single_testing(**fit_params).single_img_generator(calls = step_size * epochs)
    val_generator = da.crop_dataGenerator(**val_params).return_generator(calls = (step_size) * (epochs*2))
    
    model_multi = model
    # model.compile(optimizer = "rmsprop", loss = root_mean_squared_error, 
    #               metrics =["accuracy"])

    # adam = optimizers.Adam(lr=10**-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=.5, amsgrad=False)
    learn = 0.001
    Nadam = optimizers.Nadam(lr=learn, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model_multi.compile(loss='mse', 
    #                     optimizer='Adagrad',
                        optimizer = Nadam,
                        metrics=['mse', 'mae', 'mape', 'cosine', 'accuracy', 'msle'])

    # chk_pt_filepath="DosePred_wghts_imprv-{epoch:02d}-{mse:.2f}.hdf5" #save best as different file
    csv_logger = callbacks.CSVLogger(csv_filename)
    callbacks_list = [callbacks.ModelCheckpoint(chk_pt_filepath, 
                                                monitor='val_loss',
                                                verbose=1,
                                                save_weights_only=True,
                                                save_best_only=True, 
                                                mode='min'),
                      callbacks.ModelCheckpoint(chk_pt_filepath_rec, 
                                                monitor='val_loss',
                                                verbose=0,
                                                save_weights_only=True,
                                                save_best_only=False, 
                                                mode='min'),
                      csv_logger]
    #---------------Save Model: saving now allows you to run intermediate testing with the recent model---------
    # serialize model to JSON
    model_json = model.to_json()
    with open(fold_0 + kfold_savename + '_model.json', "w") as json_file:
        json_file.write(model_json)
    #--------------

    history_1 = model_multi.fit_generator(generator = fit_generator,
                                          validation_data = val_generator,
                                          validation_steps=step_size,
                                          steps_per_epoch=step_size,
                                          epochs=epochs,
                                          callbacks = callbacks_list)
    
    # ------------ save the template model rather than the gpu_mode ----------------
    # serialize weights to HDF5
    model.save_weights(fold_0 + kfold_savename + '_weights.h5')
    print("Saved model to disk: ", kfold_savename)
    print('')
    print('completed @:', time.strftime("%Y-%m-%d %H:%M"))
    # print("file_name0 = " + "'" + file_name0 + "'")
    print("save_name1 = " + "'" + kfold_savename + "'")
    print('')
    
    fold_num += 1
