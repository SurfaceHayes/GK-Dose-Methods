#Created by David Solis 08/03/18

import numpy as np
# import h5py
import matplotlib.pyplot as plt
import pydicom
import glob
import os
import cv2
import fnmatch
import re
from collections import Counter
import scipy
import pydicom.uid
from skimage.transform import warp
from skimage.draw import polygon
from scipy import ndimage
import pickle
from pathlib import Path
from rt_utils import RTStructBuilder

def FindAllPTFolders(main_dir):
#     PTfoldernames = os.path.abspath(main_dir) for name in os.listdir(".") if os.path.isdir(main_dir)
    PTfoldernames = []
#     i_f = 0
    for folder_name in os.listdir(main_dir):
        if os.path.isdir(os.path.join(main_dir, folder_name)):
#             i_f += 1
            PTfoldernames.append(folder_name)
#             print(i_f, ' -- ', folder_name)
    return PTfoldernames

def FindAllFiles(main_dir):
    filenames = []
    for root, dirs, files in os.walk(main_dir):
        patterns = ['*.npz']
        for name in files:
            for ext in patterns:
                if fnmatch.fnmatch(name, ext):
                    filename = os.path.join(root, name)
#                     print(filename)
                    filenames.append(filename)
    return filenames

# main_directory_folder = 'C:/Users/dsolis/DSolis_MIM/2020-05_SpherePhantom/'
# main_directory_folder = '/home/pbruck/Scripts/GKA/FromDS/GK_Programs_20_0529/pts_folder/'
# main_directory_folder = '/data/PBruck/GKA/Patients/Temp/'
# main_directory_folder = '/data/PBruck/GKA/Patients/011-020/'
# main_directory_folder = '/data/PBruck/GKA/Patients/021-030/'
# main_directory_folder = '/data/PBruck/GKA/Patients/031-040/'
# main_directory_folder = '/data/PBruck/GKA/Patients/041-050/'
# main_directory_folder = '/data/PBruck/GKA/Patients/051-060/'
# main_directory_folder = '/data/PBruck/GKA/Patients/061-070/'
# main_directory_folder = '/data/PBruck/GKA/Patients/071-080/'
# main_directory_folder = '/data/PBruck/GKA/Patients/081-090/'
# main_directory_folder = '/data/PBruck/GKA/Patients/091-100/'
# main_directory_folder = '/data/PBruck/GKA/Patients/101-110/'
# main_directory_folder = '/data/PBruck/GKA/Patients/111-120/'
main_directory_folder = '/data/PBruck/GKA/Patients/121-130/'
pull_pf_opt = 0 #find patient folder option, 0 - finds folders, 1 - uses pull_pf_list
pull_pf_list = ['md'] #specific patient folder subdirectory must be a list


def LIST2DICT(pt_all_): #pt_all_ = IMG_INFO_DOSE_STRUCT[i_patient_num]
    #--------------------------------------------------
    pt_all = {}
    
    #-----------------CT CONVERSION----------------------
    CT_ = pt_all_[0]
    CT_INFO_ = pt_all_[1]
    CT_IMG = {}
    CT_INFO = {}
    for i_dict in range(np.shape(CT_)[2]):
        CT_IMG.update({i_dict: CT_[:,:,i_dict]})
        CT_INFO.update({i_dict : CT_INFO_[i_dict]})
    CT = {'CT_IMG' : CT_IMG, 
          'CT_INFO' : CT_INFO}
    
    #-----------------DOSE CONVERSION--------------------
    RTDOSE_ = pt_all_[2]
    DOSE_IMG = {}
    for i_dict in range(np.shape(RTDOSE_)[2]):
        DOSE_IMG.update({i_dict: RTDOSE_[:,:,i_dict]})
    RTDOSE = {'DOSE_IMG' : DOSE_IMG}
    
    #-----------------RTSTRUCT CONVERSION-----------------
    RTSTRUCT_ = pt_all_[3]
    NAMES = {}
    COLORS = {}
    MASKS = {}
    TAGS = {}
    for i_dict in range(len(RTSTRUCT_[0])):
        NAMES.update({i_dict : RTSTRUCT_[0][i_dict]})
        COLORS.update({i_dict : RTSTRUCT_[1][i_dict]})
        TAGS.update({i_dict : RTSTRUCT_[3][i_dict]})
        
        MASK_IMG_ = RTSTRUCT_[2][i_dict]
        MASK_IMG = {}
        if len(MASK_IMG_) == 0:
            MASKS.update({i_dict : []})
        else:
            for j_dict in range(np.shape(MASK_IMG_)[2]):
                boolean2DIMG = np.array(MASK_IMG_[:,:,j_dict], dtype = np.bool)
                MASK_IMG.update({j_dict : boolean2DIMG})
            MASKS.update({i_dict : MASK_IMG})
        
    RTSTRUCT = {'NAMES' : NAMES, 
                'COLORS' : COLORS, 
                'MASKS' : MASKS,
                'TAGS' : TAGS}   
    
    ###USE BELOW IF NAME IS DESIRED RATHER THAN ITERATION
    #     RTSTRUCT_ = pt_all_[3]
    #     RTSTRUCT = {}
    #     for i_dict in range(len(RTSTRUCT_[0])):  
    #         NAME = RTSTRUCT_[0][i_dict]
    #         COLOR = RTSTRUCT_[1][i_dict]

    #         MASK_IMG_ = RTSTRUCT_[2][i_dict]
    #         MASK_IMG = {}
    #         for j_dict in range(np.shape(MASK_IMG_)[2]):
    #             boolean2DIMG = np.array(MASK_IMG_[:,:,j_dict], dtype = np.bool)
    #             MASK_IMG.update({j_dict : boolean2DIMG})

    #         RTSTRUCT.update({NAME : {'COLOR' : COLOR, 
    #                                  'MASK' : MASK_IMG}})
    #--------------------------------------------------
    pt_all.update({'CT' : CT, 
                   'RTDOSE' : DOSE_IMG,
                   'RTSTRUCT' : RTSTRUCT})
    ##################################################################
    ## Use format below to readout entries from the nested dictionary
    ##################################################################
    #     print(type(pt_all.keys()))
    #     print(pt_all.keys())
    #     print('=-=-=-=-=-=-=-=')
    #     print('CT')
    #     print('=-=-=-=-=-=-=-=')
    #     print('----', pt_all['CT'].keys())
    #     print('----', '----', pt_all['CT']['CT_IMG'].keys())
    #     print('----', '----', '----',pt_all['CT']['CT_IMG'][0])
    #     print('----', '----', pt_all['CT']['CT_INFO'].keys())
    #     print('----', '----', '----',pt_all['CT']['CT_INFO'][0])
    #     print('=-=-=-=-=-=-=-=')
    #     print('RTDOSE')
    #     print('=-=-=-=-=-=-=-=')
    #     print('----', pt_all['RTDOSE'].keys())
    #     print('----','----',pt_all['RTDOSE']['DOSE_IMG'].keys())
    #     print('=-=-=-=-=-=-=-=')
    #     print('RTSTRUCT')
    #     print('=-=-=-=-=-=-=-=')
    #     print('----',pt_all['RTSTRUCT'].keys())
    #     print('----','----',pt_all['RTSTRUCT']['NAMES'].keys())
    #     print('----','----','----',pt_all['RTSTRUCT']['NAMES'][0])
    #     print('----','----',pt_all['RTSTRUCT']['COLORS'].keys())
    #     print('----','----', '----',pt_all['RTSTRUCT']['COLORS'][0])
    #     print('----','----',pt_all['RTSTRUCT']['MASKS'].keys())
    #     print('----','----','----',pt_all['RTSTRUCT']['MASKS'][0].keys())
    
    return pt_all

# print(pt_folders[2])

if pull_pf_opt == 0:
    pt_folders = FindAllPTFolders(main_directory_folder)
else:
    pt_folders = pull_pf_list

print(pt_folders)
pt_folders.sort()
print(pt_folders)
# total_patient_num = len(pt_folders)
total_patient_num = 10
start_patient_num = 0 #change this number to patient number it was working on when it crashed minus 1

IMG_INFO_STRUCT = []
global_i_pt = -1
keep_global_flag = 0 #if 1 keep a global list for all patients, if 0, overwrite same position (helps memory issues)
for i_patient_num in range(start_patient_num,len(pt_folders)):
    #------------------------------------------------
    #Load in the preprocessed data set
    #------------------------------------------------
    filename = FindAllFiles(main_directory_folder + pt_folders[i_patient_num])[0]
    save_name_pre = 'GK_IMG_INFO_STRUCT_DOSE'
    print('Loading file: ' + filename)
    data = np.load(filename, allow_pickle=True)
    ALL_DATA = data['ALL_DATA']
    print('File Loaded')
    # print(len(ALL_DATA))

    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    print('Begin processing Patient: ' + str(i_patient_num+1) + ' / ' + str(len(pt_folders)))
    if keep_global_flag:
        global_i_pt += 1
#         IMG_INFO_STRUCT.append([]) #enable this if you want a final list with all pt data
    else:
        global_i_pt = 0
        IMG_INFO_STRUCT=[[]]
        pt_all = []
        
#     i_patient_num = 4
#     print(patient_data[i_patient_num][1])
    patient = ALL_DATA
    patient_RTSTRUCT = pydicom.dcmread(patient[0][0], force = True)
    patient_RTDOSE = pydicom.dcmread(patient[1][0], force = True)
    patient_PLAN = pydicom.dcmread(patient[2][0], force = True)
    if patient[3][0][-3:]=='.v2': #[CT DATA][1st .dcm image]
        patient_RTSTRUCT.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian 
        patient_RTDOSE.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian     
        patient_PLAN.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian  
    
    #------------------------------------------------------------
    #Create the CT_IMG stack and corresponding CT_Info_stack
    #important output called: 
    #dcm_img_stk
    #dcm_img_stk_list
    #------------------------------------------------------------
    print('Stacking CT Image -->')
    patient_CT = patient[3:]    
    ds = pydicom.dcmread(patient_CT[0][0], force = True)
    if patient_CT[0][0][-3:]=='.v2':
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian 
    width = int(ds.Columns)
    height = int(ds.Rows)
#     print(patient_CT[0][0][-2:])
    dcm_img_stk = np.zeros((width, height, len(patient_CT)))
    dcm_img_stk_list = [0 for x in range(len(patient_CT))]

    for i_CT in range(len(patient_CT)):
        ds = pydicom.dcmread(patient_CT[i_CT][0], force = True)
        if patient_CT[i_CT][0][-3:]=='.v2':
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian  # or whatever is the correct transfer syntax for the file
        dcm_img_stk[:,:,i_CT] = ds.pixel_array
        del ds.PixelData
        dcm_img_stk_list[i_CT] = ds   
    print('---> CT Image Stacked: size = ', np.shape(dcm_img_stk))
    
    
    
    #------------------------------------------------------------
    #Create the Dose Image, rescale and cast into space of CT_IMG
    #important output called: 
    #Dose_IMG
    #------------------------------------------------------------
    # filename_test = '/mnt/sdb/home/dsolis/patients_NMyziuk/1000106_Patient106/DATA/2.16.840.1.113669.2.931128.191185664.20130724143051.543732.dcm'
    # # ds = pydicom.dcmread(filename_test)
    print('   Calculating Dose Image')
    CT_img = dcm_img_stk
    CT_info = dcm_img_stk_list[0]
    CT_info_last = dcm_img_stk_list[-1]
    CT_origin = CT_info.ImagePositionPatient
    CT_origin_last = CT_info_last.ImagePositionPatient
    CT_SliceSpacing = CT_info.SliceThickness
    CT_RowSpacing = CT_info.PixelSpacing[0]
    CT_ColumnSpacing = CT_info.PixelSpacing[1]
    CT_num_rows, CT_num_columns, CT_num_slice = np.shape(CT_img)
#     print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#     print(CT_info)
#     print('---------------------------------------')
#     print('')
    ds = patient_RTDOSE
    Dose_Img_OG = ds.pixel_array #[frames, row, columns]
    Dose_origin = ds.ImagePositionPatient
#     Dose_SliceSpacing = ds.SliceThickness  #NOT SURE IF THIS IS CORRECT; DOUBLE CHECK LATER -DS
    Dose_SliceSpacing = CT_SliceSpacing
    Dose_RowSpacing = ds.PixelSpacing[0]
    Dose_ColumnSpacing = ds.PixelSpacing[1]
    Dose_num_rows = ds.Rows
    Dose_num_columns = ds.Columns
    Dose_num_slice = ds.NumberOfFrames
    Dose_grid_scale = ds.DoseGridScaling
    Dose_Img_OG = np.multiply(Dose_Img_OG, Dose_grid_scale)
#     print(ds)
#     print('=======================================')
#     print(np.shape(Dose_Img_OG))
#     print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    Dose_Img_new = np.zeros((np.shape(Dose_Img_OG)[1],np.shape(Dose_Img_OG)[2],np.shape(Dose_Img_OG)[0]))
    for i_new in range(np.shape(Dose_Img_OG)[0]):
        Dose_Img_new[:, :, i_new] = Dose_Img_OG[np.shape(Dose_Img_OG)[0]-i_new-1, :, :]
#         Dose_Img_new[:, :, i_new] = Dose_Img_OG[i_new, :, :]
    keep_going = 0
    if [Dose_num_rows - np.shape(Dose_Img_new)[0],
        Dose_num_columns - np.shape(Dose_Img_new)[1],
        Dose_num_slice - np.shape(Dose_Img_new)[2]] == [0,0,0]:
        keep_going = 1

    xx1 = np.arange(Dose_origin[1], Dose_RowSpacing*(Dose_num_rows) + Dose_origin[1], Dose_RowSpacing)
    yy1 = np.arange(Dose_origin[0], Dose_ColumnSpacing*(Dose_num_columns) + Dose_origin[0], Dose_ColumnSpacing)
    zz1 = np.arange(Dose_origin[2], Dose_SliceSpacing*(Dose_num_slice) + Dose_origin[2], Dose_SliceSpacing)
    zz1 = Dose_origin[2]+ np.flip(np.array(ds.GridFrameOffsetVector))
#     zz1 = Dose_origin[2] - np.array(ds.GridFrameOffsetVector)
    [X1,Y1,Z1] = np.meshgrid(xx1, yy1, zz1)

    xx2 = np.arange(CT_origin[1], CT_RowSpacing*(CT_num_rows) + CT_origin[1], CT_RowSpacing)
    yy2 = np.arange(CT_origin[0], CT_ColumnSpacing*(CT_num_columns) + CT_origin[0], CT_ColumnSpacing)
    zz2 = np.arange(CT_origin[2], CT_SliceSpacing*(CT_num_slice) + CT_origin[2], CT_SliceSpacing)
    [X2,Y2,Z2] = np.meshgrid(xx2, yy2, zz2)
    
    
    
##---copied out for troubleshooting---###
    Dose2Img = scipy.interpolate.RegularGridInterpolator((xx1, yy1, zz1), Dose_Img_new, bounds_error = False, fill_value = 0)
    V2 = Dose2Img(np.array([X2, Y2, Z2]).T)

    new_V2 = np.zeros((np.shape(V2)[1], np.shape(V2)[2],np.shape(V2)[0]))
    for i_new_V2 in range(np.shape(V2)[0]):
        new_V2[:, :, i_new_V2] = V2[i_new_V2, :, :]

    Dose_IMG = new_V2
    print('   ---> Dose Image Calculated: size = ', np.shape(Dose_IMG))
    
    
    
    #----------------------------------------------------------------
    #Create the RT Structures as binary masks in the space of CT_IMG
    #important output called: 
    #contours
    #---> contours is a list with 5 entries due to the nature of multiple contours on the same image
    #---> contours[0]: ROI Name
    #---> contours[1]: ROI Display Color
    #---> contours[2]: All original Contour points (x, y, z)
    #---> contours[3]: Slice Index correlated with contours[2]
    #---> contours[4]: The 3D contour images in the same ijk system as the dcm_img_stk
    #----------------------------------------------------------------
    print('      Creating RTStructure Binary Masks')
    # RT-Utils Method
    patient_CT[0][0] # CBCT filepath
    num_contours = len(patient_RTSTRUCT.StructureSetROISequence)
    print('            # Contours: ', num_contours)
    
    Available_contours = []
    for i_contour in range(num_contours):
        Available_contours.append(patient_RTSTRUCT.StructureSetROISequence[i_contour].ROIName) 
    print('            Available Contours: ', Available_contours)
    
    AllContoursFlag = True
    VIP_ROI_ind = [VIP_ROI_i 
                   for VIP_ROI_i, s in enumerate(Available_contours) 
                   if 'PTV' == s 
                   or 'ptv' == s
                   or '*Skull'== s
                   or 'Lungs' == s             
                   or 'lungs' == s
                   or 'Lung_R'== s
                   or 'Lung_L' == s
                   or 'RT_Lung' == s
                   or 'LT_Lung' == s
                   or 'heart' == s
                   or 'Heart' == s
                   or 'Cord' == s
                   or 'cord' == s
                   or 'GTV' == s
                   or 'CTV' == s
                   or 'GTV_Ave' == s
                   or AllContoursFlag]
    VIP_ROI_ind = np.array(VIP_ROI_ind)
    VIP_ROI_ind = VIP_ROI_ind[VIP_ROI_ind < len(patient_RTSTRUCT.ROIContourSequence)] #corrects ROI names with no Contour    
    num_contours = len(VIP_ROI_ind)
    print('            # contours after empty contour removal: ', num_contours)
    
    p = Path(patient[0][0])
    ct_path = Path(patient_CT[0][0])
    rtstruct = RTStructBuilder.create_from(
        dicom_series_path = Path(*ct_path.parts[:-1]),   # CBCT filepath, used for sizing from metadata
        rt_struct_path = p                      # RT Struct path to pull contours from
        )
    contours = []
    contours.append([])
    contours.append([])
    contours.append([])
    contours.append([])
    for i_num_contours in range(num_contours):
        contour_name = rtstruct.get_roi_names()[i_num_contours]
        contours[0].append(contour_name)
        contour_color = 'N/A'
        contours[1].append(contour_color)
        contour_mask = rtstruct.get_roi_mask_by_name(contour_name)
        contours[2].append(contour_mask)
        contour_tag = patient_RTSTRUCT.RTROIObservationsSequence[i_num_contours].RTROIInterpretedType
        if contour_tag == 'PTV':
            contours[3].append(contour_tag)
        elif contour_tag == 'EXTERNAL':
            contours[3].append(contour_tag)
        else:
            contours[3].append('Avoidance')
    

    print('      ---> RTStructure Binary Masks Created')
    
    
    
    
    #----------------------------------------------------------------
    #Create the final Output for processed patient
    #----------------------------------------------------------------
    print('         Compiling Final Output List')
    IMG_INFO_STRUCT[global_i_pt].append(dcm_img_stk)
    IMG_INFO_STRUCT[global_i_pt].append(dcm_img_stk_list)
    IMG_INFO_STRUCT[global_i_pt].append(Dose_IMG)
#     IMG_INFO_STRUCT[global_i_pt].append([contours[0], contours[1], contours[4]]) #only keeping name, color, 3D masks
    IMG_INFO_STRUCT[global_i_pt].append([contours[0], contours[1], contours[2], contours[3]]) #only keeping name, color, 3D masks from rt-util, and tags
    print('         ---> Final Output List Compiled')

    
    
    
    #----------------------------------------------------------------
    #Create the final Output for processed patient
    #----------------------------------------------------------------
    print('            Converting Output to .h5 file and Saving')
#     # find the patient folder name by locating the part of the filepath that all file types share
#     for i_commonpath in range(len(patient[0][0])):
#         if (patient[0][0][:i_commonpath] == patient[1][0][:i_commonpath] 
#         and patient[0][0][:i_commonpath] == patient[2][0][:i_commonpath]):
#             main_pathname = patient[0][0][:i_commonpath]
#     main_path_savename = os.path.basename(os.path.normpath(main_pathname)) #should be the patient folder
#     save_name = save_name_pre + '__' + "{:0>3d}".format(i_patient_num) + '__' + main_path_savename + '.h5'
    save_name = save_name_pre + '_' + pt_folders[i_patient_num] + '.h5'
    pt_all = LIST2DICT(IMG_INFO_STRUCT[global_i_pt])
#     pt_all = IMG_INFO_STRUCT[global_i_pt]
    
#     d = pt_all


# ***uncomment lines below to save***    
    with open(main_directory_folder + pt_folders[i_patient_num] + '/' + save_name, 'wb') as f:
        pickle.dump(pt_all, f, protocol=1)
    
    print('            ---> .h5 file saved as: ', save_name)
#*** ***


        ####################################################
        ## Use code below to readout the saved file ('.h5')
        ####################################################
        # import deepdish as dd

        # d2 = dd.io.load(save_name + '.h5')
        # print(d2.keys())
        # print('=-=-=-=-=-=-=-=')
        # print('CT')
        # print('=-=-=-=-=-=-=-=')
        # print('----', pt_all['CT'].keys())
        # print('----', '----', pt_all['CT']['CT_IMG'].keys())
        # print('----', '----', '----',pt_all['CT']['CT_IMG'][0])
        # print('----', '----', pt_all['CT']['CT_INFO'].keys())
        # print('----', '----', '----',pt_all['CT']['CT_INFO'][0].keys())
        # print('=-=-=-=-=-=-=-=')
        # print('RTDOSE')
        # print('=-=-=-=-=-=-=-=')
        # print('----', pt_all['RTDOSE'].keys())
        # print('----','----',pt_all['RTDOSE']['DOSE_IMG'].keys())
        # print('=-=-=-=-=-=-=-=')
        # print('RTSTRUCT')
        # print('=-=-=-=-=-=-=-=')
        # print('----',pt_all['RTSTRUCT'].keys())
        # print('----','----',pt_all['RTSTRUCT']['NAMES'].keys())
        # print('----','----',pt_all['RTSTRUCT']['COLORS'].keys())
        # print('----','----',pt_all['RTSTRUCT']['MASKS'].keys())

        
        
    #----------------------------------------------------------------
    #Done with Patient, let the user know
    #----------------------------------------------------------------
    print('')
    print('Patients processed: ' + str(i_patient_num+1) + ' / ' + str(len(pt_folders)))
#     print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')

print('done')
