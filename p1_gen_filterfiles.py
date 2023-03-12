# Launch this program first
# follow by running p2, p3, and p4 in order

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

# import graphviz
# import pydot
# if you get a does not exist error, this means the full path name is too long, move the data or change folder names to shorter format
# main_directory_folder = 'C:/Users/dsolis/DSolis_MIM/2020-05_SpherePhantom/'
# main_directory_folder = '/home/pbruck/Scripts/GKA/FromDS/GK_Programs_20_0529/pts_folder/'
# main_directory_folder = '/data/PBruck/GKA/Patients/'
# main_directory_folder = '/home/pbruck/Scripts/GKA/FromDS/Multi_Pat/GKSim_001_01/'
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

img_sel_opt = 0 #when multiple CT exist, this can be used to choose which will be exported, in future export the list of list

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
        patterns = ['*.dcm','*.img','*.v2']
        for name in files:
            for ext in patterns:
                if fnmatch.fnmatch(name, ext):
                    filename = os.path.join(root, name)
#                     print(filename)
                    filenames.append(filename)
    return filenames

def split_SD(X):
    print(X)
    Y = [[]]
    res = [X[0][2]]
    print(res)
    ii = 3
    indx = [ii]
    for i in X: 
        ii += 1
        if i[2] not in res: 
            res.append(i[2])
            Y.append([i])
            indx.append(ii)
        else:
            Y[-1].extend([i])
    return Y, indx

def FilterFiles(PatientFiles):
    keep_ind = []
    allModality = []
    allInstancenum = []
    allSliceLocation = []
    allSeriesDescription = []
    allSortOrder = []
    mergedinfo = []
    
    doubles_test = 0
    RTStruct_global_Match = 0
    SD_global_Match  = 0
    CT_global_Match  = 0
    DST_global_Match = 0
    Plan_global_Match = 0
    
    for i_info in range(len(PatientFiles)):
        dcm_name = PatientFiles[i_info]
#         print('--> ', dcm_name, len(PatientFiles),'<--')
        allModality.append('')
        allInstancenum.append('')
        allSliceLocation.append('')
        allSeriesDescription.append('')
        allSortOrder.append(0)
#         print(i_info, os.path.exists(dcm_name), ' --> ', dcm_name)
        ds = pydicom.dcmread(dcm_name, force = True)

#         print(dcm_name)
#         print(ds)
#         print('-->   ', ds.ImagePositionPatient[2])
        
        if hasattr(ds, 'Modality'):
            Modality = ds.Modality
            allModality[i_info]=Modality
            
            ####################################
            #RTStruct - 1
            ####################################
            RTStruct_match = (len(re.findall('(?i)RTSTRUCT', Modality))>0)
            if RTStruct_match:
                RTStruct_global_Match += 1
                allSortOrder[i_info] = 1
                
            ####################################
            #Dose - 2
            ####################################
            RTDose_match = (len(re.findall('(?i)RTDOSE', Modality))>0)
            if RTDose_match and hasattr(ds, 'DoseSummationType'):
                DST = ds.DoseSummationType
#                 print(DST, '   --->   ', dcm_name)
                DST_match = (len(re.findall('(?i)Plan', DST))>0)
                if DST_match:
                    DST_global_Match += 1
                    allSortOrder[i_info] = 2
            else:
                DST = ''
                DST_match = False
                
            ####################################
            #Plan - 3
            ####################################
            Plan_match = (len(re.findall('(?i)Plan', Modality))>0)
            if Plan_match:
                Plan_global_Match += 1
                allSortOrder[i_info] = 3
            
            ####################################
            #CT - 4
            ####################################
            CT_match = (len(re.findall('(?i)CT', Modality))>0) and (RTStruct_match==0)
            MMN_match = False
            if CT_match and hasattr(ds, 'SeriesDescription'):
                Mftr_Model_Name = ds.ManufacturerModelName
                MMN_match = (len(re.findall('(?i)LGK ICON', Mftr_Model_Name))>0)
                CT_global_Match = 1
                allSortOrder[i_info] = 4
                allInstancenum[i_info]=ds.InstanceNumber
#                 if ds.InstanceNumber == 1:
#                     print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
#                     print(ds.InstanceNumber, '  ||  ', dcm_name)
#                     print('_________________________________________________________________________')
# #                     print(ds)
#                     print(ds.ManufacturerModelName, MMN_match)
#                     print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
#                 allSliceLocation[i_info]=ds.SliceLocation
                allSliceLocation[i_info]=ds.ImagePositionPatient[2]
                allSeriesDescription[i_info] = ds.SeriesDescription
                
            
            keep = RTStruct_match or DST_match or Plan_match or (MMN_match and CT_match)
            if keep:
                keep_ind.append(i_info)

    keep_global = (RTStruct_global_Match==1) and (CT_global_Match==1) and (DST_global_Match==1) and (Plan_global_Match==1)
    
    if keep_global == 1:
        #Filter out unwanted files from list
        PatientFiles = [i_keep for j_keep, i_keep in enumerate(PatientFiles) if j_keep in keep_ind] 
        allSortOrder = [i_keep for j_keep, i_keep in enumerate(allSortOrder) if j_keep in keep_ind] 
        allModality = [i_keep for j_keep, i_keep in enumerate(allModality) if j_keep in keep_ind]
        allInstancenum = [i_keep for j_keep, i_keep in enumerate(allInstancenum) if j_keep in keep_ind]
        allSliceLocation = [i_keep for j_keep, i_keep in enumerate(allSliceLocation) if j_keep in keep_ind]
        allSeriesDescription = [i_keep for j_keep, i_keep in enumerate(allSeriesDescription) if j_keep in keep_ind]
        
        mergedinfo = [(i_SO, i_PF, i_aM, i_SD, i_aI, i_aSL) 
                      for i0, i_SO, in enumerate(allSortOrder)
                      for i1, i_PF in enumerate(PatientFiles) if i0 == i1
                      for i2, i_aM in enumerate(allModality) if i0 == i2
                      for i3, i_SD in enumerate(allSeriesDescription) if i0 == i3
                      for i4, i_aI in enumerate(allInstancenum) if i0 == i4
                      for i5, i_aSL in enumerate(allSliceLocation) if i0 == i5]
        
        mergedinfo = sorted(mergedinfo, key=lambda tup:(tup[0], tup[3], tup[4]))
        
        for i_mi in range(len(mergedinfo)):
            mergedinfo[i_mi] = mergedinfo[i_mi][1:]

        #check for CT doubles, test for doubles of instance number for a given set of CT images
        doubles_test = mergedinfo.copy()        
        doubles_test = doubles_test[2:]
        IN_test = Counter([i_aI for (i_PF, i_aM, i_SD, i_aI, i_aSL) in doubles_test])
        doubles_test = (len(doubles_test)==len(IN_test))
        if doubles_test == True:
            doubles_test = 2
        else:
            doubles_test = 2  #change to 2 to force test to pass, default for test should be 1
            
    if (keep_global == False) or (doubles_test<2) :
        mergedinfo = []
        print('***********************************')
        if RTStruct_global_Match == False:
            print('---> ','This patient is missing RTStruct:')
        if CT_global_Match == False:
            print('---> ','This patient is missing CT data:')
        if DST_global_Match == False:
            print('---> ','This patient is missing RTDose:')
        if Plan_global_Match == False:
            print('---> ','This patient is missing Plan:')
        if doubles_test == 1:
            print('---> ','This patient has doubles of the CT Files:')
        if RTStruct_global_Match > 1:
            print('---> ','This patient has multiple RTStruct files:')

        print('---> ', PatientFiles[0])
        #         print(i, ' | ', i_info, ' | ', Modality, ' | ', SeriesDescript, ' | ', DST, ' || ', keep)
#     return keep_ind, allSortOrder, allModality, allInstancenum, allSliceLocation, allSeriesDescription
    return mergedinfo

if pull_pf_opt == 0:
    patient_folders = FindAllPTFolders(main_directory_folder)
else:
    patient_folders = pull_pf_list
    
patient_folders.sort()
patient_folders = [main_directory_folder + s + '/' for s in patient_folders]
print(patient_folders)
total_patient_num = len(patient_folders)  #use this to process all patients
# total_patient_num = 10  #total number of patient folder to process
pat_start_ind = 0      #the starting patient's index position from the sorted folder list (zero is default)

print(len(patient_folders), '/', total_patient_num)

i = pat_start_ind-1  #counter initialization
patient_num = 0      #counter initialization

if total_patient_num > len(patient_folders)-(i+1):
    print('The Total # of patient data desired is greater than the # of patient folders available')
    print('The following has been changed: total_patient_num = ' + str(len(patient_folders)-(i+1)))
    total_patient_num = len(patient_folders)-(i+1)

ALL_DATA = []
while (patient_num < total_patient_num and i < (len(patient_folders)-1)):
    patient_num += 1
    i=i+1
#     print(patient_folders[i])
    save_name = os.path.basename(os.path.normpath(patient_folders[i]))
    print(save_name)
    PatientFiles = FindAllFiles(patient_folders[i])

    mergedinfo = FilterFiles(PatientFiles)
    T, indT =  split_SD(mergedinfo[3:])
#     print(T)
    mergedinfo = mergedinfo[:3] + T[img_sel_opt] #img_sel_opt is the image to be exported default 0
    if len(mergedinfo)==0:
        patient_num -= 1            
        print('---> ',
              ' # Patients Tested: ', i+1,
              ' || ', patient_num, '/', total_patient_num)
        print('***skipping this patient***')
        print('***********************************')
    else:
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
        print('Now saving File: ' + 'FILT_FILES_' + save_name + '.npz')

        #check for/create new folder
        if not os.path.exists(main_directory_folder + save_name):
            os.makedirs(main_directory_folder + save_name)
        #save
#         import os
#         cwd = os.getcwd()
        np.savez(main_directory_folder + '/' + save_name + '/' + 'FILT_FILES_' + save_name + '.npz', 
                 ALL_DATA=mergedinfo)
            #     print('\n'.join(test))
        print('File Saved:')
        print(main_directory_folder + '/' + save_name + '/' + 'FILT_FILES_' +  save_name + '.npz')
        print('------Filtering Folder Files------')
        print('------------Completed------------')
        
#         for aTuple in ALL_DATA[patient_num-1][:3]:
#             print(aTuple[1:])
            
        print('# Patients Tested: ', i+1 , ' || ', patient_num, '/', total_patient_num)
        print('')
        

print('done')
