import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage
import pickle
import operator
import os
from zipfile import ZipFile
import fnmatch
import pydicom
import glob
import cv2
import re
from collections import Counter

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

def FindAllFiles(main_dir, ext_str): #ext_str must be [] of string like: ['*.dcm', '*.lgp']
    filenames = []
    for root, dirs, files in os.walk(main_dir):
        patterns = ext_str
        for name in files:
            for ext in patterns:
                if fnmatch.fnmatch(name, ext):
                    filename = os.path.join(root, name)
#                     print(filename)
                    filenames.append(filename)
    return filenames


def data_simple(D):
    D = D.copy()
    D_mod = []
    for D_i_export in range(len(D)):         
        for D_j_target in range(len(D[D_i_export][1])):
            t0 = D[D_i_export][1][D_j_target][0][0] #target #
            t1 = D[D_i_export][1][D_j_target][0][2] #target box width
            t2 = D[D_i_export][1][D_j_target][0][5] #target box center coordinate
            ta3 = D[D_i_export][1][D_j_target][0][3] #RX [Gy]
            ta4 = D[D_i_export][1][D_j_target][0][4] #% Iso
            tot_shot_weight = 0
            sum_corr_weight = 0
            for D_k_shot in range(len(D[D_i_export][1][D_j_target][1])):
                t3 = D[D_i_export][1][D_j_target][1][D_k_shot][1] #shot weight
                tot_shot_weight += t3
            for D_k_shot in range(len(D[D_i_export][1][D_j_target][1])):
                t3 = D[D_i_export][1][D_j_target][1][D_k_shot][1] #shot weight
                t4 = D[D_i_export][1][D_j_target][1][D_k_shot][4] #shot average size
                t5 = D[D_i_export][1][D_j_target][1][D_k_shot][5] #shot center
                t6 = D[D_i_export][1][D_j_target][1][D_k_shot][6] #relative shot center = shot center - target box center
                corr_weight = t3/tot_shot_weight
                sum_corr_weight += corr_weight
                D_mod.append([t0, t1, t2, ta3, ta4, corr_weight, t4, t5, t6])
#             print(sum_corr_weight)
    return D_mod

def data_simple2(D, D2):
    D = D.copy()
    D2 = D2.copy()
    D_mod = []
    for D_i_export in range(len(D)):         
        i_dcm = -1
        for D_j_target in range(len(D[D_i_export][1])):
            t0 = D[D_i_export][1][D_j_target][0][0] #target #
            t1 = D[D_i_export][1][D_j_target][0][2] #target box width
            t2 = D[D_i_export][1][D_j_target][0][5] #target box center coordinate
            ta3 = D[D_i_export][1][D_j_target][0][3] #RX [Gy]
            ta4 = D[D_i_export][1][D_j_target][0][4] #% Iso
            targ_name = D[D_i_export][1][D_j_target][0][1] # target name
            tot_shot_weight = 0
            sum_corr_weight = 0
            plan_name_i = D[D_i_export][0]
                        
            for D_k_shot in range(len(D[D_i_export][1][D_j_target][1])):
                t3 = D[D_i_export][1][D_j_target][1][D_k_shot][1] #shot weight
                tot_shot_weight += t3
            for D_k_shot in range(len(D[D_i_export][1][D_j_target][1])):
                i_dcm = i_dcm + 1
                t3 = D[D_i_export][1][D_j_target][1][D_k_shot][1] #shot weight
                t4 = D[D_i_export][1][D_j_target][1][D_k_shot][4] #shot average size
                t5 = D[D_i_export][1][D_j_target][1][D_k_shot][5] #shot center
                t6 = D[D_i_export][1][D_j_target][1][D_k_shot][6] #relative shot center = shot center - target box center
                tsectors = D[D_i_export][1][D_j_target][1][D_k_shot][3] #sector values
                corr_weight = t3/tot_shot_weight
                sum_corr_weight += corr_weight
                dcm_shot_coord = D2[i_dcm][-1]
                D_mod.append([t0, t1, t2, ta3, ta4, corr_weight, t4, t5, t6, dcm_shot_coord, tsectors, targ_name])
                print(plan_name_i, i_dcm, t5, dcm_shot_coord)
    return D_mod

#==================================================================================================================
#Create subfolder List
#==================================================================================================================
# main_directory = 'C:/Users/dsolis/DSolis_MIM/2020-05_SpherePhantom/'
# main_directory = '/home/pbruck/Scripts/GKA/FromDS/GK_Programs_20_0529/pts_folder/'
# main_directory = '/data/PBruck/GKA/Patients/Temp/'
# main_directory = '/data/PBruck/GKA/Patients/011-020/'
# main_directory = '/data/PBruck/GKA/Patients/021-030/'
# main_directory = '/data/PBruck/GKA/Patients/031-040/'
# main_directory = '/data/PBruck/GKA/Patients/041-050/'
# main_directory = '/data/PBruck/GKA/Patients/051-060/'
# main_directory = '/data/PBruck/GKA/Patients/061-070/'  #last here 08/17/2022
# main_directory = '/data/PBruck/GKA/Patients/071-080/'
# main_directory = '/data/PBruck/GKA/Patients/081-090/'
# main_directory = '/data/PBruck/GKA/Patients/091-100/'
# main_directory = '/data/PBruck/GKA/Patients/101-110/'
# main_directory = '/data/PBruck/GKA/Patients/111-120/'
main_directory = '/data/PBruck/GKA/Patients/121-130/'

sub_folder_list = ['md'] 

sub_folder_list = FindAllPTFolders(main_directory)
# sub_folder_list = [sub_folder_list[-1]]
#sub_folder_list = [main_directory] #use this to run code on specific patient folder
print(sub_folder_list)
sub_folder_list.sort()
print(sub_folder_list)

#==================================================================================================================
#Begin looping for each patient subfolder in the main directory
#==================================================================================================================
for i_sub_folder in range(len(sub_folder_list)):
#     if i_sub_folder == 0:   
        sub_folder = sub_folder_list[i_sub_folder] + '/'   #sub_folder = the patient specific folder
        #----------------get plan infor from .lgp------------------------
        lgp_full_file_name = FindAllFiles(main_directory + sub_folder, ['*.lgp'])[0]
        lgp_file_name = os.path.basename(os.path.normpath(lgp_full_file_name))

        # lgp_file_name = 'ZZ_DS_DS_GKsim_001_ZZ_DS_DS_GKsim_001_ZZ_DS_DS_GKsim_001.lgp'
        with ZipFile(main_directory + sub_folder + lgp_file_name, 'r') as zip: 
            # printing all the contents of the zip file 
        #     zip.printdir() 
            fdata = zip.extract('PatInfo.xml')
        #----------------get plan info from .dcm-------------------------    
        RT_PLAN_Folders = FindAllPTFolders(main_directory + sub_folder)
        for i in range(len(RT_PLAN_Folders)):
            if str.find(RT_PLAN_Folders[i], 'RTPLAN') != -1:
                RT_PLAN_FN = RT_PLAN_Folders[i] + '/'

        dcm_name = FindAllFiles(main_directory + sub_folder + RT_PLAN_FN, ['*.dcm'])[0]
        ds = pydicom.dcmread(dcm_name, force = True)


        #==================================================================================================================
        #-----NOW PULL INFOR FROM .lgp file-----#
        #==================================================================================================================
        xml_name = 'PatInfo.xml'
        # savename = lgp_file_name[:-4]+'.plan_info'
        # print(savename)
        f = open(fdata, 'r')
        fdoc = f.readlines()
        f.close()


        # print(len(fdoc))
        exam_flag = -1
        exp_flag = -1
        exp_id = 0
        data = []
        for i in range(len(fdoc)):
            line = fdoc[i][:-1]
            fdoc[i] = line
            if line == '<examinations':
                exam_flag += 1
                treat_flag = -1
                print('exam : ', exam_flag)
            if line == '<treatment_plans':
                plan_name = fdoc[i+1][:-1]
                start = ":"
                s = plan_name
                s = s.split(start)[1]
                plan_name = s

                treat_flag += 1
                target_flag = -1
                total_shots = 0
                exp_id = 0
                print('   treatment_plans : ', treat_flag, ' / ', plan_name)
            if '#approval_time' in line:
                start = "="
                end = ":"
                s = line
                s = (s.split(start))[1].split(end)[0]
                exp_id = int(s)
                if exp_id > 0:
                    exp_flag += 1
                    data.append([plan_name, [], []])
                print('      export_id / export_flag : ', exp_id, ' / ', exp_flag)
            if exp_id > 0:
#                 print('exp_id @ L72: ', exp_id, i)
                if line == '<targets':
                    target_flag += 1

                    #--Get Target Name--#
                    target_name = fdoc[i+1][:-1]

                    start = ":"
                    end = ""
                    s = target_name
                    sep = ':'
                    s = (s.split(start))[1:]
                    s = sep.join(s)
                    s = (s.split(start))[1:]
                    s = sep.join(s)
                    s=s.replace(' - ', ' ')
                    target_name = s

                    #----Get Target Center position----#
                    x_targ = fdoc[i+1+1][:-1]
                    y_targ = fdoc[i+1+2][:-1]
                    z_targ = fdoc[i+1+3][:-1]

                    start = ":"
                    s = x_targ
                    s = s.split(start)[1]
                    x_targ = round(float(s),3)

                    s = y_targ
                    s = s.split(start)[1]
                    y_targ = round(float(s),3)

                    s = z_targ
                    s = s.split(start)[1]
                    z_targ = round(float(s),3)

                    #------Get Width of box?------#
                    targ_width = fdoc[i+4+1][:-1]

                    start = ":"
                    s = targ_width
                    s = s.split(start)[1]
                    targ_width = round(float(s),3)

                    #--------Get Target Prescriptioon Dose and Isodose--------#
                    targ_RX = fdoc[i+5+1][:-1]
                    start = ":"
                    s = targ_RX
                    s = s.split(start)[1]
                    targ_RX = round(float(s),3)

                    targ_iso = fdoc[i+5+2][:-1]
                    start = ":"
                    s = targ_iso
                    s = s.split(start)[1]
                    targ_iso = round(float(s),3)

                    #---------Add Target info to Data--------#
                    data[exp_flag][1].append([[],[]]) #target info goes in 0 slot and shot infor for target goes in 1 slot
                    target_info = [target_flag, 
                                   target_name, 
                                   targ_width, 
                                   targ_RX, 
                                   targ_iso,
                                   [x_targ, y_targ, z_targ]]
                    data[exp_flag][1][target_flag][0] = target_info

                    #----------Print out Target Info----------#
                    print('         target_flag : ', target_flag, ' : ', target_name)
                    print('            width : ', targ_width)
                    print('            RX dose [Gy] / Iso [%] : ', targ_RX, 'Gy / ', targ_iso, '%')
                    print('            center [x,y,z] : ', [x_targ, y_targ, z_targ])

                    shot_flag = -1

                if line == '<shots':
                    start = ":"

                    shot_flag += 1
                    total_shots += 1
                    sectors = fdoc[i+1:i+1+8]
                    for i_sector in range(len(sectors)):
                        s = sectors[i_sector]
                        s = s.split(start)[1]
        #                 print(s)
                        sectors[i_sector] = int(s)
        #                 sectors[i_sector] = int(sectors[i_sector][-2:-1])
                    sectors_avg = np.mean(np.array(sectors[:]))
                    sectors_f = 0
                    if sectors_avg <= 6.0:
                        sectors_f = 4
                    if sectors_avg > 6.0 and sectors_avg < 10.0: 
                        sectors_f = 8
                    if sectors_avg >= 10.0:
                        sectors_f = 16

                    x_shot = fdoc[i+9+8][:-1]
                    y_shot = fdoc[i+9+8+1][:-1]
                    z_shot = fdoc[i+9+8+2][:-1]

                    s = x_shot
                    s = s.split(start)[1]
                    x_shot = round(float(s),3)

                    s = y_shot
                    s = s.split(start)[1]
                    y_shot = round(float(s),3)

                    s = z_shot
                    s = s.split(start)[1]
                    z_shot = round(float(s),3)

                    weight = fdoc[i+19+1]
                    start = ":"
                    end = "\n"
                    s = weight
                    s = (s.split(start))[1].split(end)[0]
                    weight = round(float(s),3)

                    shot_time = fdoc[i+20+3]
                    start = ":"
                    end = "\n"
                    s = shot_time
                    s = (s.split(start))[1].split(end)[0]
                    shot_time = round(float(s),4)

                    centered_shot = [round(x_shot - x_targ,3), round(y_shot - y_targ,3), round(z_shot-z_targ,3)]

                    #---------Add Shot info to Data--------#
                    data[exp_flag][1][target_flag][1].append([]) #target info goes in 0 slot and shot infor for target goes in 1 slot
                    shot_info = [shot_flag, 
                                 weight,
                                 shot_time,
                                 sectors,
                                 sectors_f,
                                 [x_shot, y_shot, z_shot],
                                 centered_shot]
                    data[exp_flag][1][target_flag][1][shot_flag] = shot_info

                    #---------Print Shot info to Data--------#
                    print('               ', total_shots,
                          ': ', shot_flag, 
                          ' : ', weight, 
                          ' : ', shot_time,
                          ' : ', sectors, 
                          ' : ', sectors_avg, ' / ', sectors_f, ' ', 
                          [x_shot, y_shot, z_shot], centered_shot)
                    data[exp_flag][2] = total_shots

        #==================================================================================================================
        #-----NOW PULL INFOR FROM PLAN .DCM-----#
        #==================================================================================================================
        target_list=[]
        for i_target in range(len(ds.FractionGroupSequence)):
            target = ds.FractionGroupSequence[i_target]
            target_num = target.FractionGroupNumber
            target_name = ds.DoseReferenceSequence[i_target].DoseReferenceDescription
            target_RX = ds.DoseReferenceSequence[i_target].TargetPrescriptionDose
            target_list.append([target_num, target_name, target_RX])

        print('-------------------------')
        coord_list = []
        for i_shot in range(len(ds.BeamSequence)):
            shot = ds.BeamSequence[i_shot]
            shot_num = shot.BeamNumber
            shot_name = shot.BeamName
            shot_namenum_0 = re.findall(r'(\w+?)(\d+)', shot_name)[0][0] #use the shot_namenum to match with .lgp info
            shot_namenum_1 = int(re.findall(r'(\w+?)(\d+)', shot_name)[0][1])
            for i_lbl in range(len(target_list)):
                target_lett = target_list[i_lbl][1][:str.find(target_list[i_lbl][1], ':')]
                if str.find(shot_name, target_lett) != -1:
                    target_lbl = target_list[i_lbl][1]
                    target_num = target_list[i_lbl][0] #sequenced for delivery
                    target_RX = target_list[i_lbl][2]
            shot_weight = shot.FinalCumulativeMetersetWeight
            shot_coord = shot.ControlPointSequence[0].IsocenterPosition
        #     shot_coord = list(np.multiply(shot_coord, np.array([1,-1, -1]))) #to correct .dcm coordinate for y, z flip of GK
            coord_list.append([target_num, shot_namenum_0, shot_namenum_1, target_lbl, shot_num, 
                               target_RX, shot_weight, shot_coord])

        coord_list = sorted(coord_list, key=lambda x: (x[0], x[2]))
        # for i in range(len(coord_list)):
        #     print(coord_list[i])

        # print('-------------------------')
        final_i = coord_list[-1][0]
        i_lbl = 0
        cum_weight = 0
        ind_tot_weight = []
        for i in range(final_i):
            cum_weight = 0
            i_lbl_start = i_lbl
            for ii in range(len(coord_list)):
                if i + 1 == coord_list[ii][0]:
                    cum_weight = cum_weight + coord_list[ii][6]
                    i_lbl += 1
            ind_tot_weight.append([i, i_lbl_start, i_lbl, cum_weight])

        i_lbl = 0
        coord_list_mod = []
        for i in range(len(coord_list)):
            if i == ind_tot_weight[i_lbl][2]:
                i_lbl += 1
            tot_cum_weight = ind_tot_weight[i_lbl][3]
            cum_weight = coord_list[i][-2]
            norm_weight = cum_weight/tot_cum_weight
            update_ph = coord_list[i][:-2] + [norm_weight] + [coord_list[i][-1]]
            coord_list_mod.append(update_ph)

        for i in range(len(coord_list_mod)):
            print(coord_list_mod[i])

        #==================================================================================================================
        #Save the shot information
        #==================================================================================================================
        for i in range(len(data)): #i indicates iteration through the plans
            #-------Save the 'data' one plan at a time------#
            savename = lgp_file_name[:-4]+ '__' + data[i][0] + '.plan_info'
            print('-----------------------------------------------------------')
            print(savename)
            print(i, exp_flag, len(data), )
            print('--Preparing to save plan:' + str(i) + ' / ' + str(exp_flag) + '--')
            print('# shots in .lgp / # shots in .dcm :  ', data[i][2], ' / ', len(coord_list_mod))
            if data[i][2] != len(coord_list_mod):
                print('Plan Shot # mismatch: SKIPPING / ', data[i][0])
            else:        
                with open(main_directory + sub_folder + savename, 'wb') as f:
                #     pickle.dump(data, f)
                    data_mod = data_simple2([data[i]], coord_list_mod) #From [[.lgp plan],[from .dcm plan]]
                    pickle.dump(data_mod,f)
                print("----> Data saved: ", savename)
