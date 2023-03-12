#create single sector on for each size
#pull and save the dose image dose distribution

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage
import pickle
import operator
from scipy.ndimage import zoom
import os
import fnmatch
import scipy.special as sc

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

#************************************************************************************************
#*************************User controls and input************************************************
#************************************************************************************************
# main_directory = 'C:/Users/dsolis/DSolis_MIM/2019-04__Studies/ShotDistributions' #folder that contains patient folders *COMPLETED with patient skull distribution
# main_directory = '/data/PBruck/GKA/SectorActivations_Fred_Sphere/000_Fred_info/'
main_directory = '/data/PBruck/GKA/SectorActivations_Fred_Sphere/000_Sphere_info/'
# f_ext = ['shot*']
f_ext = ['*.d1']

# for i_filename in range(len(DD_Files)):
#==================================================================================================================
#Create file name for the H5 Data, plan data, and the savename to be used at the end
#==================================================================================================================
# savename = 'DD_sec_210921_Fred-Skull.DD'  #if saving cropped data change at bound for correct binary shot mask
savename = 'DD_sec_210921_Fred-Sphere.DD'

#==================================================================================================================
#Open and load in the H5 image data sets (H5 is not the true H5 format, but rather a pickled data set)
#==================================================================================================================
DD_Files = FindAllFiles(main_directory, f_ext)
for i_DD in range(len(DD_Files)):
    DD_Files[i_DD] = [DD_Files[i_DD], os.path.basename(os.path.normpath(DD_Files[i_DD])), []]

DD_Files = sorted(DD_Files, key=lambda x: [x[0], x[1]])

for i_DD in range(len(DD_Files)):
    with open(DD_Files[i_DD][0], 'rb') as f:
        DD_Files[i_DD][2] = pickle.load(f)
        'unpickled data'
        DD_Files[i_DD].pop(0)
        DD_Files[i_DD].insert(1, 0.5) #mm/vxl resolution
        DD_Files[i_DD][2] = np.array(np.swapaxes(DD_Files[i_DD][2],0,1))
    print(DD_Files[i_DD][0], DD_Files[i_DD][1], np.shape(DD_Files[i_DD][2]))
#     print(DD_Files[i_DD])
print('----------------------------------------------------')
    
DD_final = []
for i in range(10):
    DD_final.append([[],[],[]])

i_sector = 0
i_size = 0
for i_final in range(len(DD_Files)):
#     print(i_final, i_sector, i_size)
    if (i_final+1)%10 == 0:
#         print('modulo')
        i_sector = 0
        DD_final[i_sector][i_size] = DD_Files[i_final]
        i_size += 1
       
    else: 
        i_sector += 1
        DD_final[i_sector][i_size] = DD_Files[i_final]

for i in range(len(DD_final)):
    print(DD_final[i][0][0], '   |   ', DD_final[i][1][0], '   |   ', DD_final[i][2][0])
    
# SAVE normalized and thresholded CROPPED DOSE DISTRIBUTION #
print(savename)
print('--Preparing to save plan:' + savename + '--')
with open(main_directory + savename, 'wb') as f:
#     pickle.dump(DD_Files,f)
    pickle.dump(DD_final,f)
    print("----> Data saved: " + main_directory + savename)
