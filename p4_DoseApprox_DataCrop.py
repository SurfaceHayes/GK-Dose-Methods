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
from scipy import ndimage
from math import nan, floor, ceil
import edt
from skimage.measure import label
import math
from scipy.ndimage.measurements import center_of_mass
from fuzzywuzzy import fuzz

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
#----------------------------------------------------------------------
# main_directory = 'C:/Users/dsolis/DSolis_MIM/2020-05_SpherePhantom/'
# sub_folder_list = ['md'] 
# main_directory = 'C:/Users/dsolis/Python Programs/3D_Gaussian_shots/GK_Sim_DATA/'
# main_directory = '/home/pbruck/Scripts/GKA/FromDS/GK_Programs_20_0529/pts_folder/'
# main_directory = '/data/PBruck/GKA/Patients/Temp/'         # Temp is 000-010
# main_directory = '/data/PBruck/GKA/Patients/011-020/'
# main_directory = '/data/PBruck/GKA/Patients/021-030/'
main_directory = '/data/PBruck/GKA/Patients/031-040/'
# main_directory = '/data/PBruck/GKA/Patients/041-050/'   
# main_directory = '/data/PBruck/GKA/Patients/051-060/'         
# main_directory = '/data/PBruck/GKA/Patients/061-070/'
# main_directory = '/data/PBruck/GKA/Patients/071-080/'
# main_directory = '/data/PBruck/GKA/Patients/081-090/'
# main_directory = '/data/PBruck/GKA/Patients/091-100/'           # Skipped 2/17
# main_directory = '/data/PBruck/GKA/Patients/101-110/'
# main_directory = '/data/PBruck/GKA/Patients/111-120/'
# main_directory = '/data/PBruck/GKA/Patients/121-130/'
# main_directory = '/home/pbruck/Scripts/GKA/FromDS/Multi_Pat/'
# sub_folder_list = ['ZZ_DS_PB_GKA_001'] 
# sub_folder_list = ['GKSim_001_01']
sub_folder_list = FindAllPTFolders(main_directory) #USE THIS AS DEFAULT
print(sub_folder_list)
sub_folder_list.sort()
print(sub_folder_list)

# sn_extra = '_md' #lets the user add in an extra identifier for the savename default to ''
sn_extra = ''
##############################
### YOU CAN USE THIS NEXT LINE TO RESIZE THE DOSE GRID TO A COARSER SCALE
ss = .5 #currently Dose distributions are only saved in 0.5 mm/vxl
vxl_size = np.array([ss, ss, ss]) #mm/voxel
vol_shot_size = np.array([224, 224, 224])  #mm dimensions for typical dose map 
                                           #(default is 224 for all, with shots scaled: small is 1/3, md is 1/3, lg is 1/3)
# vxl_size = res_val
print('shot sizes:: vxl_size [mm/vxl] = ', vxl_size, ';  vol_size [mm] = ', vol_shot_size)
##############################
#Control crop/save and plot options; disable plotting to run through data faster
crop_option = 0 # 0 - do not crop and save, 1 - crop and save
plot_opt = 0 #0 - do not plot after each save; 1 - create a plot of the last data set loaded
classic_opt = 0 #0 - enable sector shot distributions *automatically uses actual DD, 1 - classic shots only
use_DD = 1 #1 - use actual dose distributions; otherwise - uncomment desired function for single shot Dose distribution
xtnd_crop = 0 # 0 - crop dose maps to target size, 1 - extend beyond target box by 25%
#************************************************************************************************
# dd_file_loc = 'C:/Users/dsolis/DSolis_MIM/2019-04__Studies/ShotDistributionsDD_sec_200528.DD'
# dd_file_loc = '/home/pbruck/Scripts/GKA/FromDS/GK_Programs_20_0529/ShotDistributionsDD_sec_200528.DD'
dd_file_loc = '/data/PBruck/GKA/SectorActivations_Fred_Sphere/000_Fred_info/DD_sec_210921_Fred-Skull.DD'
# dd_file_loc = '/data/PBruck/GKA/SectorActivations_Fred_Sphere/000_Sphere_info/DD_sec_210921_Fred-Sphere.DD'
#************************************************************************************************
# Directory in data drive to save large Dose Sim files to
# save_dir = '/data/PBruck/GKA/DoseSims/'
save_dir = main_directory

def DictIMG2NPIMG(Dict_IMG): # Dict_IMG = dd.io.load(filenames[0], '/RTDOSE/DOSE_IMG')
    imax_Dict_IMG = np.shape(Dict_IMG[0])[0]
    jmax_Dict_IMG = np.shape(Dict_IMG[0])[1]
    kmax_Dict_IMG = len(Dict_IMG)
    NP_IMG = np.zeros((imax_Dict_IMG, jmax_Dict_IMG, kmax_Dict_IMG))
    for k_IMG in range(kmax_Dict_IMG):
        NP_IMG[:,:,k_IMG] = Dict_IMG[k_IMG]
    return NP_IMG

def Find_Specific_Mask_from_dict(d_NAMES, d_MASKS, name_str): #(d_RTSTRUCT_NAMES, d_RTSTRUCT_MASKS, name_str)
    num_names = len(d_NAMES)
    names = []
    for i_names in range(num_names):
        names.append(d_NAMES[i_names].replace(" ", "_"))
#         print(d_NAMES[i_names])
    names_indx = [names_i for names_i, s in enumerate(names) if name_str == s][0]
#     print(names_indx)
    d_STR_MASK_IMG = d_MASKS[names_indx]
    NP_MASK = DictIMG2NPIMG(d_STR_MASK_IMG)
    return NP_MASK

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)

#00000000000000--sphere_struct_xyz_old--0000000000000
#creates the shot matrix structure based on the calculation of a 3D gaussian profile
#fwhm in the x, y, z dimensions can be controlled individually using the paramater sxyz
#size of the matrix is defined in each dimension using the parameter s_size
def sphere_struct_xyz_old(sxyz, s_size):
    sx = s_size[0]
    sy = s_size[1]
    sz = s_size[2]
    
    sx_s = sxyz[0]
    sy_s = sxyz[1]
    sz_s = sxyz[2]
    
    struct_s = np.zeros((sx,sy,sz))
    
    cntr_x = (sx-1)/2
    cntr_y = (sy-1)/2
    cntr_z = (sz-1)/2
    
#     cntr_x = (sx)/2
#     cntr_y = (sy)/2
#     cntr_z = (sz)/2
    
    sx = np.arange(0, sx, 1)
    sy = np.arange(0, sy, 1)
    sz = np.arange(0, sz, 1)
    xx, yy, zz = np.meshgrid(sx, sy, sz)
    fwhm_x = sx_s
    fwhm_y = sy_s
    fwhm_z = sz_s
    sigma_x = fwhm_x/2.3548
    sigma_y = fwhm_y/2.3548
    sigma_z = fwhm_z/2.3548
    g_x = (1/(sigma_x*np.sqrt(2*np.pi))*np.exp(-(1/2)*((xx-cntr_x)/sigma_x)**2))
    g_y = (1/(sigma_y*np.sqrt(2*np.pi))*np.exp(-(1/2)*((yy-cntr_y)/sigma_y)**2))
    g_z = (1/(sigma_z*np.sqrt(2*np.pi))*np.exp(-(1/2)*((zz-cntr_z)/sigma_z)**2))                    
    struct_s = (g_x * g_y * g_z)
    
    return struct_s


#11111111111--sphere_struct_xyz--111111111111
def sphere_struct_xyz(sxyz, s_size):
    sx = s_size[0]
    sy = s_size[1]
    sz = s_size[2]
    
    sx_s = sxyz[0]
    sy_s = sxyz[1]
    sz_s = sxyz[2]
    
    struct_s = np.zeros((sx,sy,sz))
    
    cntr_x = (sx)/2
    cntr_y = (sy)/2
    cntr_z = (sz)/2
    
    rx = np.linspace(cntr_x - (cntr_x), cntr_x + (cntr_x), num=sx, endpoint=False)
    ry = np.linspace(cntr_y - (cntr_y), cntr_y + (cntr_y), num=sy, endpoint=False)
    rz = np.linspace(cntr_z - (cntr_z), cntr_z + (cntr_z), num=sz, endpoint=False)
    
    xx, yy, zz = np.meshgrid(rx, ry, rz)
    
    fwhm_x = sx_s
    fwhm_y = sy_s
    fwhm_z = sz_s
    
    sigma_x = fwhm_x/2.3548
    sigma_y = fwhm_y/2.3548
    sigma_z = fwhm_z/2.3548
    
    g_x = (1/(sigma_x*np.sqrt(2*np.pi))*np.exp(-(1/2)*((xx-cntr_x)/sigma_x)**2))
    g_y = (1/(sigma_y*np.sqrt(2*np.pi))*np.exp(-(1/2)*((yy-cntr_y)/sigma_y)**2))
    g_z = (1/(sigma_z*np.sqrt(2*np.pi))*np.exp(-(1/2)*((zz-cntr_z)/sigma_z)**2))         
    
    struct_s = (g_x * g_y * g_z)
    struct_s = struct_s/np.max(struct_s)
    return struct_s


#222222222222222--sphere_struct_dist--222222222222222
#creates the shot matrix structure based on the calculation of a 3D gaussian profile
#fwhm in the x, y, z dimensions can be controlled individually using the paramater sxyz
#size of the matrix is defined in each dimension using the parameter s_size
def sphere_struct_dist(sxyz, s_size, n_g):
    sx = s_size[0]
    sy = s_size[1]
    sz = s_size[2]
    
    sx_s = sxyz[0]
    sy_s = sxyz[1]
    sz_s = sxyz[2]
    
    struct_s = np.zeros((sx,sy,sz))

    cntr_x = (sx)/2
    cntr_y = (sy)/2
    cntr_z = (sz)/2
    
    rx = np.linspace(cntr_x - (cntr_x), cntr_x + (cntr_x), num=sx, endpoint=False)
    ry = np.linspace(cntr_y - ((cntr_y)*sx_s/sy_s), cntr_y + (cntr_y*sx_s/sy_s), num=sy, endpoint=False)
    rz = np.linspace(cntr_z - ((cntr_z)*sx_s/sz_s), cntr_z + (cntr_z*sx_s/sz_s), num=sz, endpoint=False)
        
    xx, yy, zz = np.meshgrid(rx, ry, rz)
    
    xx_d = xx - cntr_x
    yy_d = yy - cntr_y
    zz_d = zz - cntr_z
    
    DD = np.sqrt(xx_d**2 + yy_d**2 + zz_d**2)  #distance from center mesh grid in 3D
        
    fwhm_x = sx_s
    fwhm_y = sy_s
    fwhm_z = sz_s
    
    sigma_x = fwhm_x/2.3548
    sigma_y = fwhm_y/2.3548
    sigma_z = fwhm_z/2.3548
    
    g_D = (1/(sigma_x*np.sqrt(2*np.pi))*np.exp(-(1/2)*((DD)/sigma_x)**n_g)) #gaussian if n=2, supergauss if n>2         
#     g_D = (1/(1+DD/fwhm_x)**M_l)*(M_l*np.sin(2*np.pi/M_l)/(2*np.pi**2*fwhm_x**2))
    
    struct_s = g_D
    struct_s = struct_s/np.max(struct_s)
                    
    return struct_s

#3333333333333333333--sphere_struct_dist_super_Lorentz--33333333333333333333333
#creates the shot matrix structure based on the calculation of a 3D gaussian profile
#fwhm in the x, y, z dimensions can be controlled individually using the paramater sxyz
#size of the matrix is defined in each dimension using the parameter s_size
def sphere_struct_dist_super_Lorentz(sxyz, s_size, M_l, M_l2):
    sx = s_size[0]
    sy = s_size[1]
    sz = s_size[2]
    
    sx_s = sxyz[0]
    sy_s = sxyz[1]
    sz_s = sxyz[2]
    
    struct_s = np.zeros((sx,sy,sz))

    cntr_x = (sx)/2
    cntr_y = (sy)/2
    cntr_z = (sz)/2
    ####
    rx = np.linspace(cntr_x - (cntr_x), cntr_x + (cntr_x), num=sx, endpoint=False)
    ry = np.linspace(cntr_y - ((cntr_y)*sx_s/sy_s), cntr_y + (cntr_y*sx_s/sy_s), num=sy, endpoint=False)
    rz = np.linspace(cntr_z - ((cntr_z)*sx_s/sz_s), cntr_z + (cntr_z*sx_s/sz_s), num=sz, endpoint=False)
        
    xx, yy, zz = np.meshgrid(rx, ry, rz)
    
    xx_d = xx - cntr_x
    yy_d = yy - cntr_y
    zz_d = zz - cntr_z
    
    DD = np.sqrt(xx_d**2 + yy_d**2 + zz_d**2)  #distance from center mesh grid in 3D
        
    fwhm_x = sx_s/2
    fwhm_y = sy_s/2
    fwhm_z = sz_s/2
    
    g_D = np.abs(  (1/(1+(DD    /fwhm_x)**M_l[0]))*(M_l[0]*np.sin(2*np.pi/M_l[0])/(2*np.pi**2*fwhm_x**2)) )
    g_Dx = np.abs( (1/(1+(np.abs(xx_d)/fwhm_x)**M_l[0]))*(M_l[0]*np.sin(2*np.pi/M_l[0])/(2*np.pi**2*fwhm_x**2)) )
    g_Dy = np.abs( (1/(1+(np.abs(yy_d)/fwhm_x)**M_l[1]))*(M_l[1]*np.sin(2*np.pi/M_l[1])/(2*np.pi**2*fwhm_x**2)) )
    g_Dz = np.abs( (1/(1+(np.abs(zz_d)/fwhm_x)**M_l[2]))*(M_l[2]*np.sin(2*np.pi/M_l[2])/(2*np.pi**2*fwhm_x**2)) )

    g_D_2 = np.abs( (1/(1+(DD/fwhm_x)**M_l2[0]))*(M_l2[0]*np.sin(2*np.pi/M_l2[0])/(2*np.pi**2*fwhm_x**2)) )
    g_Dx2 = np.abs( (1/(1+(np.abs(xx_d)/fwhm_x)**M_l2[0]))*(M_l2[0]*np.sin(2*np.pi/M_l2[0])/(2*np.pi**2*fwhm_x**2)) )
    g_Dy2 = np.abs( (1/(1+(np.abs(yy_d)/fwhm_x)**M_l2[1]))*(M_l2[1]*np.sin(2*np.pi/M_l2[1])/(2*np.pi**2*fwhm_x**2)) )
    g_Dz2 = np.abs( (1/(1+(np.abs(zz_d)/fwhm_x)**M_l2[2]))*(M_l2[2]*np.sin(2*np.pi/M_l2[2])/(2*np.pi**2*fwhm_x**2)) )

    g_Dx = g_Dx/np.max(g_Dx)
    g_Dy = g_Dy/np.max(g_Dy)
    g_Dz = g_Dz/np.max(g_Dz)
    
    g_Dx2 = g_Dx2/np.max(g_Dx2)
    g_Dy2 = g_Dy2/np.max(g_Dy2)
    g_Dz2 = g_Dz2/np.max(g_Dz2)
    
    g_D = g_D/np.max(g_D)
    g_D_2 = g_D_2/np.max(g_D_2)
    
    g_D[g_D < .5] = 0
    g_D_2[g_D_2 >= .5] = 0
    
    g_Dx[g_Dx < .5] = 0
    g_Dy[g_Dy < .5] = 0
    g_Dz[g_Dz < .5] = 0
    
    g_Dx2[g_Dx2 >= .5] = 0
    g_Dy2[g_Dy2 >= .5] = 0
    g_Dz2[g_Dz2 >= .5] = 0
    
    g_Dx = g_Dx + g_Dx2
    g_Dy = g_Dy + g_Dy2
    g_Dz = g_Dz + g_Dz2
    
    struct_s =  (g_Dx * g_Dy * g_Dz)**(1/3) * (g_D + g_D_2)
    struct_s = struct_s/np.max(struct_s)
                    
    return struct_s

#4444444444444444444--sphere_struct_dist_gen_error_func--444444444444444444444444
#creates the shot matrix structure based on the calculation of a 3D gaussian profile
#fwhm in the x, y, z dimensions can be controlled individually using the paramater sxyz
#size of the matrix is defined in each dimension using the parameter s_size
def sphere_struct_dist_gen_error_func(sxyz, s_size, N_e):
    sx = s_size[0]
    sy = s_size[1]
    sz = s_size[2]
    
    sx_s = sxyz[0]
    sy_s = sxyz[1]
    sz_s = sxyz[2]
    
    struct_s = np.zeros((sx,sy,sz))

    cntr_x = (sx)/2
    cntr_y = (sy)/2
    cntr_z = (sz)/2
    
    rx = np.linspace(cntr_x - (cntr_x), cntr_x + (cntr_x), num=sx, endpoint=False)
    ry = np.linspace(cntr_y - ((cntr_y)*sx_s/sy_s), cntr_y + (cntr_y*sx_s/sy_s), num=sy, endpoint=False)
    rz = np.linspace(cntr_z - ((cntr_z)*sx_s/sz_s), cntr_z + (cntr_z*sx_s/sz_s), num=sz, endpoint=False)
        
    xx, yy, zz = np.meshgrid(rx, ry, rz)
    
    xx_d = xx - cntr_x
    yy_d = yy - cntr_y
    zz_d = zz - cntr_z
    
    DD = np.sqrt(xx_d**2 + yy_d**2 + zz_d**2)  #distance from center mesh grid in 3D
    
    fwhm_x = sx_s/2
    fwhm_y = sy_s/2
    fwhm_z = sz_s/2
    
#     sigma_x = fwhm_x/2.3548
#     sigma_y = fwhm_y/2.3548
#     sigma_z = fwhm_z/2.3548
    
#     g_D = (1/(sigma_x*np.sqrt(2*np.pi))*np.exp(-(1/2)*((DD)/sigma_x)**n_g))

    e_DD = np.sqrt(2) * (DD) / fwhm_x
#     g_D = (1/np.sqrt(np.pi)) * sc.gamma(N_e) * (sc.gamma(1/N_e) - (sc.gammainc(1/N_e, gen_DD**N_e)))
#     g_D = (1 - 1/np.sqrt(np.pi)) * sc.gammainc(1/2, e_DD**2)
    g_D = sc.erf(e_DD)
    P_0 = 0
    P_max = 1
    g_D = P_0 + P_max/2 * (1 - g_D**N_e)
    g_D = g_D/np.max(g_D)
    struct_s = g_D
    
    struct_s = struct_s/np.max(struct_s)
                    
    return struct_s

#5555555555555555555555--sphere_binary_struct--555555555555555555555555
#creates the shot matrix structure based on the calculation of a 3D gaussian profile
#fwhm in the x, y, z dimensions can be controlled individually using the paramater sxyz
#size of the matrix is defined in each dimension using the parameter s_size
def sphere_binary_struct(sxyz, s_size):
    sx = s_size[0] #array shape
    sy = s_size[1]
    sz = s_size[2]
    
    sx_s = sxyz[0]  #fwhm in ijk
    sy_s = sxyz[1]
    sz_s = sxyz[2]
    
    struct_s = np.zeros((sx,sy,sz))

    cntr_x = (sx)/2
    cntr_y = (sy)/2
    cntr_z = (sz)/2
    
    rx = np.linspace(cntr_x - (cntr_x), cntr_x + (cntr_x), num=sx, endpoint=False)
    ry = np.linspace(cntr_y - ((cntr_y)*sx_s/sy_s), cntr_y + (cntr_y*sx_s/sy_s), num=sy, endpoint=False)
    rz = np.linspace(cntr_z - ((cntr_z)*sx_s/sz_s), cntr_z + (cntr_z*sx_s/sz_s), num=sz, endpoint=False)
        
    xx, yy, zz = np.meshgrid(rx, ry, rz)
    
    xx_d = xx - cntr_x
    yy_d = yy - cntr_y
    zz_d = zz - cntr_z
    
    DD = np.sqrt(xx_d**2 + yy_d**2 + zz_d**2)  #distance from center mesh grid in 3D
    r_thresh = cntr_x
    print([np.min(xx_d), np.max(xx_d)], r_thresh)
    DD_thresh = np.array(DD)
    DD_thresh[DD > r_thresh] = 0
    DD_thresh[DD <= r_thresh] = 1

    struct_s = DD_thresh
    
    struct_s = struct_s/np.max(struct_s)
                    
    return struct_s

#-----------------------add_struct_xyz------------------------
#adds the shot matrix, M_shot, into the bigger matrix of the plan using the shot center defined by the start_coord
#now includes weight, w
def add_struct_xyz(M, start_coord_xyz, M_shot, w, v_sz):    
    start_coord = np.true_divide(start_coord_xyz, v_sz)
    struct_fill = w * M_shot
#     print(np.shape(struct_fill), len(struct_fill))
    print('----', start_coord)
    
    sx = np.shape(struct_fill)[0]
    sy = np.shape(struct_fill)[1]
    sz = np.shape(struct_fill)[2]
    
    M_bnd_x1 = np.around((start_coord[0]) - (sx)/2 ).astype(int)
    M_bnd_x2 = M_bnd_x1 + sx
    M_bnd_y1 = np.around((start_coord[1]) - (sy)/2).astype(int)
    M_bnd_y2 = M_bnd_y1 + sy
    M_bnd_z1 = np.around((start_coord[2]) - (sz)/2).astype(int)
    M_bnd_z2 = M_bnd_z1 + sz
#     print([M_bnd_x1, M_bnd_x2, np.absolute(M_bnd_x1), M_bnd_x2 - np.shape(M)[0]])
    if M_bnd_x1 < 0:
        diffx = np.absolute(M_bnd_x1)
        M_bnd_x1 = 0
        struct_fill = struct_fill[diffx:sx, :, :]
        
    if M_bnd_x2 > np.shape(M)[0]:
        diffx = M_bnd_x2 - np.shape(M)[0]
        M_bnd_x2 = np.shape(M)[0]
        struct_fill = struct_fill[:(np.shape(struct_fill)[0]-diffx), :, :]
    
    if M_bnd_y1 < 0:
        diffy = np.absolute(M_bnd_y1)
        M_bnd_y1 = 0
        struct_fill = struct_fill[:, diffy:sy, :]
        
    if M_bnd_y2 > np.shape(M)[1]:
        diffy = M_bnd_y2 - np.shape(M)[1]
        M_bnd_y2 = np.shape(M)[1]
        struct_fill = struct_fill[:, :(np.shape(struct_fill)[1]-diffy), :]
    
    if M_bnd_z1 < 0:
        diffz = np.absolute(M_bnd_z1)
        M_bnd_z1 = 0
        struct_fill = struct_fill[:, :, diffz:sz]
        
    if M_bnd_z2 > np.shape(M)[2]:
        diffz = M_bnd_z2 - np.shape(M)[2]
        M_bnd_z2 = np.shape(M)[2]
        struct_fill = struct_fill[:, :, :(np.shape(struct_fill)[2]-diffz)]
    
    M[np.int(M_bnd_x1):np.int(M_bnd_x2),
      np.int(M_bnd_y1):np.int(M_bnd_y2),
      np.int(M_bnd_z1):np.int(M_bnd_z2)] += struct_fill
    return M



#==================================================================================================================
#Create the single shot distribution
#*In the future, this will be replaced with a call to the file containing the captured dose distributions from
#*the tps
#==================================================================================================================
shot_size = (np.int(np.round(vol_shot_size[0]/vxl_size[0])), 
          np.int(np.round(vol_shot_size[1]/vxl_size[1])), 
          np.int(np.round(vol_shot_size[2]/vxl_size[2]))) # big matrix i,j,k size (Length, width, height)

#shot matrix creation
sm_vxl_sz = vxl_size[0] #mm/vox
md_vxl_sz = vxl_size[1] #mm/vox
lg_vxl_sz = vxl_size[2] #mm/vox

sm_fwhm_sz  = [6.16,  6.16,  5.04]  #fwhm; from gammaplan [x, y, z] mm from release memorandum
md_fwhm_sz  = [11.06, 11.06, 9.8]   #fwhm; from gammaplan [x, y, z] mm from release memorandum
lg_fwhm_sz  = [21.75, 21.75, 17.44] #fwhm; from gammaplan [x, y, z] mm from release memorandum

sm_eof = 0.8140 # effective output factor from release memorandum
md_eof = 0.9001 # effective output factor from release memorandum
lg_eof = 1.0000 # effective output factor from release memorandum

sm_fwhm_ijk = np.divide(sm_fwhm_sz,sm_vxl_sz) #fwhm in ijk
md_fwhm_ijk = np.divide(md_fwhm_sz,md_vxl_sz) #fwhm in ijk
lg_fwhm_ijk = np.divide(lg_fwhm_sz,lg_vxl_sz) #fwhm in ijk

sm_vol_sz = np.around(np.divide(shot_size, 3)).astype(int) #shot volume, voxel size in i,j,k
md_vol_sz = np.around(np.divide(shot_size, 3)).astype(int) #shot volume, voxel size in i,j,k
lg_vol_sz = np.around(np.divide(shot_size, 2)).astype(int) #shot volume, voxel size in i,j,k

# sm_vol_sz = M_size #shot volume, voxel size in i,j,k
# md_vol_sz = M_size #shot volume, voxel size in i,j,k
# lg_vol_sz = M_size #shot volume, voxel size in i,j,k

print(sm_vol_sz, md_vol_sz, lg_vol_sz)

#------------------------
#Choose shot distribution generation technique
#------------------------
# M_shot_sm = sphere_struct_xyz(sm_fwhm_ijk, sm_vol_sz)
# M_shot_md = sphere_struct_xyz(md_fwhm_ijk, md_vol_sz)
# M_shot_lg = sphere_struct_xyz(lg_fwhm_ijk, lg_vol_sz)
# print('initial sphere creation complete')

M_shot_sm = sphere_struct_dist(sm_fwhm_ijk, sm_vol_sz, 2)
M_shot_md = sphere_struct_dist(md_fwhm_ijk, md_vol_sz, 2)
M_shot_lg = sphere_struct_dist(lg_fwhm_ijk, lg_vol_sz, 4)  #n = 3.5, sz = 2.7
print('initial sphere creation complete')

# M_shot_sm = sphere_struct_dist_super_Lorentz(sm_fwhm_ijk, sm_vol_sz, [4, 4, 5], [2, 2, 7])
# M_shot_md = sphere_struct_dist_super_Lorentz(md_fwhm_ijk, md_vol_sz, [4, 4, 6], [2, 2, 8])
# M_shot_lg = sphere_struct_dist_super_Lorentz(lg_fwhm_ijk, lg_vol_sz, [8.5, 8.5, 18], [1.9, 1.9, 12])  #n = 8.5, sz = 2
# print('initial sphere creation complete')

# M_shot_sm = sphere_binary_struct(sm_fwhm_ijk, sm_vol_sz)
# M_shot_md = sphere_binary_struct(md_fwhm_ijk, md_vol_sz)
# M_shot_lg = sphere_binary_struct(lg_fwhm_ijk, lg_vol_sz)
# print('initial sphere creation complete')

#----------------------------------------------------------------------------------------
with open(dd_file_loc, 'rb') as f:
        DD_DATA = pickle.load(f)
        'unpickled data'
# for i in range(len(DD_DATA)):
#     print(DD_DATA[i][0][0], '   |   ', DD_DATA[i][1][0], '   |   ', DD_DATA[i][2][0])
if classic_opt != 1 or use_DD == 1:
    M_shot_sm = DD_DATA[0][2][2]
    M_shot_md = DD_DATA[0][1][2]
    M_shot_lg = DD_DATA[0][0][2]
#----------------------------------------------------------------------------------------
M_shot_sm = M_shot_sm/np.max(M_shot_sm[:]) #* sm_eof
M_shot_md = M_shot_md/np.max(M_shot_md[:]) #* md_eof
M_shot_lg = M_shot_lg/np.max(M_shot_lg[:]) #* lg_eof

ind1 = list(ndimage.measurements.center_of_mass(M_shot_sm))
ind1[0], ind1[1] = ind1[1], ind1[0]
ind2 = list(ndimage.measurements.center_of_mass(M_shot_md))
ind2[0], ind2[1] = ind2[1], ind2[0]
ind3 = list(ndimage.measurements.center_of_mass(M_shot_lg))
ind3[0], ind3[1] = ind3[1], ind3[0]

print('-->', ind1)
print('-->', ind2)
print('-->', ind3)

print(np.max(M_shot_sm[:]), np.max(M_shot_md[:]), np.max(M_shot_lg[:]))

#==================================================================================================================
#Plot the each dose distribution
#==================================================================================================================
%matplotlib inline
M_sm = np.swapaxes(np.squeeze(M_shot_sm[:, :, np.int(np.shape(M_shot_sm)[2]/2)]),0,1) #axial (Y vs X)
M_md = np.swapaxes(np.squeeze(M_shot_md[:, :, np.int(np.shape(M_shot_md)[2]/2)]),0,1) #axial (Y vs X)
M_lg = np.swapaxes(np.squeeze(M_shot_lg[:, :, np.int(np.shape(M_shot_lg)[2]/2)]),0,1) #axial (Y vs X)
# M_lg_flat = np.swapaxes(np.squeeze(M_shot_lg_flat[:, :, np.int(np.shape(M_shot_lg_flat)[2]/2)]),0,1) #axial (Y vs X)

fig, axs = plt.subplots(1, 3, figsize=(10, 5))

axs[0].imshow(M_sm/np.max(M_sm), origin='lower', cmap='nipy_spectral')
axs[1].imshow(M_md/np.max(M_md), origin='lower', cmap='nipy_spectral')
axs[2].imshow(M_lg/np.max(M_lg), origin='lower', cmap='nipy_spectral')
# axs[2].imshow(M_lg_flat, origin='lower', cmap='nipy_spectral')
# axs[3].imshow(M_lg_flat1, origin='lower', cmap='nipy_spectral')

M_sm = np.swapaxes(np.squeeze(M_shot_sm[:, np.int(np.shape(M_shot_sm)[1]/2),:]),0,1) #axial (Y vs X)
M_md = np.swapaxes(np.squeeze(M_shot_md[:, np.int(np.shape(M_shot_md)[1]/2),:]),0,1) #axial (Y vs X)
M_lg = np.swapaxes(np.squeeze(M_shot_lg[:, np.int(np.shape(M_shot_lg)[1]/2),:]),0,1) #axial (Y vs X)
# M_lg_flat = np.swapaxes(np.squeeze(M_shot_lg_flat[:, :, np.int(np.shape(M_shot_lg_flat)[2]/2)]),0,1) #axial (Y vs X)

fig, axs = plt.subplots(1, 3, figsize=(10, 5))

axs[0].imshow(M_sm, origin='lower', cmap='nipy_spectral')
axs[1].imshow(M_md, origin='lower', cmap='nipy_spectral')
axs[2].imshow(M_lg, origin='lower', cmap='nipy_spectral')

# temptest = []
# for i_sub_folder in range(len(sub_folder_list)):
#     temptest.append(sub_folder_list[i_sub_folder])
#     fake_insert = 123
    
# print(type(temptest[0]))



%%time

#******************************************************************************************************************
#******************************************************************************************************************
#Begin looping for each patient subfolder in the main directory

pat_list = []             # List of lists: patient number (folder name) -> dose maps (OG and sim)
for i_sub_folder in range(len(sub_folder_list)):
#     if i_sub_folder == 1:
#         pass
#     else:
        sub_folder = sub_folder_list[i_sub_folder] + '/'   #sub_folder = the patient specific folder
        dose_maps = []
        dose_maps.append(sub_folder_list[i_sub_folder])  
                # Temp list to contain both dose maps after creation
                # First append: dose_maps(0) = patient number (folder name)
                # Second and third appends: dose_maps[1] = OG, dose_maps[2] = sim
                # Fourth and fifth appends: dose_maps[3] = crop_og[], dose_maps[4] = crop_sim[]
                # Sixth and seventh appends: dose_maps[5] = rx_og[], dose_maps[6] = rx_rc[]
                    # 6th & 7th are also cropped

                # Eigth append: dose_maps[7] = dist_crop[]
                    # Distance from external, cropped to target

    #             Eigth append: dose_maps[7] = dist_og[]
    #                 Distance from external, full array

    #******************************************************************************************************************
    #******************************************************************************************************************
        #==================================================================================================================
        #Create file name for the H5 Data, plan data, and the savename to be used at the end
        #==================================================================================================================
        H5_full_fname = FindAllFiles(main_directory + sub_folder, ['*.h5'])[0]
    #     H5_full_fname = 'GK_IMG_INFO_STRUCT_DOSE_GKSim_001_01.h5'
        H5_DATA_fname = os.path.basename(os.path.normpath(H5_full_fname))

        plan_full_fname = FindAllFiles(main_directory + sub_folder, ['*.plan_info'])[0]
    #     plan_full_fname = 'ZZ_DS_DS_GKsim_001_ZZ_DS_DS_GKsim_001_ZZ_DS_DS_GKsim_001__sm_md_lg.plan_info'
        plan_fname = os.path.basename(os.path.normpath(plan_full_fname))

        savename = sub_folder[:-1] + sn_extra + '_Dose_Sim_CROP_TEST.d1'  #if saving cropped data change at bound for correct binary shot mask
        # RT_struct2pull = ['Heart', 'Lung_L', 'SpinalCord']


        #==================================================================================================================
        #Open and load in the H5 image data sets (H5 is not the true H5 format, but rather a pickled data set)
        #==================================================================================================================
        with open(main_directory + sub_folder + H5_DATA_fname, 'rb') as f:
            DATA1 = pickle.load(f)
            'unpickled data'
        #     print(DATA1['RTDOSE'].keys())
            d_CT_IMG = DATA1['CT']['CT_IMG']
            CT0_INFO = DATA1['CT']['CT_INFO'][0]
            d_RTDOSE_IMG = DATA1['RTDOSE']
            d_STRUCT_INFO = DATA1['RTSTRUCT']

        CT_IMG = DictIMG2NPIMG(d_CT_IMG).astype(np.float16)
        print('      ---> CT_IMG loaded: ', np.shape(CT_IMG))

        RTDOSE_IMG = DictIMG2NPIMG(d_RTDOSE_IMG).astype(np.float16)
        print('      ---> DOSE_IMG loaded: ', np.shape(RTDOSE_IMG))
        
        contour_masks = []
        contour_names = []
        mask_CoM = []
        all_targets_mask = np.zeros((np.shape(CT_IMG)[0], np.shape(CT_IMG)[1], np.shape(CT_IMG)[2])) 
        for i_mask in range(len(d_STRUCT_INFO['MASKS'])):
            contour_masks.append(DictIMG2NPIMG(d_STRUCT_INFO['MASKS'][i_mask]).astype(np.uint8))
            contour_names.append(d_STRUCT_INFO['NAMES'][i_mask])
            mask_CoM.append(scipy.ndimage.center_of_mass(contour_masks[i_mask]))
            contour_tag = d_STRUCT_INFO['TAGS'][i_mask]
            if contour_tag == 'PTV':
                all_targets_mask = np.add(all_targets_mask, contour_masks[i_mask])
#         print(contour_names)
        
        with open(main_directory + sub_folder + plan_fname, 'rb') as f:
            Plan = pickle.load(f)
            'unpickled data'
        print('      ---> Plan data loaded: #Targets // # of shots = ', Plan[-1][0]+1, '//', len(Plan))

        num_target = Plan[-1][0]   # Number of targets, ranging from 0 to (n-1)

        #==================================================================================================================
        #check for the max_RX value from the plan, could be used for normalization, likely this is useless
        #==================================================================================================================
        max_RX_dose = []
        for i_shot in range(len(Plan)):
            i_shot
            max_RX_dose.append(Plan[i_shot][3] / (Plan[i_shot][4]/100))
        max_RX_dose = np.max(np.array(max_RX_dose))

        #==================================================================================================================
        #Create axes for coordinate system of the primary (sterotactic definition) image set; important for shot placement
        #==================================================================================================================
        X1 = np.array(CT0_INFO.ImagePositionPatient)
        X2 = np.array(CT0_INFO.PixelSpacing)
        X3 = np.array(CT0_INFO.SliceThickness)

        xmin = X1[0]
        xmax = xmin + X2[0]*np.shape(CT_IMG)[0]
        xspace = X2[0]
        X = np.array([xmin, xmax, xspace])

        #===================================
        # #for .dcm input use these lines because there is a - in y and z coordinate b/t gk and mim
        ymin = X1[1]
        ymax = ymin + X2[1]*np.shape(CT_IMG)[1]
        yspace = X2[1]
        Y = np.array([ymin, ymax, yspace])

        zmin = X1[2]
        zmax = zmin + X3*np.shape(CT_IMG)[2]
        zspace = X3
        Z = np.array([zmin, zmax, zspace])

        #===================================
        # #for .lgp input use these lines because there is a - in y and z coordinate
        # ymax = -X1[1]
        # ymin = ymax - X2[1]*np.shape(CT_IMG)[1]
        # yspace = X2[1]
        # Y = np.array([ymin, ymax, yspace])

        # zmax = -X1[2]
        # zmin = zmax - X3*np.shape(CT_IMG)[2]
        # zspace = X3
        # Z = np.array([zmin, zmax, zspace])
        #===================================
        print(X1)
        print(X, Y, Z)

        #==================================================================================================================
        #Create axes for coordinate system of the primary (sterotactic definition) image set; important for shot placement
        #==================================================================================================================
        #Uses sizing differences in the x, y, and z
        vol_size = (X[1]-X[0], Y[1]-Y[0], Z[1]-Z[0]) #mm defined by image patient position pulled from CT
        origin_coord = np.array([X[0], Y[0], Z[0]])
        print(origin_coord)
        res_val = np.array([X[2], Y[2], Z[2]])

        if vxl_size[0] == res_val[0] and vxl_size[1] == res_val[1] and vxl_size[2] == res_val[2]:
            resize_dose_img = RTDOSE_IMG
        else:
            print('resizing dose image -->')
            resize_dose_img = zoom(RTDOSE_IMG, np.divide(res_val, vxl_size)).astype(np.float16)
            print(np.shape(RTDOSE_IMG), '   --->resized--->   ', np.shape(resize_dose_img))

        # Appending to dose_maps[0] = OG dose map
        dose_maps.append(resize_dose_img)

        M_size = (np.int(np.round(vol_size[0]/vxl_size[0])), 
                  np.int(np.round(vol_size[1]/vxl_size[1])), 
                  np.int(np.round(vol_size[2]/vxl_size[2]))) # big matrix i,j,k size (Length, width, height)
        #==================================================================================================================
        #Use the plan info to create the plan simulation shot list info
        #==================================================================================================================
        imp_val = Plan.copy()
        imp_val_mod = [[]]
        targ_i = 0
        for i_imp_val in range(len(imp_val)):
            targ_num = imp_val[i_imp_val][0]
#             targ_name = imp_val[i_imp_val][-1]
#             print('HEREx:', targ_name)
            if targ_num != targ_i or i_imp_val == (len(imp_val)):
                imp_val_mod.append([])
                targ_i += 1
                print('----> ', i_imp_val, ' / ', len(imp_val), '  |  ', len(imp_val_mod)-1, len(imp_val_mod[targ_num-1]))

            targ_dose = imp_val[i_imp_val][3]/(imp_val[i_imp_val][4]/100)

            sectors = imp_val[i_imp_val][-2]
            if classic_opt == 1:
                if imp_val[i_imp_val][6] == 16: #use -1 not 6 to use max sector size of the shot
                    sectors[:] = [16] * 8
                if imp_val[i_imp_val][6]== 8:
                    sectors[:] = [8] * 8
                if imp_val[i_imp_val][6] == 4:
                    sectors[:] = [4] * 8

            coords = list(np.array([imp_val[i_imp_val][9][0], imp_val[i_imp_val][9][1], imp_val[i_imp_val][9][2]])) #use coord from dcm
            w_Rx_iso_mod = imp_val[i_imp_val][5] #normalized to shot
            print([coords, imp_val[i_imp_val][-2], imp_val[i_imp_val][5]])
            imp_val_mod[targ_num].append([coords, sectors, w_Rx_iso_mod, targ_dose])
        print('----> ', i_imp_val+1, ' / ', len(imp_val), '  |  ', len(imp_val_mod), ' / ', len(imp_val_mod[targ_num]))

        #==================================================================================================================
        #Use the plan info to create the plan simulation shot list info
        #==================================================================================================================
        M_mod = np.zeros(M_size)
        for mod_i in range(len(imp_val_mod)):
            s_pos_list_mod = imp_val_mod[mod_i].copy()  #uses imported values from previous cell
            ijk_shot_list = []
            M_mod_i = np.zeros(M_size)
            for i in range(len(s_pos_list_mod)):
                s_pos_i = s_pos_list_mod[i]
                print(s_pos_i[0], origin_coord)
                print(np.array(s_pos_i[0]-origin_coord))

                if classic_opt == 1:
                    if s_pos_i[1].count(s_pos_i[1][0]) == len(s_pos_i[1]):
                        if s_pos_i[1][0] == 4:
                            shot_ij = M_shot_sm * sm_eof
                        if s_pos_i[1][0] == 8:
                            shot_ij = M_shot_md * md_eof
                        if s_pos_i[1][0] == 16:
                            shot_ij = M_shot_lg * lg_eof
                        M_mod_i = add_struct_xyz(M_mod_i, #matrix to be updated
                                                 np.array(s_pos_i[0]-origin_coord), #shifted to match origin
                                                 shot_ij, #sectors
                                                 s_pos_i[2], #shot weight global (dose)
                                                 vxl_size) #vxl_size
                else:
                    if s_pos_i[1].count(s_pos_i[1][0]) == len(s_pos_i[1]):
                        if s_pos_i[1][0] == 4:
                            shot_ij = DD_DATA[0][2][2] * sm_eof
                        if s_pos_i[1][0] == 8:
                            shot_ij = DD_DATA[0][1][2] * md_eof
                        if s_pos_i[1][0] == 16:
                            shot_ij = DD_DATA[0][0][2] * lg_eof
                        M_mod_i = add_struct_xyz(M_mod_i, #matrix to be updated
                                                 np.array(s_pos_i[0]-origin_coord), #shifted to match origin
                                                 shot_ij, #sectors
                                                 s_pos_i[2], #shot weight global (dose)
                                                 vxl_size) #vxl_size
                    else:
                        for j in range(8):
                            if s_pos_i[1][j] == 4:
                                shot_ij = DD_DATA[j+1][2][2] * sm_eof/8
                            if s_pos_i[1][j] == 8:
                                shot_ij = DD_DATA[j+1][1][2] * md_eof/8
                            if s_pos_i[1][j] == 16:
                                shot_ij = DD_DATA[j+1][0][2] * lg_eof/8
                            if s_pos_i[1][j] != 0:
                                M_mod_i = add_struct_xyz(M_mod_i, #matrix to be updated
                                                         np.array(s_pos_i[0]-origin_coord), #shifted to match origin
                                                         shot_ij, #sectors
                                                         s_pos_i[2], #shot weight global (dose)
                                                         vxl_size) #vxl_size
                ijk_shot_list.append(np.true_divide(np.array(s_pos_i[0]-origin_coord), vxl_size))
            ijk_shot_list = np.array(ijk_shot_list)
            M_mod_i = M_mod_i/np.max(M_mod_i) * s_pos_i[-1] #normalize accumulating target specific shots to RX
            #it is possible to add target specific thresholding or tagging here
            M_mod = M_mod + M_mod_i
            # print(np.array(ijk_shot_list))

        #==================================================================================================================
        #Use the plan info to create the plan simulation shot list info
        #==================================================================================================================
        M = np.array(np.swapaxes(M_mod,0,1)) #Necessary to correlate simulated dose with all other images [y,x] issue
        dose_M = np.array(resize_dose_img)

        ind1 = np.unravel_index(np.argmax(dose_M, axis=None), dose_M.shape)
        ind2 = np.unravel_index(np.argmax(M, axis=None), M.shape)
        ind1 = np.array([ind1[1], ind1[0], ind1[2]])
        ind2 = np.array([ind2[1], ind2[0], ind2[2]])

        ind1 = list(ndimage.measurements.center_of_mass(dose_M))
        ind1[0], ind1[1] = ind1[1], ind1[0]
        ind2 = list(ndimage.measurements.center_of_mass(M))
        ind2[0], ind2[1] = ind2[1], ind2[0]

        print('-->', ind1, ind2)

        M_bound = np.array(np.swapaxes(shot_ij,0,1)) #M_bound based on last shot of list (useful for collecting dose distr.)
        # M_bound = np.array(M_mod)
        if crop_option == 1:
            print('--> cropping for collimator size:   ', np.max(imp_val[i_imp_val][-1])) #tells user what size collimator last shot was
            bnd_sz = np.around(np.divide(np.divide(np.shape(M_bound), 2), 1)).astype(int)

            j_bnd1 = np.around(np.min(ijk_shot_list[:, 0]) - bnd_sz[0]).astype(int)
            i_bnd1 = np.around(np.min(ijk_shot_list[:, 1]) - bnd_sz[1]).astype(int)
            k_bnd1 = np.around(np.min(ijk_shot_list[:, 2]) - bnd_sz[2]).astype(int)

            j_bnd2 = np.around(np.max(ijk_shot_list[:, 0]) + bnd_sz[0]).astype(int)
            i_bnd2 = np.around(np.max(ijk_shot_list[:, 1]) + bnd_sz[1]).astype(int)
            k_bnd2 = np.around(np.max(ijk_shot_list[:, 2]) + bnd_sz[2]).astype(int)

            M = M[i_bnd1:i_bnd2, j_bnd1:j_bnd2, k_bnd1:k_bnd2]
            dose_M = dose_M[i_bnd1:i_bnd2, j_bnd1:j_bnd2, k_bnd1:k_bnd2]
            dose_M_sel = np.array(dose_M)
            dose_M_sel[M < 1] = 0 #useful when M is the binary shot or you can modify for thresholding
            dose_M_sel = dose_M_sel/np.max(dose_M_sel)
            
        dose_maps.append(M.astype(np.float16))
        
        #==================================================================================================================
        # Create cropped dose maps around each target
        #==================================================================================================================
        crop_og=[]
        crop_rc=[]
        rx_og=[]
        rx_rc=[]
        dist_crop=[]
        full_dist=[]
    #     dist_rc=[]
        crop_trg = []
#         og_10p = []
#         rc_10p = []
        og_gi = []
        rc_gi = []
        og_tar_norm = []
        crop_edt = []
        outer_edt = []
        targ_rx_info = []
        
        # Establish crop size and pad arrays
        crop_size = 140
        rdi_pad = np.pad(resize_dose_img, ((int(crop_size/2), int(crop_size/2)), (int(crop_size/2), int(crop_size/2)), (int(crop_size/2), int(crop_size/2))), 'constant')
        M_pad = np.pad(M, ((int(crop_size/2), int(crop_size/2)), (int(crop_size/2), int(crop_size/2)), (int(crop_size/2), int(crop_size/2))), 'constant')
        
        # Establish the binary used to find the distance from the external contour
        og_washout = np.array(rdi_pad)
        og_washout[og_washout > 0] = 1
        og_external = np.array(og_washout, dtype=np.uint8)
#         og_dist_mat = edt.edt(og_external)
        
        og_washout_prepad = np.array(resize_dose_img)
        og_washout_prepad[og_washout_prepad > 0] = 1

        temp_id = 0
        pix_spc = X2[0]    # mm/voxel
        for n in range(len(Plan)):
            targ_id = Plan[n][0]
            if targ_id == temp_id:
                tar_width = Plan[n][1]    # Target box width [mm]
                tar_width = tar_width/pix_spc   # Target width in voxels
                tar_D = Plan[n][3]            # Target Dose [Gy]
                tar_rx = Plan[n][4]/100       # Target Rx [%]
                tar_name = Plan[n][-1]        # Target Name (string)
#                 print('HEREz:', tar_name)
                tar_max_Gy = tar_D/tar_rx
                og_tar_norm.append(tar_max_Gy)
                if xtnd_crop == 0:
                    tar_width = int(tar_width)
                elif xtnd_crop == 1:
                    tar_width = int(tar_width + tar_width*0.25)
                tar_center = Plan[n][2]   # Target center coord      
#                 print('TARG CENTER: ', tar_center)
                dist_diff = [np.abs(origin_coord[0]-tar_center[0]), np.abs(origin_coord[1]+tar_center[1]), 
                             np.abs(origin_coord[2]+tar_center[2])] # plus in 2nd dimension bc of sign flip
                vox_diff = np.rint(dist_diff/pix_spc)
#                 num_z_slice = 50
                
#                 crop_size = tar_width
#                 M_crop_og = np.zeros((np.shape(resize_dose_img)[0], np.shape(resize_dose_img)[1], num_z_slice))
                M_crop_og = np.zeros((crop_size, crop_size, crop_size))
                M_crop_rc = np.zeros(np.shape(M_crop_og))
        
                # padded image cropped to crop size
                M_crop_og = rdi_pad[int(floor((vox_diff[1]))):int(ceil((vox_diff[1]+crop_size))), 
                            int(floor((vox_diff[0]))):int(ceil((vox_diff[0]+crop_size))), 
                            int(floor((vox_diff[2]))):int(ceil((vox_diff[2]+crop_size)))]
                crop_og.append(M_crop_og.astype(np.float16))
                M_crop_rc = M_pad[int(floor((vox_diff[1]))):int(ceil((vox_diff[1]+crop_size))), 
                              int(floor((vox_diff[0]))):int(ceil((vox_diff[0]+crop_size))), 
                              int(floor((vox_diff[2]))):int(ceil((vox_diff[2]+crop_size)))]
                crop_rc.append(M_crop_rc.astype(np.float16))

                # =========== Distance from External ==============

#                 edt_targ = og_dist_mat[int(floor((vox_diff[1]))):int(ceil((vox_diff[1]+crop_size))), 
#                             int(floor((vox_diff[0]))):int(ceil((vox_diff[0]+crop_size))), 
#                             int(floor((vox_diff[2]))):int(ceil((vox_diff[2]+crop_size)))]
#                 crop_edt.append(edt_targ)
                
                # =========== Distance Cropping ===================

                ext = np.array(resize_dose_img, dtype=np.float64)
                ext[ext > 0] = 1
                ext_ero = ndimage.morphology.binary_erosion(ext, iterations=1)
                ext_bord = ext - ext_ero
                
                s_size = np.shape(resize_dose_img)
                
#                 num_rlevels = 30 # manual control, works with rlevels to control speed of delta radius
                num_rlevels = 100
                
                sx = s_size[0] # matrix size
                sy = s_size[1]
                sz = s_size[2]
                
                rlevels = np.linspace(0, sx, num_rlevels+1, endpoint=True)
                r_delta = 1.2**(rlevels/10)   # n^r gives a tight packing near the center then quickly grows as you move out
                r_high = r_delta[1:]
                r_low = r_delta[:-1]
                r_low[0] = 0                  # necessary for n^r growth because n^0 = 1
                r_centers = r_low + (r_high - r_low)/2
                
                struct_s = np.zeros((int(sx),int(sy),int(sz)))
                
                cntr = np.round(vox_diff)
                cntr_x = int(cntr[0])
                cntr_y = int(cntr[1])
                cntr_z = int(cntr[2])
#                 cntr_x = (sx-1)/2           # this would set center to center of array
#                 cntr_y = (sy-1)/2
#                 cntr_z = (sz-1)/2
                
                sx = np.arange(0, sx, 1)  #THIS PORTION IS MOST LIKELY THE KEY COMPONENTS FOR PIXEL SPACING UPDATE.
                sy = np.arange(0, sy, 1)
                sz = np.arange(0, sz, 1)
                
                xx, yy, zz = np.meshgrid(sx, sy, sz)
                r_map = np.sqrt((xx-cntr_x)**2 + (yy-cntr_y)**2 + (zz-cntr_z)**2)
                
                for i in range(num_rlevels):
#                     print(i+1, r_low[i], r_centers[i], r_high[i], r_delta[i])
#                     struct_s[(r_map>= r_low[i]) & (r_map < r_high[i])] = i+1
#                     struct_s[(r_map>= r_low[i]) & (r_map < r_high[i])] = r_centers[i]
                    struct_s[(r_map>= r_low[i]) & (r_map < r_high[i])] = r_low[i]
                
                # This portion would create a labelmap for the rad sections
#                 i_remap = np.unique(struct_s)    
#                 remap = np.zeros(np.shape(struct_s))
#                 l_i = 0
#                 for i in i_remap:
#                     l_i += 1
#                     remap[struct_s==i] = l_i
                
                xx_d = xx
                yy_d = yy
                zz_d = zz
                xy = (xx_d-cntr_x)**2 + (yy_d-cntr_y)**2
                r = np.sqrt(xy + (zz_d-cntr_z)**2)
                theta = np.arctan2(np.sqrt(xy), (zz_d-cntr_z))/np.pi*180  # defined from Z-axis down (physics convention)
                phi = np.arctan2((yy_d-cntr_y), (xx_d-cntr_x))/np.pi*180+180
                
                SectorMap = np.zeros(np.shape(resize_dose_img))
#                 # Coarse
#                 delta_th = 45
#                 delta_ph = 45
                # Standard
#                 delta_th = 30
#                 delta_ph = 30
#                 # Fine
                delta_th = 15
                delta_ph = 15

                theta_angs = np.arange(0, 180+delta_th, delta_th)
                phi_angs = np.arange(0, 360+delta_ph, delta_ph)
                incr_val = 100
                ang_list = []
                for i in range(len(theta_angs)-1):
                    for j in range(len(phi_angs)-1):
                        SectorMap[(theta <= theta_angs[i+1]) & (theta >= theta_angs[i]) 
                                  & (phi <= phi_angs[j+1]) & (phi >= phi_angs[j])] = incr_val
                        incr_val += 100
                        ang_list.append((theta_angs[i], theta_angs[i+1], phi_angs[j], phi_angs[j+1]))
                SectorMap[cntr_y, cntr_x, cntr_z] = 0
                
                sect_mask = np.multiply(SectorMap, ext, dtype=float)
                rad_mask = np.multiply(struct_s, ext, dtype=float)
                sect_rad_map = sect_mask + rad_mask
                sect_unique = np.unique(sect_mask)[1:]    # removes 0 from list
#                 print(sect_unique)
                bound_find = SectorMap + ext_bord
#                 avg_pt = []
                sect_max_map = np.zeros(np.shape(sect_mask))
                for i_sect in range(len(sect_unique)):
                    bound_coords = np.argwhere(bound_find==(sect_unique[i_sect] + 1))
                    max_bound = []
                    for j_coord in range(len(bound_coords)):
                        temp_dist = np.sqrt((bound_coords[j_coord][0] - cntr_y)**2 + 
                                            (bound_coords[j_coord][1] - cntr_x)**2 +   
                                            (bound_coords[j_coord][2] - cntr_z)**2)
#                         print(temp_dist)
                        max_bound.append(temp_dist)
#                     temp_avg = np.mean(bound_coords, axis=0)
#                     avg_pt.append(temp_avg)
#                     temp_dist = np.sqrt((temp_avg[0] - cntr_y)**2 + 
#                                         (temp_avg[1] - cntr_x)**2 +   
#                                         (temp_avg[2] - cntr_z)**2)
                #     temp_dist = np.linalg.norm(temp_avg - (cntr_y, cntr_x, cntr_z))
                    sect_max_map[sect_mask == sect_unique[i_sect]] = np.max(max_bound)
#                 print(avg_pt[-1])
#                 sect_rad_unique = np.unique(sect_rad_map)[1:]
                
                dist_map = sect_max_map - rad_mask
        
                dist_map_pad = np.pad(dist_map, ((int(crop_size/2), int(crop_size/2)), (int(crop_size/2), int(crop_size/2)), (int(crop_size/2), int(crop_size/2))), 'constant')
                M_cropdist_og = np.zeros(np.shape(M_crop_og))
                M_cropdist_og = dist_map_pad[int(floor((vox_diff[1]))):int(ceil((vox_diff[1]+crop_size))), 
                              int(floor((vox_diff[0]))):int(ceil((vox_diff[0]+crop_size))), 
                              int(floor((vox_diff[2]))):int(ceil((vox_diff[2]+crop_size)))]
                dist_crop.append(M_cropdist_og.astype(np.float16))
                full_dist.append(dist_map.astype(np.float16))
                
                
                # ============== End Distance Crop =============
                # ============== Start Rx Masking ==============

    #             M_rx_og = np.array(M_crop_og/np.percentile(resize_dose_img, 95))
#                 M_rx_og = np.array(M_crop_og)/np.max(resize_dose_img)
#                 M_rx_og = np.array(M_crop_og)/np.max(og_max)
                M_rx_og = np.array(M_crop_og)/tar_max_Gy
#                 M_rx_og = np.array(M_crop_og)/np.max(M_crop_og)
                M_rx_og[M_rx_og <= tar_rx] = 0
                M_rx_og[M_rx_og >  tar_rx] = 1
                M_rx_og = np.array(M_rx_og, dtype=np.uint8)
                rx_og.append(M_rx_og)
                
                M_rx_rc = np.array(M_crop_rc)/(tar_max_Gy)
#                 M_rx_rc = np.array(M_crop_rc)/np.max(M_crop_rc)
                M_rx_rc[M_rx_rc <= tar_rx] = 0
                M_rx_rc[M_rx_rc >  tar_rx] = 1
                M_rx_rc = np.array(M_rx_rc, dtype=np.uint8)
                rx_rc.append(M_rx_rc)
                
                # Threshold at 50% of Rx to get GI values
                gi_thresh = tar_rx/2
                
#                 M_gi_og = np.array(M_crop_og)/np.max(og_max)
                M_gi_og = np.array(M_crop_og)/tar_max_Gy
                M_gi_og[M_gi_og <= gi_thresh] = 0
                M_gi_og[M_gi_og >  gi_thresh] = 1
                M_gi_og = np.array(M_gi_og, dtype=np.uint8)
                og_gi.append(M_gi_og)
                
                M_gi_rc = np.array(M_crop_rc)/(tar_max_Gy)
                M_gi_rc[M_gi_rc <= gi_thresh] = 0
                M_gi_rc[M_gi_rc >  gi_thresh] = 1
                M_gi_rc = np.array(M_gi_rc, dtype=np.uint8)
                rc_gi.append(M_gi_rc)
                
                # ============== TARGET CONTOUR SELECION ====================
                close_pt = closest_node((vox_diff[1],vox_diff[0],vox_diff[2]), mask_CoM)
                current_targ = contour_masks[close_pt]
                print()
                print('Processing Target: ', tar_name)
                print('Matched Target: ', contour_names[close_pt])

                # https://pythoninoffice.com/how-to-find-similar-strings-using-python/
#                 print()
#                 fuzzy_score = []
#                 print('Processing target: ', tar_name)
#                 for i_cont in range(len(contour_masks)):
#                     fuzzy_score.append(fuzz.ratio(tar_name.lower(), contour_names[i_cont].lower()))
#                 fuzzy_max = max(fuzzy_score)
#                 fuzzy_ind = fuzzy_score.index(fuzzy_max)
#                 print('Matched target: ', contour_names[fuzzy_ind])
#                 current_targ = contour_masks[fuzzy_ind]
                
                crnt_trg_pad = np.pad(current_targ, ((int(crop_size/2), int(crop_size/2)), (int(crop_size/2), int(crop_size/2)), (int(crop_size/2), int(crop_size/2))), 'constant')
                croptarg = np.zeros(np.shape(M_crop_og))
                croptarg = crnt_trg_pad[int(floor((vox_diff[1]))):int(ceil((vox_diff[1]+crop_size))), 
                              int(floor((vox_diff[0]))):int(ceil((vox_diff[0]+crop_size))), 
                              int(floor((vox_diff[2]))):int(ceil((vox_diff[2]+crop_size)))]
#                 croptarg = current_targ[int(floor((vox_diff[1]-tar_width/2))):int(ceil((vox_diff[1]+tar_width/2))), 
#                               int(floor((vox_diff[0]-tar_width/2))):int(ceil((vox_diff[0]+tar_width/2))), 
#                               int(floor((vox_diff[2]-tar_width/2))):int(ceil((vox_diff[2]+tar_width/2)))]
                crop_trg.append(croptarg.astype(np.uint8))
    
                # ============== TARGET EDT CALCULATION ====================
                temp_targ = np.array(croptarg, dtype=np.uint8)
                targ_edt = edt.edt(temp_targ)
                crop_edt.append(targ_edt.astype(np.float16))
                
                # ============ OUTSIDE TARGET EDT CALC =====================
#                 targ_inv = np.array(crnt_trg_pad, dtype=np.uint8)+1
                atm_pad = np.pad(all_targets_mask, ((int(crop_size/2), int(crop_size/2)), (int(crop_size/2), int(crop_size/2)), (int(crop_size/2), int(crop_size/2))), 'constant')
                targ_inv = np.array(atm_pad, dtype=np.uint8)+1
                targ_inv[targ_inv > 1] = 0
                targ_outer_edt = edt.edt(targ_inv)
#                 targ_outer_edt_mask = np.multiply(targ_outer_edt, og_external)
                # Don't mask outer because then 0 corresponds to target and outside external
                crop_targ_outer_edt = targ_outer_edt[int(floor((vox_diff[1]))):int(ceil((vox_diff[1]+crop_size))), 
                              int(floor((vox_diff[0]))):int(ceil((vox_diff[0]+crop_size))), 
                              int(floor((vox_diff[2]))):int(ceil((vox_diff[2]+crop_size)))]
                outer_edt.append(crop_targ_outer_edt.astype(np.float16))
                
                
                temp_id += 1
            else:
                pass
            
        dose_maps.append(crop_og)     # [3]
        dose_maps.append(crop_rc)
        dose_maps.append(rx_og)
        dose_maps.append(rx_rc)       
        dose_maps.append(dist_crop)
        dose_maps.append(full_dist)
        dose_maps.append(crop_trg)
        dose_maps.append(og_gi)      # [10]
        dose_maps.append(rc_gi)
        dose_maps.append(og_tar_norm)
        dose_maps.append(crop_edt)
        dose_maps.append(outer_edt)
        dose_maps.append((tar_D, tar_rx))

        # Appending both dose maps to specific patient:
        pat_list.append(dose_maps)
       
        print('------------------------------------------------------------------')
        print('----------------Sub Folder Processing Complete--------------------')
        print('------------------------------------------------------------------')
#     else:
#         pass
#******************************************************************************************************************
#******************************************************************************************************************
#Subfolder loop done
print('==================================================================')
print('----------------------------DONE----------------------------------')
print('==================================================================')
#******************************************************************************************************************
#******************************************************************************************************************

dmap_savename = '2022-09-03_Pats_031-040___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCrop_' \
                '8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo' \
                '___Crops140Uniform_Dist-Standard-tp1515-60step-n^r.dmaps'
# dmap_savename = '2022-09-03_Pats_021-030___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCrop_' \
#                 '8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo' \
#                 '___Crops140Uniform_Dist-Standard-tp1515-60step-n^r.dmaps'
# dmap_savename = '2022-09-03_Pats_011-020___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCrop_' \
#                 '8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo' \
#                 '___Crops140Uniform_Dist-Standard-tp1515-60step-n^r.dmaps'
# dmap_savename = '2022-09-03_Pats_001-010___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCrop_' \
#                 '8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo' \
#                 '___Crops140Uniform_Dist-Standard-tp1515-60step-n^r.dmaps'


# dmap_savename = '2022-08-18_Pats_041-050___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCrop_' \
#                 '8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT___Crops140Uniform_' \
#                 'Dist-Standard-tp3030-n^r.dmaps'
# dmap_savename = '2022-08-18_Pats_031-040___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCrop_' \
#                 '8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT___Crops140Uniform_' \
#                 'Dist-Standard-tp3030-n^r.dmaps'
# dmap_savename = '2022-08-18_Pats_021-030___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCrop_' \
#                 '8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT___Crops140Uniform_' \
#                 'Dist-Standard-tp3030-n^r.dmaps'
# dmap_savename = '2022-08-18_Pats_011-020___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCrop_' \
#                 '8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT___Crops140Uniform_' \
#                 'Dist-Standard-tp3030-n^r.dmaps'
# dmap_savename = '2022-08-18_Pats_001-010___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop_5-RxOG_6-RxRC_7-DistCrop_' \
#                 '8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD_13-TargEDT_14-AllTargOuterEDT___Crops140Uniform_' \
#                 'Dist-Standard-tp3030-n^r.dmaps'

with open(save_dir + dmap_savename, 'wb') as f:
    pickle.dump(pat_list, f)

print('Saved as: ', dmap_savename)

print('finished')
