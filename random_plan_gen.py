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

# iv0  = 0                                               #   target #,
# iv1  = 40                                              #   target box size in mm,
# iv2  = [iv1/2] * 3                                     #   [LGP target box center coordinate],
# iv3  = 20                                              #   RX Dose [Gy],
# iv4  = 50                                              #   Rx Isodose %,
# iv5  = .5                                              #   Norm. shot weight (prop. to shot time, not max dose),
# iv6  = 8                                               #   Classic Sector Size,
# iv7  = [15, 15, 15]                                    #   [LGP shot center Coordinate],
# iv8  = list(np.array(iv7) - np.array(iv2))             #   [LGP relative shot center coordinate],
# iv9  = list(np.multiply(np.array(iv7), [1, -1, -1]))   #   [.dcm shot center coordinate],
# iv10 = [8, 4, 0, 16, 4, 8, 4, 4]                       #   [shot sector sizes]
# imp_val_man = [[iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7, iv8, iv9, iv10]]

# iv0  = 0                                               #   target #,
# iv1  = 40                                              #   target box size in mm,
# iv2  = [targ_x, targ_y, targ_z]                        #   [LGP target box center coordinate],
# iv3  = 20                                              #   RX Dose [Gy],
# iv4  = 50                                              #   Rx Isodose %,
# iv5  = .5                                              #   Norm. shot weight (prop. to shot time, not max dose),
# iv8  = [rel_x, rel_y, rel_z]                           #   [LGP relative shot center coordinate],
# iv10 = [8, 4, 0, 16, 4, 8, 4, 4]                       #   [shot sector sizes]

# iv6  = 8                                               #   Classic Sector Size,
# iv7  = list(np.array(iv8) + np.array(iv2))             #   [LGP shot center Coordinate],
# iv9  = list(np.multiply(np.array(iv7), [1, -1, -1]))   #   [.dcm shot center coordinate],
# imp_val_man = [[iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7, iv8, iv9, iv10]]

def plan_prep(iv0, iv1, iv2, iv3, iv4, iv5, iv8, iv10):
    iv7  = list(np.array(iv8) + np.array(iv2))             #   [LGP shot center Coordinate],
    iv9  = list(np.multiply(np.array(iv7), [1, -1, -1]))   #   [.dcm shot center coordinate],

    sectors_avg = np.mean(np.array(iv10))
    sectors_f = 0
    if sectors_avg <= 6.0:
        sectors_f = 4
    if sectors_avg > 6.0 and sectors_avg < 10.0: 
        sectors_f = 8
    if sectors_avg >= 10.0:
        sectors_f = 16
    iv6 = sectors_f
    
    plan_param = [iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7, iv8, iv9, iv10]
    return plan_param

#===================================================================#
#Random shot selection
#===================================================================
imp_val = []
shot_num_max = 30
targ_num = 5

for i in range(targ_num):
#     origin_coord_lgp = np.array([-13.352162, 210.936058, 161.018454])
    shot_num = np.random.randint(1, shot_num_max, size = 1)[0]
    coord_sys_width = 224
    targ_width = np.random.choice([30, 40, 50, 60, 70], size = 1)[0]
    RX_Dose = np.random.randint(18, 24, size = 1)[0]
    RX_Iso = np.random.randint(50, 70, size = 1)[0]
    targ_xyz = np.random.rand(3)*(coord_sys_width-.3*coord_sys_width)+.3/2*coord_sys_width# + origin_coord_lgp
    iv2  = list(targ_xyz)                        #   [LGP target box center coordinate],
#     print(shot_num, targ_width, RX_Dose, RX_Iso)
    for j in range(shot_num):
        iv0 = i                                              #   target #,
        iv1 = targ_width                                     #   target box size in mm,
        iv3 = RX_Dose                                        #   RX Dose [Gy],
        iv4 = RX_Iso                                         #   Rx Isodose %,
        iv5 = np.abs(np.random.rand(1,1))[0][0]*(1-.1)+.05      #   Norm. shot weight (prop. to shot time, not max dose),
                
        rel_xyz = np.random.random(3)*(np.array(iv1)*.4) - np.array(iv1)/2*.4
        iv8  = list(rel_xyz)
        
        sector_opt = [0, 4, 8, 16]
        prob_sec = [0.05, 1, 1, 0.4]
        prob_sec = list(np.array(prob_sec)/np.sum(np.array(prob_sec)))
        iv10 = list(np.random.choice(sector_opt, 8, p=prob_sec))                      #   [shot sector sizes]
        
        imp_val_j = plan_prep(iv0, iv1, iv2, iv3, iv4, iv5, iv8, iv10)
        imp_val.append(imp_val_j)

for i in range(len(imp_val)):
    print(imp_val[i])
#     print(imp_val[i][1], imp_val[i][2], imp_val[i][8])
        

#===================================================================#
#Manualish shot selection
#===================================================================
imp_val = []

targ_width = 50
RX_Dose = 20
RX_Iso = 50

# iv0 = 0                                              #   target #,
iv1 = targ_width                                     #   target box size in mm,
iv3 = RX_Dose                                        #   RX Dose [Gy],
iv4 = RX_Iso                                         #   Rx Isodose %,
iv5 = 1                                              #   Shot weight
iv10 = [4]*8                                         #   [shot sector sizes]
shift_dist = 4

coord_sys_width = 224
targ_xyz = coord_sys_width/2
iv2  = list(targ_xyz * np.array([1, 1, 1]))      #   [LGP target box center coordinate],

#---------------1-------------------#
# rel_xyz = shift_dist * np.array([1, -2, 0])
# iv8  = list(rel_xyz)
# iv10 = [4]*8
# imp_val_j = plan_prep(iv0, iv1, iv2, iv3, iv4, iv5, iv8, iv10)
# imp_val.append(imp_val_j)
#---------------2-------------------#
# # rel_xyz = shift_dist * np.array([-1, 0, 0])
# rel_xyz = shift_dist * np.array([-2, -2, 0])
# iv8  = list(rel_xyz)
# # iv10 = [0, 4, 8, 16, 0, 0, 0, 4]
# iv10 = [8]*8
# imp_val_j = plan_prep(iv0, iv1, iv2, iv3, iv4, iv5, iv8, iv10)
# imp_val.append(imp_val_j)
#---------------3-------------------#
# # rel_xyz = shift_dist * np.array([0, 1, 0])
# rel_xyz = shift_dist * np.array([0, 2, 0])
# iv8  = list(rel_xyz)
# # iv10 = [4]*8
# iv10 = [16]*8
# imp_val_j = plan_prep(iv0, iv1, iv2, iv3, iv4, iv5, iv8, iv10)
# imp_val.append(imp_val_j)
#---------------4-------------------#
# rel_xyz = shift_dist * np.array([0, -1, 0])
# iv8  = list(rel_xyz)
# # iv10 = [4]*8
# imp_val_j = plan_prep(iv0, iv1, iv2, iv3, iv4, iv5, iv8, iv10)
# imp_val.append(imp_val_j)
#------------SingleSector-----------#
# rel_xyz = shift_dist * np.array([0, 0, 0])
# iv8  = list(rel_xyz)
# # iv10 = [8]*8
# iv10 = [8, 0, 0, 0, 0, 0, 0, 0]
# # iv10 = [8, 4, 0, 16, 0, 4, 8, 4]         # Mixed for fig
# imp_val_j = plan_prep(iv0, iv1, iv2, iv3, iv4, iv5, iv8, iv10)
# imp_val.append(imp_val_j)
#------------4/8 Alt-----------#
# rel_xyz = shift_dist * np.array([0, 0, 0])
# iv8  = list(rel_xyz)
# # iv10 = [8]*8
# # iv10 = [8, 0, 8, 0, 8, 0, 8, 0]
# # iv10 = [0, 4, 0, 4, 0, 4, 0, 4]
# iv10 = [8, 4, 8, 4, 8, 4, 8, 4]
# imp_val_j = plan_prep(iv0, iv1, iv2, iv3, iv4, iv5, iv8, iv10)
# imp_val.append(imp_val_j)
#------------Cycle for Sector Activations-----------#
iv0 = 1                                              #   target #,
rel_xyz = shift_dist * np.array([1.25, 0, 0])
iv8  = list(rel_xyz)
# iv10 = [8]*8
# iv10 = [8, 0, 8, 0, 8, 0, 8, 0]
# iv10 = [0, 4, 0, 4, 0, 4, 0, 4]
# iv10 = [0, 0, 0, 0, 0, 0, 0, 16]
iv10 = [8, 4, 0, 16, 0, 4, 8, 4]
imp_val_j = plan_prep(iv0, iv1, iv2, iv3, iv4, iv5, iv8, iv10)
imp_val.append(imp_val_j)
#---------------end shot list-------------------#

for i in range(len(imp_val)):
    print(imp_val[i])

#-------Save the 'data' one plan at a time------#
# savename = 'randomtest_061520' + '_rand.plan_info'
# savename = 'manualtest_042221' + '_3shot_interaction.plan_info'
# savename = 'manualtest_050921' + '_SingleSector.plan_info'
# savename = 'manualtest_050921' + '_Alt4-8.plan_info'
# savename = 'manualtest_210921' + '_16mmSectorsAndShot.plan_info'
savename = 'manualtest_211121' + '_Classic+Mixed_DoseAccumFigs.plan_info'
print('-----------------------------------------------------------')
print(savename)

with open(savename, 'wb') as f:
    pickle.dump(imp_val,f)
    print("----> Data saved: ", savename)
