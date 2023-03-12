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
from skimage.filters import gaussian
import pydicom as py
from scipy.spatial.distance import directed_hausdorff
import seaborn as sns
from matplotlib.transforms import Affine2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import medpy

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600
import cmasher as cmr

import cv2

#SNC performs as follows
#Perc Diff test: if passes, sets DTA to 0; if fails, proceeds with DTA
#DTA test: if passes, sets Perc Diff to 0; if fails, proceeds with full gamma <= 1, strict looks for sign change in vicinity of dta search

from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,\
    generate_binary_structure
from scipy.ndimage.measurements import label, find_objects
from scipy.stats import pearsonr, wilcoxon


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
gpu_0_files = ['Prediction2023-01-29_DosePred_Pats111-120_SmallTarg_Stride20-20-20_Mdl-UNet_Inp-Targ_NoNorm___5fold_23-01-27',
               'Prediction2023-01-29_DosePred_Pats121-130_SmallTarg_Stride20-20-20_Mdl-UNet_Inp-Targ_NoNorm___5fold_23-01-27']
gpu_1_files = ['Prediction2023-01-29_DosePred_Pats111-120_SmallTarg_Stride20-20-20_Mdl-UNet_Inp-InnerOuterEDT_NoNorm___5fold_23-01-27',
               'Prediction2023-01-29_DosePred_Pats121-130_SmallTarg_Stride20-20-20_Mdl-UNet_Inp-InnerOuterEDT_NoNorm___5fold_23-01-27']

# # ----- HD-UNET -----
# gpu_0_files = ['Prediction2023-01-29_DosePred_Pats111-120_SmallTarg_Stride20-20-20_Mdl-HD-UNet_Inp-Targ_NoNorm___5fold_23-01-27',
#                'Prediction2023-01-29_DosePred_Pats121-130_SmallTarg_Stride20-20-20_Mdl-HD-UNet_Inp-Targ_NoNorm___5fold_23-01-27']
# gpu_1_files = ['Prediction2023-01-29_DosePred_Pats111-120_SmallTarg_Stride20-20-20_Mdl-HD-UNet_Inp-InnerOuterEDT_NoNorm___5fold_23-01-27',
#                'Prediction2023-01-29_DosePred_Pats121-130_SmallTarg_Stride20-20-20_Mdl-HD-UNet_Inp-InnerOuterEDT_NoNorm___5fold_23-01-27']

# # ===== Large TARGET TRAINING =====
# # ----- UNET -----
# gpu_0_files = ['Prediction2023-01-29_DosePred_Pats111-120_LargeTarg_Stride20-20-20_Mdl-UNet_Inp-Targ_NoNorm___5fold_23-01-27',
#                'Prediction2023-01-29_DosePred_Pats121-130_LargeTarg_Stride20-20-20_Mdl-UNet_Inp-Targ_NoNorm___5fold_23-01-27']
# gpu_1_files = ['Prediction2023-01-29_DosePred_Pats111-120_LargeTarg_Stride20-20-20_Mdl-UNet_Inp-InnerOuterEDT_NoNorm___5fold_23-01-27',
#                'Prediction2023-01-29_DosePred_Pats121-130_LargeTarg_Stride20-20-20_Mdl-UNet_Inp-InnerOuterEDT_NoNorm___5fold_23-01-27']

# # ----- HD-UNET -----
# gpu_0_files = ['Prediction2023-01-29_DosePred_Pats111-120_LargeTarg_Stride20-20-20_Mdl-HD-UNet_Inp-Targ_NoNorm___5fold_23-01-27',
#                'Prediction2023-01-29_DosePred_Pats121-130_LargeTarg_Stride20-20-20_Mdl-HD-UNet_Inp-Targ_NoNorm___5fold_23-01-27']
# gpu_1_files = ['Prediction2023-01-29_DosePred_Pats111-120_LargeTarg_Stride20-20-20_Mdl-HD-UNet_Inp-InnerOuterEDT_NoNorm___5fold_23-01-27',
#                'Prediction2023-01-29_DosePred_Pats121-130_LargeTarg_Stride20-20-20_Mdl-HD-UNet_Inp-InnerOuterEDT_NoNorm___5fold_23-01-27']


# all_dmaps = []
# for i in range(len(filenames)):
#     dmaps_file = open(main_dir + filenames[i], 'rb')
#     dmaps = pickle.load(dmaps_file)
#     all_dmaps.append(dmaps)

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

# Prediction structure:
# [0]  - OG dmap
# [1]  - Target Contour
# [2]  - Input 1 (Sup Approx)
# [3]  - Input 2 (Depth)   (for gpu1)
# [-1] - Prediction

# dmaps structure:
# 031-040/2022-09-03_Pats_031-040___0-Index_1-OGfull_2-RCfull_3-OGcrop_4-RCcrop
# _5-RxOG_6-RxRC_7-DistCrop_8-FullDist_9-CropTarg_10-OGgi_11-RCgi_12-NormMaxD
# _13-TargEDT_14-AllTargOuterEDT_15-TargRxInfo___Crops140Uniform_Dist-Standard-tp1515-60step-n^r.dmaps

# Set the font dictionaries (for plot title and axis titles)
title_font = {'fontname':'Arial', 'size':'18', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} # Bottom vertical alignment for more space
axis_font = {'fontname':'Arial', 'size':'18'}
tick_font = {'fontname':'Arial', 'size':'10'}

print(np.shape(preproc_dmaps[0][0]))
print(preproc_dmaps[0][0][15])  # Rx info

# Good target index options for viewing: 1, 2, 6
targ_select = 2
ind = 70
lblsize = 14

og_dmap = gpu_0_dmaps[0][targ_select][0]
og_dmap = og_dmap/np.max(og_dmap)
targ_contour = gpu_0_dmaps[0][targ_select][1]
inner = gpu_1_dmaps[0][targ_select][2]
inner = inner/np.max(inner)
outer = gpu_1_dmaps[0][targ_select][3]
outer = outer/np.max(outer)
pred_0 = gpu_0_dmaps[0][targ_select][-1]
pred_0 = pred_0/np.max(pred_0)
pred_1 = gpu_1_dmaps[0][targ_select][-1]
pred_1 = pred_1/np.max(pred_1)

combined_data = np.array([og_dmap, pred_0, pred_1])
_min, _max = np.amin(combined_data), np.amax(combined_data)

combined_inp = np.array([inner, outer, targ_contour])
_mininp, _maxinp = np.amin(combined_inp), np.amax(combined_inp)

fig = plt.figure(facecolor='white')
im = plt.imshow(og_dmap[:,:,ind].astype(float), cmap='turbo',
          vmin = _min, vmax = _max)
plt.axis('off')
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=lblsize)
cbar.set_label(label='Normalized Dose', **axis_font)
plt.tight_layout()
# cbar.set_clim(0.0, 1.0)
# cax = fig.add_axes([left, bottom, width, height])
# cax = fig.add_axes([0.85, 0, 0.025, 1])


plt.figure(facecolor='white')
im = plt.imshow(targ_contour[:,:,ind].astype(float), cmap='gray', 
          vmin = _mininp, vmax = _maxinp)
plt.axis('off')
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=lblsize)
cbar.set_label(label='Contour', **axis_font)
plt.tight_layout()

plt.figure(facecolor='white')
im = plt.imshow(pred_0[:,:,ind].astype(float), cmap='turbo',
          vmin = _min, vmax = _max)
plt.axis('off')
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=lblsize)
cbar.set_label(label='Normalized Dose', **axis_font)
plt.tight_layout()

plt.figure(facecolor='white')
im = plt.imshow(inner[:,:,ind].astype(float), cmap='turbo', 
          vmin = _mininp, vmax = _maxinp)
plt.axis('off')
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=lblsize)
cbar.set_label(label='Relative Distance', **axis_font)
plt.tight_layout()

plt.figure(facecolor='white')
im = plt.imshow(outer[:,:,ind].astype(float), cmap='turbo', 
          vmin = _mininp, vmax = _maxinp)
plt.axis('off')
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=lblsize)
cbar.set_label(label='Relative Distance', **axis_font)
plt.tight_layout()

plt.figure(facecolor='white')
im = plt.imshow(pred_1[:,:,ind].astype(float), cmap='turbo',
          vmin = _min, vmax = _max)
plt.axis('off')
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=lblsize)
cbar.set_label(label='Normalized Dose', **axis_font)
plt.tight_layout()

og_pred0_diff = og_dmap - pred_0
og_pred1_diff = og_dmap - pred_1
# combined_diff = np.array([og_pred0_diff, og_pred1_diff])
# _min, _max = np.amin(combined_diff), np.amax(combined_diff)

# plt.figure()
# plt.imshow(og_pred0_diff[:,:,ind].astype(float), 
#            cmap='cmr.redshift', vmin = _min, vmax = _max)
# plt.axis('off')
# cbar = plt.colorbar()

# plt.figure()
# plt.imshow(og_pred1_diff[:,:,ind].astype(float), 
#            cmap='cmr.redshift', vmin = _min, vmax = _max)
# plt.axis('off')
# cbar = plt.colorbar()

# MSE
# mse_0 = ((og_pred0_diff)**2).mean(axis=2)
# mse_1 = ((og_pred1_diff)**2).mean(axis=2)
# MAE
mse_0 = (np.abs(og_pred0_diff)).mean(axis=2)
mse_1 = (np.abs(og_pred1_diff)).mean(axis=2)
combined_mse = np.array([mse_0, mse_1])
_min, _max = np.amin(combined_mse), np.amax(combined_mse)

plt.figure(facecolor='white')
# use cmasher ('cmr.name') cmaps for difference images
im = plt.imshow(mse_0.astype(float), 
           cmap='inferno', vmin = _min, vmax = _max)
plt.axis('off')
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=lblsize)
# cbar.set_label(label='Mean Squared Error', **axis_font)
cbar.set_label(label='Mean Absolute Error', **axis_font)
plt.tight_layout()

plt.figure(facecolor='white')
im = plt.imshow(mse_1.astype(float), 
           cmap='inferno', vmin = _min, vmax = _max)
plt.axis('off')
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=lblsize)
# cbar.set_label(label='Mean Squared Error', **axis_font)
cbar.set_label(label='Mean Absolute Error', **axis_font)
plt.tight_layout()

# inner = gpu_1_dmaps[0][targ_select][2]
# outer = gpu_1_dmaps[0][targ_select][3]
# plt.figure()
# plt.imshow(inner[:,:,70].astype(float), cmap='nipy_spectral')
# plt.axis('off')
# plt.figure()
# plt.imshow(outer[:,:,70].astype(float), cmap='nipy_spectral')
# plt.axis('off')



def lst_avg(lst):
    return sum(lst)/len(lst)

def lst_avg_std(lst):
    avg = sum(lst)/len(lst)
    variance = sum([((x - avg) ** 2) for x in lst]) / len(lst) 
    std = variance ** 0.5
    return avg, std


# https://loli.github.io/medpy/_modules/medpy/metric/binary.html
import numpy

def dc(result, reference):
    r"""
    Dice coefficient
    
    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.
    
    The metric is defined as
    
    .. math::
        
        DC=\frac{2|A\cap B|}{|A|+|B|}
        
    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).
    
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    
    Returns
    -------
    dc : float
        The Dice coefficient between the object(s) in ```result``` and the
        object(s) in ```reference```. It ranges from 0 (no overlap) to 1 (perfect overlap).
        
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))
    
    intersection = numpy.count_nonzero(result & reference)
    
    size_i1 = numpy.count_nonzero(result)
    size_i2 = numpy.count_nonzero(reference)
    
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0
    
    return dc

def hd(result, reference, voxelspacing=None, connectivity=1):
    """
    Hausdorff Distance.
    
    Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. It is defined as the maximum surface distance between the objects.
    
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.
        
    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of 
        elements along each dimension, which is usually given in mm.
        
    See also
    --------
    :func:`assd`
    :func:`asd`
    
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity).max()
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd

def hd95(result, reference, voxelspacing=None, connectivity=1):
    """
    95th percentile of the Hausdorff Distance.

    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of 
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`hd`

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = numpy.percentile(numpy.hstack((hd1, hd2)), 95)
    return hd95

def maxmin(result, reference, voxelspacing=None, connectivity=1):
    """
    Distance Maximum:Minimum Ratio.
    
    Computes the maximum:minimum distance ratio between the binary objects in two
    images. It is defined as the maximum/minimum surface distance between the 
    objects in one direction (result to reference).
    
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.
        
    Returns
    -------
    maxmin ratio : float
        The distance ratio (max:min) between the object(s) in ```result``` and the
        object(s) in ```reference```. 
        
    See also
    --------
    :func:`assd`
    :func:`asd`
    
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    distmax = __surface_distances(result, reference, voxelspacing, connectivity).max()
    distmin = __surface_distances(result, reference, voxelspacing, connectivity).min()
    maxmin = distmax/distmin
    return maxmin

def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=numpy.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
            
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    # test for emptiness
    if 0 == numpy.count_nonzero(result): 
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == numpy.count_nonzero(reference): 
        raise RuntimeError('The second supplied array does not contain any binary object.')    
            
    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    
    return sds

def precision(result, reference):
    """
    Precison.
    
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    
    Returns
    -------
    precision : float
        The precision between two binary datasets, here mostly binary objects in images,
        which is defined as the fraction of retrieved instances that are relevant. The
        precision is not symmetric.
    
    See also
    --------
    :func:`recall`
    
    Notes
    -----
    Not symmetric. The inverse of the precision is :func:`recall`.
    High precision means that an algorithm returned substantially more relevant results than irrelevant.
    
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Precision_and_recall
    .. [2] http://en.wikipedia.org/wiki/Confusion_matrix#Table_of_confusion
    """
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))
        
    tp = numpy.count_nonzero(result & reference)
    fp = numpy.count_nonzero(result & ~reference)
    
    try:
        precision = tp / float(tp + fp)
    except ZeroDivisionError:
        precision = 0.0
    
    return precision


def recall(result, reference):
    """
    Recall.
    
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    
    Returns
    -------
    recall : float
        The recall between two binary datasets, here mostly binary objects in images,
        which is defined as the fraction of relevant instances that are retrieved. The
        recall is not symmetric.
    
    See also
    --------
    :func:`precision`
    
    Notes
    -----
    Not symmetric. The inverse of the recall is :func:`precision`.
    High recall means that an algorithm returned most of the relevant results.
    
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Precision_and_recall
    .. [2] http://en.wikipedia.org/wiki/Confusion_matrix#Table_of_confusion
    """
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))
        
    tp = numpy.count_nonzero(result & reference)
    fn = numpy.count_nonzero(~result & reference)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0
    
    return recall

pat_ind = 0
print('Displaying Patient ', dmaps[pat_ind][0])
print('Possible Channels: ', len(dmaps[pat_ind])-1)
print('Patient has ', np.shape(dmaps[pat_ind][3])[0], ' target(s)')
og_dmap = dmaps[pat_ind][1]
print('og_dmap loaded, size: ', np.shape(og_dmap))
rc_dmap = dmaps[pat_ind][2]
print('rc_dmap loaded, size: ', np.shape(rc_dmap))
og_crop_dmap = dmaps[pat_ind][3]
print('og_crop_dmap loaded, size: ', np.shape(og_crop_dmap))
rc_crop_dmap = dmaps[pat_ind][4]
print('rc_crop_dmap loaded, size: ', np.shape(rc_crop_dmap))
og_rx_dmap = dmaps[pat_ind][5]
print('og_rx_dmap loaded, size: ', np.shape(og_rx_dmap))
rc_rx_dmap = dmaps[pat_ind][6]
print('rc_rx_dmap loaded, size: ', np.shape(rc_rx_dmap))
dist_crop = dmaps[pat_ind][7]
print('dist_crop loaded, size: ', np.shape(dist_crop))
# dist_full = dmaps[pat_ind][8]
# print('dist_full loaded, size: ', np.shape(dist_full))
# targ_contour = dmaps[pat_ind][9]
# print('targ_contour loaded, size: ', np.shape(targ_contour))
# full_dist = dmaps[pat_ind][7]
# full_dist_crop = dmaps[pat_ind][8]
# full_dist_crop = dmaps[pat_ind][7]
# print('full_dist_crop loaded, size: ', np.shape(full_dist_crop))

# Calculate the ratio of the maximum distance (target contour edge -> external) 
# to the minimum distance (target contour edge -> external) within the axial
# slice containing the target contour CoM. A 2D calculation avoids the potential
# of inconsistencies caused by the extent of the external in the patient inferior
# direction.
# Also calculate the volume of each target

maxmin_scores = []
maxmin_targ_scores = []
targ_vol = []
targ_track = 0
pat_track = 0
for i_file in range(len(preproc_dmaps)):
    for pat_num in range(len(preproc_dmaps[i_file])):
    # for pat_num in range(8):
        print('===============================================================================')
        print('Starting new patient [', pat_num+1, '/', len(preproc_dmaps[i_file]), ']:  (', preproc_dmaps[i_file][pat_num][0], ')')
        print('===============================================================================')
        pat_maxmin = []
        pat_maxmin.append(preproc_dmaps[i_file][pat_num][0])
    #     HD_scores.append(dmaps[pat_num][0])
    #     HD_scores[pat_num] = []
    #     DSC_scores.append(dmaps[pat_num][0])
    #     DSC_scores[pat_num] = []
        for tar_num in range(len(preproc_dmaps[i_file][pat_num][3])):
            if targ_track == 1000:
                pass
            else:
                targ_maxmin = []
        #         targ_hd.append(len(dmaps[pat_num][3]))
                print('     -------------------------------------------------------------------------------')
                print('     Starting new target calculations [', tar_num+1, '/', len(preproc_dmaps[i_file][pat_num][3]), ']:')
                # Target Contour
                targ = np.array(preproc_dmaps[i_file][pat_num][16][tar_num])
                targ_CoM = scipy.ndimage.center_of_mass(targ)
                mm_img1 = targ[:,:,int(targ_CoM[2])]
                # LGP dmap
                mm_img2 = np.array(preproc_dmaps[i_file][pat_num][1])[:,:,int(targ_CoM[2])]
                print('          --------------------------------------------------------------------------')
                maxmin_temp = maxmin(mm_img1, mm_img2, voxelspacing=0.5)
                targ_maxmin.append(maxmin_temp)
                maxmin_targ_scores.append(maxmin_temp)
                print('          Target max:min ratio: ', maxmin_temp)
                
                print('          --------------------------------------------------------------------------')
                vol = np.sum(targ)*(0.5**3)*0.001 # vol in cm^3
                print('          Target volume: ', vol)
                targ_vol.append(vol)
                
                
                pat_maxmin.append(targ_maxmin)
            targ_track += 1
        pat_track += 1
        maxmin_scores.append(pat_maxmin)
    
print('========================')
print('       Finished         ')
print('========================')
print(targ_track)




targ_diff = [[],[]]

targ_track = 0                      # Useful for skipping problem targets
for i_file in range(len(gpu_0_files)):
    print('Starting new fileset')
    for tar_num in range(len(gpu_0_dmaps[i_file])):
        print('===============================================================================')
        print('Starting new target [', tar_num+1, '/', len(gpu_0_dmaps[i_file]), ']:')
        print('===============================================================================')
        og_dmap   = gpu_0_dmaps[i_file][tar_num][0]
        pred_dmap = gpu_0_dmaps[i_file][tar_num][-1]
        
        targ = gpu_0_dmaps[i_file][tar_num][1]
        targ = np.abs(targ - 1.0)
        og_mask   = og_dmap * targ
        pred_mask = pred_dmap * targ
        
#         diff = og_dmap - pred_dmap
        diff = og_mask - pred_mask
        diff_rav = diff[targ==1].ravel()

        targ_track += 1
        if i_file == 0:
            targ_diff[0].append(diff_rav)
        elif i_file == 1:
            targ_diff[1].append(diff_rav)
#         elif i_file == 2:
#             gpu_0_final_DSC_scores.append(targ_dsc)
    print()
    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    print()
print('--------------')
print('===Finished===')
print('--------------')

inout_diff = [[],[]]

targ_track = 0                      # Useful for skipping problem targets
for i_file in range(len(gpu_1_files)):
    print('Starting new fileset')
    for tar_num in range(len(gpu_1_dmaps[i_file])):
        print('===============================================================================')
        print('Starting new target [', tar_num+1, '/', len(gpu_1_dmaps[i_file]), ']:')
        print('===============================================================================')
        og_dmap   = gpu_1_dmaps[i_file][tar_num][0]
        pred_dmap = gpu_1_dmaps[i_file][tar_num][-1]
        
        targ = gpu_1_dmaps[i_file][tar_num][1]
        targ = np.abs(targ - 1.0)
        print(np.sum(targ))
        og_mask   = og_dmap * targ
        pred_mask = pred_dmap * targ
        
#         diff = og_dmap - pred_dmap
        diff = og_mask - pred_mask
        print(np.shape(diff), np.shape(diff.ravel()))
        diff_rav = diff[targ==1].ravel()
        print(np.shape(diff_rav))

        targ_track += 1
        if i_file == 0:
            inout_diff[0].append(diff_rav)
        elif i_file == 1:
            inout_diff[1].append(diff_rav)
#         elif i_file == 2:
#             gpu_0_final_DSC_scores.append(targ_dsc)
    print()
    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    print()
print('--------------')
print('===Finished===')
print('--------------')

targ_diff_all = targ_diff[0] + targ_diff[1]
print(np.shape(targ_diff_all))
targ_diff_all_flat = np.concatenate(targ_diff_all).ravel()
print(np.shape(targ_diff_all_flat))

inout_diff_all = inout_diff[0] + inout_diff[1]
print(np.shape(inout_diff_all))
inout_diff_all_flat = np.concatenate(inout_diff_all).ravel()
print(np.shape(inout_diff_all_flat))

bin_edge = np.linspace(-1, 1, num=100)

targ_mean = targ_diff_all_flat.mean()
targ_std = targ_diff_all_flat.std()
textstr = '\n'.join((
    r'$\mu=%.4f$' % (targ_mean, ),
    r'$\sigma=%.4f$' % (targ_std, )))
counts, bins = np.histogram(targ_diff_all_flat, bins=bin_edge)
fig, ax = plt.subplots(1,1, facecolor='white', dpi=600)
# plt.figure(facecolor='white', dpi=600)
ax.stairs(counts, bins, fill=False, edgecolor='black', linewidth=1, 
           linestyle='-')
plt.text(.95, .95, textstr, ha='right', va='top', transform=ax.transAxes)

inout_mean = inout_diff_all_flat.mean()
inout_std = inout_diff_all_flat.std()
textstr = '\n'.join((
    r'$\mu=%.4f$' % (inout_mean, ),
    r'$\sigma=%.4f$' % (inout_std, )))
counts, bins = np.histogram(inout_diff_all_flat, bins=bin_edge)
fig, ax = plt.subplots(1,1, facecolor='white', dpi=600)
# plt.figure(facecolor='white', dpi=600)
ax.stairs(counts, bins, fill=False, edgecolor='black', linewidth=1, 
           linestyle='-')
plt.text(.95, .95, textstr, ha='right', va='top', transform=ax.transAxes)





# DSC_scores = []     # Same thing but for Dice Similarity Coeff.
# iso_perc = np.arange(10, 100, 10)   # Defining the thresholding percentages
# targ_track = 0                      # Useful for skipping problem targets
# for i_file in range(len(preproc_dmaps)):
#     for pat_num in range(len(preproc_dmaps[i_file])):
#     # for pat_num in range(8):
#         print('===============================================================================')
#         print('Starting new patient [', pat_num+1, '/', len(preproc_dmaps[i_file]), ']:')
#         print('===============================================================================')
#     #     DSC_scores.append(dmaps[pat_num][0])
#         pat_dsc = []
#         pat_dsc.append(preproc_dmaps[i_file][pat_num][0])
#         for tar_ind in range(len(preproc_dmaps[i_file][pat_num][3])):
#             if targ_track == 16:
#                 pass
#             else:
#                 print('-------------------------------------------------------------------------------')
#                 print('Starting new target DSC calculation [', tar_ind+1, '/', len(preproc_dmaps[i_file][pat_num][3]), ']:')
#         #         print('-------------------------------------------------------------------------------')
#         #         print('Starting DSC Calculation:')
#                 targ_dsc = []        
#         #         targ_dsc.append(len(dmaps[pat_num][3]))
#         #         targ_dsc.append(tar_ind)
#                 for i_iso in range(len(iso_perc)):
#         #             iso_dsc = []
#                     dsc_img1 = np.array(preproc_dmaps[i_file][pat_num][3][tar_ind]).astype(float)
#                     dsc_img1 = dsc_img1/np.max(dsc_img1)

#         #             print(np.shape(dsc_img1))
#                     dsc_img2 = np.array(preproc_dmaps[i_file][pat_num][4][tar_ind]).astype(float)
#                     dsc_img2 = dsc_img2/np.max(dsc_img2)
#         #             print(np.shape(dsc_img2))

#                     thresh1 = np.array(dsc_img1)
#                     thresh1[thresh1 <= iso_perc[i_iso]/100] = 0
#                     thresh1[thresh1 >  0] = 1

#                     thresh2 = np.array(dsc_img2)
#                     thresh2[thresh2 <= iso_perc[i_iso]/100] = 0
#                     thresh2[thresh2 >  0] = 1

#                     intersection = np.sum(thresh1[thresh2==1])
#                     dsc = intersection * 2.0 / (np.sum(thresh1) + np.sum(thresh2))
#         #             iso_dsc.append(dsc)
#                     if iso_perc[i_iso] % 15 == 0:
#                         print('Iso %: ', iso_perc[i_iso], '; DSC: ', dsc)
#                     targ_dsc.append(dsc)
#         #             targ_dsc.append(iso_dsc)
#                 pat_dsc.append(targ_dsc)
#             targ_track += 1
#         DSC_scores.append(pat_dsc)
# print('--------------')
# print('===Finished===')
# print('--------------')

# iso_DSC = []
# for iso_ind in range(len(iso_perc)):
#     print('Current Step: ', iso_ind+1, '/', len(iso_perc))
# #     iso_DSC.append(iso_perc[iso_ind])
# #     iso_DSC[iso_ind] = []
#     thresh_lvl = []
#     for pat_step in range(len(DSC_scores)):
#         for tar_step in range(len(DSC_scores[pat_step])):
#             if tar_step != 0:
#                 thresh_lvl.append(DSC_scores[pat_step][tar_step][iso_ind])
#     iso_DSC.append(thresh_lvl)
    
# print(len(iso_DSC))
# print('Total Targets:', len(iso_DSC[2]))
# AvgSD_DSC_iso = []
# for i in range(len(iso_DSC)):
#     avg, sd = lst_avg_std(iso_DSC[i])
#     AvgSD_DSC_iso.append((avg,sd))
# #     print('HD [vox/mm]: ', avg_HD_iso[i], '/', avg_HD_iso[i]*0.5)
#     print('Avg DSC (SD) at ', iso_perc[i], '% threshold: ', AvgSD_DSC_iso[i])
    
# DSC_just_avg = [avg[0] for avg in AvgSD_DSC_iso]
# print(DSC_just_avg)
# DSC_just_sd = [sd[1] for sd in AvgSD_DSC_iso]
# print(DSC_just_sd)
# # avg_HD_iso_mm = [i * 0.5 for i in avg_HD_iso]
# fig = plt.figure()
# fig.patch.set_facecolor('white')
# plt.errorbar(iso_perc, DSC_just_avg, yerr=DSC_just_sd, fmt='bo')
# plt.title('Average DSC at Given Threshold Levels for 26 Targets')
# plt.ylabel('Dice Similarity Coefficient')
# plt.xlabel('Threshold Percentage')
# ax = plt.gca()
# ax.set_ylim([0.5,1])

gpu_0_DSC  = [[],[]]    # dice similarity
gpu_0_prec = [[],[]]   # precision
gpu_0_rec  = [[],[]]    # recall

# gpu_0_initial_DSC_scores = []     # Same thing but for Dice Similarity Coeff.
# gpu_0_resume_DSC_scores = []
# gpu_0_final_DSC_scores = []
iso_perc = np.arange(10, 100, 10)   # Defining the thresholding percentages
targ_track = 0                      # Useful for skipping problem targets
for i_file in range(len(gpu_0_files)):
    print('Starting new fileset')
    for tar_num in range(len(gpu_0_dmaps[i_file])):
        print('===============================================================================')
        print('Starting new target [', tar_num+1, '/', len(gpu_0_dmaps[i_file]), ']:')
        print('===============================================================================')
        targ_dsc = []
        targ_prec = []
        targ_rec = []
        for i_iso in range(len(iso_perc)):
            dsc_img1 = np.array(gpu_0_dmaps[i_file][tar_num][0]).astype(float)
            dsc_img1 = dsc_img1/np.max(dsc_img1)

#             print(np.shape(dsc_img1))
            dsc_img2 = np.array(gpu_0_dmaps[i_file][tar_num][-1]).astype(float)
            dsc_img2 = dsc_img2/np.max(dsc_img2)
#             print(np.shape(dsc_img2))

            thresh1 = np.array(dsc_img1)
            thresh1[thresh1 <= iso_perc[i_iso]/100] = 0
            thresh1[thresh1 >  0] = 1

            thresh2 = np.array(dsc_img2)
            thresh2[thresh2 <= iso_perc[i_iso]/100] = 0
            thresh2[thresh2 >  0] = 1

            intersection = np.sum(thresh1[thresh2==1])
            dsc = intersection * 2.0 / (np.sum(thresh1) + np.sum(thresh2))
            
#             iso_dsc.append(dsc)
            if iso_perc[i_iso] % 15 == 0:
                print('Iso %: ', iso_perc[i_iso], '; DSC: ', dsc)
            targ_dsc.append(dsc)
            
            prec = precision(thresh2, thresh1)
            targ_prec.append(prec)
            rec  = recall(thresh2, thresh1)
            targ_rec.append(rec)
            
        targ_track += 1
        if i_file == 0:
            gpu_0_DSC[0].append(targ_dsc)
            gpu_0_prec[0].append(targ_prec)
            gpu_0_rec[0].append(targ_rec)
        elif i_file == 1:
            gpu_0_DSC[1].append(targ_dsc)
            gpu_0_prec[1].append(targ_prec)
            gpu_0_rec[1].append(targ_rec)
#         elif i_file == 2:
#             gpu_0_final_DSC_scores.append(targ_dsc)
    print()
    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    print()
print('--------------')
print('===Finished===')
print('--------------')

gpu_1_DSC  = [[],[]]    # dice similarity
gpu_1_prec = [[],[]]   # precision
gpu_1_rec  = [[],[]]    # recall

# gpu_1_initial_DSC_scores = []     # Same thing but for Dice Similarity Coeff.
# gpu_1_resume_DSC_scores = []
# gpu_1_final_DSC_scores = []
iso_perc = np.arange(10, 100, 10)   # Defining the thresholding percentages
targ_track = 0                      # Useful for skipping problem targets
for i_file in range(len(gpu_1_files)):
    print('Starting new fileset')
    for tar_num in range(len(gpu_1_dmaps[i_file])):
    # for pat_num in range(8):
        print('===============================================================================')
        print('Starting new target [', tar_num+1, '/', len(gpu_1_dmaps[i_file]), ']:')
        print('===============================================================================')
        targ_dsc = []    
        targ_prec = []
        targ_rec = []
#         targ_dsc.append(len(dmaps[pat_num][3]))
#         targ_dsc.append(tar_ind)
        for i_iso in range(len(iso_perc)):
#             iso_dsc = []
            dsc_img1 = np.array(gpu_1_dmaps[i_file][tar_num][0]).astype(float)
            dsc_img1 = dsc_img1/np.max(dsc_img1)

#             print(np.shape(dsc_img1))
            dsc_img2 = np.array(gpu_1_dmaps[i_file][tar_num][-1]).astype(float)
            dsc_img2 = dsc_img2/np.max(dsc_img2)
#             print(np.shape(dsc_img2))

            thresh1 = np.array(dsc_img1)
            thresh1[thresh1 <= iso_perc[i_iso]/100] = 0
            thresh1[thresh1 >  0] = 1

            thresh2 = np.array(dsc_img2)
            thresh2[thresh2 <= iso_perc[i_iso]/100] = 0
            thresh2[thresh2 >  0] = 1

            intersection = np.sum(thresh1[thresh2==1])
            dsc = intersection * 2.0 / (np.sum(thresh1) + np.sum(thresh2))
#             iso_dsc.append(dsc)
            if iso_perc[i_iso] % 15 == 0:
                print('Iso %: ', iso_perc[i_iso], '; DSC: ', dsc)
            targ_dsc.append(dsc)
            
            prec = precision(thresh2, thresh1)
            targ_prec.append(prec)
            rec  = recall(thresh2, thresh1)
            targ_rec.append(rec)
            
        targ_track += 1
        if i_file == 0:
            gpu_1_DSC[0].append(targ_dsc)
            gpu_1_prec[0].append(targ_prec)
            gpu_1_rec[0].append(targ_rec)
        elif i_file == 1:
            gpu_1_DSC[1].append(targ_dsc)
            gpu_1_prec[1].append(targ_prec)
            gpu_1_rec[1].append(targ_rec)
#         elif i_file == 2:
#             gpu_1_final_DSC_scores.append(targ_dsc)
    print()
    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    print()
print('--------------')
print('===Finished===')
print('--------------')

print('gpu0')
print(len(gpu_0_DSC))
gpu_0_DSC_all = gpu_0_DSC[0] + gpu_0_DSC[1]
gpu_0_prec_all = gpu_0_prec[0] + gpu_0_prec[1]
gpu_0_rec_all = gpu_0_rec[0] + gpu_0_rec[1]
print(len(gpu_0_DSC_all))

print('gpu1')
print(len(gpu_1_DSC))
gpu_1_DSC_all = gpu_1_DSC[0] + gpu_1_DSC[1]
gpu_1_prec_all = gpu_1_prec[0] + gpu_1_prec[1]
gpu_1_rec_all = gpu_1_rec[0] + gpu_1_rec[1]
print(len(gpu_1_DSC_all))



gpu_0_iso_DSC = []
for iso_ind in range(len(iso_perc)):
    print('Current Step: ', iso_ind+1, '/', len(iso_perc))
#     iso_DSC.append(iso_perc[iso_ind])
#     iso_DSC[iso_ind] = []
    thresh_lvl = []
    for tar_step in range(len(gpu_0_DSC_all)):
#         if tar_step != 0:
            thresh_lvl.append(gpu_0_DSC_all[tar_step][iso_ind])
    gpu_0_iso_DSC.append(thresh_lvl)

print()
print()
    
gpu_1_iso_DSC = []
for iso_ind in range(len(iso_perc)):
    print('Current Step: ', iso_ind+1, '/', len(iso_perc))
#     iso_DSC.append(iso_perc[iso_ind])
#     iso_DSC[iso_ind] = []
    thresh_lvl = []
    for tar_step in range(len(gpu_1_DSC_all)):
#         if tar_step != 0:
            thresh_lvl.append(gpu_1_DSC_all[tar_step][iso_ind])
    gpu_1_iso_DSC.append(thresh_lvl)

gpu_0_AvgSD_DSC_iso = []
for i in range(len(gpu_0_iso_DSC)):
    avg, sd = lst_avg_std(gpu_0_iso_DSC[i])
    gpu_0_AvgSD_DSC_iso.append((avg,sd))
#     print('HD [vox/mm]: ', avg_HD_iso[i], '/', avg_HD_iso[i]*0.5)
    print('Avg DSC (SD) at ', iso_perc[i], '% threshold: ', gpu_0_AvgSD_DSC_iso[i])

print()
print()

gpu_1_AvgSD_DSC_iso = []
for i in range(len(gpu_1_iso_DSC)):
    avg, sd = lst_avg_std(gpu_1_iso_DSC[i])
    gpu_1_AvgSD_DSC_iso.append((avg,sd))
#     print('HD [vox/mm]: ', avg_HD_iso[i], '/', avg_HD_iso[i]*0.5)
    print('Avg DSC (SD) at ', iso_perc[i], '% threshold: ', gpu_1_AvgSD_DSC_iso[i])

# Set the font dictionaries (for plot title and axis titles)
title_font = {'fontname':'Arial', 'size':'18', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} # Bottom vertical alignment for more space
axis_font = {'fontname':'Arial', 'size':'14'}
tick_font = {'fontname':'Arial', 'size':'10'}

gpu_0_DSC_just_avg = [avg[0] for avg in gpu_0_AvgSD_DSC_iso]
print(gpu_0_DSC_just_avg)
gpu_0_DSC_just_sd = [sd[1] for sd in gpu_0_AvgSD_DSC_iso]
print(gpu_0_DSC_just_sd)
# avg_HD_iso_mm = [i * 0.5 for i in avg_HD_iso]
fig = plt.figure(figsize=(6,4))
fig.patch.set_facecolor('white')
# plt.errorbar(iso_perc, gpu_0_DSC_just_avg, yerr=gpu_0_DSC_just_sd, fmt='bo')
dsc_bp_t = plt.boxplot(gpu_0_iso_DSC, showmeans=True, meanline=True,
            labels=['10','20','30','40','50','60','70','80','90'])
    # plt.title('Average DSC at Given Threshold Levels \n for 26 Targets (gpu0 initial pats 01-10)')
    # plt.title('Average DSC at Given Threshold Levels for 49 Targets \n (UNET InnerOuterDepth Pats 111-130)')

plt.title('DSC at Given Threshold Levels', **title_font)
# plt.title('DSC at Given Threshold Levels for Large Cohort \n (HD-UNET, Input: Target, n=20)', **title_font)
# plt.title('DSC at Given Threshold Levels for Small Cohort \n (HD-UNET, Input: Target, n=49)', **title_font)
# plt.title('DSC at Given Threshold Levels for Large Cohort \n (UNET, Input: Target, n=20)', **title_font)
# plt.title('DSC at Given Threshold Levels for Small Cohort \n (UNET, Input: Target, n=49)', **title_font)

    # plt.title('Average DSC at Given Threshold Levels for 49 Targets \n (UNET TargetDepth Pats 111-130)')
    # plt.title('Average DSC at Given Threshold Levels for 20 Targets \n (UNET InnerOuter Pats 111-130)')
    # plt.title('Average DSC at Given Threshold Levels for 49 Targets \n (UNET InnerOuterDepth Pats 111-130)')
plt.ylabel('DSC', **axis_font)
plt.xlabel('Threshold Percentage', **axis_font)
plt.xticks(**tick_font)
plt.yticks(**tick_font)
ax = plt.gca()
ax.set_ylim([-0.05,1])
print()



gpu_1_DSC_just_avg = [avg[0] for avg in gpu_1_AvgSD_DSC_iso]
print(gpu_1_DSC_just_avg)
gpu_1_DSC_just_sd = [sd[1] for sd in gpu_1_AvgSD_DSC_iso]
print(gpu_1_DSC_just_sd)
# avg_HD_iso_mm = [i * 0.5 for i in avg_HD_iso]
fig = plt.figure(figsize=(6,4))
fig.patch.set_facecolor('white')
# plt.errorbar(iso_perc, gpu_1_DSC_just_avg, yerr=gpu_1_DSC_just_sd, fmt='bo')
dsc_bp_io = plt.boxplot(gpu_1_iso_DSC, showmeans=True, meanline=True,
            labels=['10','20','30','40','50','60','70','80','90'])
        # plt.title('Average DSC at Given Threshold Levels \n for 26 Targets (gpu1 initial pats 01-10)')
        # plt.title('Average DSC at Given Threshold Levels for 49 Targets \n (HD_UNET InnerOuterDepth 111-130)')

plt.title('DSC at Given Threshold Levels', **title_font)
# plt.title('DSC at Given Threshold Levels for Large Cohort \n (HD-UNET, Input: InOut, n=20)', **title_font)
# plt.title('DSC at Given Threshold Levels for Small Cohort \n (HD-UNET, Input: InOut, n=49)', **title_font)
# plt.title('DSC at Given Threshold Levels for Large Cohort \n (UNET, Input: InOut, n=20)', **title_font)
# plt.title('DSC at Given Threshold Levels for Small Cohort \n (UNET, Input: InOut, n=49)', **title_font)
    
        # plt.title('Average DSC at Given Threshold Levels for 49 Targets \n (HD-UNET TargetDepth Pats 111-130)')
        # plt.title('Average DSC at Given Threshold Levels for 20 Targets \n (HD-UNET InnerOuter Pats 111-130)')
        # plt.title('Average DSC at Given Threshold Levels for 49 Targets \n (HD-UNET InnerOuterDepth Pats 111-130)')
plt.ylabel('DSC', **axis_font)
plt.xlabel('Threshold Percentage', **axis_font)
plt.xticks(**tick_font)
plt.yticks(**tick_font)
ax = plt.gca()
ax.set_ylim([-0.05,1])
print()

dsc_t_medians = [item.get_ydata()[0] for item in dsc_bp_t['medians']]
dsc_t_means = [item.get_ydata()[0] for item in dsc_bp_t['means']]
print(f'Medians: {dsc_t_medians}\n'
      f'Means:   {dsc_t_means}')

dsc_t_q1 = [min(item.get_ydata()) for item in dsc_bp_t['boxes']]
dsc_t_q3 = [max(item.get_ydata()) for item in dsc_bp_t['boxes']]
print(f'Q1: {dsc_t_q1}\n'
      f'Q3: {dsc_t_q3}')

dsc_t_iqr = [q3_i - q1_i for q3_i, q1_i in zip(dsc_t_q3, dsc_t_q1)]
print(f'IQR: {dsc_t_iqr}')

dsc_io_medians = [item.get_ydata()[0] for item in dsc_bp_io['medians']]
dsc_io_means = [item.get_ydata()[0] for item in dsc_bp_io['means']]
print(f'Medians: {dsc_io_medians}\n'
      f'Means:   {dsc_io_means}')

dsc_io_q1 = [min(item.get_ydata()) for item in dsc_bp_io['boxes']]
dsc_io_q3 = [max(item.get_ydata()) for item in dsc_bp_io['boxes']]
print(f'Q1: {dsc_io_q1}\n'
      f'Q3: {dsc_io_q3}')

dsc_io_iqr = [q3_i - q1_i for q3_i, q1_i in zip(dsc_io_q3, dsc_io_q1)]
print(f'IQR: {dsc_io_iqr}')

dsc_stats = []
for i_iso in range(len(gpu_0_iso_DSC)):
    targ_set = gpu_0_iso_DSC[i_iso]
    inout_set = gpu_1_iso_DSC[i_iso]
    # Want to test that inout > targ
    # Wilcoxon tests (d = targ - inout) based on order provided
    # Want to test if d is less than a distr symm about 0 
    dsc_stats.append(wilcoxon(targ_set, inout_set, alternative='less'))
    print(dsc_stats[i_iso][1])
# print(dsc_stats)





gpu_0_HD95 = [[],[]]

# gpu_0_HD95_initial = []      # Initialze a list to store the Hausdorff Distances for each patient 
#                             # at varying thresholds
#                             # HD_scores[patient number]
# gpu_0_HD95_resume = []
# gpu_0_HD95_final = []
# DSC_scores = []     # Same thing but for Dice Similarity Coeff.
iso_perc = np.arange(10, 100, 10)   # Defining the thresholding percentages
targ_track = 0                      # Useful for skipping problem targets
for i_file in range(len(gpu_1_files)):
    print('Starting new fileset')
    file_HD = []
    for tar_num in range(len(gpu_0_dmaps[i_file])):
        print('     ===============================================================================')
        print('     Starting new target [', tar_num+1, '/', len(gpu_0_dmaps[i_file]), ']:')
        print('     ===============================================================================')
        targ_hd = []
        hd_img1 = np.array(gpu_0_dmaps[i_file][tar_num][0])
        hd_img2 = np.array(gpu_0_dmaps[i_file][tar_num][-1])
        print('     Starting Hausdorff Calculation:')
        for i_iso in range(len(iso_perc)):
            iso_1_coord = []
            iso_2_coord = []

            iso_1 = np.zeros(np.shape(hd_img1))
            iso_2 = np.zeros(np.shape(hd_img1))

            iso_1[hd_img1 >= iso_perc[i_iso]/100*np.amax(hd_img1)] = 1
            iso_2[hd_img2 >= iso_perc[i_iso]/100*np.amax(hd_img2)] = 1
#             iso_2[hd_img2 >= iso_perc[i_iso]] = 1

            hd95_temp = hd95(iso_1, iso_2)
            hd95_mm = hd95_temp * 0.5
            targ_hd.append(hd95_mm)

#                     iso1_rmv = ndimage.binary_erosion(iso_1)
#                     iso2_rmv = ndimage.binary_erosion(iso_2)

#                     iso1_prm = iso_1 - iso1_rmv
#                     iso2_prm = iso_2 - iso2_rmv

#                     i1, j1, k1 = np.where(iso1_prm == 1)
#                     i2, j2, k2 = np.where(iso2_prm == 1)

#                     for i in range(len(i1)):
#                         iso_1_coord.append(tuple([i1[i], j1[i], k1[i]]))
#                     for i in range(len(i2)):
#                         iso_2_coord.append(tuple([i2[i], j2[i], k2[i]]))
            if iso_perc[i_iso] % 15 == 0:
                print('     Hausdorff distance Calculating...')

#                     HD1 = directed_hausdorff(iso_1_coord, iso_2_coord)
#                     HD2 = directed_hausdorff(iso_2_coord, iso_1_coord)
#                     HD = max(HD1[0], HD2[0])
#                     targ_hd.append(HD)
            if iso_perc[i_iso] % 15 == 0:
                print('          Iso %: ', iso_perc[i_iso], 
                      '; Hausdorff distance [ voxels // mm ]= [ ', hd95_temp, ' // ', hd95_temp * 0.5, ' ]')
        targ_track += 1
        if i_file == 0:
            gpu_0_HD95[0].append(targ_hd)
        elif i_file == 1:
            gpu_0_HD95[1].append(targ_hd)
#         elif i_file == 2:
#             gpu_0_HD95_final.append(targ_hd)
    print()
    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    print()
print('========================')
print('       Finished         ')
print('========================')
print(targ_track)

gpu_1_HD95 = [[],[]]

# gpu_1_HD95_initial = []      # Initialze a list to store the Hausdorff Distances for each patient 
#                             # at varying thresholds
#                             # HD_scores[patient number]
# gpu_1_HD95_resume = []
# gpu_1_HD95_final = []
# DSC_scores = []     # Same thing but for Dice Similarity Coeff.
iso_perc = np.arange(10, 100, 10)   # Defining the thresholding percentages
targ_track = 0                      # Useful for skipping problem targets
for i_file in range(len(gpu_1_files)):
    print('Starting new fileset')
    file_HD = []
    for tar_num in range(len(gpu_1_dmaps[i_file])):
        print('     ===============================================================================')
        print('     Starting new target [', tar_num+1, '/', len(gpu_1_dmaps[i_file]), ']:')
        print('     ===============================================================================')
        targ_hd = []
        hd_img1 = np.array(gpu_1_dmaps[i_file][tar_num][0])
        hd_img2 = np.array(gpu_1_dmaps[i_file][tar_num][-1])
        print('     Starting Hausdorff Calculation:')
        for i_iso in range(len(iso_perc)):
            iso_1_coord = []
            iso_2_coord = []

            iso_1 = np.zeros(np.shape(hd_img1))
            iso_2 = np.zeros(np.shape(hd_img1))

            iso_1[hd_img1 >= iso_perc[i_iso]/100*np.amax(hd_img1)] = 1
            iso_2[hd_img2 >= iso_perc[i_iso]/100*np.amax(hd_img2)] = 1

            hd95_temp = hd95(iso_1, iso_2)
            hd95_mm = hd95_temp * 0.5
            targ_hd.append(hd95_mm)

#                     iso1_rmv = ndimage.binary_erosion(iso_1)
#                     iso2_rmv = ndimage.binary_erosion(iso_2)

#                     iso1_prm = iso_1 - iso1_rmv
#                     iso2_prm = iso_2 - iso2_rmv

#                     i1, j1, k1 = np.where(iso1_prm == 1)
#                     i2, j2, k2 = np.where(iso2_prm == 1)

#                     for i in range(len(i1)):
#                         iso_1_coord.append(tuple([i1[i], j1[i], k1[i]]))
#                     for i in range(len(i2)):
#                         iso_2_coord.append(tuple([i2[i], j2[i], k2[i]]))
            if iso_perc[i_iso] % 15 == 0:
                print('     Hausdorff distance Calculating...')

#                     HD1 = directed_hausdorff(iso_1_coord, iso_2_coord)
#                     HD2 = directed_hausdorff(iso_2_coord, iso_1_coord)
#                     HD = max(HD1[0], HD2[0])
#                     targ_hd.append(HD)
            if iso_perc[i_iso] % 15 == 0:
                print('          Iso %: ', iso_perc[i_iso], 
                      '; Hausdorff distance [ voxels // mm ]= [ ', hd95_temp, ' // ', hd95_temp * 0.5, ' ]')
        targ_track += 1
        if i_file == 0:
            gpu_1_HD95[0].append(targ_hd)
        elif i_file == 1:
            gpu_1_HD95[1].append(targ_hd)
#         elif i_file == 2:
#             gpu_1_HD95_final.append(targ_hd)
    print()
    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    print()
print('========================')
print('       Finished         ')
print('========================')
print(targ_track)

# HD95_scores = []      # Initialze a list to store the Hausdorff Distances for each patient 
#                     # at varying thresholds
#                     # HD_scores[patient number]
# # DSC_scores = []     # Same thing but for Dice Similarity Coeff.
# iso_perc = np.arange(10, 100, 10)   # Defining the thresholding percentages
# targ_track = 0                      # Useful for skipping problem targets
# for i_file in range(len(preproc_dmaps)):
#     for pat_num in range(len(preproc_dmaps[i_file])):
#     # for pat_num in range(8):
#         print('===============================================================================')
#         print('Starting new patient [', pat_num+1, '/', len(preproc_dmaps[i_file]), ']:')
#         print('===============================================================================')
#         pat_hd = []
#         pat_hd.append(preproc_dmaps[i_file][pat_num][0])
#     #     HD_scores.append(dmaps[pat_num][0])
#     #     HD_scores[pat_num] = []
#     #     DSC_scores.append(dmaps[pat_num][0])
#     #     DSC_scores[pat_num] = []
#         for tar_num in range(len(preproc_dmaps[i_file][pat_num][3])):
#             if targ_track == 16:
#                 pass
#             else:
#                 targ_hd = []
#         #         targ_hd.append(len(dmaps[pat_num][3]))
#                 print('-------------------------------------------------------------------------------')
#                 print('Starting new target calculation [', tar_num+1, '/', len(preproc_dmaps[i_file][pat_num][3]), ']:')
#                 hd_img1 = np.array(preproc_dmaps[i_file][pat_num][3][tar_num])
#                 hd_img2 = np.array(preproc_dmaps[i_file][pat_num][4][tar_num])
#                 print('-------------------------------------------------------------------------------')
#                 print('Starting Hausdorff Calculation:')
#                 for i_iso in range(len(iso_perc)):
#                     iso_1_coord = []
#                     iso_2_coord = []

#                     iso_1 = np.zeros(np.shape(hd_img1))
#                     iso_2 = np.zeros(np.shape(hd_img1))

#                     iso_1[hd_img1 >= iso_perc[i_iso]/100*np.amax(hd_img1)] = 1
#                     iso_2[hd_img2 >= iso_perc[i_iso]/100*np.amax(hd_img2)] = 1
                    
#                     hd95_temp = hd95(iso_1, iso_2)
#                     targ_hd.append(hd95_temp)

# #                     iso1_rmv = ndimage.binary_erosion(iso_1)
# #                     iso2_rmv = ndimage.binary_erosion(iso_2)

# #                     iso1_prm = iso_1 - iso1_rmv
# #                     iso2_prm = iso_2 - iso2_rmv

# #                     i1, j1, k1 = np.where(iso1_prm == 1)
# #                     i2, j2, k2 = np.where(iso2_prm == 1)

# #                     for i in range(len(i1)):
# #                         iso_1_coord.append(tuple([i1[i], j1[i], k1[i]]))
# #                     for i in range(len(i2)):
# #                         iso_2_coord.append(tuple([i2[i], j2[i], k2[i]]))
#                     if iso_perc[i_iso] % 15 == 0:
#                         print('Hausdorff distance Calculating...')

# #                     HD1 = directed_hausdorff(iso_1_coord, iso_2_coord)
# #                     HD2 = directed_hausdorff(iso_2_coord, iso_1_coord)
# #                     HD = max(HD1[0], HD2[0])
# #                     targ_hd.append(HD)
#                     if iso_perc[i_iso] % 15 == 0:
#                         print('Iso %: ', iso_perc[i_iso], 
#                               '; Hausdorff distance [ voxels // mm ]= [ ', HD, ' // ', HD * 0.5, ' ]')
#                 pat_hd.append(targ_hd)
#             targ_track += 1
#         HD95_scores.append(pat_hd)
    
# print('========================')
# print('       Finished         ')
# print('========================')
# print(targ_track)

# iso_HD95 = []
# for iso_ind in range(len(iso_perc)):
#     print('Current Step: ', iso_ind+1, '/', len(iso_perc))
#     thresh_lvl = []
#     for pat_step in range(len(HD95_scores)):
#         for tar_step in range(len(HD95_scores[pat_step])):
#             if tar_step != 0:
#                 thresh_lvl.append(HD95_scores[pat_step][tar_step][iso_ind])
#     iso_HD95.append(thresh_lvl)
    
# AvgSD_HD95_iso = []
# for i in range(len(iso_HD95)):
#     avg, sd = lst_avg_std(iso_HD95[i])
#     AvgSD_HD95_iso.append((avg,sd))
# #     print('HD [vox/mm]: ', avg_HD_iso[i], '/', avg_HD_iso[i]*0.5)
#     print('Avg HD (SD) at ', iso_perc[i], '% threshold: ', AvgSD_HD95_iso[i], '[vox]')

# HD95_just_avg = [avg[0] * 0.5 for avg in AvgSD_HD95_iso]
# print(HD95_just_avg)
# HD95_just_sd = [sd[1] * 0.5 for sd in AvgSD_HD95_iso]
# print(HD95_just_sd)
# # avg_HD_iso_mm = [i * 0.5 for i in avg_HD_iso]

# fig = plt.figure()
# fig.patch.set_facecolor('white')
# plt.errorbar(iso_perc, HD95_just_avg, yerr=HD95_just_sd, fmt='rD')
# plt.title('Average HD95 at Given Threshold Levels for 26 Targets')
# plt.ylabel('Hausdorff Distance [mm]')
# plt.xlabel('Threshold Percentage')
# ax = plt.gca()
# ax.set_ylim([0,15])

print('gpu0')
print(len(gpu_0_HD95))
gpu_0_HD95_all = gpu_0_HD95[0] + gpu_0_HD95[1]
print(len(gpu_0_HD95_all))

print('gpu1')
print(len(gpu_1_HD95))
gpu_1_HD95_all = gpu_1_HD95[0] + gpu_1_HD95[1]
print(len(gpu_1_HD95_all))

gpu_0_iso_HD95 = []
for iso_ind in range(len(iso_perc)):
    print('Current Step: ', iso_ind+1, '/', len(iso_perc))
    thresh_lvl = []
    for tar_step in range(len(gpu_0_HD95_all)):
#         if tar_step != 0:
            thresh_lvl.append(gpu_0_HD95_all[tar_step][iso_ind])
    gpu_0_iso_HD95.append(thresh_lvl)
    
print()
print()
    
gpu_1_iso_HD95 = []
for iso_ind in range(len(iso_perc)):
    print('Current Step: ', iso_ind+1, '/', len(iso_perc))
    thresh_lvl = []
    for tar_step in range(len(gpu_1_HD95_all)):
#         if tar_step != 0:
            thresh_lvl.append(gpu_1_HD95_all[tar_step][iso_ind])
    gpu_1_iso_HD95.append(thresh_lvl)

AvgSD_HD95_iso_gpu_0 = []
for i in range(len(gpu_0_iso_HD95)):
    avg, sd = lst_avg_std(gpu_0_iso_HD95[i])
    AvgSD_HD95_iso_gpu_0.append((avg,sd))
    print('Avg HD95 (SD) at ', iso_perc[i], '% threshold: ', AvgSD_HD95_iso_gpu_0[i], '[mm]')
    
print()
print()
    
AvgSD_HD95_iso_gpu_1 = []
for i in range(len(gpu_1_iso_HD95)):
    avg, sd = lst_avg_std(gpu_1_iso_HD95[i])
    AvgSD_HD95_iso_gpu_1.append((avg,sd))
    print('Avg HD95 (SD) at ', iso_perc[i], '% threshold: ', AvgSD_HD95_iso_gpu_1[i], '[mm]')

# HD95_just_avg_gpu_0 = [avg[0] * 0.5 for avg in AvgSD_HD95_iso_gpu_0]
# print(HD95_just_avg_gpu_0)
# HD95_just_sd_gpu_0 = [sd[1] * 0.5 for sd in AvgSD_HD95_iso_gpu_0]
# print(HD95_just_sd_gpu_0)
fig = plt.figure(figsize=(6,4))
fig.patch.set_facecolor('white')
# plt.errorbar(iso_perc, HD95_just_avg_gpu_0, yerr=HD95_just_sd_gpu_0, fmt='rD')
hd_bp_t = plt.boxplot(gpu_0_iso_HD95, showmeans=True, meanline=True,
            labels=['10','20','30','40','50','60','70','80','90'])
    # plt.title('Average HD95 at Given Threshold Levels \n for 26 Targets (gpu0 initial pats 01-10)')
    # plt.title('Average HD95 at Given Threshold Levels for 49 Targets \n (UNET InnerOuterDepth Pats 111-130)')
    # plt.title('Average HD95 at Given Threshold Levels for 20 Targets \n (UNET Target Pats 111-130)')
    # plt.title('Average HD95 at Given Threshold Levels for 49 Targets \n (UNET TargetDepth Pats 111-130)')

plt.title('HD95 at Given Threshold Levels', **title_font)
# plt.title('HD95 at Given Threshold Levels for Large Cohort \n (HD-UNET, Input: Target, n=20)', **title_font)
# plt.title('HD95 at Given Threshold Levels for Small Cohort \n (HD-UNET, Input: Target, n=49)', **title_font)
# plt.title('HD95 at Given Threshold Levels for Large Cohort \n (UNET, Input: Target, n=20)', **title_font)
# plt.title('HD95 at Given Threshold Levels for Small Cohort \n (UNET, Input: Target, n=49)', **title_font)

    # plt.title('Average HD95 at Given Threshold Levels for 49 Targets \n (UNET InnerOuterDepth Pats 111-130)')
plt.ylabel('HD95 [mm]', **axis_font)
plt.xlabel('Threshold Percentage', **axis_font)
plt.xticks(**tick_font)
plt.yticks(**tick_font)
ax = plt.gca()
ax.set_ylim([0,65])

print()
print()


# HD95_just_avg_gpu_1 = [avg[0] * 0.5 for avg in AvgSD_HD95_iso_gpu_1]
# print(HD95_just_avg_gpu_1)
# HD95_just_sd_gpu_1 = [sd[1] * 0.5 for sd in AvgSD_HD95_iso_gpu_1]
# print(HD95_just_sd_gpu_1)
fig = plt.figure(figsize=(6,4))
fig.patch.set_facecolor('white')
# plt.errorbar(iso_perc, HD95_just_avg_gpu_1, yerr=HD95_just_sd_gpu_1, fmt='rD')
hd_bp_io = plt.boxplot(gpu_1_iso_HD95, showmeans=True, meanline=True,
            labels=['10','20','30','40','50','60','70','80','90'])
    # plt.title('Average HD95 at Given Threshold Levels \n for 26 Targets (gpu1 initial pats 01-10)')
    # plt.title('Average HD95 at Given Threshold Levels for 49 Targets \n (HD-UNET InnerOuterDepth Pats 111-130)')
    # plt.title('Average HD95 at Given Threshold Levels for 20 Targets \n (HD-UNET Target Pats 111-130)')
    # plt.title('Average HD95 at Given Threshold Levels for 49 Targets \n (HD-UNET TargetDepth Pats 111-130)')

plt.title('HD95 at Given Threshold Levels', **title_font)
# plt.title('HD95 at Given Threshold Levels for Large Cohort \n (HD-UNET, Input: InOut, n=20)', **title_font)
# plt.title('HD95 at Given Threshold Levels for Small Cohort \n (HD-UNET, Input: InOut, n=49)', **title_font)
# plt.title('HD95 at Given Threshold Levels for Large Cohort \n (UNET, Input: InOut, n=20)', **title_font)
# plt.title('HD95 at Given Threshold Levels for Small Cohort \n (UNET, Input: InOut, n=49)', **title_font)

    # plt.title('Average HD95 at Given Threshold Levels for 49 Targets \n (HD-UNET InnerOuterDepth Pats 111-130)')
plt.ylabel('HD95 [mm]', **axis_font)
plt.xlabel('Threshold Percentage', **axis_font)
plt.xticks(**tick_font)
plt.yticks(**tick_font)
ax = plt.gca()
ax.set_ylim([0,65])


hd_t_medians = [item.get_ydata()[0] for item in hd_bp_t['medians']]
hd_t_means = [item.get_ydata()[0] for item in hd_bp_t['means']]
print(f'Medians: {hd_t_medians}\n'
      f'Means:   {hd_t_means}')

hd_t_q1 = [min(item.get_ydata()) for item in hd_bp_t['boxes']]
hd_t_q3 = [max(item.get_ydata()) for item in hd_bp_t['boxes']]
print(f'Q1: {hd_t_q1}\n'
      f'Q3: {hd_t_q3}')

hd_t_iqr = [q3_i - q1_i for q3_i, q1_i in zip(hd_t_q3, hd_t_q1)]
print(f'IQR: {hd_t_iqr}')

hd_io_medians = [item.get_ydata()[0] for item in hd_bp_io['medians']]
hd_io_means = [item.get_ydata()[0] for item in hd_bp_io['means']]
print(f'Medians: {hd_io_medians}\n'
      f'Means:   {hd_io_means}')

hd_io_q1 = [min(item.get_ydata()) for item in hd_bp_io['boxes']]
hd_io_q3 = [max(item.get_ydata()) for item in hd_bp_io['boxes']]
print(f'Q1: {hd_io_q1}\n'
      f'Q3: {hd_io_q3}')

hd_io_iqr = [q3_i - q1_i for q3_i, q1_i in zip(hd_io_q3, hd_io_q1)]
print(f'IQR: {hd_io_iqr}')

hd95_stats = []
for i_iso in range(len(gpu_0_iso_HD95)):
    targ_set = gpu_0_iso_HD95[i_iso]
    inout_set = gpu_1_iso_HD95[i_iso]
    # Want to test that inout < targ
    # Wilcoxon tests (d = targ - inout) based on order provided
    # Want to test if d is greater than a distr symm about 0 
    hd95_stats.append(wilcoxon(targ_set, inout_set, alternative='greater'))
    print(hd95_stats[i_iso][1])

