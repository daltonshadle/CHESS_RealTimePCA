#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 21:12:02 2021

@author: djs522
"""

#%% ***************************************************************************
# IMPORTS
try:
    import dill as cpl
except(ImportError):
    import pickle as cpl

import yaml

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn.preprocessing import Normalizer
from sklearn.manifold import LocallyLinearEmbedding, Isomap, MDS, TSNE


import time

from scipy import ndimage

from hexrd import config
from hexrd import xrdutil
from hexrd import instrument
from hexrd import imageutil
from hexrd import imageseries
from hexrd import material


import sys 
IMPORT_HEXRD_SCRIPT_DIR = '/home/djs522/bin/view_snapshot_series'
sys.path.insert(0, IMPORT_HEXRD_SCRIPT_DIR)
import preprocess_dexela_h5 as ppdex
import pp_dexela

#%% EXTRA FUNCTIONS
# *****************************************************************************

# plane data
def load_pdata(cpkl, key):
    with open(cpkl, "rb") as matf:
        mat_list = cpl.load(matf)
    return dict(list(zip([i.name for i in mat_list], mat_list)))[key].planeData

# plane data
def load_pdata_hexrd3(h5, key):
    temp_mat = material.Material(material_file=h5, name=key)
    return temp_mat.planeData


# images
def load_images(yml):
    return imageseries.open(yml, format="image-files")


# instrument
def load_instrument(yml):
    with open(yml, 'r') as f:
        icfg = yaml.safe_load(f)
    return instrument.HEDMInstrument(instrument_config=icfg)

def pca2ipca(pca):
    ipca = decomposition.IncrementalPCA(n_components=pca.n_components_)
    ipca.components_ = pca.components_
    ipca.explained_variance_ = pca.explained_variance_
    ipca.explained_variance_ratio_ = pca.explained_variance_ratio_
    ipca.singular_values_ = pca.singular_values_
    ipca.mean_ = pca.mean_
    #ipca.var_ = pca.var_ #not sure about this
    ipca.var_ = np.zeros(pca.mean_.shape)
    ipca.noise_variance_ = pca.noise_variance_
    ipca.n_components_ = pca.n_components_
    ipca.n_samples_seen_ = pca.n_samples_
    
    return ipca

def get_eta_tth_spot_stats(polar_img):         
    # input:
    #   polar image (m x n) [eta x tth] m = # eta pixels, n = # tth pixels
    # output:
    #   spot_stats (r x 5) [eta_mean, tth_mean, eta_std, tth_std, total_inten] r = # spots
    
    labels, n_feats = ndimage.label(polar_img)
    
    if n_feats > 0:
        spot_stats = np.zeros([n_feats, 5])
    else:
        spot_stats = np.array([[-1, -1, -1, -1, 1]])
    
    for i in range(n_feats):
        feat_idx = np.array(np.where(labels == i+1)).T #[row_idx, col_idx]
        row_min = feat_idx[:, 0].min()
        row_max = feat_idx[:, 0].max() + 1
        col_min = feat_idx[:, 1].min()
        col_max = feat_idx[:, 1].max() + 1
        
        feat_label = np.copy(labels[row_min:row_max, col_min:col_max])
        feat_img = np.copy(polar_img[row_min:row_max, col_min:col_max])
        feat_img[feat_label != i+1] = 0
        
        eta_inten = np.sum(feat_img, axis=1)
        tth_inten = np.sum(feat_img, axis=0)
        
        eta_pos = np.arange(row_min, row_max)
        tth_pos = np.arange(col_min, col_max)
        eta_mean = np.average(eta_pos, weights=eta_inten)
        tth_mean = np.average(tth_pos, weights=tth_inten)
        eta_std = np.sqrt(np.average((eta_pos-eta_mean)**2, weights=eta_inten))
        tth_std = np.sqrt(np.average((tth_pos-tth_mean)**2, weights=tth_inten))
        total_inten = np.sum(feat_img)
        
        spot_stats[i, :] = np.array([eta_mean, tth_mean, eta_std, tth_std, total_inten])
    
    return spot_stats

def get_weighted_eta_tth_metrics(spot_stats, max_pos=None, eta_max=1, tth_max=1):
    if max_pos is None:
        max_stats = spot_stats[spot_stats[:, 4] == spot_stats[:, 4].max(), :]
    else:
        temp_pos = np.copy(max_pos)
        temp_pos[0] = temp_pos[0] / eta_max
        temp_pos[1] = temp_pos[1] / tth_max
        other_pos = np.copy(spot_stats[:, [0, 1]])
        other_pos[:, 0] = other_pos[:, 0] / eta_max
        other_pos[:, 1] = other_pos[:, 1] / tth_max
        
        dist = np.linalg.norm(other_pos - max_pos, axis=1)
        max_stats = spot_stats[dist == dist.min(), :]
    
    tot_eta_mean = np.average(spot_stats[:, 0], weights=spot_stats[:, 4])
    tot_tth_mean = np.average(spot_stats[:, 1], weights=spot_stats[:, 4])
    tot_eta_std = np.average(spot_stats[:, 2], weights=spot_stats[:, 4])
    tot_tth_std = np.average(spot_stats[:, 3], weights=spot_stats[:, 4])
    
    return [tot_eta_mean, tot_tth_mean, tot_eta_std, tot_tth_std, spot_stats.shape[0],
            max_stats[0,0], max_stats[0,1], max_stats[0,2], max_stats[0,3], max_stats[0,4]]
        

#%% ***************************************************************************
# FUNCTION DECLARATION
def animate_frames(scan_list, image_list, reg1_bnd, reg2_bnd, max_cmap=None, save_gif=False, name_gif='temp.gif', time_per_frame=1):
    numframes = len(scan_list)

    # plot entire volume fundamental region
    fig, axs = plt.subplots(1, 3)
    
    if max_cmap is None:
        max_cmap = 8000
    
    # plot images
    init_scan = scan_list[0]
    init_image = image_list[0]
    axs[0].imshow(init_image, vmax=max_cmap)
    axs[1].imshow(init_image[reg1_bnd[0]:reg1_bnd[1], reg1_bnd[2]:reg1_bnd[3]], vmax=max_cmap)
    axs[2].imshow(init_image[reg2_bnd[0]:reg2_bnd[1], reg2_bnd[2]:reg2_bnd[3]], vmax=max_cmap)
    fig.suptitle('Scan %i' %init_scan)
    
    
    # do function animation
    ani = animation.FuncAnimation(fig, update_frames_ani, interval=time_per_frame*1000,
                                  frames=numframes, 
                                  fargs=(scan_list, image_list, reg1_bnd, reg2_bnd, max_cmap, axs, fig))
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    if save_gif:
        ani.save(name_gif, writer='imagemagick', dpi=200)

def update_frames_ani(i, scan_list, image_list, reg1_bnd, reg2_bnd, max_cmap, axs, fig):
    
    i_scan = scan_list[i]
    i_image = image_list[i]
    
    axs[0].cla()
    axs[1].cla()
    axs[2].cla()
    
    axs[0].imshow(i_image, vmax=max_cmap)
    axs[1].imshow(i_image[reg1_bnd[0]:reg1_bnd[1], reg1_bnd[2]:reg1_bnd[3]], vmax=max_cmap)
    axs[2].imshow(i_image[reg2_bnd[0]:reg2_bnd[1], reg2_bnd[2]:reg2_bnd[3]], vmax=max_cmap)
    fig.suptitle('Scan %i' %i_scan)

    return axs


def simple_anim_frames(image_list, max_cmap=None, save_gif=False, name_gif='temp.gif', time_per_frame=1):
    numframes = len(image_list)

    # plot entire volume fundamental region
    fig, axs = plt.subplots(1, 1)
    
    if max_cmap is None:
        max_cmap = 1
    
    # plot images
    init_image = image_list[0]
    axs.imshow(init_image, vmax=max_cmap)
    
    # do function animation
    ani = animation.FuncAnimation(fig, simple_update_frames, interval=time_per_frame*1000,
                                  frames=numframes, 
                                  fargs=(image_list, max_cmap, axs, fig))
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    if save_gif:
        ani.save(name_gif, writer='imagemagick', dpi=200)

def simple_update_frames(i, image_list, max_cmap, axs, fig):
    i_image = image_list[i]
    
    axs.cla()
    
    axs.imshow(i_image, vmax=max_cmap)

    return axs

def animate_frames_stress_strain_with_spots(scan_list, image_list, stress_strain_list, 
                                            max_cmap=None, save_gif=False, 
                                            name_gif='temp.gif', time_per_frame=1):
    
    numframes = len(scan_list)

    # plot entire volume fundamental region
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    
    if max_cmap is None:
        max_cmap = 8000
    
    # plot images
    init_scan = scan_list[0]
    init_image = image_list[0]
    init_ss = stress_strain_list[0, :]
    axs[0].scatter(init_ss[1], init_ss[0], c='blue')
    axs[1].imshow(init_image, vmax=max_cmap)
    #fig.suptitle('Scan %i' %init_scan)
    
    
    # do function animation
    ani = animation.FuncAnimation(fig, update_frames_ani_ss_with_spots, interval=time_per_frame*1000,
                                  frames=numframes, 
                                  fargs=(scan_list, image_list, stress_strain_list, max_cmap, axs, fig))
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    if save_gif:
        ani.save(name_gif, writer='imagemagick', dpi=200)

def update_frames_ani_ss_with_spots(i, scan_list, image_list, stress_strain_list, 
                                            max_cmap, axs, fig):
    
    i_scan = scan_list[i]
    i_image = image_list[i]
    i_ss = stress_strain_list[:i, :]
    
    axs[0].cla()
    axs[1].cla()
    
    axs[0].scatter(i_ss[:, 1], i_ss[:, 0], c='blue')
    if i > 1:
        axs[0].scatter(i_ss[-1, 1], i_ss[-1, 0], c='red', s=100)
    axs[1].imshow(i_image, vmax=max_cmap)
    #fig.suptitle('Scan %i' %i_scan)
    
    axs[0].set_xlabel('Macroscopic Strain', fontsize=16)
    axs[0].set_ylabel('Macroscopic Stress (MPa)', fontsize=16)
    axs[1].set_title('Subset of Detector', fontsize=16)
    
    axs[0].set_xlim([-0.0055, 0.0055])
    axs[0].set_ylim([-400, 400])

    return axs

def scale_0_1(data):
    return (data - data.min()) / (data.max() - data.min())

#%% ***************************************************************************
# CONSTANTS AND PARAMETERS
'''
debug = True 

raw_fmt = 'hdf5'
raw_path = '/imageseries'
key = 'ff1'
raw_data_dir_template = '/nfs/chess/raw/current/id3a/%s/%s/%d/ff/'
expt_name = 'miller-881-2'
samp_name = 'ss718-1'

start_scan = 0
end_scan = 400
scan_step = 1
frame_num = 0
inten_max = 2500
time_per_frame = 0.2 #seconds

# coarse refined region for viewing
row_start1 = 1800
row_stop1 = 2300
col_start1 = 1800
col_stop1 = 2400
reg1_bounds = [row_start1, row_stop1, col_start1, col_stop1]

# fine refined region for viewing
row_start2 = 1600
row_stop2 = 2200
col_start2 = 1970
col_stop2 = 2140
reg2_bounds = [row_start2, row_stop2, col_start2, col_stop2]

# cycle indexing info by dic image number
# ss718-1
init_load = [2013, 2069]
cyc1 = [2069, 2237]
cyc2 = [2237, 2331]
cyc3 = [2331, 2409]
cyc4 = [2409, 2456]
cyc5 = [2456, 2483]
cyc6 = [2483, 2512]
cyc7 = [2512, 2554]
cyc8 = [2554, 2615]
cyc9_15 = [2615, 2678]
cyc16 = [2678, 2729]
cyc17_31 = [2732, 2773]
cyc32 = [2773, 2814]
cyc33_63 = [2817, 2893]
cyc64 = [2901, 2953]
cyc65_127 = [2954, 3125]
cyc128 = [3125, 3187]

c0_ls1 = [2014, 2030] # start
c0_ls2 = [2030, 2041] # 150 MPa
c0_ls3 = [2048, 2053] # 250 MPa
c0_ls3 = [2062, 2071] # 0.25%

c1_ls1 = [2071, 2095] # 0.5%
c1_ls2 = [2097, 2123] # 0 MPa
c1_ls3 = [2135, 2147] # 0 %
c1_ls4 = [2155, 2163] # -0.25%
c1_ls5 = [2163, 2182] # -0.5%
c1_ls6 = [2185, 2204] # 0 MPa
c1_ls7 = [2210, 2218] # 0 %
c1_ls8 = [2228, 2237] # 0.25%

c2_ls1 = [2237, 2272] # 0.5%
c2_ls2 = [2282, 2288] # -0.25%
c2_ls3 = [2288, 2318] # -0.5%
c2_ls4 = [2327, 2333] # 0.25%

cyc_list = [init_load, cyc1, cyc2, cyc3, cyc4, cyc5, cyc6, cyc7, cyc8, cyc16, cyc32, cyc64, cyc128]
cyc_list = [c1_ls1, c1_ls2, c1_ls3, c1_ls4, c1_ls5, c1_ls6, c1_ls7, c1_ls8]
cyc_list = [c2_ls1, c2_ls2, c2_ls3, c2_ls4]

cyc_list = [init_load, cyc1]

bad_scan_list = [181]
'''
            

#%% ***************************************************************************
# STRESS STRAIN PREPROCESSING

ss718 = True

if ss718:
    temp_dir = '/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/'
    raw_data_dir_template = temp_dir + 'new_ss718_ct_thresh360/'
    # [img_num, stress, strain_xx, strain_yy, strain_xy, screw_pos]
    dic = np.loadtxt(temp_dir+'ss718-1_nov2020.txt')
    # [img_num, screw_pos, stress, image_ind, ct_scan_num]
    ct2dic = np.load(temp_dir+'sync_ct_scans_2_dic_ss718.npy')
    save_name = 'new_ss718-1_ct2dic.npy'
    sample_name = 'ss718-1'
    
    first_img_base = 'ss718-1_33_%s_000027-cachefile.npz'
    deform_img_base = 'ss718-1_1331_%s_001317-cachefile.npz'
    
    first_img_base = 'ss718-1_81_%s_000075-cachefile.npz'
    deform_img_base = 'ss718-1_92_%s_000085-cachefile.npz'
    
    dark_image_base = 'ss718-1_7_%s_000019-cachefile.npz'
    
    # fine refined region for viewing
    row_start2 = 1680
    row_stop2 = 2120
    col_start2 = 980
    col_stop2 = 1140
    reg_bounds = [row_start2, row_stop2, col_start2, col_stop2]
    
    scan_step = 1
    inten_max = 500
    inten_min = 50
    time_per_frame = 0.5 #seconds
    
    start_dic = 2013 #2237 #2013
    end_dic = 2375 #2375 till end of cycle 2
    
    bad_scan_list = [61, 181, 236, 318, 547, 573] #ss
else:
    temp_dir = '/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/'
    raw_data_dir_template = temp_dir + 'new_dp718_ct_thresh360/'
    # [img_num, stress, strain_xx, strain_yy, strain_xy, screw_pos]
    dic = np.loadtxt(temp_dir+'dp718-1_nov2020_dic.txt')
    # [img_num, screw_pos, stress, image_ind, ct_scan_num]
    ct2dic = np.load(temp_dir+'sync_ct_scans_2_dic_dp718.npy')
    save_name = 'new_dp718-1_ct2dic.npy'
    sample_name = 'dp718-1'
    
    first_img_base = 'dp718-1_22_%s_001340-cachefile.npz'
    deform_img_base = 'dp718-1_961_%s_002277-cachefile.npz'
    
    # fine refined region for viewing
    row_start2 = 1680
    row_stop2 = 2120
    col_start2 = 980
    col_stop2 = 1140
    reg_bounds = [row_start2, row_stop2, col_start2, col_stop2]
    
    scan_step = 1
    inten_max = 500
    inten_min = 50
    time_per_frame = 0.5 #seconds
    
    start_dic = 3216 
    end_dic = 3300#3407
    
    bad_scan_list = [] #ss

raw_fmt = 'frame-cache'
raw_path = '/imageseries'
key = 'ff1'
frame_num = 0

# main directory
base_path = '/media/djs522/djs522_nov2020/chess_2020_11/'
analysis_path = os.path.join(base_path, 'analysis')
instrument_fname = 'dexela_nov_2020_ceo2_ruby.yml'
material_fname = 'materials_61_332_36000.h5'
instr = load_instrument(os.path.join(analysis_path, instrument_fname))
plane_data = load_pdata_hexrd3(os.path.join(analysis_path, material_fname), 'in718')

# grab panel
panel = instr.detectors[key]
# native_area = panel.pixel_area  # pixel ref area

# pixel angular coords for the detector panel
panel_tth, panel_eta = panel.pixel_angles()

# [dic_img_num, ct_scan_num, stress, strain_xx, strain_yy, strain_xy]
new_ct2dic = []
for i in range(ct2dic.shape[0]):
    temp_img_num = ct2dic[i, 0]
    ind = np.where(dic[:, 0] == temp_img_num)[0]
    if ind.size != 0:
        temp_array = np.zeros(6)
        temp_array[0] = ct2dic[i, 0]
        temp_array[1] = ct2dic[i, 4]
        temp_array[2:] = dic[ind, 1:5]
        new_ct2dic.append(temp_array)
new_ct2dic = np.array(new_ct2dic)

# dic_ind = np.in1d(dic[:, 0].astype(int), ct2dic[:, 0].astype(int))
# ct2dic_ind = np.in1d(ct2dic[:, 0].astype(int), dic[:, 0].astype(int))

# dic_uni, dic_ind = np.unique(dic[dic_ind, 0].astype(int), return_index=True)
# ct2dic_uni, ct2dic_ind = np.unique(ct2dic[ct2dic_ind, 0].astype(int), return_index=True)

# condensed_dic = dic[dic_ind, :]
# condensed_ct2dic = ct2dic[ct2dic_ind, :]

# new_ct2dic = np.zeros([condensed_ct2dic.shape[0], 6])
# new_ct2dic[:, 0] = condensed_ct2dic[:, 0]
# new_ct2dic[:, 1] = condensed_ct2dic[:, 4]
# new_ct2dic[:, 2:] = condensed_dic[:, 1:5]

np.save(save_name, new_ct2dic)


ind = np.where(((new_ct2dic[:, 0] >= 2013) & (new_ct2dic[:, 0] <= 2375)) | 
               ((new_ct2dic[:, 0] >= 3124) & (new_ct2dic[:, 0] <= 3190)))
ind = np.where(((new_ct2dic[:, 0] >= 2013) & (new_ct2dic[:, 0] <= 2375)) | 
               ((new_ct2dic[:, 0] >= 2894) & (new_ct2dic[:, 0] <= 2953)))
good_dic_list = new_ct2dic[ind, 0]


#%%

scan_im_series = imageseries.open(raw_data_dir_template+first_img_base%(key), raw_fmt, path=raw_path)
mask_scan_im_series = np.zeros(panel_tth.shape)

deg2rad = np.pi / 180.0
rad2deg = 180.0 / np.pi

delta_tth = 0.25 * deg2rad # degrees to radians (smallest resolution ~0.0075 deg)
mask_tth = np.array([5.57, 6.43, 9.1, 10.68, 11.16, 12.89, 14.05]) * deg2rad
#ring = 1
#mask_tth = np.array([mask_tth[ring]])
mask_tth = mask_tth[:6]

max_eta = 60 * deg2rad
min_eta = -60 * deg2rad
step_eta = 0.25 * deg2rad

panel_mask = np.zeros(panel_tth.shape)
for tth in mask_tth:
    print(tth)
    min_tth = tth - delta_tth / 2.0
    max_tth = tth + delta_tth / 2.0
    temp_ind = np.where((panel_tth > min_tth) & (panel_tth < max_tth) & (panel_eta < max_eta) & (panel_eta > min_eta))
    panel_mask[temp_ind] = 1
panel_mask = panel_mask.astype(int)
mask_scan_im_series[panel_mask == 1] = scan_im_series[frame_num][panel_mask == 1]
mask_scan_im_series[mask_scan_im_series < inten_min] = 0   
    
fig = plt.figure()
temp_p = 0
plt.imshow(panel_mask, vmax=temp_p+1, vmin=temp_p)
#plt.imshow(panel_mask, vmax=np.size(panel_mask[panel_mask > 0]), vmin=0)
plt.show()
print(np.size(panel_mask[panel_mask > 0]))



#%%
raw_fmt_h5 = 'hdf5'
raw_path_h5 = '/imageseries'
dark_series_ff1 = imageseries.open('/home/djs522/bin/ff/ff1_001337.h5', raw_fmt_h5, path=raw_path_h5)
dark_series_ff2 = imageseries.open('/home/djs522/bin/ff/ff2_001337.h5', raw_fmt_h5, path=raw_path_h5)


print("Images loaded")

num_dark = 600
dark_ff1 = imageseries.stats.median(dark_series_ff1, nframes=num_dark)
dark_ff2 = imageseries.stats.median(dark_series_ff2, nframes=num_dark)

fig = plt.figure()
plt.imshow(dark_ff1)

fig = plt.figure()
plt.imshow(dark_ff2)
plt.show()

# if flip in ('y', 'v'):  # about y-axis (vertical)
#     pimg = img[:, ::-1]
# elif flip in ('x', 'h'):  # about x-axis (horizontal)
#     pimg = img[::-1, :]

#%%
temp_dark_ff1 = dark_ff1[:, ::-1]
temp_dark_ff2 = dark_ff2[::-1, :]
temp_dark_ff1 = np.insert(temp_dark_ff1, 1, 0, axis=0)
temp_dark_ff1 = np.insert(temp_dark_ff1, 1, 0, axis=1)
temp_dark_ff2 = np.insert(temp_dark_ff2, 1, 0, axis=0)
temp_dark_ff2 = np.insert(temp_dark_ff2, 1, 0, axis=1)
#temp_dark = np.hstack([np.flip(temp_dark_1, axis=1), np.flip(temp_dark_2, axis=1)])
temp_dark = np.hstack([temp_dark_ff1, temp_dark_ff2])
fig = plt.figure()
plt.imshow(temp_dark, aspect='auto', vmax=500)


#%%

raw_thresh = 5

plane_data = load_pdata_hexrd3(os.path.join(analysis_path, material_fname), 'in718')
plane_data.set_exclusions([4, 5])

eta_pix_size = 0.05
tth_pix_size = 0.005
my_polar_ff1 = xrdutil.PolarView(plane_data, instr,
                 eta_min=-60., eta_max=60.,
                 pixel_size=(tth_pix_size, eta_pix_size))

my_polar_ff2 = xrdutil.PolarView(plane_data, instr,
                 eta_min=120., eta_max=240.,
                 pixel_size=(tth_pix_size, eta_pix_size))

ff1_img = imageseries.open(raw_data_dir_template+first_img_base%('ff1'), raw_fmt, path=raw_path)
ff2_img = imageseries.open(raw_data_dir_template+first_img_base%('ff2'), raw_fmt, path=raw_path)

t_ff1_img = ff1_img[0] - temp_dark_ff1
t_ff1_img[t_ff1_img < raw_thresh] = 0
t_ff2_img = ff2_img[0] - temp_dark_ff2
t_ff2_img[t_ff2_img < raw_thresh] = 0

img_dict = {'ff1':t_ff1_img, 'ff2':t_ff2_img}
eta_tth_ff1_img1 = my_polar_ff1.warp_image(img_dict, pad_with_nans=True,
                   do_interpolation=False)
eta_tth_ff2_img1 = my_polar_ff2.warp_image(img_dict, pad_with_nans=True,
                   do_interpolation=False)

ff1_img = imageseries.open(raw_data_dir_template+deform_img_base%('ff1'), raw_fmt, path=raw_path)
ff2_img = imageseries.open(raw_data_dir_template+deform_img_base%('ff2'), raw_fmt, path=raw_path)

t_ff1_img = ff1_img[0] - temp_dark_ff1
t_ff1_img[t_ff1_img < raw_thresh] = 0
t_ff2_img = ff2_img[0] - temp_dark_ff2
t_ff2_img[t_ff2_img < raw_thresh] = 0

img_dict = {'ff1':t_ff1_img, 'ff2':t_ff2_img}
eta_tth_ff1_img2 = my_polar_ff1.warp_image(img_dict, pad_with_nans=True,
                   do_interpolation=True)
eta_tth_ff2_img2 = my_polar_ff2.warp_image(img_dict, pad_with_nans=True,
                   do_interpolation=True)
     

bin_img1 = np.vstack([np.copy(eta_tth_ff1_img1.data) / eta_tth_ff1_img1.data.max(), 
                      np.copy(eta_tth_ff2_img1.data) / eta_tth_ff2_img1.data.max()])
bin_img2 = np.vstack([np.copy(eta_tth_ff1_img2.data) / eta_tth_ff1_img2.data.max(), 
                      np.copy(eta_tth_ff2_img2.data) / eta_tth_ff2_img2.data.max()])


#%%
temp_thresh = 0
temp_up_thresh = 0.001

fig = plt.figure()
plt.imshow(bin_img1, aspect='auto', vmin=temp_thresh, vmax=temp_thresh+temp_up_thresh)

fig = plt.figure()
plt.imshow(bin_img2, aspect='auto', vmin=temp_thresh, vmax=temp_thresh+temp_up_thresh)


fig = plt.figure()
plt.imshow(bin_img1 - bin_img2, aspect='auto', vmin=-0.1, vmax=0.1)

# max_pos = np.array([[1.23346298e+03, 1.68408656e+01], # ring 0 spot
#                    [2.18550716e+03, 1.74760274e+01], # ring 1 spot
#                    [8.96659982e+02, 1.38890915e+01]]) # ring 2 spot



# tth_ring_rng = int((plane_data.getTThRanges()[0, 1] - plane_data.getTThRanges()[0, 0]) / (tth_pix_size * deg2rad) + 1)  
# for i in range(plane_data.getNHKLs()):
#     img1_stats = get_eta_tth_spot_stats(bin_img1[:, i*tth_ring_rng:(i+1)*tth_ring_rng])  
#     img1_tot_stats = get_weighted_eta_tth_metrics(img1_stats, max_pos=max_pos[i, :])
#     print(i)
#     print(img1_stats.shape)
#     #print(img1_tot_stats)
    
#     img2_stats = get_eta_tth_spot_stats(bin_img2[:, i*tth_ring_rng:(i+1)*tth_ring_rng])  
#     img2_tot_stats = get_weighted_eta_tth_metrics(img2_stats, max_pos=max_pos[i, :])
#     print(img2_stats.shape)
#     #print(img2_tot_stats)


plt.show()



#%%

raw_thresh = 5 #300

eta_ome_scan_image_list = []
raw_scan_image_list = []
scan_list = []
stress_list = []
strain_list = []

shift_eta_list = []
shift_tth_list = []
spread_eta_list = []
spread_tth_list = []
num_spot_list = []

all_stats = []

#max_pos = max_pos

b_inten_list = []
inten_list = []
polar_inten_list = []

ring0_stats = []
ring1_stats = []
ring2_stats = []

pca_time = []
ipca_time = []
pca_error = []

n_cmpts = 3
PCA_func = decomposition.PCA(n_components=n_cmpts)
IPCA_func = decomposition.IncrementalPCA(n_components=n_cmpts)

pca_switch = 4

for i, row in enumerate(new_ct2dic):
    if (i % scan_step) == 0:
        dic_num = row[0]
        scan_num = row[1]
        scan_stress = row[2]
        scan_strain = row[3]
        
        if (dic_num in good_dic_list) and (scan_num not in bad_scan_list):
            
            #ss718-1_33_ff1_000027-cachefile
            file_name_key = (sample_name+'_%i_%s_' %(scan_num, 'ff1'))
            for file in os.listdir(raw_data_dir_template):
                if file_name_key in file:
                    scan_series_dir_ff1 = os.path.join(raw_data_dir_template, file)
                    scan_series_dir_ff2 = os.path.join(raw_data_dir_template, file.replace('ff1', 'ff2'))
                    
                    if os.path.isfile(scan_series_dir_ff1):
                        scan_im_series_ff1 = imageseries.open(scan_series_dir_ff1, raw_fmt, path=raw_path)
                        scan_im_series_ff2 = imageseries.open(scan_series_dir_ff2, raw_fmt, path=raw_path)
                        full_img_ff1 = scan_im_series_ff1[frame_num] - temp_dark_ff1
                        full_img_ff2 = scan_im_series_ff2[frame_num] - temp_dark_ff2
                        full_img_ff1[full_img_ff1 < raw_thresh] = 0
                        full_img_ff2[full_img_ff2 < raw_thresh] = 0
                        
                        print(dic_num, scan_num, scan_stress)
                        
                        '''
                        if scan_num < 90:
                            full_img = full_img
                        elif scan_num < 111:
                            full_img = full_img * (0.38685845730140467) #(165.0 / 374.0)
                        elif scan_num < 136:
                            full_img = full_img * (0.2884083723233839) #(165.0 / 657.0)
                        else:
                            full_img = full_img * (0.15955562762132985) #(165.0 / 816.0)
                        '''
                        
                        img_dict = {'ff1':full_img_ff1, 'ff2':full_img_ff2}
                        eta_tth_ff1_img = my_polar_ff1.warp_image(img_dict, pad_with_nans=True,
                                           do_interpolation=False)
                        eta_tth_ff2_img = my_polar_ff2.warp_image(img_dict, pad_with_nans=True,
                                           do_interpolation=False)
                        
                        eta_tth_tot_img = np.vstack([eta_tth_ff1_img, eta_tth_ff2_img])
                        raw_tot_img = np.hstack([full_img_ff2, full_img_ff1])
                        
                        eta_ome_scan_image_list.append(eta_tth_tot_img)
                        raw_scan_image_list.append(raw_tot_img)
                        scan_list.append(scan_num)
                        stress_list.append(scan_stress)
                        strain_list.append(scan_strain)
                        
                        
                        
                        # timing
                        # PCA_mat = np.zeros([len(scan_image_list), small_img.size])
                        # for i, image in enumerate(scan_image_list):
                        #     PCA_mat[i, :] = image.ravel(order='C') # C = Row, F = Column
                        # transformer = Normalizer().fit(PCA_mat)
                        # PCA_mat = np.atleast_2d(transformer.transform(PCA_mat))
                        # print(PCA_mat.shape)
                        
                        
                        # if len(scan_image_list) > n_cmpts:
                            
                        #     start = time.time()
                        #     Z = PCA_func.fit_transform(PCA_mat)
                        #     end = time.time()
                        #     pca_time.append(end - start)
                            
                            
                        #     Zi = Z
                        #     if len(scan_image_list) < pca_switch:
                        #         ipca_time.append(end - start)
                        #     elif len(scan_image_list) == pca_switch:
                        #         IPCA_func = pca2ipca(PCA_func)
                                
                        #         start = time.time()
                        #         IPCA_func.partial_fit(np.atleast_2d(PCA_mat[-n_cmpts:, :]))
                        #         Zi = IPCA_func.transform(PCA_mat)
                        #         end = time.time()
                        #         ipca_time.append(end - start)
                        #     else:
                        #         start = time.time()
                        #         IPCA_func.partial_fit(np.atleast_2d(PCA_mat[-n_cmpts:, :]))
                        #         Zi = IPCA_func.transform(PCA_mat)
                        #         end = time.time()
                        #         ipca_time.append(end - start)
                            
                            
                            
                        #     pca_error.append(np.linalg.norm(Z - Zi))
                        
                        # num_rings = plane_data.getNHKLs()
                        # ring_tth_range = int(eta_tth_img.data.shape[1] / num_rings)
                        
                        # shft_eta = []
                        # shft_tth = []
                        # spr_eta = []
                        # spr_tth = []
                        # num_spots = []
                        # stats = []
                        
                        # inten_list.append(np.sum(full_img))
                        # polar_inten_list.append(np.sum(small_img))
                        
                        # for j in range(plane_data.getNHKLs()):
                        #     img_stats = get_eta_tth_spot_stats(small_img[:, j*ring_tth_range:(j+1)*ring_tth_range])
                        #     if len(all_stats) > 2:
                        #         temp_pos = all_stats[-1][j][5:7]
                        #     else:
                        #         temp_pos = max_pos[j, :]
                        #     img_tot_stats = get_weighted_eta_tth_metrics(img_stats, max_pos=temp_pos)  
                            
                        #     if j == 0:
                        #         ring0_stats.append(img_stats)
                        #     elif j == 1:
                        #         ring1_stats.append(img_stats)
                        #     else:
                        #         ring2_stats.append(img_stats)
                        #         print(img_tot_stats[-1])
                            
                        #     shft_eta.append(img_tot_stats[0])
                        #     shft_tth.append(img_tot_stats[1])
                        #     spr_eta.append(img_tot_stats[2])
                        #     spr_tth.append(img_tot_stats[3])
                        #     num_spots.append(img_tot_stats[4])
                        #     stats.append(img_tot_stats)
                        #     #print(img_tot_stats)
                        
                        # all_stats.append(stats)
                        # shift_eta_list.append(shft_eta)
                        # shift_tth_list.append(shft_tth)
                        # spread_eta_list.append(spr_eta)
                        # spread_tth_list.append(spr_tth)
                        # num_spot_list.append(num_spots)
                        
                        #print(dic_num, scan_num)
pca_eta_ome_image_list = eta_ome_scan_image_list
pca_raw_image_list = raw_scan_image_list

#%%

fig = plt.figure(figsize=(10,10))
plt.imshow(raw_tot_img[:, 3073:2*3073], vmax=100, cmap='bone')
plt.show()



#%%

'''
ss_list = np.vstack([stress_list, strain_list]).T
animate_frames_stress_strain_with_spots(scan_list, scan_image_list, ss_list, 
                                            max_cmap=1000, save_gif=True, 
                                            name_gif=temp_dir+'temp5.gif', time_per_frame=0.1)
'''

#%%

my_sc = [81, 92, 107, 113, 132, 138]
my_arr = np.array(scan_list)
my_int = np.array(b_inten_list)
my_int_p = np.array(polar_inten_list)
my_list = np.where(np.in1d(my_arr, my_sc))[0]

r_int = my_int_p[my_list]

print(my_list)
print(my_arr[my_list])
print(r_int)

i = r_int[0] / r_int[1]
j = i * r_int[2] / r_int[3]
k = j * r_int[4] / r_int[5]
print(i, j, k)


for i in my_list:
    fig = plt.figure()
    plt.imshow(pca_image_list[i], aspect='auto')
    print(np.sum(pca_image_list[i]), np.max(pca_image_list[i]))
'''
fig = plt.figure()
plt.scatter(scan_list, my_int_p)


temp_stats = np.array(all_stats)
fig = plt.figure()
plt.scatter(scan_list, temp_stats[:, :, 4])
'''
plt.show()


#%%

from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit

def FWHM(X,Y):
    half_max = max(Y) / 2.
    #find when function crosses line half_max (when sign of diff flips)
    #take the 'derivative' of signum(half_max - Y[])
    d = np.sign(half_max - np.array(Y[0:-1])) - np.sign(half_max - np.array(Y[1:]))
    #plot(X[0:len(d)],d) #if you are interested
    #find the left and right most indexes
    left_idx = np.where(d > 0)[0]
    right_idx = np.where(d < 0)[-1]
    return X[right_idx] - X[left_idx] #return the difference (full width)


def gauss(x, a, mu, sigma):
    return a * np.exp(-(x-mu)**2 / 2 * sigma**2)

def fit_cake(polar_img, rings=3, plot=False, bounds=None, gauss_fit=False):
    cake = np.sum(polar_img, axis=0)
    inds = int(cake.shape[0] / rings)
    cake = cake / cake.max()
    
    fits = []
    mean = inds / 2
    sigma = 2
    
    for i in range(rings):
        x = np.arange(inds)
        y = cake[(i)*inds:(i+1)*inds]
        if gauss_fit:
            if bounds is not None:
                fit, cov = curve_fit(gauss, x, y, p0=[1,mean,sigma], bounds=bounds)
            else:
                fit, cov = curve_fit(gauss, x, y, p0=[1,mean,sigma])
        else:
            avg = np.average(x, weights=y)
            #fwhm = FWHM(y, y)[0]
            std = np.average((x - avg)**2, weights=y)
            fit = np.array([1, avg, std])
            
        
        fits.append(fit)
    fits = np.array(fits)
    
    
    if plot:
        fig = plt.figure()
        for i in range(rings):
            x = np.arange(inds)
            y = cake[(i)*inds:(i+1)*inds]
            plt.plot(x + i*inds, y)
            if gauss_fit:
                plt.plot(x + i*inds, gauss(x, fits[i, 0], fits[i,1], fits[i,2]))
                plt.scatter(fits[i,1] + i*inds, fits[i,0])
    
    return fits # rows = rings, cols = [const, mu, sigma]

#%%
# initialize PCA and matrix variables

pca_image_list = pca_eta_ome_image_list # pca_raw_image_list, pca_eta_ome_image_list


rings = 3
k = 0
l = 51 * rings
m = 0
n = 4800

# k = 4000
# l = 4250
# m = 1500
# n = 2400

ind = np.hstack([np.arange(900,1500), np.arange(3300, 3900)])
#ind = np.hstack([np.arange(750,1650), np.arange(3150, 4050)])
#ind = np.hstack([np.arange(600,1800), np.arange(3000, 4200)])

#ind = np.arange(4800)

num_cmpts = 3
num_images = len(pca_image_list)
num_pixels = pca_image_list[0][m:n, k:l][ind, :].size

# assemble PCA matri

bigPCA_mat = np.zeros([num_images, num_pixels])
PCA_cmpts = np.zeros([num_images, num_cmpts])
all_fits = np.zeros([num_images, 3 * rings])
for i, image in enumerate(pca_image_list):
    temp_img = image[m:n, k:l]
    temp_img = temp_img[ind, :]
    #bigPCA_mat[i, :] = np.log(temp_img.ravel(order='C') / temp_img[temp_img > 0].max())  # C = Row, F = Column
    for j in range(rings):
        temp_img[:, j*51:(j+1)*51] = temp_img[:, j*51:(j+1)*51] / np.max(temp_img[:, j*51:(j+1)*51])
    bigPCA_mat[i, :] = temp_img.ravel(order='C')  # C = Row, F = Column
    fits = fit_cake(temp_img, rings=rings, plot=False, gauss_fit=False)
    all_fits[i, :] = fits.ravel()

#%%


k = 52 
l = 206

fig = plt.figure()
plt.plot(scale_0_1(all_fits[k:l, 1]))
plt.plot(scale_0_1(all_fits[k:l, 2]))
plt.show()

#%%
bounds = [np.array([0, 0, 0]), np.array([100, 100, 100])]
bounds = None
#fits = fit_cake(pca_image_list[52+98][600:1800, :], rings=4, plot=True, bounds=bounds)
fits = fit_cake(temp_img, rings=rings, plot=True, bounds=bounds, gauss_fit=False)

print(fits)

#%%
k = 4080
l = 4170
m = 1700
n = 2120

k = 0
l = 51 * 1
m = 1000
n = 1330

temp_ind = [0, 20, 31, 40, 52, 76, 105, 127, 140, 153, 176, 190, 205]

for ind in temp_ind:
    fig = plt.figure()
    plt.imshow(pca_eta_ome_image_list[ind][m:n, k:l], cmap='bone', vmax=100)#, aspect='auto')


#%%
nrows = 4
fig, ax = plt.subplots(nrows=nrows, ncols=num_cmpts, figsize=(4.5*num_cmpts, 4.5*nrows))
cmap = 'viridis_r'#'RdYlBu'

pc_lim = [[-0.6, 0.6], [-0.45, 0.45], [-0.2, 0.2]]
pc_lim = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
stress_lim = [-400, 400]
strain_lim = [-0.6, 0.6]

# fit and transform PCA matrix
PCA_func = decomposition.PCA(n_components=num_cmpts)
#PCA_func = TSNE(n_components=num_cmpts,learning_rate=300,perplexity = 30,early_exaggeration = 12,init = 'random',  random_state=2019)
#PCA_func = MDS(n_components=num_cmpts, n_init=12, max_iter=10, metric=True, n_jobs=4, random_state=2019)
#PCA_func = Isomap(n_components=num_cmpts, n_jobs=10, n_neighbors=10)
#decomposition.PCA(n_components=num_cmpts)
#PCA_func = LocallyLinearEmbedding(n_neighbors=15, n_components=num_cmpts, reg=1e-3, tol=1e-6, max_iter=100)

# 50-210 cycle 1, 210-300 cycle 2 # 89 comp tip
#k = 77
#l = 223
# k = 0
# l = 76

k = 0 # 23 = elastic, 31 = macroscopic yield, 40 = post-yield full measurement, 52 = tension tip, 76 = unload
l = 206#52 #216 is end of first cycle
PCA_mat = bigPCA_mat[k:l, :]
plot_stress = np.array(stress_list)[k:l]
plot_strain = np.array(strain_list)[k:l] * 100
plot_scan = np.array(scan_list)[k:l]

transformer = Normalizer().fit(PCA_mat)
PCA_mat = transformer.transform(PCA_mat)
Z = PCA_func.fit_transform(PCA_mat)
var_ratio = PCA_func.explained_variance_ratio_
print(var_ratio)

Z[:, 0] = scale_0_1(Z[:, 0])
Z[:, 1] = scale_0_1(Z[:, 1])
Z[:, 2] = scale_0_1(Z[:, 2])

title = ""
for i in range(num_cmpts):
    pc1 = ax[0, i].scatter(plot_strain, plot_stress, c=Z[:, i], s=50, vmin=pc_lim[i][0], vmax=pc_lim[i][1], cmap=cmap)
    ax[0, i].set_xlabel("Macroscopic Strain (%)")
    ax[0, i].set_xlim(strain_lim)
    ax[0, i].set_ylim(stress_lim)
    cbar = fig.colorbar(pc1, ax=ax[0, i])
    cbar.set_label('PC %i' %(i+1), rotation=270, labelpad=10)

#%%

k = 52
l = 206
PCA_mat = bigPCA_mat[k:l, :]
plot_stress = np.array(stress_list)[k:l]
plot_strain = np.array(strain_list)[k:l] * 100
plot_scan = np.array(scan_list)[k:l]

transformer = Normalizer().fit(PCA_mat)
PCA_mat = transformer.transform(PCA_mat)
Z = PCA_func.fit_transform(PCA_mat)
#var_ratio = PCA_func.explained_variance_ratio_
#print(var_ratio)

Z[:, 0] = scale_0_1(Z[:, 0])
Z[:, 1] = scale_0_1(Z[:, 1])
Z[:, 2] = scale_0_1(Z[:, 2])

for i in range(num_cmpts):
    pc1 = ax[1, i].scatter(plot_strain, plot_stress, c=Z[:, i], s=50, vmin=pc_lim[i][0], vmax=pc_lim[i][1], cmap=cmap)
    ax[1, i].set_xlabel("Macroscopic Strain (%)")
    ax[1, i].set_xlim(strain_lim)
    ax[1, i].set_ylim(stress_lim)
    cbar = fig.colorbar(pc1, ax=ax[1, i])
    cbar.set_label('PC %i' %(i+1), rotation=270, labelpad=10)

k = 206
l = 292
PCA_mat = bigPCA_mat[k:l, :]
plot_stress = np.array(stress_list)[k:l]
plot_strain = np.array(strain_list)[k:l] * 100
plot_scan = np.array(scan_list)[k:l]

transformer = Normalizer().fit(PCA_mat)
PCA_mat = transformer.transform(PCA_mat)
#Z = PCA_func.fit_transform(PCA_mat)
Z = PCA_func.transform(PCA_mat)
#var_ratio = PCA_func.explained_variance_ratio_
#print(var_ratio)

Z[:, 0] = scale_0_1(Z[:, 0])
Z[:, 1] = scale_0_1(Z[:, 1])
Z[:, 2] = scale_0_1(Z[:, 2])

for i in range(num_cmpts):
    pc1 = ax[2, i].scatter(plot_strain, plot_stress, c=Z[:, i], s=50, vmin=pc_lim[i][0], vmax=pc_lim[i][1], cmap=cmap)
    ax[2, i].set_xlabel("Macroscopic Strain (%)")
    ax[2, i].set_xlim(strain_lim)
    ax[2, i].set_ylim(stress_lim)
    cbar = fig.colorbar(pc1, ax=ax[2, i])
    cbar.set_label('PC %i' %(i+1), rotation=270, labelpad=10)


k = 332
l = -1
# k = 206
# l = 292
PCA_mat = bigPCA_mat[k:l, :]
plot_stress = np.array(stress_list)[k:l]
plot_strain = np.array(strain_list)[k:l] * 100
plot_scan = np.array(scan_list)[k:l]

transformer = Normalizer().fit(PCA_mat)
PCA_mat = transformer.transform(PCA_mat)
#Z = PCA_func.fit_transform(PCA_mat)
Z = PCA_func.transform(PCA_mat)
var_ratio = PCA_func.explained_variance_ratio_
print(var_ratio)

Z[:, 0] = scale_0_1(Z[:, 0])
Z[:, 1] = scale_0_1(Z[:, 1])
Z[:, 2] = scale_0_1(Z[:, 2])

for i in range(num_cmpts):
    pc1 = ax[3, i].scatter(plot_strain, plot_stress, c=Z[:, i], s=50, vmin=pc_lim[i][0], vmax=pc_lim[i][1], cmap=cmap)
    ax[3, i].set_xlabel("Macroscopic Strain (%)")
    ax[3, i].set_xlim(strain_lim)
    ax[3, i].set_ylim(stress_lim)
    cbar = fig.colorbar(pc1, ax=ax[3, i])
    cbar.set_label('PC %i' %(i+1), rotation=270, labelpad=10)

plt.show()

#%%

# cycle 1
k = 52 # 23 = elastic, 31 = macroscopic yield, 40 = post-yield full measurement, 52 = tension tip, 76 = unload
l = 206#52 #216 is end of first cycle
PCA_mat = bigPCA_mat[k:l, :]
c1_plot_stress = np.array(stress_list)[k:l]
c1_plot_strain = np.array(strain_list)[k:l] * 100
c1_plot_scan = np.array(scan_list)[k:l]
c1_fits = all_fits[k:l, :]

transformer = Normalizer().fit(PCA_mat)
PCA_mat = transformer.transform(PCA_mat)
c1_Z = PCA_func.fit_transform(PCA_mat)
var_ratio = PCA_func.explained_variance_ratio_
print(var_ratio)

# c1_Z[:, 0] = scale_0_1(c1_Z[:, 0])
# c1_Z[:, 1] = scale_0_1(c1_Z[:, 1])
# c1_Z[:, 2] = scale_0_1(c1_Z[:, 2])


# cycle 2 fit
k = 206
l = 292
PCA_mat = bigPCA_mat[k:l, :]
c2_plot_stress = np.array(stress_list)[k:l]
c2_plot_strain = np.array(strain_list)[k:l] * 100
c2_plot_scan = np.array(scan_list)[k:l]
c2_fits = all_fits[k:l, :]

transformer = Normalizer().fit(PCA_mat)
PCA_mat = transformer.transform(PCA_mat)
c2_Z = PCA_func.transform(PCA_mat)
var_ratio = PCA_func.explained_variance_ratio_
print(var_ratio)

# c2_Z[:, 0] = scale_0_1(c2_Z[:, 0])
# c2_Z[:, 1] = scale_0_1(c2_Z[:, 1])
# c2_Z[:, 2] = scale_0_1(c2_Z[:, 2])


# cycle 64 fit
k = 332 # 23 = elastic, 31 = macroscopic yield, 40 = post-yield full measurement, 52 = tension tip, 76 = unload
l = -1 #52 #216 is end of first cycle
PCA_mat = bigPCA_mat[k:l, :]
c64_plot_stress = np.array(stress_list)[k:l]
c64_plot_strain = np.array(strain_list)[k:l] * 100
c64_plot_scan = np.array(scan_list)[k:l]
c64_fits = all_fits[k:l, :]

transformer = Normalizer().fit(PCA_mat)
PCA_mat = transformer.transform(PCA_mat)
c64_Z = PCA_func.transform(PCA_mat)
var_ratio = PCA_func.explained_variance_ratio_
print(var_ratio)

# c64_Z[:, 0] = scale_0_1(c64_Z[:, 0])
# c64_Z[:, 1] = scale_0_1(c64_Z[:, 1])
# c64_Z[:, 2] = scale_0_1(c64_Z[:, 2])




#%%
k = 0 # 23 = elastic, 31 = macroscopic yield, 40 = post-yield full measurement, 52 = tension tip, 76 = unload
l = 76
PCA_mat = bigPCA_mat[k:l, :]
plot_stress = np.array(stress_list)[k:l]
plot_strain = np.array(strain_list)[k:l] * 100
plot_scan = np.array(scan_list)[k:l]

# fit and transform PCA matrix
transformer = Normalizer().fit(PCA_mat)
PCA_mat = transformer.transform(PCA_mat)
Z = PCA_func.fit_transform(PCA_mat)
Z[:, 1] = -Z[:, 1]
#Z[:, 0] = -Z[:, 0]
#Z[:, 2] = -Z[:, 2]

#Z = Zi[k:l]

Z[:, 0] = scale_0_1(Z[:, 0])
Z[:, 1] = scale_0_1(Z[:, 1])
Z[:, 2] = scale_0_1(Z[:, 2])


for i in range(num_cmpts):
    pc1 = ax[3, i].scatter(plot_strain, plot_stress, c=Z[:, i], s=50, vmin=pc_lim[i][0], vmax=pc_lim[i][1], cmap=cmap)
    ax[3, i].set_xlabel("Macroscopic Strain (%)")
    # ax[3, i].set_xlim(strain_lim)
    # ax[3, i].set_ylim(stress_lim)
    cbar = fig.colorbar(pc1, ax=ax[3, i])
    cbar.set_label('PC %i' %(i+1), rotation=270, labelpad=10)


fig.suptitle(title, fontsize=16)
ax[0, 0].set_ylabel("Macroscopic Stress (MPa)")
ax[1, 0].set_ylabel("Macroscopic Stress (MPa)")
ax[2, 0].set_ylabel("Macroscopic Stress (MPa)")
ax[3, 0].set_ylabel("Macroscopic Stress (MPa)")
ax[4, 0].set_ylabel("Macroscopic Stress (MPa)")
ax[5, 0].set_ylabel("Macroscopic Stress (MPa)")




# k = 0 # 23 = elastic, 31 = macroscopic yield, 40 = post-yield full measurement, 52 = tension tip, 76 = unload
# l = 76
PCA_mat = PCA_mat[k:l, :]
plot_stress = np.array(stress_list)[k:l]
plot_strain = np.array(strain_list)[k:l] * 100
plot_scan = np.array(scan_list)[k:l]
plot_all_stats = np.copy(np.array(all_stats)[k:l, :, :])

r=0
tth_range = plane_data.getTThRanges() * 180 / np.pi
plot_all_stats[:, r, 6] = tth_pix_size * plot_all_stats[:, r, 6] + tth_range[0, 0]
plot_all_stats[:, r, 8] = tth_pix_size * plot_all_stats[:, r, 8]
plot_all_stats[:, r, 5] = eta_pix_size * plot_all_stats[:, r, 5] - 65
plot_all_stats[:, r, 7] = eta_pix_size * plot_all_stats[:, r, 7]

pc1 = ax[4, 0].scatter(plot_strain, plot_stress, c=((plot_all_stats[:, r, 6])), 
                       s=50, cmap=cmap)
pc2 = ax[4, 1].scatter(plot_strain, plot_stress, c=plot_all_stats[:, r, 8], 
                       s=50, cmap=cmap)
pc3 = ax[5, 0].scatter(plot_strain, plot_stress, c=-plot_all_stats[:, r, 5], 
                       s=50, cmap=cmap)
pc4 = ax[5, 1].scatter(plot_strain, plot_stress, c=plot_all_stats[:, r, 7], 
                       s=50, cmap=cmap)

ax[4, 0].set_xlabel("Macroscopic Strain (%)")
ax[4, 1].set_xlabel("Macroscopic Strain (%)")
ax[5, 0].set_xlabel("Macroscopic Strain (%)")
ax[5, 1].set_xlabel("Macroscopic Strain (%)")
cbar = fig.colorbar(pc1, ax=ax[4, 0])
cbar.set_label('2theta Shift', rotation=270, labelpad=10)
cbar.set_ticks(np.linspace(5.555, 5.564, 10))
cbar.set_ticklabels(np.linspace(5.555, 5.564, 10))

cbar = fig.colorbar(pc2, ax=ax[4, 1])
cbar.set_label('2theta Spread', rotation=270, labelpad=10)
cbar = fig.colorbar(pc3, ax=ax[5, 0])
cbar.set_label('eta Shift', rotation=270, labelpad=10)
cbar.set_ticks(np.linspace(3.2, 3.4, 5))
cbar.set_ticklabels(np.linspace(3.2, 3.4, 5))

cbar = fig.colorbar(pc4, ax=ax[5, 1])
cbar.set_label('eta Spread', rotation=270, labelpad=10)


# ax[4, 0].set_xlim(strain_lim)
# ax[4, 0].set_ylim(stress_lim)
# ax[4, 1].set_xlim(strain_lim)
# ax[4, 1].set_ylim(stress_lim)
# ax[5, 0].set_xlim(strain_lim)
# ax[5, 0].set_ylim(stress_lim)
# ax[5, 1].set_xlim(strain_lim)
# ax[5, 1].set_ylim(stress_lim)

fig.subplots_adjust(top=0.95,
bottom=0.1,
left=0.11,
right=0.9,
hspace=0.4,
wspace=0.5)


fig.savefig("PCA_at_diff_steps_with_spot_stats_values_0_1_cyclic.png")


fig, ax = plt.subplots(nrows=3, ncols=num_cmpts, figsize=(4.5*num_cmpts, 12))

ax[0, 0].plot(plot_strain, scale_0_1(-Z[:, 0]))
ax[0, 1].plot(plot_strain, scale_0_1(Z[:, 1]))


r=0
ax[1, 0].plot(plot_strain, (scale_0_1(plot_all_stats[:, r, 6])))
ax[1, 1].plot(plot_strain, (scale_0_1(plot_all_stats[:, r, 8])))
ax[2, 0].plot(plot_strain, (scale_0_1(plot_all_stats[:, r, 5])))
ax[2, 1].plot(plot_strain, (scale_0_1(plot_all_stats[:, r, 7])))



plt.show()

#%%
#fig.savefig(title + ".png")

#[eta_mean, tth_mean, eta_spread, tth_spread, num_spots, max_eta_pos, max_tth_pos, max_eta_std, max_tth_std]
for r in range(num_rings):
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(18, 5))
    
    # total
    pc1 = ax[0].scatter(plot_strain, plot_stress, c=plot_all_stats[:, r, 1], s=50, cmap='RdYlBu')
    pc2 = ax[1].scatter(plot_strain, plot_stress, c=plot_all_stats[:, r, 3], s=50, cmap='RdYlBu')
    pc3 = ax[2].scatter(plot_strain, plot_stress, c=plot_all_stats[:, r, 0], s=50, cmap='RdYlBu')
    pc4 = ax[3].scatter(plot_strain, plot_stress, c=plot_all_stats[:, r, 2], s=50, cmap='RdYlBu')
    
    # one spot
    pc1 = ax[0].scatter(plot_strain, plot_stress, c=plot_all_stats[:, r, 6], s=50, cmap='RdYlBu')
    pc2 = ax[1].scatter(plot_strain, plot_stress, c=plot_all_stats[:, r, 8], s=50, cmap='RdYlBu')
    pc3 = ax[2].scatter(plot_strain, plot_stress, c=plot_all_stats[:, r, 5], s=50, cmap='RdYlBu')
    pc4 = ax[3].scatter(plot_strain, plot_stress, c=plot_all_stats[:, r, 7], s=50, cmap='RdYlBu')
    
    fig.suptitle('Ring %i' %(r+1), fontsize=16)
    
    ax[0].set_xlabel("Strain")
    ax[1].set_xlabel("Strain")
    ax[2].set_xlabel("Strain")
    ax[3].set_xlabel("Strain")
    ax[0].set_ylabel("Stress")
    cbar = fig.colorbar(pc1, ax=ax[0])
    cbar.set_label('2theta Shift', rotation=270, labelpad=10)
    cbar = fig.colorbar(pc2, ax=ax[1])
    cbar.set_label('2theta Spread', rotation=270, labelpad=10)
    cbar = fig.colorbar(pc3, ax=ax[2])
    cbar.set_label('eta Shift', rotation=270, labelpad=10)
    cbar = fig.colorbar(pc4, ax=ax[3])
    cbar.set_label('eta Spread', rotation=270, labelpad=10)
    
    fig.subplots_adjust(top=0.88,
        bottom=0.2,
        left=0.11,
        right=0.9,
        hspace=0.4,
        wspace=0.5)

fig, ax = plt.subplots(nrows=1, ncols=7, figsize=(18, 5))
ax[0].scatter(plot_scan, Z[:, 0])
ax[1].scatter(plot_scan, Z[:, 1])
ax[2].scatter(plot_scan, plot_all_stats[:, r, 6])
ax[3].scatter(plot_scan, plot_all_stats[:, r, 8])
ax[4].scatter(plot_scan, plot_all_stats[:, r, 5])
ax[5].scatter(plot_scan, plot_all_stats[:, r, 7])
ax[6].scatter(plot_scan, plot_stress)

plt.show()


#%%


fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
ax[0, 0].plot(-scale_0_1(Z[:, 0]))
ax[0, 1].plot(scale_0_1(Z[:, 1]))
ax[0, 2].plot(scale_0_1(Z[:, 2]))


ind_stats = [6, 8, 9, 5, 7, 9]
for r in range(num_rings):
    for j, ind in enumerate(ind_stats):
        stat_to_plot = scale_0_1(plot_all_stats[:, r, ind])
        ax[int(j / 3 + 1), int(j % 3)].plot(stat_to_plot / stat_to_plot.max())


plt.show()

#%%



fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
ax[0].plot(scale_0_1(Z[:, 0]))
ax[1].plot(scale_0_1(Z[:, 1]))
ax[2].plot(scale_0_1(-Z[:, 2]))

ind_stats = [5, 6]
for j, ind in enumerate(ind_stats):
    stat_to_plot = (plot_all_stats[:, r, ind])
    ax[0].plot(scale_0_1(stat_to_plot))
 
ind_stats = [5, 6]
for j, ind in enumerate(ind_stats):
    stat_to_plot = (plot_all_stats[:, r, ind])
    ax[1].plot(scale_0_1(stat_to_plot))   
 
ind_stats = [7, 8]
for j, ind in enumerate(ind_stats):
    stat_to_plot = (plot_all_stats[:, r, ind])
    ax[2].plot(scale_0_1(stat_to_plot))

plt.show()

#%%


ind_stats = [6, 8, 5, 7]
#ind_stats = [1, 3, 0, 2]

nrow = len(ind_stats) + 1
ncol = num_cmpts
fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(12, 12))

temp_s = [0]
for j, ind in enumerate(ind_stats):
    stat_to_plot = plot_all_stats[:, temp_s, ind]
    stat_to_plot.shape = [stat_to_plot.shape[0], stat_to_plot.shape[1]]
    stat_to_plot = (stat_to_plot - np.min(stat_to_plot, axis=0))
    stat_to_plot = stat_to_plot / np.max(stat_to_plot, axis=0)
    
    for i in range(3):
        c_stat = np.linalg.lstsq(stat_to_plot, Z[:, i], rcond=-1)
        
        ax[j, i].plot(stat_to_plot @ c_stat[0])
        ax[j, i].plot(Z[:, i])


stat_to_plot = plot_all_stats[:, :, ind_stats]
stat_to_plot = np.reshape(stat_to_plot, [stat_to_plot.shape[0], stat_to_plot.shape[1]*stat_to_plot.shape[2]])
stat_to_plot = (stat_to_plot - np.min(stat_to_plot, axis=0))
stat_to_plot = stat_to_plot / np.max(stat_to_plot, axis=0)
for i in range(3):
    c_stat = np.linalg.lstsq(stat_to_plot, Z[:, i], rcond=-1)
    
    ax[-1, i].plot(stat_to_plot @ c_stat[0])
    ax[-1, i].plot(Z[:, i])
    
    print(c_stat[0])

plt.show()


#%%

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))

temp_shift_tth = plot_shift_tth_list / np.max(plot_shift_tth_list, axis=0)
temp_spread_tth = plot_spread_tth_list / np.max(plot_spread_tth_list, axis=0)
temp_shift_eta = plot_shift_eta_list / np.max(plot_shift_eta_list, axis=0)
temp_spread_eta = plot_spread_eta_list / np.max(plot_spread_eta_list, axis=0)

# PC's
for i in range(3):
    # rings
    for j in range(3):
        ring_mat = np.vstack([temp_shift_tth[:, j], temp_spread_tth[:, j], 
                              temp_shift_eta[:, j], temp_spread_eta[:, j]]).T
        c_ring = np.linalg.lstsq(ring_mat, Z[:, i], rcond=-1)
        
        ax[j, i].plot(Z[:, i])
        ax[j, i].plot(ring_mat @ c_ring[0])
    
        print(c_ring[0])
plt.show()

#%%
fig = plt.figure()
ind = np.arange(bigPCA_mat.shape[0])
#ind = np.delete(ind, [23, 157])
plt.imshow(bigPCA_mat[ind, :], aspect='auto', vmax=np.mean(PCA_mat))
plt.show()

#%%
#fig = plt.figure()
#plt.imshow(np.atleast_2d(pca_image_list[50] - pca_image_list[89]), vmax=1e2, vmin=-1e2, aspect='auto')
#plt.imshow((pca_image_list[50] - pca_image_list[89]).reshape(eta_tth_img.data.shape), vmax=1e2, vmin=-1e2, aspect='auto')

fig = plt.figure()
plt.imshow((pca_image_list[90]).reshape(eta_tth_img.data.shape), vmax=1e2, aspect='auto')
plt.xlabel('2theta')
plt.ylabel('eta')
plt.show()


#%%

plane_data = load_pdata_hexrd3(os.path.join(analysis_path, material_fname), 'in718')
plane_data.set_exclusions([1, 2, 3, 4, 5])

my_polar = xrdutil.PolarView(plane_data, instr,
                 eta_min=-10., eta_max=10.,
                 pixel_size=(0.005, 0.05))

sc = 81
ff1_img = imageseries.open(raw_data_dir_template+'ss718-1_%i_%s_0000%i-cachefile.npz'%(sc, 'ff1', sc-6), raw_fmt, path=raw_path)
ff2_img = imageseries.open(raw_data_dir_template+'ss718-1_%i_%s_0000%i-cachefile.npz'%(sc, 'ff2', sc-6), raw_fmt, path=raw_path)

ff1_img1_cp = np.copy(ff1_img[0])
#ff1_img_cp[ff1_img_cp < 350] = 0


sc = 92
ff1_img = imageseries.open(raw_data_dir_template+'ss718-1_%i_%s_0000%i-cachefile.npz'%(sc, 'ff1', sc-7), raw_fmt, path=raw_path)
ff2_img = imageseries.open(raw_data_dir_template+'ss718-1_%i_%s_0000%i-cachefile.npz'%(sc, 'ff2', sc-7), raw_fmt, path=raw_path)

ff1_img2_cp = np.copy(ff1_img[0])
#ff1_img2_cp[ff1_img2_cp < 400] = 0
#ff1_img2_cp = ff1_img2_cp * 2

ratio_ind =  np.where((ff1_img1_cp > 1000) & (ff1_img1_cp < 16000) & (ff1_img2_cp > 1000) & (ff1_img2_cp < 16000))
ratio = np.divide(ff1_img1_cp[ratio_ind], ff1_img2_cp[ratio_ind])
old2new_ratio = ratio.mean()

ff1_img1_cp = ff1_img1_cp / old2new_ratio
ff1_img1_cp[ff1_img1_cp > 16023] = 16023

ff1_img1_cp[ff1_img1_cp < 800] = 0
ff1_img2_cp[ff1_img2_cp < 800] = 0



img_dict = {'ff1':ff1_img1_cp, 'ff2':ff2_img[0]}
eta_tth_img1 = my_polar.warp_image(img_dict, pad_with_nans=True,
                   do_interpolation=False)
img_dict = {'ff1':ff1_img2_cp, 'ff2':ff2_img[0]}
eta_tth_img2 = my_polar.warp_image(img_dict, pad_with_nans=True,
                   do_interpolation=False)

img1 = np.copy(eta_tth_img1.data)
img2 = np.copy(eta_tth_img2.data)



diff = ff1_img1_cp - ff1_img2_cp

fig = plt.figure()
plt.imshow(diff[1800:2150, 1045:1085], aspect='auto', vmin=-100, vmax=100)
#plt.imshow(img1 - img2, aspect='auto', vmin=-100, vmax=100)

fig = plt.figure()
plt.imshow(ff1_img1_cp[1800:2150, 1045:1085], aspect='auto', vmax=12000)

fig = plt.figure()
plt.imshow(ff1_img2_cp[1800:2150, 1045:1085], aspect='auto', vmax=12000)

print(ff1_img1_cp[1800:2150, 1045:1085].mean(), ff1_img1_cp[1800:2150, 1045:1085].max())
print(ff1_img2_cp[1800:2150, 1045:1085].mean(), ff1_img2_cp[1800:2150, 1045:1085].max())

print(ff1_img1_cp.mean(), ff1_img1_cp[1800:2150, 1045:1085].max())
print(ff1_img2_cp.mean(), ff1_img2_cp[1800:2150, 1045:1085].max())

plt.show()


#%%
sc5_5 = np.loadtxt('/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/scan_summary/summary_%i.dat' %(68+1))
sc4_5 = np.loadtxt('/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/scan_summary/summary_%i.dat' %(90+1))
sc4 = np.loadtxt('/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/scan_summary/summary_%i.dat' %(111+1))
sc3_5 = np.loadtxt('/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/scan_summary/summary_%i.dat' %(136+1))
sc3_25 = np.loadtxt('/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/scan_summary/summary_%i.dat' %(632+1))

# index = 9 for ic3
print(np.mean(sc5_5[:, 9]))
print(np.mean(sc4_5[:, 9]))
print(np.mean(sc4[:, 9]))
print(np.mean(sc3_5[:, 9]))
print(np.mean(sc3_25[:, 9]))



#%%

def simple_anim_pca(num_points=20, save_gif=False, name_gif='temp.gif', time_per_frame=1):
    
    def y(x, m, b):
        return m*x + b
    
    X = np.linspace(-1, 1, num_points)
    Y = [y(x, 1, 0) + np.random.uniform(low=-0.3, high=0.3) for x in X]
    XY = np.vstack([X, Y]).T
    print(XY.shape)
    numframes = num_points

    # plot entire volume fundamental region
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    axs.set_xlim([-1.5, 1.5])
    axs.set_ylim([-1.5, 1.5])
    
    num_cmpts = 2
    PCA_func = decomposition.PCA(n_components=num_cmpts)
    PCA_func.fit(XY)
    Z = PCA_func.transform(XY)
    
    var_ratio = PCA_func.explained_variance_ratio_
    PCA_dir = PCA_func.components_
    
    print(var_ratio)
    print(PCA_dir)
    print(Z)
    
    X_mean = (XY - np.mean(XY, axis=0))
    U, S, V = np.linalg.svd(X_mean, full_matrices=False)
    print(U @ np.diag(S))
    print(V)
    
    explained_variance_ = (S ** 2) / num_points
    explained_variance_ratio_ = (explained_variance_ /
                                 explained_variance_.sum())
    print(explained_variance_ratio_)
    
    
    # do function animation
    ani = animation.FuncAnimation(fig, simple_update_pca, interval=time_per_frame*1000,
                                  frames=numframes, 
                                  fargs=(XY, axs, fig))
    plt.show()
    if save_gif:
        ani.save(name_gif, writer='imagemagick', dpi=200)

def simple_update_pca(i, XY, axs, fig):
    axs.cla()
    
    tempXY = XY[:i+1, :]
    
    # plot images
    axs.set_xlim([-1.5, 1.5])
    axs.set_ylim([-1.5, 1.5])
    axs.scatter(tempXY[:, 0], tempXY[:, 1])
    
    # fit and transform PCA matrix
    if i > 1:
        num_cmpts = 2
        PCA_func = decomposition.PCA(n_components=num_cmpts)
        PCA_func.fit(tempXY)
        Z = PCA_func.transform(tempXY)
        
        var_ratio = PCA_func.explained_variance_ratio_
        PCA_dir = PCA_func.components_
        for j in range(num_cmpts):
            if PCA_dir[j, 0] < 0:
                PCA_dir[j, :] = PCA_dir[j, :] * -1
            axs.plot([0 + np.mean(tempXY[:, 0]), 0.3 * PCA_dir[j, 0] + np.mean(tempXY[:, 0])],
                     [0 + np.mean(tempXY[:, 1]), 0.3 * PCA_dir[j, 1] + np.mean(tempXY[:, 1])])
        axs.legend(['PC1 Direction', 'PC2 Direction', 'Observations'], loc='lower right')
    else:
        axs.legend(['Observations'], loc='lower right')
        
    

    return axs

simple_anim_pca(save_gif=True, name_gif='pca_temp.gif')
 

#%%

import time

start_size = 1
n_feats = 50000
num_cmpts = 1
XY = np.random.uniform(low=-1.0, high=1.0, size=(start_size, n_feats))

start = time.time()

PCA_func = decomposition.PCA(n_components=num_cmpts)
PCA_func.fit(XY)
Z = PCA_func.transform(XY)

end = time.time()
print(end - start)

start = time.time()

pca = decomposition.IncrementalPCA(n_components=num_cmpts)
pca.fit(XY)
Z_new = pca.transform(XY)

end = time.time()
print(end - start)

for i in np.arange(1, 20):
    new_XY = np.random.uniform(low=-1.0, high=1.0, size=(10, n_feats))
    XY = np.vstack([XY, new_XY])
    
    start = time.time()
    
    PCA_func.fit(XY)
    Z = PCA_func.transform(XY)
    
    end = time.time()
    print(end - start)
    
    start = time.time()
    
    pca.partial_fit(new_XY)
    Z_new = pca.transform(XY)
    
    end = time.time()
    print(end - start)

print(PCA_func.explained_variance_ratio_)
print(Z[:10, :])

print(pca.explained_variance_ratio_)
print(Z_new[:10, :])


#%% create a diffraction image synthetically

deg2rad = np.pi / 180.0
rad2deg = 180.0 / np.pi

delta_tth = 0.01 * deg2rad # degrees to radians (smallest resolution ~0.0075 deg)
mask_tth = np.array([5.57, 6.43, 9.1]) * deg2rad
num_rings = 1

max_eta = 15 * deg2rad
min_eta = 5 * deg2rad

# generate random points on the rings
num_points = 3
rand_polar_points = np.zeros([num_rings*num_points, 2]) #tth, eta

for i in range(num_rings):
    rand_tth = np.random.uniform(low=mask_tth[i] - delta_tth, high=mask_tth[i] + delta_tth, size=(num_points,))
    rand_eta = np.random.uniform(low=min_eta, high=max_eta, size=(num_points,))
    
    rand_polar_points[int(i*num_points):int((i+1)*num_points), :] = np.vstack([rand_tth, rand_eta]).T
    
rand_pix_points = panel.cartToPixel(panel.angles_to_cart(rand_polar_points)).astype(int)

my_polar_ff1 = xrdutil.PolarView(plane_data, instr,
                 eta_min=-0., eta_max=20.,
                 pixel_size=(0.005, 0.05))

temp_img = np.zeros(panel_tth.shape)
temp_img[rand_pix_points[:, 0], rand_pix_points[:, 1]] = 1000
img_dict = {'ff1':temp_img, 'ff2':temp_img}
temp_polar_img = my_polar_ff1.warp_image(img_dict, pad_with_nans=True,
                   do_interpolation=False)

rand_polar_pix_points = np.array(np.where(temp_polar_img > 0)).T
print(rand_pix_points)
print(rand_polar_pix_points)

plt.imshow(temp_polar_img, aspect='auto')

#%%

temp_legend = []
temp_img_list = []
num_steps = 1
for i_steps in range(num_steps):
    # adjust all points
    total_img = np.zeros(temp_polar_img.shape)
    for j_points in range(2):
        temp_pos = np.copy(rand_polar_pix_points[j_points, :])
        if j_points == 0:
            # adjust position
            temp_pos[0] = int(temp_pos[0] + i_steps * 0)
            temp_pos[1] = int(temp_pos[1] + i_steps * 0)
            
            # adjust intensity
            temp_img = np.zeros(temp_polar_img.shape)
            temp_img[temp_pos[0], temp_pos[1]] = 300 + i_steps * 0
            
            # adjust spread
            temp_img = ndimage.gaussian_filter(temp_img, (5 + i_steps*0.0, 0.5))
            
        elif j_points == 1:
            # adjust position
            temp_pos[0] = int(temp_pos[0] + i_steps * 0)
            temp_pos[1] = int(temp_pos[1] + i_steps * 0)
            
            # adjust intensity
            temp_img = np.zeros(temp_polar_img.shape)
            temp_img[temp_pos[0], temp_pos[1]] = 2000
            
            # adjust spread
            temp_img = ndimage.gaussian_filter(temp_img, (2, 0.5))
        
        # add to total image
        total_img = total_img + temp_img
    temp_img_list.append(total_img)
   


    
#%%
num_cmpts = 2
num_images = len(temp_img_list)
num_pixels = temp_img_list[0].size

# assemble PCA matrix
PCA_mat = np.zeros([num_images, num_pixels])
PCA_cmpts = np.zeros([num_images, num_cmpts])
for i, image in enumerate(temp_img_list):
    PCA_mat[i, :] = image.ravel(order='C') # C = Row, F = Column

fig = plt.figure()
temp_leg = []

# fit and transform PCA matrix
transformer = Normalizer().fit(PCA_mat)
PCA_mat = transformer.transform(PCA_mat)
PCA_func = decomposition.PCA(n_components=num_cmpts)
PCA_func.fit(PCA_mat)
Z = PCA_func.transform(PCA_mat)
plt.plot(Z[:, 0] / np.max(Z[:, 0]))
plt.plot(Z[:, 1] / np.max(Z[:, 1]))
temp_leg.append("PCA 1")
temp_leg.append("PCA 2")


test_range = num_steps
if test_range > 10:
    test_range = 10
for i in np.arange(2, test_range):
    embedding = LocallyLinearEmbedding(n_components=num_cmpts, n_neighbors=i)
    Z_lle = embedding.fit_transform(PCA_mat)
    plt.plot(Z_lle[:, 0])
    temp_leg.append("LLE %i" %i)

plt.legend(temp_leg)

# for i in range(num_images):
#     fig = plt.figure()
#     plt.imshow(temp_img_list[i], aspect='auto')


plt.show()   




