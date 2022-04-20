#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 11:26:51 2021

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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn.preprocessing import Normalizer
from sklearn.manifold import LocallyLinearEmbedding, Isomap, MDS, TSNE

from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit

import time

from scipy import ndimage

from hexrd import config
from hexrd import xrdutil
from hexrd import instrument
from hexrd import imageutil
from hexrd import imageseries
from hexrd import material

import pandas as pd
import seaborn as sns


import sys 
IMPORT_HEXRD_SCRIPT_DIR = '/home/djs522/bin/view_snapshot_series'
sys.path.insert(0, IMPORT_HEXRD_SCRIPT_DIR)
import preprocess_dexela_h5 as ppdex
import pp_dexela

IMPORT_HEXRD_SCRIPT_DIR = '/home/djs522/djs522/hexrd_utils'
sys.path.insert(0, IMPORT_HEXRD_SCRIPT_DIR)
import post_process_stress as pp_stress
import post_process_goe as pp_GOE

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

def scale_0_1(data, n1_p1=True):
    if n1_p1:
        return 2*(data - data.min()) / (data.max() - data.min()) - 1.0
    else:
        return (data - data.min()) / (data.max() - data.min())

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
    #cake = cake / cake.max()
    
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
    
    first_img_base = 'ss718-1_81_%s_000075-cachefile.npz'
    deform_img_base = 'ss718-1_92_%s_000085-cachefile.npz'
    
    dark_image_base = 'ss718-1_7_%s_000019-cachefile.npz'
    
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
    
    scan_step = 1
    inten_max = 500
    inten_min = 50
    time_per_frame = 0.5 #seconds
    
    start_dic = 3216 
    end_dic = 3300#3407
    
    bad_scan_list = [] #dp

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

# if plotting for debuging
do_plot = True

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

np.save(save_name, new_ct2dic)


if ss718: # grab cycle 0 , 1, 2, 64
    ind = np.where(((new_ct2dic[:, 0] >= 2013) & (new_ct2dic[:, 0] <= 2375)) | 
                   ((new_ct2dic[:, 0] >= 2894) & (new_ct2dic[:, 0] <= 2953)))
else:
    ind = np.arange(new_ct2dic.shape[0])
good_dic_list = new_ct2dic[ind, 0]



#%% Gerenate dark images

calc = False
if calc:
    raw_fmt_h5 = 'hdf5'
    raw_path_h5 = '/imageseries'
    dark_series_ff1 = imageseries.open('/home/djs522/bin/ff/ff1_001337.h5', raw_fmt_h5, path=raw_path_h5)
    dark_series_ff2 = imageseries.open('/home/djs522/bin/ff/ff2_001337.h5', raw_fmt_h5, path=raw_path_h5)
    
    print("Images loaded")
    
    num_dark = 600
    dark_ff1 = imageseries.stats.median(dark_series_ff1, nframes=num_dark)
    dark_ff2 = imageseries.stats.median(dark_series_ff2, nframes=num_dark)
    
    
    temp_dark_ff1 = dark_ff1[:, ::-1]
    temp_dark_ff2 = dark_ff2[::-1, :]
    temp_dark_ff1 = np.insert(temp_dark_ff1, 1, 0, axis=0)
    temp_dark_ff1 = np.insert(temp_dark_ff1, 1, 0, axis=1)
    temp_dark_ff2 = np.insert(temp_dark_ff2, 1, 0, axis=0)
    temp_dark_ff2 = np.insert(temp_dark_ff2, 1, 0, axis=1)
    temp_dark = np.hstack([temp_dark_ff1, temp_dark_ff2])
    
    
    np.save(base_path+'ff1_dark.npy', temp_dark_ff1)
    np.save(base_path+'ff2_dark.npy', temp_dark_ff2)
else:
    temp_dark_ff1 = np.load(base_path+'ff1_dark.npy')
    temp_dark_ff2 = np.load(base_path+'ff2_dark.npy')
    temp_dark = np.hstack([temp_dark_ff1, temp_dark_ff2])

if do_plot:
    fig = plt.figure()
    plt.imshow(temp_dark, aspect='auto', vmax=500)
    plt.show()


#%% Define polar mapping transformation for both detectors

raw_thresh = 5
eta_pix_size = 0.05
tth_pix_size = 0.005

plane_data = load_pdata_hexrd3(os.path.join(analysis_path, material_fname), 'in718')
plane_data.set_exclusions([4, 5])

my_polar_ff1 = xrdutil.PolarView(plane_data, instr,
                 eta_min=-60., eta_max=60.,
                 pixel_size=(tth_pix_size, eta_pix_size))

my_polar_ff2 = xrdutil.PolarView(plane_data, instr,
                 eta_min=120., eta_max=240.,
                 pixel_size=(tth_pix_size, eta_pix_size))

if do_plot:
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
    
    fig = plt.figure()
    temp_image = np.vstack([eta_tth_ff1_img1, eta_tth_ff2_img1])
    plt.imshow(temp_image, aspect='auto', vmax=100)
    plt.show()


#%% Perform data collecting from raw ff-HEDM images

raw_thresh = 5 #300

eta_ome_scan_image_list = []
raw_scan_image_list = []
scan_list = []
stress_list = []
strain_list = []

for i, row in enumerate(new_ct2dic):
    if (i % scan_step) == 0:
        dic_num = row[0]
        scan_num = row[1]
        scan_stress = row[2]
        scan_strain = row[3]
        
        if (dic_num in good_dic_list) and (scan_num not in bad_scan_list):
            
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
                        
                        
pca_eta_ome_image_list = eta_ome_scan_image_list
pca_raw_image_list = raw_scan_image_list


#%% Animate frames with stress strain curve

'''
ss_list = np.vstack([stress_list, strain_list]).T
animate_frames_stress_strain_with_spots(scan_list, scan_image_list, ss_list, 
                                            max_cmap=1000, save_gif=True, 
                                            name_gif=temp_dir+'temp5.gif', time_per_frame=0.1)
'''


#%% initialize PCA and matrix variables

pca_image_list = pca_eta_ome_image_list # pca_raw_image_list, pca_eta_ome_image_list


rings = 3
k = 0
l = 51 * rings
m = 0
n = 2400

# k = 4000
# l = 4250
# m = 1500
# n = 2400

#ind = np.hstack([np.arange(900,1500), np.arange(3300, 3900)])
#ind = np.hstack([np.arange(750,1650), np.arange(3150, 4050)])
#ind = np.hstack([np.arange(600,1800), np.arange(3000, 4200)])
#ind = np.arange(2400)

num_cmpts = 3
num_images = len(pca_image_list)
#num_pixels = pca_image_list[0][m:n, k:l][ind, :].size
num_pixels = pca_image_list[0][m:n, k:l].size

# assemble PCA matri

bigPCA_mat = np.zeros([num_images, num_pixels])
PCA_cmpts = np.zeros([num_images, num_cmpts])
comb_all_fits = np.zeros([num_images, 3 * rings])
for i, image in enumerate(pca_image_list):
    temp_img = image[m:n, k:l]
    #temp_img = temp_img / temp_img.max()
    #temp_img = temp_img[ind, :]
    #for j in range(rings):
    #    temp_img[:, j*51:(j+1)*51] = temp_img[:, j*51:(j+1)*51] / np.max(temp_img[:, j*51:(j+1)*51])
    bigPCA_mat[i, :] = temp_img.ravel(order='C')  # C = Row, F = Column
    fits = fit_cake(temp_img, rings=rings, plot=False, gauss_fit=False)
    comb_all_fits[i, :] = fits.ravel()


all_stress = np.array(stress_list)
all_strain = np.array(strain_list) * 100
all_scan = np.array(scan_list)
all_fits = np.array(comb_all_fits)

macro_youngs_mod = 190e3
all_e_strain = all_stress / macro_youngs_mod * 100
all_p_strain = all_strain - all_e_strain
all_ISE = all_stress * all_strain # instantaneous strain energy



lw_big = 4
lw_small = 2.5
colors = ['#D81B60', '#1E88E5', '#FFC107', '#004D40']
markers = ['.', 'x', 's']
lines = ['-', '-.', ':', '--']

label_long = {'macro_stress':'Macroscopic Stress in LD (MPa)',
              'macro_strain':'Macroscopic Strain in LD (%)',
              'macro_p_strain':'Macroscopic Plastic Strain in LD (%)',
              'macro_stress_mag':'Macroscopic Stress Magnitude in LD (MPa)',
              'macro_strain_mag':'Macroscopic Strain Magnitude in LD (%)',
              'macro_p_strain_mag':'Macroscopic Plastic Strain in LD Magnitude (%)',
              'grain_stress_LD':'Grain Avg. Stress in LD (MPa)',
              'grain_strain_LD':'Grain Avg. Strain in LD (%)',
              'grain_misorientation':'Grain Avg. Misorientation (°)',
              'grain_dsgod_Sigma':'Grain Avg. $| \Sigma |$',
              'grain_von_mises_stress':'Grain Avg. Von Mises Stress (MPa)',
              'PC1':'Principal Component 1',
              'PC2':'Principal Component 2',
              'PC3':'Principal Component 3',
              'grain_tau_mrss':'Maximum Resolved Shear Stress'
              }
label_short = {'macro_stress':'${}^{macro}\sigma_{LD}$',
              'macro_strain':'${}^{macro}\epsilon_{LD}$',
              'macro_p_strain':'${}^{macro}\epsilon_{LD}^p$',
              'macro_stress_mag':'$| {}^{macro}\sigma_{LD} |$',
              'macro_strain_mag':'$| {}^{macro}\epsilon_{LD} |$',
              'macro_p_strain_mag':'$| {}^{macro}\epsilon_{LD}^p |$',
              'grain_stress_LD':'${}^{grain}\sigma_{LD}$',
              'grain_strain_LD':'${}^{grain}\epsilon^{e}_{LD}$',
              'grain_misorientation':'${}^{grain}\\theta_{misorientation}$',
              'grain_dsgod_Sigma':'${}^{grain}| \Sigma |$',
              'grain_von_mises_stress':'${}^{grain}\sigma_{VM}$',
              'PC1':'PC 1',
              'PC2':'PC 2',
              'PC3':'PC 3',
              'grain_tau_mrss':'${}^{grain}\\tau_{mrss}$'
              }

#%%
bounds = [np.array([0, 0, 0]), np.array([100, 100, 100])]
bounds = None
#fits = fit_cake(pca_image_list[52+98][600:1800, :], rings=4, plot=True, bounds=bounds)

fig = plt.figure()
plt.imshow(temp_img, vmax=0.1, aspect='auto')

fits = fit_cake(temp_img, rings=rings, plot=True, bounds=bounds, gauss_fit=False)
print(fits)

plt.show()

#%%
nrows = 4
fig, ax = plt.subplots(nrows=nrows, ncols=num_cmpts, figsize=(4.5*num_cmpts, 4.5*nrows))
cmap = 'viridis_r'#'RdYlBu'

if ss718:
    pc_lim = [[-0.6, 0.6], [-0.45, 0.45], [-0.2, 0.2]]
    pc_lim = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
    stress_lim = [-400, 400]
    stress_lim = [-600, 400]
    strain_lim = [-0.6, 0.6]

# fit and transform PCA matrix
PCA_func = decomposition.PCA(n_components=num_cmpts)
#PCA_func = TSNE(n_components=num_cmpts,learning_rate=300,perplexity = 30,early_exaggeration = 12,init = 'random',  random_state=2019)
#PCA_func = MDS(n_components=num_cmpts, n_init=12, max_iter=10, metric=True, n_jobs=4, random_state=2019)
#PCA_func = Isomap(n_components=num_cmpts, n_jobs=10, n_neighbors=10)
#decomposition.PCA(n_components=num_cmpts)
#PCA_func = LocallyLinearEmbedding(n_neighbors=15, n_components=num_cmpts, reg=1e-3, tol=1e-6, max_iter=100)

#% Initial load to cycle 1 fit
k = 0 # 23 = elastic, 31 = macroscopic yield, 40 = post-yield full measurement, 52 = tension tip, 76 = unload
l = 209#52 #216 is end of first cycle
PCA_mat = bigPCA_mat[k:l, :]
init_plot_stress = all_stress[k:l]
init_plot_strain = all_strain[k:l]
init_plot_scan = all_scan[k:l]
init_plot_all_fits = all_fits[k:l, :]
init_plot_p_strain = all_p_strain[k:l]
init_plot_ISE = all_ISE[k:l]

transformer = Normalizer().fit(PCA_mat)
PCA_mat = transformer.transform(PCA_mat)
init_Z = PCA_func.fit_transform(PCA_mat)
var_ratio = PCA_func.explained_variance_ratio_
print(var_ratio)

init_Z = -init_Z

title = ""
for i in range(num_cmpts):
    pc1 = ax[0, i].scatter(init_plot_strain, init_plot_stress, c=scale_0_1(init_Z[:, i]), s=50, vmin=pc_lim[i][0], vmax=pc_lim[i][1], cmap=cmap)
    ax[0, i].set_xlabel(label_long['macro_strain'])
    ax[0, i].set_xlim(strain_lim)
    ax[0, i].set_ylim(stress_lim)
    cbar = fig.colorbar(pc1, ax=ax[0, i])
    cbar.set_label('PC %i' %(i+1), rotation=270, labelpad=10)

#% cycle 1 fit

k = 52 # 23 = elastic, 31 = macroscopic yield, 40 = post-yield full measurement, 52 = tension tip, 76 = unload
l = 209#52 #216 is end of first cycle
PCA_mat = bigPCA_mat[k:l, :]
c1_plot_stress = all_stress[k:l]
c1_plot_strain = all_strain[k:l]
c1_plot_scan = all_scan[k:l]
c1_plot_all_fits = all_fits[k:l, :]
c1_plot_p_strain = all_p_strain[k:l]
c1_plot_ISE = all_ISE[k:l]

transformer = Normalizer().fit(PCA_mat)
PCA_mat = transformer.transform(PCA_mat)
c1_Z = PCA_func.fit_transform(PCA_mat)
c1_Z[:, 1] = -c1_Z[:, 1]
var_ratio = PCA_func.explained_variance_ratio_
print(var_ratio)

for i in range(num_cmpts):
    pc1 = ax[1, i].scatter(c1_plot_strain, c1_plot_stress, c=scale_0_1(c1_Z[:, i]), s=50, vmin=pc_lim[i][0], vmax=pc_lim[i][1], cmap=cmap)
    ax[1, i].set_xlabel(label_long['macro_strain'])
    ax[1, i].set_xlim(strain_lim)
    ax[1, i].set_ylim(stress_lim)
    cbar = fig.colorbar(pc1, ax=ax[1, i])
    cbar.set_label('PC %i' %(i+1), rotation=270, labelpad=10)

#% cycle 2 model
k = 206
l = 292
PCA_mat = bigPCA_mat[k:l, :]
c2_plot_stress = all_stress[k:l]
c2_plot_strain = all_strain[k:l]
c2_plot_scan = all_scan[k:l]
c2_plot_all_fits = all_fits[k:l, :]
c2_plot_p_strain = all_p_strain[k:l]
c2_plot_ISE = all_ISE[k:l]

transformer = Normalizer().fit(PCA_mat)
PCA_mat = transformer.transform(PCA_mat)
c2_Z = PCA_func.transform(PCA_mat)
c2_Z[:, 1] = -c2_Z[:, 1]
var_ratio = PCA_func.explained_variance_ratio_
print(var_ratio)

for i in range(num_cmpts):
    pc1 = ax[2, i].scatter(c2_plot_strain, c2_plot_stress, c=scale_0_1(c2_Z[:, i]), s=50, vmin=pc_lim[i][0], vmax=pc_lim[i][1], cmap=cmap)
    ax[2, i].set_xlabel(label_long['macro_strain'])
    ax[2, i].set_xlim(strain_lim)
    ax[2, i].set_ylim(stress_lim)
    cbar = fig.colorbar(pc1, ax=ax[2, i])
    cbar.set_label('PC %i' %(i+1), rotation=270, labelpad=10)

#% cycle 64 fit
k = 332 # 23 = elastic, 31 = macroscopic yield, 40 = post-yield full measurement, 52 = tension tip, 76 = unload
l = -1 #52 #216 is end of first cycle
PCA_mat = bigPCA_mat[k:l, :]
c64_plot_stress = all_stress[k:l]
c64_plot_strain = all_strain[k:l]
c64_plot_scan = all_scan[k:l]
c64_plot_all_fits = all_fits[k:l, :]
c64_plot_p_strain = all_p_strain[k:l]
c64_plot_ISE = all_ISE[k:l]

transformer = Normalizer().fit(PCA_mat)
PCA_mat = transformer.transform(PCA_mat)
c64_Z = PCA_func.transform(PCA_mat)
c64_Z[:, 1] = -c64_Z[:, 1]
var_ratio = PCA_func.explained_variance_ratio_
print(var_ratio)

for i in range(num_cmpts):
    pc1 = ax[3, i].scatter(c64_plot_strain, c64_plot_stress, c=scale_0_1(c64_Z[:, i]), s=50, vmin=pc_lim[i][0], vmax=pc_lim[i][1], cmap=cmap)
    ax[3, i].set_xlabel(label_long['macro_strain'])
    ax[3, i].set_xlim(strain_lim)
    ax[3, i].set_ylim([stress_lim[0] - 200, stress_lim[1] + 200])
    cbar = fig.colorbar(pc1, ax=ax[3, i])
    cbar.set_label('PC %i' %(i+1), rotation=270, labelpad=10)





# ****************************************************************************
#%% Start of Analysis of PC's
if ss718:
    temp_dir = '/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/'
    raw_data_dir_template = temp_dir + 'new_ss718_ct_thresh360/'
    raw_ff_data_dir_template = temp_dir + 'ff/'
    sample_name = 'ss718-1'
    
    load_step_list = ['c0_0_gripped', 'c0_1', 'c0_2', 'c0_3', 'c1_0', 'c1_1',
                  'c1_2', 'c1_3', 'c1_4', 'c1_5', 'c1_6', 'c1_7', 'c2_0',
                  'c2_3', 'c2_4', 'c2_7', 'c3_0', 'c64_0', 'c64_3', 'c64_4', 'c64_7' ,'c65_0']
    full_scan_list = [30, 50, 68, 90, 111, 136, 173, 198, 216, 234, 260, 280, 300, 337, 356, 387, 405, 1012, 1033, 1043, 1064, 1077]
    full_ct_scan_list = [34, 52, 70, 92, 113, 138, 175, 200, 218, 237, 262, 282, 297, 339, 358, 389, 397, 1014, 1035, 1045, 1066, 1072]
    full_stress_list = [0, 150, 247, 283, 300, 0, -263, -310, -327, 0, 264, 314, 326, -328, -336, 325, 333, 445, -390, -456, 382, 442]
    full_strain_list = [0, 0.08, 0.13, 0.25, 0.51, 0.37, 0, -0.25, -0.51, -0.35, 0, 0.26, 0.51, -0.25, -0.5, 0.26, 0.5, 0.5, -0.25, -0.5, 0.25, 0.5]
    scan_stem = '/%s_sc%i/grains.out'
    
else:
    temp_dir = '/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/'
    raw_data_dir_template = temp_dir + 'new_dp718_ct_thresh360/'
    raw_ff_data_dir_template = temp_dir + 'ff/'
    sample_name = 'dp718-1'

comp_thresh = 0.8
chi2_thresh = 1.0e-2
buffer = 0.125

pca_grains = np.load(temp_dir+'%s_pca_grains_from_spots_buffer%0.3f.npy' %(sample_name, buffer))
new_pca_grains = pca_grains # pca grains that maintain comp and chi2 thresh through cycles

#%%
num_scans = len(full_scan_list)
for i, scan in enumerate(full_scan_list):
    temp_grain_mat = np.loadtxt(temp_dir + load_step_list[i] + scan_stem %(load_step_list[i], scan))
    pca_grain_ind = np.searchsorted(temp_grain_mat[:, 0], pca_grains)
    temp_grain_mat = temp_grain_mat[pca_grain_ind, :]
    print(temp_grain_mat.shape)
    ind = np.where((temp_grain_mat[:, 1] >= comp_thresh) & (temp_grain_mat[:, 2] <= chi2_thresh))[0]
    temp_grain_mat = temp_grain_mat[ind, :]
    print(ind.shape)
    
    new_pca_grains = np.intersect1d(temp_grain_mat[:, 0], new_pca_grains)

print(new_pca_grains.shape)


np.save(temp_dir+'%s_pca_grains_c0_0_to_c3_0.npy' %(sample_name), np.array(new_pca_grains))


#%%

new_pca_grains = np.load(temp_dir+'%s_pca_grains_c0_0_to_c3_0.npy' %(sample_name))
first_grain_mat = np.loadtxt(temp_dir + load_step_list[0] + scan_stem %(load_step_list[0], full_scan_list[0]))
pca_grain_ind = np.searchsorted(first_grain_mat[:, 0], new_pca_grains)
first_grain_mat = first_grain_mat[pca_grain_ind, :]

stress_list =  np.array(stress_list)
strain_list = np.array(strain_list)
scan_list = np.array(scan_list)
all_fits = np.array(all_fits)


SX_STIFF = np.copy(pp_stress.INCONEL_718_STIFF)
SX_STIFF[0, 0] = 232e3
SX_STIFF[1, 1] = 232e3
SX_STIFF[2, 2] = 232e3
SX_STIFF[3, 3] = 112e3
SX_STIFF[4, 4] = 112e3
SX_STIFF[5, 5] = 112e3

data_in_full_ff_HEDM = np.zeros([len(full_scan_list), 29]) #29
# scan, macro_stress, macro_strain, r0_mu, r0_sigma, r1_mu, r1_sigma, r2_mu, r2_sigma, (9)
# avg e strain (6), avg sigma, avg gamma, avg_kappa, avg misori, vm_stress, coaxiality, triaxiality (13)
# avg grain stress (6)

for i, scan in enumerate(full_scan_list):
    temp_grain_mat = np.loadtxt(temp_dir + load_step_list[i] + scan_stem %(load_step_list[i], scan))
    pca_grain_ind = np.searchsorted(temp_grain_mat[:, 0], new_pca_grains)
    temp_grain_mat = temp_grain_mat[pca_grain_ind, :]
    temp_grain_mat[:, 15:] = temp_grain_mat[:, 15:] - first_grain_mat[:, 15:]
    
    schmid_T = pp_stress.gen_schmid_tensors_from_pd(plane_data, np.array([[1, 1, 0]]).T, np.array([[1, 1, 1]]).T)
    
    stress_data = pp_stress.post_process_stress(temp_grain_mat, SX_STIFF, 
                                                schmid_T_list=schmid_T, stress_macro=np.array([[0, full_stress_list[i], 0, 0, 0, 0]]).T)
    
    data_in_full_ff_HEDM[i, 0] = full_ct_scan_list[i]
    data_in_full_ff_HEDM[i, 1] = stress_list[scan_list == full_ct_scan_list[i]]
    data_in_full_ff_HEDM[i, 2] = strain_list[scan_list == full_ct_scan_list[i]]
    data_in_full_ff_HEDM[i, 3:9] = all_fits[scan_list == full_ct_scan_list[i], [1, 2, 4, 5, 7, 8]]
    data_in_full_ff_HEDM[i, 9:15] = np.mean(temp_grain_mat[:, 15:], axis=0).flatten() * 100
    data_in_full_ff_HEDM[i, 19] = np.mean(stress_data['von_mises'], axis=0).flatten()
    data_in_full_ff_HEDM[i, 20] = np.mean(stress_data['coaxiality'], axis=0).flatten()
    data_in_full_ff_HEDM[i, 21] = np.mean(stress_data['triaxiality'], axis=0).flatten()
    data_in_full_ff_HEDM[i, 22:28] = np.mean(stress_data['stress_S'], axis=0).flatten()
    data_in_full_ff_HEDM[i, 28] = np.mean(np.max(np.abs(stress_data['RSS']), axis=1), axis=0).flatten()
    #data_in_full_ff_HEDM[i, 28:] = np.mean(np.abs(stress_data['RSS']), axis=0).flatten()
    


#%%
comp_thresh = 0.85
sigma_stats = np.zeros([new_pca_grains.size, len(load_step_list)])
gamma_stats = np.zeros([new_pca_grains.size, len(load_step_list)])
kappa_stats = np.zeros([new_pca_grains.size, len(load_step_list)])
misoir = np.zeros([new_pca_grains.size, len(load_step_list)])
reori = np.zeros([new_pca_grains.size, len(load_step_list)])
init_avg_quat = np.zeros([new_pca_grains.size, 4])

# /media/djs522/djs522_nov2020/chess_2020_11/ss718-1/c3_0/dsgod/dsgod/dsgod_ss718-1_405
for i, scan in enumerate(load_step_list):
    dsgod_output_dir = '/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/%s/dsgod' %(scan)
    scan_num = full_scan_list[i]
    print(scan_num)
    for j, grain_id in enumerate(new_pca_grains):
        dsgod_npz_file = 'dsgod/dsgod_%s_%i/grain_%i_dsgod_map_data_inv.npz' %(sample_name, scan_num, grain_id)
        dsgod_npz_dir = os.path.join(dsgod_output_dir, dsgod_npz_file)
        
        # calculate moments
        [grain_quat, grain_mis_quat, grain_odf, sum_grain_inten] = pp_GOE.process_dsgod_file(dsgod_npz_dir, 
                                                          scan=scan_num, comp_thresh=comp_thresh,
                                                          do_avg_ori=True, do_conn_comp=True, save=False,
                                                          connectivity_type=26)
        
        avg_quat = np.atleast_2d(np.average(grain_quat, axis=1, weights=grain_odf)).T
        [a, misori_ang_deg, a] = pp_GOE.calc_misorientation(grain_quat[:, grain_odf > 0], avg_quat=avg_quat, disp_stats=False)
        if i == 0:
            init_avg_quat[j, :] = avg_quat.T
        else:
            [a, reori_ang_deg, a] = pp_GOE.calc_misorientation(grain_quat[:, grain_odf > 0], avg_quat=np.atleast_2d(init_avg_quat[j, :]).T, disp_stats=False)
            reori[j, i] = reori_ang_deg.max()
        
        
        norm_sigma, norm_gamma, norm_kappa, sigma, gamma, kappa = pp_GOE.calc_misorient_moments(grain_mis_quat[:, grain_odf > 0].T, grain_odf[grain_odf > 0])
        #print('|Sigma| = %0.2E \t |Gamma| = %0.2E \t |Kappa| = %0.2E' %(norm_sigma, norm_gamma, norm_kappa))
        
        sigma_stats[j, i] = norm_sigma
        gamma_stats[j, i] = norm_gamma
        kappa_stats[j, i] = norm_kappa
        misoir[j, i] = misori_ang_deg.max()
        

#%%

sigma_ave = []
gamma_ave = []
kappa_ave = []
mis_ave = []
reori_ave = []

for i, scan in enumerate(load_step_list):
    sigma_ave.append(np.mean(sigma_stats[sigma_stats[:, i] != 0, i], axis=0))
    gamma_ave.append(np.mean(gamma_stats[gamma_stats[:, i] != 0, i], axis=0))
    kappa_ave.append(np.mean(kappa_stats[kappa_stats[:, i] != 0, i], axis=0))
    mis_ave.append(np.mean(misoir[sigma_stats[:, i] != 0, i], axis=0))
    reori_ave.append(np.mean(reori[sigma_stats[:, i] != 0, i], axis=0))

sigma_ave = np.array(sigma_ave)
gamma_ave = np.array(gamma_ave)
kappa_ave = np.array(kappa_ave)
mis_ave = np.array(mis_ave)
reori_ave = np.array(reori_ave)

data_in_full_ff_HEDM[:, 15] = sigma_ave
data_in_full_ff_HEDM[:, 16] = gamma_ave
data_in_full_ff_HEDM[:, 17] = kappa_ave
data_in_full_ff_HEDM[:, 18] = mis_ave

k = 4
fig = plt.figure()
'''
plt.plot(scale_0_1(sigma_ave[k:]))
plt.plot(scale_0_1(gamma_ave[k:]))
plt.plot(scale_0_1(kappa_ave[k:]))
plt.plot(scale_0_1(mis_ave[k:]))
plt.plot(scale_0_1(reori_ave[k:]))
'''
plt.plot((sigma_ave[k:]))
plt.plot((gamma_ave[k:]))
plt.plot((kappa_ave[k:]))
plt.plot((mis_ave[k:]))
plt.plot((reori_ave[k:]))
plt.legend(['sigma', 'gamma', 'kappa', 'misori', 'reori'])
plt.show()

#%%
Z = init_Z
plot_against = full_scan_list # full_scan_list, full_stress_list, full_strain_list
plot_against_snap = init_plot_scan # plot_scan, plot_stress, plot_strain
plot_scan = init_plot_scan
plot_stress = init_plot_stress
plot_strain = init_plot_strain
plot_p_strain = init_plot_p_strain
plot_all_fits = init_plot_all_fits


Z = c1_Z
plot_against = full_scan_list # full_scan_list, full_stress_list, full_strain_list
plot_against_snap = c1_plot_scan # plot_scan, plot_stress, plot_strain
plot_scan = c1_plot_scan
plot_stress = c1_plot_stress
plot_strain = c1_plot_strain
plot_p_strain = c1_plot_p_strain
plot_all_fits = c1_plot_all_fits


Z = c2_Z
plot_against = full_scan_list # full_scan_list, full_stress_list, full_strain_list
plot_against_snap = c2_plot_scan # plot_scan, plot_stress, plot_strain
plot_scan = c2_plot_scan
plot_stress = c2_plot_stress
plot_strain = c2_plot_strain
plot_p_strain = c2_plot_p_strain
plot_all_fits = c2_plot_all_fits

ind1 = 13#4#13#0
ind2 = 17#13#18#13

d = 0
e = -1#-100#-18

size = 3.5
small = 15
big = 100
fs = 10

stress_lim = [-650, 400]
strain_lim = [-0.6, 0.6]

all_strain


matplotlib.rcParams.update({'font.size': fs})
plt.set_cmap('viridis')

fig, ax = plt.subplots(figsize=(size, size))
plt.subplots_adjust(top=0.95,
bottom=0.18,
left=0.22,
right=0.95,
hspace=0.2,
wspace=0.3)
temp_full_stress = np.array([0, 150, 235, 250, 300, 0, -210, -280, -327, 0, 235, 295, 326])
temp_full_strain = np.array([0, 0.08, 0.13, 0.25, 0.51, 0.37, 0.015, -0.255, -0.52, -0.35, -0.01, 0.25, 0.51])
pm1, = ax.plot(all_strain[:292], all_stress[:292], label='Material Response', c='k', linewidth=2, zorder=0)
p0 = ax.scatter(all_strain[:292], all_stress[:292], label='Limited Omega Diffraction Image (LODI)', c=colors[0], s=8)#small)
#pm1, = ax.plot(init_plot_strain, init_plot_stress, label='Material Response', c='k', linewidth=2, zorder=0)
#p0 = ax.scatter(init_plot_strain, init_plot_stress, label='Limited Omega Diffraction Image (LODI)', c=colors[0], s=small)
ind = [0, 2, 3, 4, 5, 6, 8, 9, 10, 12]
p1 = ax.scatter(temp_full_strain[ind], temp_full_stress[ind], label='Select Examples of LODI', c=np.arange(len(ind)), cmap='bone_r', s=100, marker='s', edgecolors='k')
p2 = ax.scatter(temp_full_strain[:], temp_full_stress[:], label='ff-HEDM Measurements', c=colors[2], s=40, marker='^', edgecolors='k')
#plt.legend(handles=[p0, p1, p2], loc='upper left', fontsize=10, frameon=False)
plt.legend(handles=[pm1, p0, p1, p2], loc='lower right', fontsize=7.5, frameon=False)
plt.xlim(strain_lim)
plt.ylim(stress_lim)
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.xlabel(label_long['macro_strain'], fontsize=fs)
plt.ylabel(label_long['macro_stress'], fontsize=fs)
plt.show()



#%%


grey_color = np.arange(Z.shape[0])
fig = plt.figure(figsize=(size, size))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(top=0.95,
bottom=0.18,
left=0.22,
right=0.95,
hspace=0.2,
wspace=0.3)

ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=grey_color, s=50, cmap='copper')
ax.set_xlabel('PC1', labelpad=5)
ax.set_ylabel('PC2', labelpad=5)
ax.set_zlabel('PC3', labelpad=5)


fig, ax = plt.subplots(figsize=(size, size))
plt.subplots_adjust(top=0.95,
bottom=0.18,
left=0.22,
right=0.95,
hspace=0.2,
wspace=0.3)
p0 = ax.scatter(plot_strain, plot_stress, label='Limited Omega Diffraction Image (LODI)', c=grey_color, s=small, cmap='copper')
plt.legend(handles=[p0], loc='lower right', fontsize=7.5, frameon=False)
plt.xlim(strain_lim)
plt.ylim(stress_lim)
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.xlabel(label_long['macro_strain'], fontsize=fs)
plt.ylabel(label_long['macro_stress'], fontsize=fs)
plt.show()



#%%
do_abs = True
s = 15
for i in range(3):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(right=0.75)
    twin1 = ax.twinx()
    twin2 = ax.twinx()
    twin3 = ax.twinx()
    twin1.spines['right'].set_position(("axes", 1.0))
    twin2.spines['right'].set_position(("axes", 1.13))
    twin3.spines['right'].set_position(("axes", 1.26))
    p0, = ax.plot(plot_against_snap[d:e], Z[d:e, i], c=colors[0], linestyle=lines[0], linewidth=lw_big, label='PC %i' %(i+1))
    if do_abs:
        p1, = twin1.plot(plot_against_snap[d:e], np.abs(plot_stress[d:e]), c=colors[1], linestyle=lines[1], linewidth=lw_small, label=label_short['macro_stress_mag'])
        p2, = twin2.plot(plot_against_snap[d:e], np.abs(plot_strain[d:e]), c=colors[2], linestyle=lines[2], linewidth=lw_small, label=label_short['macro_strain_mag'])
        p3, = twin3.plot(plot_against_snap[d:e], np.abs(plot_p_strain[d:e]), c=colors[3], linestyle=lines[3], linewidth=lw_small, label=label_short['macro_p_strain_mag'])
    else:
        p1, = twin1.plot(plot_against_snap[d:e], plot_stress[d:e], c=colors[1], linestyle=lines[1], linewidth=lw_small, label=label_short['macro_stress'])
        p2, = twin2.plot(plot_against_snap[d:e], plot_strain[d:e], c=colors[2], linestyle=lines[2], linewidth=lw_small, label=label_short['macro_strain'])
        p3, = twin3.plot(plot_against_snap[d:e], plot_p_strain[d:e], c=colors[3], linestyle=lines[3], linewidth=lw_small, label=label_short['macro_p_strain'])
    
    ax.set_xlabel("Scans")
    ax.set_ylabel(label_long["PC%i" %(i+1)])
    if do_abs:
        twin1.set_ylabel(label_long['macro_stress_mag'])
        twin2.set_ylabel(label_long['macro_strain_mag'])
        twin3.set_ylabel(label_long['macro_p_strain_mag'])
    else:
        twin1.set_ylabel(label_long['macro_stress'])
        twin2.set_ylabel(label_long['macro_strain'])
        twin3.set_ylabel(label_long['macro_p_strain'])
    
    ax.yaxis.label.set_color(p0.get_color())
    twin1.yaxis.label.set_color(p1.get_color())
    twin2.yaxis.label.set_color(p2.get_color())
    twin3.yaxis.label.set_color(p3.get_color())
    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', colors=p0.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin2.tick_params(axis='y', colors=p2.get_color(), **tkw)
    twin3.tick_params(axis='y', colors=p3.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw) 
    if do_abs:
        ax.legend(handles=[p0, p1, p2, p3], loc='upper left', fontsize=10)
    else:
        ax.legend(handles=[p0, p1, p2, p3], loc='lower right', fontsize=10)



plt.show()


#%% calculating correlations between PCs and features
#%% macro correlations cycle 1
feat = np.hstack([Z, np.atleast_2d(plot_stress).T, np.atleast_2d(plot_strain).T, 
                  plot_all_fits[:, [1, 2, 4, 5, 7, 8]], np.atleast_2d(plot_p_strain).T, 
                  np.abs(np.atleast_2d(plot_stress).T), np.abs(np.atleast_2d(plot_strain).T), 
                  np.abs(np.atleast_2d(plot_p_strain).T)])
feat_names = [label_short['PC1'], label_short['PC2'], label_short['PC3'], label_short['macro_stress'], label_short['macro_strain'], 'Ring0_mu', 'Ring0_sigma', 
              'Ring1_mu', 'Ring1_sigma', 'Ring2_mu', 'Ring2_sigma', label_short['macro_p_strain'], 
              label_short['macro_stress_mag'], label_short['macro_strain_mag'], label_short['macro_p_strain_mag']]
df = pd.DataFrame(feat, columns=feat_names)
df_small = df.iloc[:, [0, 1, 2, 3, 4, 11, 12, 13, 14]]

corr_mat = df_small.corr()

fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(111, aspect='equal')
plt.subplots_adjust(top=0.95,
bottom=0.3,
left=0.11,
right=0.85,
hspace=0.2,
wspace=0.3)


from scipy.stats import pearsonr

def corr_sig(df=None):
    p_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    for col in df.columns:
        for col2 in df.drop(col,axis=1).columns:
            _ , p = pearsonr(df[col],df[col2])
            p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = p
    return p_matrix
p_values = corr_sig(df_small)

sns.heatmap(corr_mat, annot=True, fmt='0.2f', cmap='RdBu', cbar_kws={'label': 'Pearson Pairwise Correlation (R-value)'}, vmax=1, vmin=-1)
# sns.heatmap(corr_mat, annot=False, cmap='RdBu', cbar_kws={'label': 'Pearson Pairwise Correlation'}, vmax=1, vmin=-1)
# sns.heatmap(corr_mat, annot=True, annot_kws={'va':'bottom'}, fmt="0.2f", cbar=False, mask=p_values > 0.005, alpha=0)
# sns.heatmap(corr_mat, annot=True, annot_kws={'va':'top'}, fmt="0.2f", cbar=False, mask=p_values < 0.005, alpha=0)

fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(111, aspect='equal')
plt.subplots_adjust(top=0.95,
bottom=0.3,
left=0.11,
right=0.85,
hspace=0.2,
wspace=0.3)
sns.heatmap(p_values, annot=True, fmt='0.2f', cmap='RdBu', cbar_kws={'label': 'Pearson Pairwise Correlation (R-value)'}, vmax=1, vmin=-1)

#%%
def corrdot(*args, **kwargs):
    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca()
    ax.set_axis_off()
    marker_size = 2000 #abs(corr_r) * 10000
    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
               vmin=-1, vmax=1, transform=ax.transAxes, marker='s')
    font_size = 16 #abs(corr_r) * 40 + 5
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)
    
def corrdot_diag(*args, **kwargs):
    corr_r = args[0].corr(args[0], 'pearson')
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca()
    ax.set_axis_off()
    marker_size = 2000 #abs(corr_r) * 10000
    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
               vmin=-1, vmax=1, transform=ax.transAxes, marker='s')
    font_size = 16 #abs(corr_r) * 40 + 5
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)

df = pd.DataFrame(feat, columns=feat_names)
df_small = df.iloc[:, [0, 1, 2, 4, 11, 3, 13, 14, 12]]
sns.set(style='white', font_scale=1.0)
g = sns.PairGrid(df_small, aspect=1.0, diag_sharey=False, height=1.5, layout_pad=0.25)

#g.map_diag(sns.histplot, kde_kws={'color': 'black'})
g.map_diag(corrdot_diag)
g.map_upper(corrdot)
#g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'black'})
g.map_lower(sns.scatterplot, edgecolor='none')
plt.show()




#%% micro correlations cycle 1 from full ff-HEDM measurements
# scan, macro_stress, macro_strain, r0_mu, r0_sigma, r1_mu, r1_sigma, r2_mu, r2_sigma, (9)
# avg e strain (6), avg sigma, avg gamma, avg_kappa, avg misori, vm_stress, coaxiality, triaxiality (13)
# avg grain stress (6)
matplotlib.rcParams.update({'font.size': 12})
feat = np.hstack([Z[np.in1d(plot_scan, np.array(full_ct_scan_list)), :], data_in_full_ff_HEDM[ind1:ind2, [23, 10, 19, 28, 18, 15, 1, 2]] ]) # init and cycle 1
#feat = np.hstack([c1_Z[np.in1d(c1_plot_scan, np.array(full_ct_scan_list)), :], data_in_full_ff_HEDM[4:12, [23, 10, 18, 19, 15, 1, 2]] ]) # init and cycle 1

feat_names = [label_short['PC1'], label_short['PC2'], label_short['PC3'], label_short['grain_stress_LD'], label_short['grain_strain_LD'], 
              label_short['grain_von_mises_stress'], label_short['grain_tau_mrss'], label_short['grain_misorientation'], label_short['grain_dsgod_Sigma'], 'Macro Stress', 'Macro Strain']


df = pd.DataFrame(feat, columns=feat_names)
df_small = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]

corr_mat = df_small.corr()
fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(111, aspect='equal')
plt.subplots_adjust(top=0.95,
bottom=0.3,
left=0.11,
right=0.85,
hspace=0.2,
wspace=0.3)
sns.heatmap(corr_mat, annot=True, fmt='0.2f', cmap='RdBu', cbar_kws={'label': 'Pearson Pairwise Correlation (R-value)'}, vmax=1, vmin=-1)
plt.show()


#%%
df = pd.DataFrame(feat, columns=feat_names)
df_small = df.iloc[:, [0, 1, 2, 4, 3, 5, 6, 7]]
sns.set(style='white', font_scale=1.0)
g = sns.PairGrid(df_small, aspect=1.0, diag_sharey=False, height=1.5, layout_pad=0.25)

#g.map_diag(sns.histplot, kde_kws={'color': 'black'})
g.map_diag(corrdot_diag)
g.map_upper(corrdot)
#g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'black'})
g.map_lower(sns.scatterplot, edgecolor='none')
plt.show()


#%% plotting scatters to show correlation with PC

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(size*3, size))
fig.subplots_adjust(top=0.85,
bottom=0.2,
left=0.07,
right=0.95,
hspace=0.2,
wspace=0.5)

# PC 1
twin1 = ax[0].twinx()
twin1.spines['right'].set_position(("axes", 1.0))
p0 = ax[0].scatter(scale_0_1(Z[d:e, 0]), plot_strain[d:e], s=small, c=colors[0], marker=markers[0])
p1 = twin1.scatter(scale_0_1(Z[d:e, 0]), plot_p_strain[d:e], s=small, c=colors[1], marker=markers[1])
ax[0].set_xlabel(label_long['PC1'])
ax[0].set_ylabel(label_long['macro_strain'])
twin1.set_ylabel(label_long['macro_p_strain'])
ax[0].yaxis.label.set_color(colors[0])
twin1.yaxis.label.set_color(colors[1])
tkw = dict(size=4, width=1.5)
ax[0].tick_params(axis='y', colors=colors[0], **tkw)
twin1.tick_params(axis='y', colors=colors[1], **tkw)

do_init = False
if do_init:
    # PC 2
    twin2 = ax[1].twinx()
    twin2.spines['right'].set_position(("axes", 1.0))
    p2 = ax[1].scatter(scale_0_1(Z[d:e, 1]), np.abs(plot_strain[d:e]), s=small, c=colors[0], marker=markers[0])
    p3 = twin2.scatter(scale_0_1(Z[d:e, 1]), np.abs(plot_p_strain[d:e]), s=small, c=colors[1], marker=markers[1])
    ax[1].set_xlabel(label_long['PC2'])
    ax[1].set_ylabel(label_long['macro_strain_mag'])
    twin2.set_ylabel(label_long['macro_p_strain_mag'])
    ax[1].yaxis.label.set_color(colors[0])
    twin2.yaxis.label.set_color(colors[1])
    tkw = dict(size=4, width=1.5)
    ax[1].tick_params(axis='y', colors=colors[0], **tkw)
    twin2.tick_params(axis='y', colors=colors[1], **tkw)
    
    
    # PC 3
    p4 = ax[2].scatter(scale_0_1(Z[d:e, 2]), plot_stress[d:e], s=small, c=colors[0], marker=markers[0])
    ax[2].set_xlabel(label_long['PC3'])
    ax[2].set_ylabel(label_long['macro_stress'])
    ax[2].yaxis.label.set_color(colors[0])
    tkw = dict(size=4, width=1.5)
    ax[2].tick_params(axis='y', colors=colors[0], **tkw)
else:
    # PC 2
    twin2 = ax[2].twinx()
    twin2.spines['right'].set_position(("axes", 1.0))
    p2 = ax[2].scatter(scale_0_1(Z[d:e, 2]), np.abs(plot_strain[d:e]), s=small, c=colors[0], marker=markers[0])
    p3 = twin2.scatter(scale_0_1(Z[d:e, 2]), np.abs(plot_p_strain[d:e]), s=small, c=colors[1], marker=markers[1])
    ax[2].set_xlabel(label_long['PC3'])
    ax[2].set_ylabel(label_long['macro_strain_mag'])
    twin2.set_ylabel(label_long['macro_p_strain_mag'])
    ax[2].yaxis.label.set_color(colors[0])
    twin2.yaxis.label.set_color(colors[1])
    tkw = dict(size=4, width=1.5)
    ax[2].tick_params(axis='y', colors=colors[0], **tkw)
    twin2.tick_params(axis='y', colors=colors[1], **tkw)
    
    
    # PC 3
    p4 = ax[1].scatter(scale_0_1(Z[d:e, 1]), plot_stress[d:e], s=small, c=colors[0], marker=markers[0])
    ax[1].set_xlabel(label_long['PC2'])
    ax[1].set_ylabel(label_long['macro_stress'])
    ax[1].yaxis.label.set_color(colors[0])
    tkw = dict(size=4, width=1.5)
    ax[1].tick_params(axis='y', colors=colors[0], **tkw)
plt.show()


#%% plotting the init PCs on stress strain curve

vert=False

plt.set_cmap('viridis')

if vert:
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(size, size*3))
    fig.subplots_adjust(top=0.97,
    bottom=0.07,
    left=0.25,
    right=0.95,
    hspace=0.3,
    wspace=0.3)
else:
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(size*3, size))
    fig.subplots_adjust(top=0.88,
    bottom=0.15,
    left=0.1,
    right=0.95,
    hspace=0.3,
    wspace=0.3)
ax[0].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(Z[d:e, 0]), s=small)
ax[1].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(Z[d:e, 1]), s=small)
ax[2].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(Z[d:e, 2]), s=small)
ax[0].legend([label_short['PC1']], loc='lower right')
ax[1].legend([label_short['PC2']], loc='lower right')
ax[2].legend([label_short['PC3']], loc='lower right')


if vert:
    #ax[0].set_xlabel(label_long['macro_strain'])
    ax[0].set_ylabel(label_long['macro_stress'])
    #ax[1].set_xlabel(label_long['macro_strain'])
    ax[1].set_ylabel(label_long['macro_stress'])
    ax[2].set_xlabel(label_long['macro_strain'])
    ax[2].set_ylabel(label_long['macro_stress'])
else:
    ax[0].set_xlabel(label_long['macro_strain'])
    ax[0].set_ylabel(label_long['macro_stress'])
    ax[1].set_xlabel(label_long['macro_strain'])
    #ax[1].set_ylabel(label_long['macro_stress'])
    ax[2].set_xlabel(label_long['macro_strain'])
    #ax[2].set_ylabel(label_long['macro_stress'])
ax[0].set_xlim([strain_lim[0], strain_lim[1]])
ax[0].set_ylim([stress_lim[0], stress_lim[1]])
ax[1].set_xlim([strain_lim[0], strain_lim[1]])
ax[1].set_ylim([stress_lim[0], stress_lim[1]])
ax[2].set_xlim([strain_lim[0], strain_lim[1]])
ax[2].set_ylim([stress_lim[0], stress_lim[1]]) 


for i in range(num_cmpts):
    fig, ax = plt.subplots(figsize=(size, size))
    plt.subplots_adjust(top=0.95,
    bottom=0.18,
    left=0.22,
    right=0.95,
    hspace=0.2,
    wspace=0.3)
    ax.scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(Z[d:e, i]), s=small)
    ax.legend([label_short['PC%i'%(i+1)]], loc='lower right')
    plt.xlim(strain_lim)
    plt.ylim(stress_lim)
    plt.tick_params(axis='both', which='major', labelsize=fs)
    plt.xlabel(label_long['macro_strain'], fontsize=fs)
    plt.ylabel(label_long['macro_stress'], fontsize=fs)
plt.show()
    

#%% plotting similar correlations of PCs on stress strain

split = 3
do_init = False
# MICRO
fig, ax = plt.subplots(figsize=(size, size))
plt.subplots_adjust(top=0.95,
bottom=0.18,
left=0.22,
right=0.95,
hspace=0.2,
wspace=0.3)
ax.scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(Z[d:e, 0]), s=small)
ax.scatter(full_strain_list[ind1:ind2], np.array(full_stress_list[ind1:ind2])+split, c=data_in_full_ff_HEDM[ind1:ind2, 10], s=big, marker=10)
ax.scatter(full_strain_list[ind1:ind2], np.array(full_stress_list[ind1:ind2])-split, c=data_in_full_ff_HEDM[ind1:ind2, 23], s=big, marker=11)
ax.legend([label_short['PC1'], label_short['grain_strain_LD'], label_short['grain_stress_LD']], loc='lower right', fontsize=10, frameon=False)
plt.xlim(strain_lim)
plt.ylim(stress_lim)
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.xlabel(label_long['macro_strain'], fontsize=fs)
plt.ylabel(label_long['macro_stress'], fontsize=fs)


fig, ax = plt.subplots(figsize=(size, size))
plt.subplots_adjust(top=0.95,
bottom=0.18,
left=0.22,
right=0.95,
hspace=0.2,
wspace=0.3)
if do_init:
    ax.scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(Z[d:e, 1]), s=small)
else:
    ax.scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(Z[d:e, 2]), s=small)
ax.scatter(full_strain_list[ind1:ind2], np.array(full_stress_list[ind1:ind2])+split, c=data_in_full_ff_HEDM[ind1:ind2, 18], s=big, marker=10)
ax.scatter(full_strain_list[ind1:ind2], np.array(full_stress_list[ind1:ind2])-split, c=data_in_full_ff_HEDM[ind1:ind2, 15], s=big, marker=11)
if do_init:
    ax.legend([label_short['PC2'], label_short['grain_misorientation'], label_short['grain_dsgod_Sigma']], loc='lower right', fontsize=10, frameon=False)
else:
    ax.legend([label_short['PC3'], label_short['grain_misorientation'], label_short['grain_dsgod_Sigma']], loc='lower right', fontsize=10, frameon=False)
    
plt.xlim(strain_lim)
plt.ylim(stress_lim)
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.xlabel(label_long['macro_strain'], fontsize=fs)
plt.ylabel(label_long['macro_stress'], fontsize=fs)

fig, ax = plt.subplots(figsize=(size, size))
plt.subplots_adjust(top=0.95,
bottom=0.18,
left=0.22,
right=0.95,
hspace=0.2,
wspace=0.3)
if do_init:
    ax.scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(Z[d:e, 2]), s=small)
else:
    ax.scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(Z[d:e, 1]), s=small)
ax.scatter(full_strain_list[ind1:ind2], np.array(full_stress_list[ind1:ind2])+split, c=data_in_full_ff_HEDM[ind1:ind2, 10], s=big, marker=10)
ax.scatter(full_strain_list[ind1:ind2], np.array(full_stress_list[ind1:ind2])-split, c=data_in_full_ff_HEDM[ind1:ind2, 23], s=big, marker=11)
if do_init:
    ax.legend([label_short['PC3'], label_short['grain_strain_LD'], label_short['grain_stress_LD']], loc='lower right', fontsize=10, frameon=False)
else:
    ax.legend([label_short['PC2'], label_short['grain_strain_LD'], label_short['grain_stress_LD']], loc='lower right', fontsize=10, frameon=False)
plt.xlim(strain_lim)
plt.ylim(stress_lim)
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.xlabel(label_long['macro_strain'], fontsize=fs)
plt.ylabel(label_long['macro_stress'], fontsize=fs)


#%% MACRO
fig, ax = plt.subplots(figsize=(size, size))
plt.subplots_adjust(top=0.95,
bottom=0.18,
left=0.22,
right=0.95,
hspace=0.2,
wspace=0.3)
ax.scatter(plot_strain[d:e], plot_stress[d:e], c=(plot_stress[d:e]), s=small)
ax.legend([label_short['macro_stress']], loc='lower right', fontsize=10, frameon=False)
plt.xlim(strain_lim)
plt.ylim(stress_lim)
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.xlabel(label_long['macro_strain'], fontsize=fs)
plt.ylabel(label_long['macro_stress'], fontsize=fs)

fig, ax = plt.subplots(figsize=(size, size))
plt.subplots_adjust(top=0.95,
bottom=0.18,
left=0.22,
right=0.95,
hspace=0.2,
wspace=0.3)
ax.scatter(plot_strain[d:e], plot_stress[d:e], c=(plot_strain[d:e]), s=small)
ax.legend([label_short['macro_strain']], loc='lower right', fontsize=10, frameon=False)
plt.xlim(strain_lim)
plt.ylim(stress_lim)
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.xlabel(label_long['macro_strain'], fontsize=fs)
plt.ylabel(label_long['macro_stress'], fontsize=fs)

fig, ax = plt.subplots(figsize=(size, size))
plt.subplots_adjust(top=0.95,
bottom=0.18,
left=0.22,
right=0.95,
hspace=0.2,
wspace=0.3)
ax.scatter(plot_strain[d:e], plot_stress[d:e], c=(plot_p_strain[d:e]), s=small)
ax.legend([label_short['macro_p_strain']], loc='lower right', fontsize=10, frameon=False)
plt.xlim(strain_lim)
plt.ylim(stress_lim)
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.xlabel(label_long['macro_strain'], fontsize=fs)
plt.ylabel(label_long['macro_stress'], fontsize=fs)

fig, ax = plt.subplots(figsize=(size, size))
plt.subplots_adjust(top=0.95,
bottom=0.18,
left=0.22,
right=0.95,
hspace=0.2,
wspace=0.3)
ax.scatter(plot_strain[d:e], plot_stress[d:e], c=np.abs(plot_stress[d:e]), s=small)
ax.legend([label_short['macro_stress_mag']], loc='lower right', fontsize=10, frameon=False)
plt.xlim(strain_lim)
plt.ylim(stress_lim)
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.xlabel(label_long['macro_strain'], fontsize=fs)
plt.ylabel(label_long['macro_stress'], fontsize=fs)

fig, ax = plt.subplots(figsize=(size, size))
plt.subplots_adjust(top=0.95,
bottom=0.18,
left=0.22,
right=0.95,
hspace=0.2,
wspace=0.3)
ax.scatter(plot_strain[d:e], plot_stress[d:e], c=np.abs(plot_strain[d:e]), s=small)
ax.legend([label_short['macro_strain_mag']], loc='lower right', fontsize=10, frameon=False)
plt.xlim(strain_lim)
plt.ylim(stress_lim)
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.xlabel(label_long['macro_strain'], fontsize=fs)
plt.ylabel(label_long['macro_stress'], fontsize=fs)

fig, ax = plt.subplots(figsize=(size, size))
plt.subplots_adjust(top=0.95,
bottom=0.18,
left=0.22,
right=0.95,
hspace=0.2,
wspace=0.3)
ax.scatter(plot_strain[d:e], plot_stress[d:e], c=np.abs(plot_p_strain[d:e]), s=small)
ax.legend([label_short['macro_p_strain_mag']], loc='lower right', fontsize=10, frameon=False)
plt.xlim(strain_lim)
plt.ylim(stress_lim)
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.xlabel(label_long['macro_strain'], fontsize=fs)
plt.ylabel(label_long['macro_stress'], fontsize=fs)


plt.show()

#%%

fig_micro, ax_micro = plt.subplots(nrows=1, ncols=3, figsize=(size*3, size))
fig_micro.subplots_adjust(top=0.88,
bottom=0.15,
left=0.07,
right=0.95,
hspace=0.3,
wspace=0.3)
ax_micro[0].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(Z[d:e, 0]), s=small)
ax_micro[0].scatter(full_strain_list[ind1:ind2], np.array(full_stress_list[ind1:ind2])+1.5, c=data_in_full_ff_HEDM[ind1:ind2, 10], s=big, marker=10)
ax_micro[0].scatter(full_strain_list[ind1:ind2], np.array(full_stress_list[ind1:ind2])-1.5, c=data_in_full_ff_HEDM[ind1:ind2, 23], s=big, marker=11)
ax_micro[0].legend([label_short['PC1'], label_short['grain_strain_LD'], label_short['grain_stress_LD']], loc='lower right', fontsize=10, frameon=False)

ax_micro[1].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(Z[d:e, 1]), s=small)
ax_micro[1].scatter(full_strain_list[ind1:ind2], np.array(full_stress_list[ind1:ind2])+1.5, c=data_in_full_ff_HEDM[ind1:ind2, 18], s=big, marker=10)
ax_micro[1].scatter(full_strain_list[ind1:ind2], np.array(full_stress_list[ind1:ind2])-1.5, c=data_in_full_ff_HEDM[ind1:ind2, 15], s=big, marker=11)
ax_micro[1].legend([label_short['PC2'], label_short['grain_misorientation'], label_short['grain_dsgod_Sigma']], loc='lower right', fontsize=10, frameon=False)

ax_micro[2].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(Z[d:e, 2]), s=small)
ax_micro[2].scatter(full_strain_list[ind1:ind2], np.array(full_stress_list[ind1:ind2])+1.5, c=data_in_full_ff_HEDM[ind1:ind2, 10], s=big, marker=10)
ax_micro[2].scatter(full_strain_list[ind1:ind2], np.array(full_stress_list[ind1:ind2])-1.5, c=data_in_full_ff_HEDM[ind1:ind2, 23], s=big, marker=11)
ax_micro[2].legend([label_short['PC3'], label_short['grain_strain_LD'], label_short['grain_stress_LD']], loc='lower right', fontsize=10, frameon=False)


ax_micro[0].set_xlabel(label_long['macro_strain'])
ax_micro[0].set_ylabel(label_long['macro_stress'])
ax_micro[1].set_xlabel(label_long['macro_strain'])
ax_micro[1].set_ylabel(label_long['macro_stress'])
ax_micro[2].set_xlabel(label_long['macro_strain'])
ax_micro[2].set_ylabel(label_long['macro_stress'])
ax_micro[0].set_xlim([strain_lim[0], strain_lim[1]])
ax_micro[0].set_ylim([stress_lim[0], stress_lim[1]])
ax_micro[1].set_xlim([strain_lim[0], strain_lim[1]])
ax_micro[1].set_ylim([stress_lim[0], stress_lim[1]])
ax_micro[2].set_xlim([strain_lim[0], strain_lim[1]])
ax_micro[2].set_ylim([stress_lim[0], stress_lim[1]])

plt.show() 


#%%

plot_micro = False
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(size*3, size))
fig.subplots_adjust(top=0.88,
bottom=0.15,
left=0.07,
right=0.95,
hspace=0.3,
wspace=0.3)
ax[0].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(Z[d:e, 0]), s=small)
ax[1].scatter(plot_strain[d:e], plot_stress[d:e], c=plot_strain[d:e], s=small)
ax[2].scatter(plot_strain[d:e], plot_stress[d:e], c=plot_p_strain[d:e], s=small)
if plot_micro:
    ax[0].scatter(full_strain_list[ind1:ind2], np.array(full_stress_list[ind1:ind2])+1.5, c=data_in_full_ff_HEDM[ind1:ind2, 10], s=big, marker=10)
    ax[0].scatter(full_strain_list[ind1:ind2], np.array(full_stress_list[ind1:ind2])-1.5, c=data_in_full_ff_HEDM[ind1:ind2, 23], s=big, marker=11)
ax[0].legend([label_short['PC1'], label_short['grain_strain_LD'], label_short['grain_stress_LD']], loc='lower right', fontsize=10, frameon=False)
ax[0].set_xlabel(label_long['macro_strain'])
ax[0].set_ylabel(label_long['macro_stress'])
ax[1].legend([label_short['macro_strain']], loc='lower right', fontsize=10, frameon=False)
ax[1].set_xlabel(label_long['macro_strain'])
ax[1].set_ylabel(label_long['macro_stress'])
ax[2].legend([label_short['macro_p_strain']], loc='lower right', fontsize=10, frameon=False)
ax[2].set_xlabel(label_long['macro_strain'])
ax[2].set_ylabel(label_long['macro_stress'])
ax[0].set_xlim([strain_lim[0], strain_lim[1]])
ax[0].set_ylim([stress_lim[0], stress_lim[1]])
ax[1].set_xlim([strain_lim[0], strain_lim[1]])
ax[1].set_ylim([stress_lim[0], stress_lim[1]])
ax[2].set_xlim([strain_lim[0], strain_lim[1]])
ax[2].set_ylim([stress_lim[0], stress_lim[1]]) 

if do_init:
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(size*3, size))
    fig.subplots_adjust(top=0.88,
    bottom=0.15,
    left=0.07,
    right=0.95,
    hspace=0.3,
    wspace=0.3)
    ax[0].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(Z[d:e, 1]), s=small)
    ax[1].scatter(plot_strain[d:e], plot_stress[d:e], c=np.abs(plot_strain[d:e]), s=small)
    ax[2].scatter(plot_strain[d:e], plot_stress[d:e], c=np.abs(plot_p_strain[d:e]), s=small)
    if plot_micro:
        ax[0].scatter(full_strain_list[ind1:ind2], np.array(full_stress_list[ind1:ind2])+1.5, c=data_in_full_ff_HEDM[ind1:ind2, 18], s=big, marker=10)
        ax[0].scatter(full_strain_list[ind1:ind2], np.array(full_stress_list[ind1:ind2])-1.5, c=data_in_full_ff_HEDM[ind1:ind2, 15], s=big, marker=11)
    ax[0].legend([label_short['PC2'], label_short['grain_misorientation'], label_short['grain_dsgod_Sigma']], loc='lower right', fontsize=10, frameon=False)
    ax[0].set_xlabel(label_long['macro_strain'])
    ax[0].set_ylabel(label_long['macro_stress'])
    ax[1].legend([label_short['macro_strain_mag']], loc='lower right', fontsize=10, frameon=False)
    ax[1].set_xlabel(label_long['macro_strain'])
    ax[1].set_ylabel(label_long['macro_stress'])
    ax[2].legend([label_short['macro_p_strain_mag']], loc='lower right', fontsize=10, frameon=False)
    ax[2].set_xlabel(label_long['macro_strain'])
    ax[2].set_ylabel(label_long['macro_stress'])
    ax[0].set_xlim([strain_lim[0], strain_lim[1]])
    ax[0].set_ylim([stress_lim[0], stress_lim[1]])
    ax[1].set_xlim([strain_lim[0], strain_lim[1]])
    ax[1].set_ylim([stress_lim[0], stress_lim[1]])
    ax[2].set_xlim([strain_lim[0], strain_lim[1]])
    ax[2].set_ylim([stress_lim[0], stress_lim[1]])
    
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(size*3, size))
    fig.subplots_adjust(top=0.88,
    bottom=0.15,
    left=0.07,
    right=0.95,
    hspace=0.3,
    wspace=0.3)
    ax[0].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(Z[d:e, 2]), s=small)
    ax[1].scatter(plot_strain[d:e], plot_stress[d:e], c=plot_stress[d:e], s=small)
    if plot_micro:
        ax[0].scatter(full_strain_list[ind1:ind2], np.array(full_stress_list[ind1:ind2])+1.5, c=data_in_full_ff_HEDM[ind1:ind2, 10], s=big, marker=10)
        ax[0].scatter(full_strain_list[ind1:ind2], np.array(full_stress_list[ind1:ind2])-1.5, c=data_in_full_ff_HEDM[ind1:ind2, 23], s=big, marker=11)
    ax[0].legend([label_short['PC3'], label_short['grain_strain_LD'], label_short['grain_stress_LD']], loc='lower right', fontsize=10, frameon=False)
    ax[1].legend([label_short['macro_stress']], loc='lower right', fontsize=10, frameon=False)
    ax[0].set_xlabel(label_long['macro_strain'])
    ax[0].set_ylabel(label_long['macro_stress'])
    ax[1].set_xlabel(label_long['macro_strain'])
    ax[1].set_ylabel(label_long['macro_stress'])
    ax[0].set_xlim([strain_lim[0], strain_lim[1]])
    ax[0].set_ylim([stress_lim[0], stress_lim[1]])
    ax[1].set_xlim([strain_lim[0], strain_lim[1]])
    ax[1].set_ylim([stress_lim[0], stress_lim[1]])
else:
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(size*3, size))
    fig.subplots_adjust(top=0.88,
    bottom=0.15,
    left=0.07,
    right=0.95,
    hspace=0.3,
    wspace=0.3)
    ax[0].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(Z[d:e, 2]), s=small)
    ax[1].scatter(plot_strain[d:e], plot_stress[d:e], c=np.abs(plot_strain[d:e]), s=small)
    ax[2].scatter(plot_strain[d:e], plot_stress[d:e], c=np.abs(plot_p_strain[d:e]), s=small)
    if plot_micro:
        ax[0].scatter(full_strain_list[ind1:ind2], np.array(full_stress_list[ind1:ind2])+1.5, c=data_in_full_ff_HEDM[ind1:ind2, 18], s=big, marker=10)
        ax[0].scatter(full_strain_list[ind1:ind2], np.array(full_stress_list[ind1:ind2])-1.5, c=data_in_full_ff_HEDM[ind1:ind2, 15], s=big, marker=11)
    ax[0].legend([label_short['PC3'], label_short['grain_misorientation'], label_short['grain_dsgod_Sigma']], loc='lower right', fontsize=10, frameon=False)
    ax[0].set_xlabel(label_long['macro_strain'])
    ax[0].set_ylabel(label_long['macro_stress'])
    ax[1].legend([label_short['macro_strain_mag']], loc='lower right', fontsize=10, frameon=False)
    ax[1].set_xlabel(label_long['macro_strain'])
    ax[1].set_ylabel(label_long['macro_stress'])
    ax[2].legend([label_short['macro_p_strain_mag']], loc='lower right', fontsize=10, frameon=False)
    ax[2].set_xlabel(label_long['macro_strain'])
    ax[2].set_ylabel(label_long['macro_stress'])
    ax[0].set_xlim([strain_lim[0], strain_lim[1]])
    ax[0].set_ylim([stress_lim[0], stress_lim[1]])
    ax[1].set_xlim([strain_lim[0], strain_lim[1]])
    ax[1].set_ylim([stress_lim[0], stress_lim[1]])
    ax[2].set_xlim([strain_lim[0], strain_lim[1]])
    ax[2].set_ylim([stress_lim[0], stress_lim[1]])
    
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(size*3, size))
    fig.subplots_adjust(top=0.88,
    bottom=0.15,
    left=0.07,
    right=0.95,
    hspace=0.3,
    wspace=0.3)
    ax[0].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(Z[d:e, 1]), s=small)
    ax[1].scatter(plot_strain[d:e], plot_stress[d:e], c=plot_stress[d:e], s=small)
    if plot_micro:
        ax[0].scatter(full_strain_list[ind1:ind2], np.array(full_stress_list[ind1:ind2])+1.5, c=data_in_full_ff_HEDM[ind1:ind2, 10], s=big, marker=10)
        ax[0].scatter(full_strain_list[ind1:ind2], np.array(full_stress_list[ind1:ind2])-1.5, c=data_in_full_ff_HEDM[ind1:ind2, 23], s=big, marker=11)
    ax[0].legend([label_short['PC2'], label_short['grain_strain_LD'], label_short['grain_stress_LD']], loc='lower right', fontsize=10, frameon=False)
    ax[1].legend([label_short['macro_stress']], loc='lower right', fontsize=10, frameon=False)
    ax[0].set_xlabel(label_long['macro_strain'])
    ax[0].set_ylabel(label_long['macro_stress'])
    ax[1].set_xlabel(label_long['macro_strain'])
    ax[1].set_ylabel(label_long['macro_stress'])
    ax[0].set_xlim([strain_lim[0], strain_lim[1]])
    ax[0].set_ylim([stress_lim[0], stress_lim[1]])
    ax[1].set_xlim([strain_lim[0], strain_lim[1]])
    ax[1].set_ylim([stress_lim[0], stress_lim[1]])


plt.show()











































#%%

temp_save_path = '/home/djs522/Downloads/2021_10_15_pca_fig/'
matplotlib.rcParams.update({'font.size':16})
size = 4.5
temp_fs = 10
stress_lim = [-550, 550]
strain_lim = [-0.6, 0.6]
transp = True
dpi=400

save_fig = False
base1 = 'pc_and_macro'
base2 = 'pc_and_micro'
base3 = 'pc_c1_2_64'

# macroscopic plotting
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(size*3, size*2))
fig.subplots_adjust(top=0.88,
bottom=0.11,
left=0.07,
right=0.95,
hspace=0.3,
wspace=0.3)
ax[0, 0].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(-Z[d:e, 0]), s=small)
ax[0, 1].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(-Z[d:e, 1]), s=small)
ax[0, 2].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(-Z[d:e, 2]), s=small)
ax[1, 0].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(plot_stress[d:e]), s=small)
ax[1, 1].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(plot_strain[d:e]), s=small)
#ax[0].scatter(full_strain_list[ind1:ind2], full_stress_list[ind1:ind2], c=scale_0_1(np.mean(val_to_plot, axis=1)[ind1:ind2]), s=big, marker='s')
#ax[1].scatter(full_strain_list[ind1:ind2], full_stress_list[ind1:ind2], c=scale_0_1(np.mean(val_to_plot, axis=1)[ind1:ind2]), s=big, marker='s')
#ax[2].scatter(full_strain_list[ind1:ind2], full_stress_list[ind1:ind2], c=scale_0_1(np.mean(val_to_plot, axis=1)[ind1:ind2]), s=big, marker='s')
ax[0, 0].legend(['PC1', 'Grain LD Elastic Strain'], loc='lower right', fontsize=temp_fs)
ax[0, 1].legend(['PC2', 'Grain LD Elastic Strain'], loc='lower right', fontsize=temp_fs)
ax[0, 2].legend(['PC3', 'Grain LD Elastic Strain'], loc='lower right', fontsize=temp_fs)
ax[1, 0].legend(['Macro LD Stress', 'Grain LD Elastic Strain'], loc='lower right', fontsize=temp_fs)
ax[1, 1].legend(['Macro LD Strain ', 'Grain LD Elastic Strain'], loc='lower right', fontsize=temp_fs)

for i in range(2):
    for j in range(3):
        ax[i,j].set_xlim([strain_lim[0], strain_lim[1]])
        ax[i,j].set_ylim([stress_lim[0], stress_lim[1]])
if save_fig:
    if transp:
        plt.savefig(temp_save_path + '%s_t.png' %base1, dpi=dpi, transparent=transp )
        plt.savefig(temp_save_path + '%s_t.svg' %base1, dpi=dpi, transparent=transp )
    else:
        plt.savefig(temp_save_path + '%s.png' %base1, dpi=dpi, transparent=transp )
        plt.savefig(temp_save_path + '%s.svg' %base1, dpi=dpi, transparent=transp )


# microscopic plotting
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(size*3, size*2))
fig.subplots_adjust(top=0.88,
bottom=0.11,
left=0.07,
right=0.95,
hspace=0.3,
wspace=0.3)

ind1 = 0
ind2 = 13

# PC1
ax[0, 0].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(-Z[d:e, 0]), s=small)
ax[0, 0].scatter(full_strain_list[ind1:ind2], full_stress_list[ind1:ind2], c=scale_0_1(np.mean(val_to_plot, axis=1)[ind1:ind2]), s=big, marker='s')

# PC2
ax[0, 1].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(-Z[d:e, 1]), s=small)
ax[0, 1].scatter(full_strain_list[ind1:ind2], full_stress_list[ind1:ind2], c=scale_0_1(np.mean(val_to_plot, axis=1)[ind1:ind2]), s=big, marker='s')

# PC3
ax[1, 0].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(-Z[d:e, 2]), s=small)
ax[1, 1].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(-Z[d:e, 2]), s=small)
ax[1, 2].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(-Z[d:e, 2]), s=small)
ax[1, 0].scatter(np.array(full_strain_list)[ind1:ind2], full_stress_list[ind1:ind2], c=-np.log(1/sigma_ave[ind1:ind2]), s=big, marker='s')
#ax[0].scatter(np.array(full_strain_list)[ind1:ind2], full_stress_list[ind1:ind2], c=-scale_0_1(mis_ave[ind1:ind2]), s=big, marker='s')
ax[1, 1].scatter(np.array(full_strain_list)[ind1:ind2], full_stress_list[ind1:ind2], c=-np.log(gamma_ave[ind1:ind2]), s=big, marker='s')
ax[1, 2].scatter(np.array(full_strain_list)[ind1:ind2], full_stress_list[ind1:ind2], c=-np.log(kappa_ave[ind1:ind2]), s=big, marker='s')

ax[0, 0].legend(['PC1', 'Grain E LD Strain'], loc='lower right', fontsize=temp_fs)
ax[0, 1].legend(['PC2', 'Grain E LD Strain'], loc='lower right', fontsize=temp_fs)
ax[1, 0].legend(['PC3', 'Average $|\Sigma|$'], loc='lower right', fontsize=temp_fs)
ax[1, 1].legend(['PC3', 'Average $|\Gamma|$'], loc='lower right', fontsize=temp_fs)
ax[1, 2].legend(['PC3', 'Average $|K|$'], loc='lower right', fontsize=temp_fs)

for i in range(2):
    for j in range(3):
        ax[i,j].set_xlim([strain_lim[0], strain_lim[1]])
        ax[i,j].set_ylim([stress_lim[0], stress_lim[1]])

if save_fig:
    if transp:
        plt.savefig(temp_save_path + '%s_t.png' %base2, dpi=dpi, transparent=transp )
        plt.savefig(temp_save_path + '%s_t.svg' %base2, dpi=dpi, transparent=transp )
    else:
        plt.savefig(temp_save_path + '%s.png' %base2, dpi=dpi, transparent=transp )
        plt.savefig(temp_save_path + '%s.svg' %base2, dpi=dpi, transparent=transp )



# c1, c2, c64 model and fit
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(size*3, size*3))
fig.subplots_adjust(top=0.88,
bottom=0.11,
left=0.07,
right=0.95,
hspace=0.3,
wspace=0.3)

# cycle 1
ind1 = 4
ind2 = 13
ax[0, 0].scatter(c1_plot_strain, c1_plot_stress, c=scale_0_1(c1_Z[:, 0]), s=small)
ax[0, 1].scatter(c1_plot_strain, c1_plot_stress, c=scale_0_1(-c1_Z[:, 1]), s=small)
ax[0, 2].scatter(c1_plot_strain, c1_plot_stress, c=scale_0_1(c1_Z[:, 2]), s=small)
ax[0, 0].scatter(full_strain_list[ind1:ind2], full_stress_list[ind1:ind2], c=scale_0_1(np.mean(val_to_plot, axis=1)[ind1:ind2]), s=big, marker='s')
ax[0, 1].scatter(full_strain_list[ind1:ind2], full_stress_list[ind1:ind2], c=scale_0_1(np.mean(val_to_plot, axis=1)[ind1:ind2]), s=big, marker='s')
ax[0, 2].scatter(np.array(full_strain_list)[ind1:ind2], full_stress_list[ind1:ind2], c=-np.log(1/sigma_ave[ind1:ind2]), s=big, marker='s')

# cycle 2
ind1 = ind2
ind2 = ind1 + 4
ax[1, 0].scatter(c2_plot_strain, c2_plot_stress, c=scale_0_1(c2_Z[:, 0]), s=small)
ax[1, 1].scatter(c2_plot_strain, c2_plot_stress, c=scale_0_1(-c2_Z[:, 1]), s=small)
ax[1, 2].scatter(c2_plot_strain, c2_plot_stress, c=scale_0_1(c2_Z[:, 2]), s=small)
ax[1, 0].scatter(full_strain_list[ind1:ind2], full_stress_list[ind1:ind2], c=scale_0_1(np.mean(val_to_plot, axis=1)[ind1:ind2]), s=big, marker='s')
ax[1, 1].scatter(full_strain_list[ind1:ind2], full_stress_list[ind1:ind2], c=scale_0_1(np.mean(val_to_plot, axis=1)[ind1:ind2]), s=big, marker='s')
ax[1, 2].scatter(np.array(full_strain_list)[ind1:ind2], full_stress_list[ind1:ind2], c=-np.log(1/sigma_ave[ind1:ind2]), s=big, marker='s')

# cycle 64
ind1 = ind2
ind2 = ind1 + 5
ax[2, 0].scatter(c64_plot_strain, c64_plot_stress, c=scale_0_1(c64_Z[:, 0]), s=small)
ax[2, 1].scatter(c64_plot_strain, c64_plot_stress, c=scale_0_1(-c64_Z[:, 1]), s=small)
ax[2, 2].scatter(c64_plot_strain, c64_plot_stress, c=scale_0_1(c64_Z[:, 2]), s=small)
ax[2, 0].scatter(full_strain_list[ind1:ind2], full_stress_list[ind1:ind2], c=scale_0_1(np.mean(val_to_plot, axis=1)[ind1:ind2]), s=big, marker='s')
ax[2, 1].scatter(full_strain_list[ind1:ind2], full_stress_list[ind1:ind2], c=scale_0_1(np.mean(val_to_plot, axis=1)[ind1:ind2]), s=big, marker='s')
ax[2, 2].scatter(np.array(full_strain_list)[ind1:ind2], full_stress_list[ind1:ind2], c=-np.log(1/sigma_ave[ind1:ind2]), s=big, marker='s')


for i in range(3):
    for j in range(3):
        ax[i,j].set_xlim([strain_lim[0], strain_lim[1]])
        ax[i,j].set_ylim([stress_lim[0], stress_lim[1]])

if save_fig:
    if transp:
        plt.savefig(temp_save_path + '%s_t.png' %base3, dpi=dpi, transparent=transp )
        plt.savefig(temp_save_path + '%s_t.svg' %base3, dpi=dpi, transparent=transp )
    else:
        plt.savefig(temp_save_path + '%s.png' %base3, dpi=dpi, transparent=transp )
        plt.savefig(temp_save_path + '%s.svg' %base3, dpi=dpi, transparent=transp )



plt.show()

#%%
temp_save_path = '/home/djs522/Downloads/2021_10_15_pca_fig/'
matplotlib.rcParams.update({'font.size':16})
size = 4.5
temp_fs = 10
stress_lim = [-550, 550]
strain_lim = [-0.6, 0.6]
transp = True
dpi=400

plot_micro = False
save_fig = False
base3 = 'c1_pc_3d_stress'


fig = plt.figure(figsize=(8,6))
ax = Axes3D(fig)

data = c1_Z
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=c1_plot_strain, s=150)
#ax.scatter(data[0, 0], data[0, 1], data[0, 2], c=c1_plot_strain[0], s=150, marker='d')
ax.set_xlabel('PC1', labelpad=15)
ax.set_ylabel('PC2', labelpad=15)
ax.set_zlabel('PC3', labelpad=15)

ax.set_xlim([-0.6, 0.6])
ax.set_ylim([-0.6, 0.6])
ax.set_zlim([-0.6, 0.6])

if save_fig:
    if transp:
        plt.savefig(temp_save_path + '%s_t.png' %base3, dpi=dpi, transparent=transp )
        plt.savefig(temp_save_path + '%s_t.svg' %base3, dpi=dpi, transparent=transp )
    else:
        plt.savefig(temp_save_path + '%s.png' %base3, dpi=dpi, transparent=transp )
        plt.savefig(temp_save_path + '%s.svg' %base3, dpi=dpi, transparent=transp )



#%%
size = 3
fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(size*5, size*3))

ind = np.arange(c1_plot_stress.size)
#ind = np.logical_xor((ind < 51), (ind > 81))
temp_stress = c1_plot_stress[ind]
temp_strain = c1_plot_strain[ind]
temp_z = c1_Z[ind, :]
temp_fit = c1_fits[ind, :]

temp_ind = (temp_stress < -50)
temp_diff = np.diff(temp_stress[temp_ind]) / np.diff(temp_strain[temp_ind])

# ax[0].scatter(temp_strain, temp_stress, c=temp_z[:, 0], s=small)
# ax[1].scatter(temp_z[:, 0], temp_stress, c=temp_z[:, 1], s=small)
# ax[2].scatter(temp_z[:, 0], temp_stress, c=temp_z[:, 2], s=small)


for i in range(3):
    ax[i, 0].scatter(temp_strain, temp_stress, c=temp_z[:, i], s=small)

ax[0, 1].scatter(temp_strain, temp_stress, c=temp_stress, s=small)
ax[1, 1].scatter(temp_strain, temp_stress, c=temp_strain, s=small)
ax[2, 1].scatter(temp_strain, temp_z[:, 0])

ax[0, 2].scatter(temp_strain, temp_stress, c=temp_fit[:, 1], s=small)
ax[1, 2].scatter(temp_strain, temp_stress, c=temp_fit[:, 2], s=small)
#ax[2, 2].scatter(temp_fit[:, 1], temp_z[:, 1])

ax[0, 3].scatter(temp_strain, temp_stress, c=temp_fit[:, 4], s=small)
ax[1, 3].scatter(temp_strain, temp_stress, c=-temp_fit[:, 5], s=small)
#ax[2, 3].scatter(temp_fit[:, 2], temp_z[:, 2])

ax[0, 4].scatter(temp_strain, temp_stress, c=temp_fit[:, 7], s=small)
ax[1, 4].scatter(temp_strain, temp_stress, c=temp_fit[:, 8], s=small)
#ax[2, 4].scatter(temp_fit[:, 5], temp_z[:, 2])

#ax[3].scatter(temp_z[:-1], temp_diff, c=temp_z[:-1], s=small)
#ax[3].scatter(temp_z[temp_ind][:-1], temp_diff, c=temp_z[temp_ind][:-1], s=small)
#ax[2].scatter(temp_z[:-1] - temp_z[1:], temp_strain[:-1] - temp_strain[1:], c=temp_stress[:-1] - temp_stress[1:], s=small)

plt.show()


#%%

from scipy.integrate import simps

c1_e_disp_actual = simps(c1_plot_stress, scale_0_1(c1_plot_strain))
c1_e_disp_pc1 = simps(c1_plot_stress, scale_0_1(c1_Z[:, 0]))

print(c1_e_disp_actual)
print(c1_e_disp_pc1)

print(c1_e_disp_actual - c1_e_disp_pc1)

c2_e_disp_actual = simps(c2_plot_stress, scale_0_1(c2_plot_strain))
c2_e_disp_pc1 = simps(c2_plot_stress, scale_0_1(c2_Z[:, 0]))

print(c2_e_disp_actual)
print(c2_e_disp_pc1)

print(c2_e_disp_actual - c2_e_disp_pc1)

c64_e_disp_actual = simps(c64_plot_stress, scale_0_1(c64_plot_strain))
c64_e_disp_pc1 = simps(c64_plot_stress, scale_0_1(c64_Z[:, 0]))

print(c64_e_disp_actual)
print(c64_e_disp_pc1)

print(c64_e_disp_actual - c64_e_disp_pc1)


#%% c1, c2, c64 model and fit
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(size*3, size*3))
fig.subplots_adjust(top=0.88,
bottom=0.11,
left=0.07,
right=0.95,
hspace=0.3,
wspace=0.3)

# cycle 1
ind1 = 4
ind2 = 13
temp_stress = c1_plot_stress
temp_strain = c1_plot_strain
temp_z = c1_Z[:, 0]
ax[0, 0].scatter(temp_strain, temp_stress, c=temp_z, s=small)
ax[0, 1].scatter(temp_z, temp_stress, c=temp_strain, s=small)
ax[0, 2].scatter(temp_z, temp_strain, c=temp_stress, s=small)
if plot_micro:
    ax[0, 0].scatter(full_strain_list[ind1:ind2], full_stress_list[ind1:ind2], c=scale_0_1(np.mean(val_to_plot, axis=1)[ind1:ind2]), s=big, marker='s')
    ax[0, 1].scatter(full_strain_list[ind1:ind2], full_stress_list[ind1:ind2], c=scale_0_1(np.mean(val_to_plot, axis=1)[ind1:ind2]), s=big, marker='s')
    ax[0, 2].scatter(np.array(full_strain_list)[ind1:ind2], full_stress_list[ind1:ind2], c=-np.log(1/sigma_ave[ind1:ind2]), s=big, marker='s')

# cycle 2
ind1 = ind2
ind2 = ind1 + 4
temp_stress = c2_plot_stress
temp_strain = c2_plot_strain
temp_z = c2_Z[:, 0]
ax[1, 0].scatter(temp_strain, temp_stress, c=temp_z, s=small)
ax[1, 1].scatter(temp_z, temp_stress, c=temp_strain, s=small)
ax[1, 2].scatter(temp_z, temp_strain, c=temp_stress, s=small)
if plot_micro:
    ax[1, 0].scatter(full_strain_list[ind1:ind2], full_stress_list[ind1:ind2], c=scale_0_1(np.mean(val_to_plot, axis=1)[ind1:ind2]), s=big, marker='s')
    ax[1, 1].scatter(full_strain_list[ind1:ind2], full_stress_list[ind1:ind2], c=scale_0_1(np.mean(val_to_plot, axis=1)[ind1:ind2]), s=big, marker='s')
    ax[1, 2].scatter(np.array(full_strain_list)[ind1:ind2], full_stress_list[ind1:ind2], c=-np.log(1/sigma_ave[ind1:ind2]), s=big, marker='s')

# cycle 64
ind1 = ind2
ind2 = ind1 + 5
temp_stress = c64_plot_stress
temp_strain = c64_plot_strain
temp_z = c64_Z[:, 0]
ax[2, 0].scatter(temp_strain, temp_stress, c=temp_z, s=small)
ax[2, 1].scatter(temp_z, temp_stress, c=temp_strain, s=small)
ax[2, 2].scatter(temp_z, temp_strain, c=temp_stress, s=small)
if plot_micro:
    ax[2, 0].scatter(full_strain_list[ind1:ind2], full_stress_list[ind1:ind2], c=scale_0_1(np.mean(val_to_plot, axis=1)[ind1:ind2]), s=big, marker='s')
    ax[2, 1].scatter(full_strain_list[ind1:ind2], full_stress_list[ind1:ind2], c=scale_0_1(np.mean(val_to_plot, axis=1)[ind1:ind2]), s=big, marker='s')
    ax[2, 2].scatter(np.array(full_strain_list)[ind1:ind2], full_stress_list[ind1:ind2], c=-np.log(1/sigma_ave[ind1:ind2]), s=big, marker='s')


# for i in range(3):
#   

  for j in range(3):
#         ax[i,j].set_xlim([strain_lim[0], strain_lim[1]])
#         ax[i,j].set_ylim([stress_lim[0], stress_lim[1]])

if save_fig:
    if transp:
        plt.savefig(temp_save_path + '%s_t.png' %base3, dpi=dpi, transparent=transp )
        plt.savefig(temp_save_path + '%s_t.svg' %base3, dpi=dpi, transparent=transp )
    else:
        plt.savefig(temp_save_path + '%s.png' %base3, dpi=dpi, transparent=transp )
        plt.savefig(temp_save_path + '%s.svg' %base3, dpi=dpi, transparent=transp )



plt.show()





#%% A little bit of CNN work

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.optimizers import SGD
from keras.models import Sequential
from keras.utils import np_utils as utils
from keras.layers import Dropout, Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D


#pca_image_list = pca_eta_ome_image_list # pca_raw_image_list, pca_eta_ome_image_list
#plot_stress = np.array(stress_list)[k:l]
#plot_strain = np.array(strain_list)[k:l] * 100
#plot_scan = np.array(scan_list)[k:l]
#plot_all_fits = np.array(all_fits)[k:l, :]

X_all = np.array(pca_eta_ome_image_list)
y_all = np.array(plot_stress)
y_all = ((y_all > 200) | (y_all < -200))

all_ind = np.random.permutation(X_all.shape[0])
test_cut = int(0.8 * X_all)

X = X_all[:test_cut, :, :]
y = y_all[:test_cut]

X_test = X_all[test_cut:, :, :]
y_test = y_all[test_cut:]

model = Sequential()

model.add(Conv2D(3, (3, 3), input_shape=(X.shape[1], X.shape[2]), padding='same', activation='relu'))

model.add(Dropout(0.2))

model.add(Conv2D(3, (3, 3), activation='relu', padding='valid'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(momentum=0.5, decay=0.0004), metrics=['accuracy'])

model.fit(X, y, validation_data=(X_test, y_test), epochs=25, batch_size=512)

print("Accuracy: &2.f%%" %(model.evaluate(X_test, y_test)[1]*100))




