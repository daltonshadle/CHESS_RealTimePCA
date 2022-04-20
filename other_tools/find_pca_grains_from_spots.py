#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:29:26 2021

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

def scale_0_1(data, n1_p1=True):
    if n1_p1:
        return 2*(data - data.min()) / (data.max() - data.min()) - 1.0
    else:
        return (data - data.min()) / (data.max() - data.min())
        



#%% ***************************************************************************
# CONSTANTS AND PARAMETERS

ss718 = True

if ss718:
    temp_dir = '/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/'
    raw_data_dir_template = temp_dir + 'new_ss718_ct_thresh360/'
    raw_ff_data_dir_template = temp_dir + 'ff/'
    sample_name = 'ss718-1'
    
    snap_img_base = 'ss718-1_33_%s_000027-cachefile.npz'
    ff_img_base = 'ss718-1_30_%s_000024-cachefile.npz'
    
else:
    temp_dir = '/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/'
    raw_data_dir_template = temp_dir + 'new_dp718_ct_thresh360/'
    raw_ff_data_dir_template = temp_dir + 'ff/'
    sample_name = 'dp718-1'
    
    snap_img_base = 'dp718-1_22_%s_001340-cachefile.npz'
    ff_img_base = 'dp718-1_19_%s_001337-cachefile.npz'

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


#%%

snap_scan_im_series = imageseries.open(raw_data_dir_template+snap_img_base%(key), raw_fmt, path=raw_path)
ff_scan_im_series = imageseries.open(raw_ff_data_dir_template+ff_img_base%(key), raw_fmt, path=raw_path)

    
fig = plt.figure()
temp_p = 0
plt.imshow(snap_scan_im_series[0], vmax=temp_p+1, vmin=temp_p)
plt.show()

#%%

snap_omega_deg = 344.868
snap_omega_rad = snap_omega_deg * np.pi / 180.0
snap_img_num = 1379

buffer = 0#0.125
ff_img_omegas = ff_scan_im_series.metadata['omega']
ff_snap_omegas = ff_img_omegas[snap_img_num]
ff_snap_omegas[0] -= buffer
ff_snap_omegas[1] += buffer

if ss718:
    ff1_spots_out_dir = '/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/c0_0_gripped/c0_0_gripped_sc30/ff1/'
    ff2_spots_out_dir = '/media/djs522/djs522_nov2020/chess_2020_11/ss718-1/c0_0_gripped/c0_0_gripped_sc30/ff2/'
    num_grains = 2848
else:
    ff1_spots_out_dir = '/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/c0_0_gripped/c0_0_gripped_sc19_nf/ff1/'
    ff2_spots_out_dir = '/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/c0_0_gripped/c0_0_gripped_sc19_nf/ff2/'
    num_grains = 2125


col = 12 # measured omega
snap_grains = []
for i in range(num_grains):
    ff1_spot = np.loadtxt(ff1_spots_out_dir + 'spots_%0*d.out' % (5, i))
    ff2_spot = np.loadtxt(ff2_spots_out_dir + 'spots_%0*d.out' % (5, i))
    
    ff1_omegas = ff1_spot[:, col] * 180.0 / np.pi
    ff2_omegas = ff2_spot[:, col] * 180.0 / np.pi
    
    if np.any((ff1_omegas >= ff_snap_omegas[0]) & (ff1_omegas <= ff_snap_omegas[1])):
        snap_grains.append(i)
    elif np.any((ff2_omegas >= ff_snap_omegas[0]) & (ff2_omegas <= ff_snap_omegas[1])):
        snap_grains.append(i)

print(len(snap_grains))
#np.save(temp_dir+'%s_pca_grains_from_spots_buffer%0.3f.npy' %(sample_name, buffer), np.array(snap_grains))

#%%


load_step_list = ['c0_0_gripped', 'c0_1', 'c0_2', 'c0_3', 'c1_0', 'c1_1',
                  'c1_2', 'c1_3', 'c1_4', 'c1_5', 'c1_6', 'c1_7', 'c2_0',
                  'c2_3', 'c2_4', 'c2_7', 'c3_0', 'c64_0', 'c64_3', 'c64_4', 'c64_7' ,'c65_0']
full_scan_list = [30, 50, 68, 90, 111, 136, 173, 198, 216, 234, 260, 280, 300, 337, 356, 387, 405, 1012, 1033, 1043, 1064, 1077]
full_scan_list = [30, 50, 68, 90, 111, 136, 173, 198, 216, 234, 260, 280, 300]
full_stress_list = [0, 150, 247, 283, 300, 0, -263, -310, -327, 0, 264, 314, 326, -328, -336, 325, 333, 445, -390, -456, 382, 442]
full_strain_list = [0, 0.08, 0.13, 0.25, 0.51, 0.37, 0, -0.25, -0.51, -0.35, 0, 0.26, 0.51, -0.25, -0.5, 0.26, 0.5, 0.5, -0.25, -0.5, 0.25, 0.5]
scan_stem = '/%s_sc%i/grains.out'

comp_thresh = 0.8
chi2_thresh = 1.0e-2

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

val_to_plot = []
val_to_plot2 = []
ind_to_plot = [16]
ind_to_plot2 = [15,17]

for i, scan in enumerate(full_scan_list):
    temp_grain_mat = np.loadtxt(temp_dir + load_step_list[i] + scan_stem %(load_step_list[i], scan))
    pca_grain_ind = np.searchsorted(temp_grain_mat[:, 0], new_pca_grains)
    temp_grain_mat = temp_grain_mat[pca_grain_ind, :]
    temp_grain_mat[:, 15:] = temp_grain_mat[:, 15:] - first_grain_mat[:, 15:]
    
    val_to_plot.append(temp_grain_mat[:, ind_to_plot].flatten())
    val_to_plot2.append(np.mean(temp_grain_mat[:, ind_to_plot2], axis=1).flatten())
    
val_to_plot = np.array(val_to_plot)
val_to_plot = scale_0_1(val_to_plot)
val_to_plot2 = np.array(val_to_plot2)
val_to_plot2 = scale_0_1(val_to_plot2)


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
        [grain_quat, grain_mis_quat, grain_odf] = pp_GOE.process_dsgod_file(dsgod_npz_dir, 
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

k = 4
fig = plt.figure()
plt.plot(scale_0_1(sigma_ave[k:]))
plt.plot(scale_0_1(gamma_ave[k:]))
plt.plot(scale_0_1(kappa_ave[k:]))
plt.plot(scale_0_1(mis_ave[k:]))
plt.plot(scale_0_1(reori_ave[k:]))
plt.legend(['sigma', 'gamma', 'kappa', 'misori', 'reori'])
plt.show()

#%%

# plot_stress = np.array(stress_list)[k:l]
# plot_strain = np.array(strain_list)[k:l] * 100
# plot_scan = np.array(scan_list)[k:l]

# Z[:, 0] = scale_0_1(Z[:, 0])
# Z[:, 1] = scale_0_1(Z[:, 1])
# Z[:, 2] = scale_0_1(Z[:, 2])

plot_against = full_scan_list # full_scan_list, full_stress_list, full_strain_list
plot_against_snap = plot_scan # plot_scan, plot_stress, plot_strain

ind1 = 0#13#5
ind2 = ind1+13#ind1+8

d = 0
e = -1#-100#-18
lw_big = 4
lw_small = 2

stress_offset = 0
strain_offset = 1
ds_de = plot_stress[d:e]#((plot_stress[d:e] + stress_offset) / (plot_strain[d:e] + strain_offset))
stress_offset = 800
strain_offset = 0
de_ds = plot_strain[d:e]#((plot_strain[d:e] + strain_offset) / (plot_stress[d:e] + stress_offset))

#de_ds =((plot_strain[d+1:e+1] - plot_strain[d:e]) * (plot_stress[d+1:e+1] - plot_stress[d:e]))


fig, ax = plt.subplots(figsize=(10, 6))
fig.subplots_adjust(right=0.75)
twin1 = ax.twinx()
twin2 = ax.twinx()
twin2.spines['right'].set_position(("axes", 1.2))
p1, = ax.plot(plot_against_snap[d:e], plot_stress[d:e], '--r', linewidth=lw_small, label='$\sigma_{Macro LD}$')
p2, = twin1.plot(plot_against_snap[d:e], plot_strain[d:e], '--g', linewidth=lw_small, label='$\epsilon_{Macro LD}$')
p3, = twin2.plot(plot_against_snap[d:e], -scale_0_1(Z[d:e, 0]), '-.b', linewidth=lw_big, label='PC1')
#ax.set_xlim(0, 2)
# ax.set_ylim(-350, 350)
# twin1.set_ylim(-0.55, 0.55)
# twin2.set_ylim(-1.1, 1.1)
ax.set_xlabel("Scans")
ax.set_ylabel("Stress (MPa)")
twin1.set_ylabel("Strain (%)")
twin2.set_ylabel("PC1 (Unitless)")
ax.yaxis.label.set_color(p1.get_color())
twin1.yaxis.label.set_color(p2.get_color())
twin2.yaxis.label.set_color(p3.get_color())
tkw = dict(size=4, width=1.5)
ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
ax.tick_params(axis='x', **tkw) 
ax.legend(handles=[p1, p2, p3])


fig, ax = plt.subplots(figsize=(10, 6))
fig.subplots_adjust(right=0.75)
twin1 = ax.twinx()
twin2 = ax.twinx()
twin2.spines['right'].set_position(("axes", 1.2))
p1, = ax.plot(plot_against_snap[d:e], ds_de, '--r', linewidth=lw_small, label='$\sigma_{Macro LD}$')
p2, = twin1.plot(plot_against_snap[d:e], de_ds, '--g', linewidth=lw_small, label='$\epsilon_{Macro LD}$')
p3, = twin2.plot(plot_against_snap[d:e], -scale_0_1(Z[d:e, 1]), '-.b', linewidth=lw_big, label='PC2')
#ax.set_xlim(0, 2)
# ax.set_ylim(650, 1600)
# twin1.set_ylim(-0.0015, -0.00055)
# twin2.set_ylim(-1.1, 1.1)
ax.set_xlabel("Scans")
ax.set_ylabel("Stress (MPa)")
twin1.set_ylabel("Strain (%)")
twin2.set_ylabel("PC2 (Unitless)")
ax.yaxis.label.set_color(p1.get_color())
twin1.yaxis.label.set_color(p2.get_color())
twin2.yaxis.label.set_color(p3.get_color())
tkw = dict(size=4, width=1.5)
ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
ax.tick_params(axis='x', **tkw)
ax.legend(handles=[p1, p2, p3])


fig, ax = plt.subplots(figsize=(10, 6))
fig.subplots_adjust(right=0.75)
twin1 = ax.twinx()
twin2 = ax.twinx()
twin2.spines['right'].set_position(("axes", 1.2))
# p1, = ax.plot(plot_against[ind1:ind2], mis_ave[ind1:ind2], '--r', linewidth=lw_small, label='Average Grain Misorientation')
# p2, = twin1.plot(plot_against[ind1:ind2], reori_ave[ind1:ind2], '--g', linewidth=lw_small, label='Average Grain $|\Sigma|$')
p1, = ax.plot(plot_against_snap[d:e], ds_de, '--r', linewidth=lw_small, label='$\sigma_{Macro LD}$')
p2, = twin1.plot(plot_against_snap[d:e], de_ds, '--g', linewidth=lw_small, label='$\epsilon_{Macro LD}$')
p3, = twin2.plot(plot_against_snap[d:e], -scale_0_1(Z[d:e, 2]), '-.b', linewidth=lw_big, label='PC3')
#ax.set_xlim(0, 2)
#ax.set_ylim(650, 1600)
#twin1.set_ylim(-0.0015, -0.00055)
#twin2.set_ylim(-1.1, 1.1)
ax.set_xlabel("Scans")
# ax.set_ylabel("Misorientation (°)")
# twin1.set_ylabel("$|\Sigma|$")
ax.set_ylabel("Stress (MPa)")
twin1.set_ylabel("Strain (%)")
twin2.set_ylabel("PC3 (Unitless)")
ax.yaxis.label.set_color(p1.get_color())
twin1.yaxis.label.set_color(p2.get_color())
twin2.yaxis.label.set_color(p3.get_color())
tkw = dict(size=4, width=1.5)
ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
ax.tick_params(axis='x', **tkw)
ax.legend(handles=[p1, p2, p3])


fig, ax = plt.subplots(figsize=(10, 6))
fig.subplots_adjust(right=0.75)
twin1 = ax.twinx()
twin2 = ax.twinx()
twin2.spines['right'].set_position(("axes", 1.2))
p1, = ax.plot(plot_against_snap[d:e], -scale_0_1(Z[d:e, 0]), '--r', linewidth=lw_big, label='PC1')
p2, = twin1.plot(plot_against_snap[d:e], -scale_0_1(Z[d:e, 1]), '--g', linewidth=lw_big, label='PC2')
p3, = twin2.plot(plot_against_snap[d:e], -scale_0_1(Z[d:e, 2]), '--b', linewidth=lw_big, label='PC3')
#ax.set_xlim(0, 2)
#ax.set_ylim(650, 1600)
#twin1.set_ylim(-0.0015, -0.00055)
#twin2.set_ylim(-1.1, 1.1)
ax.set_xlabel("Scans")
ax.set_ylabel("PC1 (Unitless)")
twin1.set_ylabel("PC2 (Unitless)")
twin2.set_ylabel("PC3 (Unitless)")
ax.yaxis.label.set_color(p1.get_color())
twin1.yaxis.label.set_color(p2.get_color())
twin2.yaxis.label.set_color(p3.get_color())
tkw = dict(size=4, width=1.5)
ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
ax.tick_params(axis='x', **tkw)
ax.legend(handles=[p1, p2, p3])



big = 200
small = 20


stress_lim = [-500, 500]
strain_lim = [-0.6, 0.6]
size = 5.0


#%%

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(size*3, size))
fig.subplots_adjust(top=0.88,
bottom=0.11,
left=0.07,
right=0.95,
hspace=0.2,
wspace=0.2)
ax[0].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(Z[d:e, 0]), s=small)
ax[1].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(plot_stress[d:e]), s=small)
ax[2].scatter(plot_strain[d:e], plot_stress[d:e], c=scale_0_1(plot_strain[d:e]), s=small)
#ax[0].scatter(full_strain_list[ind1:ind2], full_stress_list[ind1:ind2], c=scale_0_1(np.mean(val_to_plot, axis=1)[ind1:ind2]), s=big, marker='s')
#ax[1].scatter(full_strain_list[ind1:ind2], full_stress_list[ind1:ind2], c=scale_0_1(np.mean(val_to_plot, axis=1)[ind1:ind2]), s=big, marker='s')
#ax[2].scatter(full_strain_list[ind1:ind2], full_stress_list[ind1:ind2], c=scale_0_1(np.mean(val_to_plot, axis=1)[ind1:ind2]), s=big, marker='s')
ax[0].legend(['PC1', 'Grain LD Elastic Strain'], loc='lower right')
ax[0].set_xlabel('Macroscopic LD Strain (%)')
ax[0].set_ylabel('Macroscopic LD Stress (MPa)')
ax[1].legend(['Macroscopic LD Stress', 'Grain LD Elastic Strain'], loc='lower right')
ax[1].set_xlabel('Macroscopic LD Strain (%)')
ax[1].set_ylabel('Macroscopic LD Stress (MPa)')
ax[2].legend(['Macroscopic LD Strain ', 'Grain LD Elastic Strain'], loc='lower right')
ax[2].set_xlabel('Macroscopic LD Strain (%)')
ax[2].set_ylabel('Macroscopic LD Stress (MPa)')
ax[0].set_xlim([strain_lim[0], strain_lim[1]])
ax[0].set_ylim([stress_lim[0], stress_lim[1]])
ax[1].set_xlim([strain_lim[0], strain_lim[1]])
ax[1].set_ylim([stress_lim[0], stress_lim[1]])
ax[2].set_xlim([strain_lim[0], strain_lim[1]])
ax[2].set_ylim([stress_lim[0], stress_lim[1]]) 


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(size*3, size))
fig.subplots_adjust(top=0.88,
bottom=0.11,
left=0.07,
right=0.95,
hspace=0.2,
wspace=0.2)
ax[0].scatter(plot_strain[d:e], plot_stress[d:e], c=-Z[d:e, 1]+1, s=small)
ax[1].scatter(plot_strain[d:e], plot_stress[d:e], c=ds_de, s=small)
ax[2].scatter(plot_strain[d:e], plot_stress[d:e], c=de_ds, s=small)
#ax[0].scatter(full_strain_list[ind1:ind2], full_stress_list[ind1:ind2], c=scale_0_1(np.std(val_to_plot, axis=1)[ind1:ind2]), s=big, marker='s')
#ax[1].scatter(full_strain_list[ind1:ind2], full_stress_list[ind1:ind2], c=scale_0_1(np.std(val_to_plot, axis=1)[ind1:ind2]), s=big, marker='s')
#ax[2].scatter(full_strain_list[ind1:ind2], full_stress_list[ind1:ind2], c=scale_0_1(np.std(val_to_plot, axis=1)[ind1:ind2]), s=big, marker='s')
ax[0].legend(['PC2', 'Grain LD Elastic Strain'], loc='lower right')
ax[0].set_xlabel('Macroscopic LD Strain (%)')
ax[0].set_ylabel('Macroscopic LD Stress (MPa)')
ax[1].legend(['$(\sigma_{Macro LD} + B)/(\epsilon_{Macro LD} + C)$', 'Grain LD Elastic Strain'], loc='lower right')
ax[1].set_xlabel('Macroscopic LD Strain (%)')
ax[1].set_ylabel('Macroscopic LD Stress (MPa)')
ax[2].legend(['$-(\epsilon_{Macro LD} + B)/(\sigma _{Macro LD}+ C)$', 'Grain LD Elastic Strain'], loc='lower right')
ax[2].set_xlabel('Macroscopic LD Strain (%)')
ax[2].set_ylabel('Macroscopic LD Stress (MPa)')
ax[0].set_xlim([strain_lim[0], strain_lim[1]])
ax[0].set_ylim([stress_lim[0], stress_lim[1]])
ax[1].set_xlim([strain_lim[0], strain_lim[1]])
ax[1].set_ylim([stress_lim[0], stress_lim[1]])
ax[2].set_xlim([strain_lim[0], strain_lim[1]])
ax[2].set_ylim([stress_lim[0], stress_lim[1]])


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(size*3, size))
fig.subplots_adjust(top=0.88,
bottom=0.11,
left=0.07,
right=0.95,
hspace=0.2,
wspace=0.2)
ax[0].scatter(plot_strain[d:e], plot_stress[d:e], c=Z[d:e, 2], s=small)
ax[1].scatter(plot_strain[d:e], plot_stress[d:e], c=Z[d:e, 2], s=small)
ax[2].scatter(plot_strain[d:e], plot_stress[d:e], c=Z[d:e, 2], s=small)
#ax[0].scatter(np.array(full_strain_list)[ind1:ind2], full_stress_list[ind1:ind2], c=-np.log(1/sigma_ave[ind1:ind2]), s=big, marker='s')
#ax[0].scatter(np.array(full_strain_list)[ind1:ind2], full_stress_list[ind1:ind2], c=-scale_0_1(mis_ave[ind1:ind2]), s=big, marker='s')
#ax[1].scatter(np.array(full_strain_list)[ind1:ind2], full_stress_list[ind1:ind2], c=-np.log(gamma_ave[ind1:ind2]), s=big, marker='s')
#ax[2].scatter(np.array(full_strain_list)[ind1:ind2], full_stress_list[ind1:ind2], c=-np.log(kappa_ave[ind1:ind2]), s=big, marker='s')
ax[0].legend(['PC3', 'Average $|\Sigma|$'], loc='lower right')
ax[1].legend(['PC3', 'Average $|\Gamma|$'], loc='lower right')
ax[2].legend(['PC3', 'Average $|K|$'], loc='lower right')
ax[0].set_xlabel('Macroscopic LD Strain (%)')
ax[0].set_ylabel('Macroscopic LD Stress (MPa)')
ax[1].set_xlabel('Macroscopic LD Strain (%)')
ax[1].set_ylabel('Macroscopic LD Stress (MPa)')
ax[2].set_xlabel('Macroscopic LD Strain (%)')
ax[2].set_ylabel('Macroscopic LD Stress (MPa)')
ax[0].set_xlim([strain_lim[0], strain_lim[1]])
ax[0].set_ylim([stress_lim[0], stress_lim[1]])
ax[1].set_xlim([strain_lim[0], strain_lim[1]])
ax[1].set_ylim([stress_lim[0], stress_lim[1]])
ax[2].set_xlim([strain_lim[0], strain_lim[1]])
ax[2].set_ylim([stress_lim[0], stress_lim[1]])


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
#     for j in range(3):
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

