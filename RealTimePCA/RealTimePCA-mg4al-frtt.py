#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 14:57:43 2022

@author: djs522
"""

# CONSTANT_NAMES = all uppercase with underscores
# FunctionNames() = camel case with parentheses and without underscores
# variable_names = all lowercase with underscores


# TODO Notes
# + read in config file
# + work with multiple detectors (but only dexelas for now, frame-caches or raws)
# + work with plane data for rings check box when selecting ROI (eta angles)
# + some real-time data reading (updating path list)
# + shore up objects by including everything needed for non-gui, update load and save methods
# - outputting data
# + better masking of images
# - commenting!
# - better plotting, have a realtime aspect of PCA plots, maybe output to realtime text file, read to plot in sepearte script
# - work with DIC data, again realtime text file and plotting separate probably best
# - convolution of data before PCA to establish a basis of known behavior


#*****************************************************************************
#%% IMPORTS
import matplotlib.pyplot as plt

import os

import sys

import numpy as np

from hexrd import config

from sklearn import decomposition
from sklearn.preprocessing import Normalizer

sys.path.append(os.path.split(__file__)[0])
from RTPCA_Classes import lodi_experiment
from RTPCA_Widgets import pca_parameters_selector_widget

#*****************************************************************************
#%% USER INPUT

# NECESSARY INPUTS ***********************************************************

# sample_raw_stem is used to find the detector images of a LODI measurement
#  - use '%i' in place of scan numbers
#  - use '%s' in place of detector id in filename
# CHESS EXPERIMENT
beamtime_cycle = '2022-3'
beamline_id = 'id1a3'
exp_name = 'nygren-3527-a'
sample_name = 'mg4al-sht-2022-ff-1-frtt'
sample_raw_stem = '/nfs/chess/raw/%s/%s/%s/%s' %(beamtime_cycle, beamline_id, exp_name, sample_name)  + '/%i/ff/%s*.h5' 
sample_raw_stem = '/nfs/chess/raw/%s/%s/%s/%s' %(beamtime_cycle, beamline_id, exp_name, sample_name)  + '/%i/ff/%s_%06i.h5' 

# OPTIONAL INPUTS ************************************************************
# All other inputs can either be declared to an existing path or set to None
# If set to None, a GUI window will prompt users to point to directory or file

# sample_aux_dir is used as the output directory for results
sample_aux_dir = '/nfs/chess/user/djs522/rtpca_dec_2022/CHESS_RealTimePCA/nygren-3527-a/'

# cfg_name is used to define the location of a hexrd config file with roughly
# calibrated detector (CeO2) and materials file
cfg_fname = os.path.join(sample_aux_dir, 'mg4al_LODI_config.yml')

# pair_lodi_with_dic is a flag used to define whether RealTimePCA should use RealTimeDIC
# stress and strain information to help visualize PCA data; True = Piared, False = Not Paired
# if False, dic_output_txt_fname and dic_output_json_fname will be ignored
pair_lodi_with_dic = True

# dic_output_json_fname is the full path to the json file indexing the dic_output_file from RealTimeDIC
dic_output_json_fname = os.path.join(sample_aux_dir, 'dic_output.json') #'/nfs/chess/user/djs522/rtpca_dec_2022/CHESS_RealTimePCA/wiley_dec_2022/'

# dic_output_txt_fname is the full path to the dic_output_file from RealTimeDIC
#dic_output_txt_fname = os.path.join(sample_aux_dir, 'ff-c103-90-s1-2_dic_output.txt')
dic_output_txt_fname = '/nfs/chess/aux/cycles/2022-3/id1a3/nygren-3527-a/%s/%s_dic.txt' %(sample_name, sample_name)#os.path.join(sample_aux_dir, 'c103-90-s2-2_dic.txt')

# dic_output_cols is a list of column indices used to index dic_output file
dic_output_cols = [1, 3, 6]

# lodi_json_fname is the full path to the json file indexing the lodi_par_fname
#lodi_json_fname = '/nfs/chess/raw/2022-3/id1a3/miller-3528-a/ff-c103-90-s1-2/id1a3-rams4_lodi-dexela-ff-c103-90-s1-2.json'
lodi_json_fname = '/nfs/chess/raw/2022-3/id1a3/nygren-3527-a/%s/id1a3-rams4_lodi-dexela-%s.json'  %(sample_name, sample_name)

# lodi_par_fname is the full path to the par file for this LODI experiment
#lodi_par_fname = '/nfs/chess/raw/2022-3/id1a3/miller-3528-a/ff-c103-90-s1-2/id1a3-rams4_lodi-dexela-ff-c103-90-s1-2.par' 
lodi_par_fname = '/nfs/chess/raw/2022-3/id1a3/nygren-3527-a/%s/id1a3-rams4_lodi-dexela-%s.par'  %(sample_name, sample_name)

# lodi_par_cols is a list of column indices used to index lodi par file
lodi_par_cols = [3, 22, 12, 13, 4, 5]
lodi_par_cols = [3, 22, 12, 13, 5]

# first_img_dict is a dictionary keyed by detector keys with paths to first 
# detector images of LODI measurements
#first_img_dict = {'ff1': '/nfs/chess/raw/2022-3/id1a3/miller-3528-a/ff-c103-90-s1-2/1/ff/ff1_000204.h5',
#                  'ff2': '/nfs/chess/raw/2022-3/id1a3/miller-3528-a/ff-c103-90-s1-2/1/ff/ff2_000204.h5'}
first_img_dict = {'ff1': '/nfs/chess/raw/2022-3/id1a3/nygren-3527-a/mg4al_test-1/12/ff/ff1_000740.h5',
                  'ff2': '/nfs/chess/raw/2022-3/id1a3/nygren-3527-a/mg4al_test-1/12/ff/ff2_000740.h5'}
first_img_dict = {'ff2': '/nfs/chess/raw/2022-3/id1a3/nygren-3527-a/%s/19/ff/ff2_000972.h5' %(sample_name)}
start_loadstep_num = 506

# img_process_dict is a dictionary for any image processing that needs to be donw
# i.e. fliiping dexela raw images hortizontal or vertical
img_process_dict = {'ff1': [('flip', 'v')],
                    'ff2': [('flip', 'h')]}
img_process_dict = {'ff2': [('flip', 'h')]}

# img_rings_eta_dict is a dictionary of eta ranges (in degrees) for using ring masks
img_ring_eta_dict = {'ff1': [],
                     'ff2': []}
img_ring_eta_dict = {'ff2': []}

# mask_dict_file is the path to the detector image mask file for selecting
# data for analysis from images
mask_dict_file = None #os.path.join(sample_aux_dir, 'example_mask_file')

#*****************************************************************************
#%% INITIALIZE OBJECTS AND SET UP RAW STEM

pca_exp = lodi_experiment(img_stem=sample_raw_stem, 
                          ring_eta_dict=img_ring_eta_dict, 
                          pair_lodi_with_dic=pair_lodi_with_dic)

#*****************************************************************************
#%% SELECT OUTPUT / AUX DIR

pca_exp.open_output_dir(output_dir=sample_aux_dir)

#*****************************************************************************
#%% SELECT CONFIG FILE

pca_exp.open_config_from_file(config_dir=cfg_fname)

#*****************************************************************************
#%% SELECT DIC OUTPUT FILE

pca_exp.open_dic_output_file(dic_output_json_dir=dic_output_json_fname, 
                             dic_output_txt_dir=dic_output_txt_fname,
                             dic_output_cols=dic_output_cols)

#*****************************************************************************
#%% SELECT LODI PAR FILE

pca_exp.open_lodi_par_file(lodi_json_dir=lodi_json_fname, 
                           lodi_par_dir=lodi_par_fname,
                           lodi_par_cols=lodi_par_cols)

#*****************************************************************************
#%% SELECT FIRST IMAGE / REFERENCE IMAGE

pca_exp.open_first_image(first_img_dict=first_img_dict)

#*****************************************************************************
#%% SET ROI AND PCA PARAMTERS

if mask_dict_file is not None:
    pca_exp.load_mask_dict_from_file(mask_dict_file)
else:
    ppsw = pca_parameters_selector_widget(pca_exp, vmax=1000, img_process_dict=img_process_dict)
    [pca_exp] = ppsw.get_all_pca_objects()

#*****************************************************************************
#%% READ IN CURRENT IMAGES
# start_loadstep_num = 506 start of loading
# start_loadstep_num = 542 tension tip on init load
# might ignore ls [564] as it looked vastly different (dropped / scrambled frame)
start_loadstep_num = 506
ignore_loadstep_list = [564]
pca_exp.overwrite_img_list_new(img_process_dict=img_process_dict, 
                               frane_num_or_img_aggregation_options=[5, 7, 9, 11],
                               start_loadstep_num=start_loadstep_num,
                               ignore_loadstep_list=ignore_loadstep_list)

#%%
num_cmpts = 3
PCA_func = decomposition.PCA(n_components=num_cmpts)

#*****************************************************************************
#%% UPDATE IMAGE LIST

pca_exp.update_img_list_new(img_process_dict=img_process_dict, 
                            frane_num_or_img_aggregation_options=[5, 7, 9, 11],
                            start_loadstep_num=start_loadstep_num,
                            ignore_loadstep_list=ignore_loadstep_list)

#*****************************************************************************
# ASSEMBLE THE PCA MATRIX

pca_matrix = pca_exp.assemble_data_matrix()

#*****************************************************************************
#% FIT AND TRANSFORM PCA MATRIX


transformer = Normalizer().fit(pca_matrix)
pca_matrix = transformer.transform(pca_matrix)
#PCs = PCA_func.fit_transform(pca_matrix)
#curr_fit = PCA_func.fit(pca_matrix)
PCs = PCA_func.transform(pca_matrix)
var_ratio = PCA_func.explained_variance_ratio_
print(var_ratio)

n_pts = pca_exp.dic_output_data.shape[0]
n = -10
m = -100
alpha = 0.6
base_size = 30
max_size = 90
grow_size = np.linspace(base_size, max_size, num=np.abs(n))
all_size = np.hstack([np.ones(n_pts + n) * base_size, grow_size])
alphas = np.hstack([np.ones(n_pts + n) * alpha, np.ones(grow_size.shape)])

# plot results
plt.close(1)
fig, ax = plt.subplots(nrows=2, ncols=num_cmpts, figsize=(15, 8), num=1)
for i in range(num_cmpts):
    pc_plot = ax[0, i].scatter(pca_exp.dic_output_data[:, 1], pca_exp.dic_output_data[:, 0], 
                     c=PCs[:, i], alpha=alphas, edgecolor='none', s=all_size)
    ax[0, i].set_xlabel('LD Strain')
    ax[0, i].set_ylabel('LD Stress (MPa)')
    ax[0, i].set_title('PC %i' %(i+1))
    #fig.colorbar(pc_plot, ax=ax[0, i])
    
    ax[1, i].scatter(pca_exp.dic_output_data[:n, 2], PCs[:n, i], c='b', 
                     s=base_size, alpha=alpha, edgecolor='none')
    ax[1, i].scatter(pca_exp.dic_output_data[n:, 2], PCs[n:, i], c='b', 
                     s=grow_size, edgecolor='none')
    ax[1, i].scatter(pca_exp.dic_output_data[-1, 2], PCs[-1, i], c='r', 
                     s=base_size, edgecolor='none')
    ax[1, i].set_xlabel('Load Step Num')
    ax[1, i].set_ylabel('PC %i' %(i+1))
    #ax[1, i].set_title('PC %i' %(i+1))

plt.show()

#*****************************************************************************
#%% CHECK ROI DATA
pca_exp.plot_reassemble_image_frame_from_roi(frame_num=0)

#*****************************************************************************
#%% SAVE LODI EXPERIMENT
pca_exp.save_lodi_exp_to_file()
pca_exp.load_lodi_exp_from_file()