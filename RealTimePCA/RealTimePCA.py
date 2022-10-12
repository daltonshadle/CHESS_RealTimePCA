#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 14:57:43 2022

@author: djs522
"""

# CONSTANT_NAMES = all uppercase with underscores
# FunctionNames() = camel case with parentheses and without underscores
# variable_names = all lowercase with underscores


# Notes
# + read in config file
# + work with multiple detectors (but only dexelas for now, frame-caches or raws)
# + work with plane data for rings check box when selecting ROI (eta angles)
# - some real-time data reading (updating path list)
# - better plotting, have a realtime aspect of PCA plots, maybe output to realtime text file, read to plot in sepearte script
# - work with DIC data, again realtime text file and plotting separate probably best
# - convolution of data before PCA to establish a basis of known behavior


#*****************************************************************************
#%% IMPORTS
import matplotlib.pyplot as plt

import os

import sys

import numpy as np

from sklearn import decomposition
from sklearn.preprocessing import Normalizer

sys.path.append(os.path.split(__file__)[0])
from RTPCA_Classes import pca_paths, pca_matrices
from RTPCA_Widgets import pca_parameters_selector_widget

#*****************************************************************************
#%% USER INPUT

script_base_dir = '/home/djs522/additional_sw/RealTimePCA/CHESS_RealTimePCA/example/'

sample_raw_stem = '/media/djs522/djs522_nov2020/chess_2020_11/ceo2/*/ff/%s*.h5' 
sample_raw_stem = '/home/djs522/additional_sw/RealTimePCA/CHESS_RealTimePCA/example/*%s*.npz'
sample_raw_dir = script_base_dir
sample_aux_dir = script_base_dir
output_fname = 'dictest_output.txt'
cfg_fname = os.path.join(script_base_dir, 'example_ff_config.yml')
first_img_dict={'panel_id': 'image.npz'}
is_frame_cache = True
det_img_roi_file = None

use_gui = True

#*****************************************************************************
#%% INITIALIZE OBJECTS
exp_pca_paths = pca_paths(base_dir=script_base_dir, img_dir=sample_raw_dir,
                          first_img_dict=first_img_dict,
                          is_frame_cache=is_frame_cache,
                          output_dir=sample_aux_dir, output_fname=output_fname,
                          config_fname=cfg_fname)

exp_pca_mats = pca_matrices(det_keys=exp_pca_paths.config.instrument.hedm.detectors.keys())

#*****************************************************************************
#%% SELECT FIRST IMAGE / REFERENCE IMAGE
if use_gui:
    exp_pca_paths.open_first_image()

#*****************************************************************************
#%% SET ROI AND PCA PARAMTERS
if use_gui:
    ppsw = pca_parameters_selector_widget(exp_pca_paths, exp_pca_mats, adjust_grid=True)
    [exp_pca_paths, exp_pca_mats] = ppsw.get_all_pca_objects()
else:
    if det_img_roi_file is not None:
        exp_pca_paths.load(det_img_roi_file)
        exp_pca_mats.load(det_img_roi_file)
    else:
        raise ValueError("ROI file could not be loaded. Use GUI to select ROI or fix ROI file path.")
        
#*****************************************************************************
#%% READ IN THE CURRENT IMAGES

img_path_list_dict = exp_pca_paths.get_all_image_paths_dict(image_path_stem=sample_raw_stem)

img_mask_dict = exp_pca_mats.make_det_image_mask(exp_pca_paths.config.instrument.hedm.detectors) # this is a hard coded item, should be fixed by passing pca_paths for config

img_data = exp_pca_mats.load_img_list(img_path_list_dict, img_mask_dict=img_mask_dict, ims_length=2, 
                                      is_frame_cache=exp_pca_paths.is_frame_cache, 
                                      frane_num_or_img_aggregation_options=None)

#*****************************************************************************
#%% ASSEMBLE THE PCA MATRIX
pca_matrix = exp_pca_mats.assemble_pca_matrix()

#*****************************************************************************
#%% CHECK ROI DATA
if use_gui:
    exp_pca_mats.plot_reassemble_image_frame_from_roi(frame_num=0)

#*****************************************************************************
#%% FIT AND TRANSFORM PCA MATRIX
#num_cmpts = input("How many principle components would you like to analyze? Choose from 1 to 5 principle components.\n")
#num_cmpts = int(num_cmpts)
num_cmpts = 3
while num_cmpts > 5 or num_cmpts < 1:
    num_cmpts = input("Choose from 1 to 5 principle components.\n")
    num_cmpts = int(num_cmpts)
PCA_func = decomposition.PCA(n_components=num_cmpts)
transformer = Normalizer().fit(exp_pca_mats.pca_matrix)
exp_pca_mats.pca_matrix = transformer.transform(exp_pca_mats.pca_matrix)
PCs = PCA_func.fit_transform(exp_pca_mats.pca_matrix)
var_ratio = PCA_func.explained_variance_ratio_

# plot results
for index in range(num_cmpts):
    fig = plt.figure()
    plt.plot(PCs[:, index])
    plt.xlabel('image numbers')
    pc = "Principle Component {}".format(index+1)
    plt.ylabel(pc)

fig = plt.figure()
plt.scatter(PCs[:, 0], PCs[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.show()