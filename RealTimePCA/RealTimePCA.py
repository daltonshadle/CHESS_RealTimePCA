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
# - read in config file
# - work with multiple detectors (but only dexelas for now, frame-caches or raws)
# - work with plane data for rings check box when selecting ROI (eta angles)
# - some real-time data reading
# - better plotting, have a realtime aspect of PCA plots, maybe output to realtime text file, read to plot in sepearte script
# - work with DIC data, again realtime text file and plotting separate probably best
# - convolution of data before PCA to establish a basis of known behavior


#*****************************************************************************
#%% IMPORTS
import tkinter as tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import os
import sys

from hexrd import imageseries
from hexrd import config

import numpy as np
from sklearn import decomposition
from sklearn.preprocessing import Normalizer

sys.path.append(os.path.split(__file__)[0])
from RTPCA_Classes import pca_paths, pca_matrices
from RTPCA_Widgets import pca_parameters_selector_widget

#*****************************************************************************
#%% USER INPUT

script_base_dir = '/home/djs522/additional_sw/RealTimePCA/CHESS_RealTimePCA/example/'

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
#%% READ IN THE REST OF THE IMAGES
path = '/media/djs522/djs522_nov2020/chess_2020_11/ceo2/*/ff/%s*.h5' 
path = '/home/djs522/additional_sw/RealTimePCA/CHESS_RealTimePCA/example/*%s*.npz'
temp = exp_pca_paths.get_all_image_paths_dict(image_path_stem=path)

#%%
fname_tuple = exp_pca_paths.open_remaining_imgs()
img_files = []
for fname in fname_tuple:
    ims = imageseries.open(fname, format='frame-cache')
    img_files.append(ims[0])
first_img_fname = os.path.join(exp_pca_paths.img_dir, exp_pca_paths.first_img_fname)
ims = imageseries.open(first_img_fname, format='frame-cache')
img_files.append(ims[0])
exp_pca_mats.image_files = np.array(img_files)

#*****************************************************************************
#%% CREATE THE PCA MATRIX
pts = exp_pca_mats.box_points
mat = []

for image in exp_pca_mats.image_files:
    reduced_image = []
    for index in range(int(len(pts[0])/2)):
        reduced_image.append(image[int(pts[0][2*index+1]):int(pts[1][2*index+1]),int(pts[0][2*index]):int(pts[1][2*index])].flatten())
    reduced_image = np.array(reduced_image, dtype=object)
    mat.append(np.hstack(reduced_image))
exp_pca_mats.pca_matrix = np.array(mat)

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