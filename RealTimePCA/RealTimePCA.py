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
use_gui = True

# sample_raw_stem is used to find the detector images of a LODI measurement
#  - use '*' in place of scan numbers
#  - use '%s' in place of detector id in filename
# CHESS EXPERIMENT
beamtime_run = '2019-2'
beamline_id = 'id3a'
exp_name = 'miller-888-1'
sample_name = 'ss718-1'
sample_raw_stem = '/nfs/chess/raw/%s/%s/%s/%s' %(beamtime_run, beamline_id, exp_name, sample_name)  + '/*/ff/%s*.h5' 
# LOCALLY STORED FRAME-CACHE
sample_raw_stem = '/home/djs522/additional_sw/RealTimePCA/CHESS_RealTimePCA/example/*%s*.npz'


# OPTIONAL NON-GUI INPUTS ****************************************************
# if not using the gui to declare variables, fill in the corresponding fields

# sample_aux_dir is used as the output directory for results
sample_aux_dir = '/home/djs522/additional_sw/RealTimePCA/CHESS_RealTimePCA/example/'

# first_img_dict is a dictionary keyed by detector keys with paths to first 
# detector images of LODI measurements
first_img_dict={'ff1': '/home/djs522/additional_sw/RealTimePCA/CHESS_RealTimePCA/example/ss718-1_34_ff1_000028-cachefile.npz',
                'ff2': '/home/djs522/additional_sw/RealTimePCA/CHESS_RealTimePCA/example/ss718-1_34_ff2_000028-cachefile.npz'}

# cfg_name is used to define the location of a hexrd config file with roughly
# calibrated detector (CeO2) and materials file
cfg_fname = os.path.join(sample_aux_dir, 'example_ff_config.yml') 

# mask_dict_file is the path to the detector image mask file for selecting
# data for analysis from images
mask_dict_file = os.path.join(sample_aux_dir, 'example_mask_file')

#*****************************************************************************
#%% INITIALIZE OBJECTS

pca_exp = lodi_experiment(img_stem=sample_raw_stem)

#*****************************************************************************
#%% SELECT RAW_DIR, AUX_DIR, CFG WITH GUI
if use_gui:
    pca_exp.open_output_dir()
    pca_exp.load_config_from_file()
else:
    pca_exp.output_dir = sample_aux_dir
    pca_exp.cfg = config.open(cfg_fname)[0]

#*****************************************************************************
#%% SELECT FIRST IMAGE / REFERENCE IMAGE
if use_gui:
    pca_exp.open_first_image()
else:
    if first_img_dict.keys() == pca_exp.det_keys():
        pca_exp.first_img_dict = first_img_dict
    else:
        raise ValueError("first_img_dict is empty. Use GUI to select first image(s) or fix first_img_dict.")

#*****************************************************************************
#%% SET ROI AND PCA PARAMTERS
if use_gui:
    ppsw = pca_parameters_selector_widget(pca_exp)
    [pca_exp] = ppsw.get_all_pca_objects()
else:
    if mask_dict_file is not None:
        pca_exp.load_mask_dict_from_file(mask_dict_file)
    else:
        raise ValueError("ROI file could not be loaded. Use GUI to select ROI or fix ROI file path.")

#*****************************************************************************
#%% READ IN CURRENT IMAGES
pca_exp.update_img_list(ims_length=2, frane_num_or_img_aggregation_options=None)

#*****************************************************************************
#%% ASSEMBLE THE PCA MATRIX
pca_matrix = pca_exp.assemble_data_matrix()

#*****************************************************************************
#%% FIT AND TRANSFORM PCA MATRIX
#num_cmpts = input("How many principle components would you like to analyze? Choose from 1 to 5 principle components.\n")
#num_cmpts = int(num_cmpts)
num_cmpts = 3
while num_cmpts > 5 or num_cmpts < 1:
    num_cmpts = input("Choose from 1 to 5 principle components.\n")
    num_cmpts = int(num_cmpts)
PCA_func = decomposition.PCA(n_components=num_cmpts)
transformer = Normalizer().fit(pca_matrix)
pca_matrix = transformer.transform(pca_matrix)
PCs = PCA_func.fit_transform(pca_matrix)
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

#*****************************************************************************
#%% CHECK ROI DATA
if use_gui:
    pca_exp.plot_reassemble_image_frame_from_roi(frame_num=0)

#*****************************************************************************
#%% SAVE LODI EXPERIMENT
pca_exp.save_lodi_exp_to_file()
pca_exp.load_lodi_exp_from_file()