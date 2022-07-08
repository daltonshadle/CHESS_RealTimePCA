#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 14:57:43 2022

@author: djs522
"""

# CONSTANT_NAMES = all uppercase with underscores
# FunctionNames() = camel case with parentheses and without underscores
# variable_names = all lowercase with underscores

#*****************************************************************************
#%% IMPORTS
import numpy as np

from RTPCA_Classes import pca_paths, pca_matrices
from RTPCA_Widgets import pca_parameters_selector_widget
#*****************************************************************************
#%% USER INPUT

raw_dir = "/Users/jacksonearls/documents/GitHub/CHESS_RealTimePCA/example/"
aux_dir = "/Users/jacksonearls/documents/GitHub/CHESS_RealTimePCA/example/"
output_fname = 'dictest_output.txt'

#*****************************************************************************
#%% INITIALIZE OBJECTS
exp_pca_paths = pca_paths(base_dir=raw_dir, img_dir=raw_dir, 
                          output_dir=aux_dir, output_fname=output_fname)
exp_pca_mats = pca_matrices()

#*****************************************************************************
#%% SELECT FIRST IMAGE / REFERENCE IMAGE
exp_pca_paths.open_first_image()

#*****************************************************************************
#%% SET CONTROL POINT GRID AND pca PARAMTERS
dpsw = pca_parameters_selector_widget(exp_pca_paths, exp_pca_mats, adjust_grid=True)
[exp_pca_paths, exp_pca_mats] = dpsw.get_all_pca_objects()

#*****************************************************************************
