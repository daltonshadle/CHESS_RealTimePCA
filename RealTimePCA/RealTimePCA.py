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
import tkinter as tk
import numpy as np
import os
from hexrd import imageseries
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from RTPCA_Classes import pca_paths, pca_matrices
from RTPCA_Widgets import pca_parameters_selector_widget
from sklearn import decomposition
from sklearn.preprocessing import Normalizer
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
#%% READ IN THE REST OF THE IMAGES
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
num_cmpts = 2 # this can probably be made as user input for user to select (say that it must be <=5 for now)
PCA_func = decomposition.PCA(n_components=num_cmpts)
transformer = Normalizer().fit(exp_pca_mats.pca_matrix)
exp_pca_mats.pca_matrix = transformer.transform(exp_pca_mats.pca_matrix)
PCs = PCA_func.fit_transform(exp_pca_mats.pca_matrix)
var_ratio = PCA_func.explained_variance_ratio_

# plot results
fig = plt.figure()
plt.plot(PCs[:, 0])
plt.xlabel('image numbers')
plt.ylabel('Principal Component 1')

fig = plt.figure()
plt.plot(PCs[:, 1])
plt.xlabel('image numbers')
plt.ylabel('Principal Component 2')

fig = plt.figure()
plt.scatter(PCs[:, 0], PCs[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.show()