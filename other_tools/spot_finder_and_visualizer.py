#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 10:46:10 2021

@author: djs522
"""
#%% IMPORTS
# *****************************************************************************

import os

try:
    import dill as cpl
except(ImportError):
    import pickle as cpl

import yaml

import numpy as np
    
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib

from hexrd import config
from hexrd import xrdutil
from hexrd import instrument
from hexrd import imageutil
from hexrd import imageseries
from hexrd import material

#%% EXTRA FUNCTIONS
# *****************************************************************************

# plane data
def load_pdata(cpkl, key):
    with open(cpkl, "rb") as matf:
        mat_list = cpl.load(matf)
    return dict(list(zip([i.name for i in mat_list], mat_list)))[key].planeData


# images
def load_images(yml):
    return imageseries.open(yml, format="image-files")


# instrument
def load_instrument(yml):
    with open(yml, 'r') as f:
        icfg = yaml.safe_load(f)
    return instrument.HEDMInstrument(instrument_config=icfg)


#%% MAIN
# *****************************************************************************
base_path = '/media/djs522/djs522_nov2020/chess_2020_11/'
mruby = os.path.join(base_path, 'mruby')
ss718 = os.path.join(base_path, 'ss718-1/ff')
dp718 = os.path.join(base_path, 'dp718-1/ff')
analysis_path = os.path.join(base_path, 'analysis')
instrument_fname = 'dexela_nov_2020_ceo2_ruby.yml'
image_name = 'dp718-1_18_ff1_001336-cachefile.npz'

instr = load_instrument(os.path.join(analysis_path, instrument_fname))
ims = imageseries.open(os.path.join(dp718, image_name), format='frame-cache')
panel_tth = np.zeros([len(instr.detectors), 3889, 3073])
panel_eta = np.zeros([len(instr.detectors), 3889, 3073])
for i_d, det_key in enumerate(instr.detectors):
    print("working on detector '%s'..." % det_key)

    # grab panel
    panel = instr.detectors[det_key]
    # native_area = panel.pixel_area  # pixel ref area

    # pixel angular coords for the detector panel
    ptth, peta = panel.pixel_angles()
    
    panel_tth[i_d, :, :] = ptth
    panel_eta[i_d, :, :] = peta


#%% MAIN
# *****************************************************************************
deg2rad = np.pi / 180.0
rad2deg = 180.0 / np.pi

path = '/media/djs522/djs522_nov2020/chess_2020_11/dp718-1/c0_0_gripped/c0_0_gripped_sc18_nf/ff1/spots_00938.out'

spots = np.loadtxt(path)

tth_rad = spots[:, 10]
eta_rad = spots[:, 11]
ome_rad = spots[:, 12]

ang_deg = (180.0 / np.pi) * spots[:, 10:13]

img_nums = (1440 * (ome_rad / (2*np.pi))).astype(int)

ff1_panel_tth = panel_tth[0, :, :]
ff1_panel_eta = panel_eta[0, :, :]

tth_buffer = 0.2 * deg2rad
eta_buffer = 3.0 * deg2rad
buffer = 30

thresh = [500, 2000, 5000]
thresh = [500, 1500, 3000]
#thresh = [250, 500, 1000]
ids = [6] 

det_key = 'ff1'
panel = instr.detectors[det_key]

for i in ids:
    # both_ind = np.where((ff1_panel_tth > (tth_rad[i] - tth_buffer)) & (ff1_panel_tth < (tth_rad[i] + tth_buffer))
    #                     & (ff1_panel_eta > (eta_rad[i] - eta_buffer)) & (ff1_panel_eta < (eta_rad[i] + eta_buffer)))
    
    # temp_im = ims[img_nums[i]]
    # temp_im = temp_im[both_ind]
    
    tth_eta = np.atleast_2d([tth_rad[i], eta_rad[i]])
    temp_cart = panel.angles_to_cart(tth_eta)
    temp_pix = panel.cartToPixel(temp_cart).astype(int)
    
    
    temp_im = ims[img_nums[i]]
    temp_im = temp_im[temp_pix[0, 0]-buffer:temp_pix[0, 0]+buffer, temp_pix[0, 1]-buffer:temp_pix[0, 1]+buffer]
    
    fig, ax = plt.subplots(1, len(thresh), figsize=(14, 6))
    for j, t in enumerate(thresh):
        ax[j].set_title(t)
        ax[j].imshow(temp_im, vmax = t)
plt.show()
