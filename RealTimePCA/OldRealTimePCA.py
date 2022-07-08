#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 16:02:23 2020

@author: djs522
"""

#%% ***************************************************************************
# IMPORTS
import os
import glob
import numpy as np

try:
    import dill as cpl
except(ImportError):
    import pickle as cpl

import yaml

import matplotlib
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from sklearn import decomposition
from sklearn.preprocessing import Normalizer

from scipy import ndimage

from hexrd import imageseries
from hexrd import config
from hexrd import xrdutil
from hexrd import instrument
from hexrd import imageutil
from hexrd import imageseries
from hexrd import material

import pandas as pd
import seaborn as sns

import cv2

import tkinter as tk


#%% ***************************************************************************
# CLASS DECLARATION

class roi_selector_widget():
    def __init__(self, img):
        
        self.window = tk.Tk()
        self.window.geometry("1000x800")
        
        self.img = img
        
        self.fig = Figure(figsize=(6,6))
        self.fig.suptitle("Use the left mouse button to select a region of interest")
        self.img_ax = self.fig.add_subplot(111)
        self.inten_lim = [self.img.min(), self.img.max()]
        self.img_ax.imshow(self.img, cmap='Greys_r', vmin=self.inten_lim[0], vmax=self.inten_lim[1])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().place(x=0, y=0, height=800, width=800)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.window)
        
        self.bound_box_list = []
        self.curr_bound_box = np.zeros([2, 2])
        
        # handle mouse cursor button for figure grid selection
        self.num_clicks = 0
        def on_fig_cursor_press(event):
            if event.inaxes is not None and str(event.button) == 'MouseButton.LEFT' and str(self.toolbar.mode) != 'zoom rect':
                if self.num_clicks == 0:
                    self.curr_bound_box[0, :] = np.array([event.xdata, event.ydata])
                    self.img_ax.scatter(event.xdata, event.ydata, c='r', s=200, marker='+')
                    self.canvas.draw()
                    # update plot here with on point scatter
                else:
                    self.curr_bound_box[1, :] = np.array([event.xdata, event.ydata])
                    self.bound_box_list.append(np.copy(self.curr_bound_box.astype(int)))
                    self.update_plot()
                    # update plot here with on point scatter and with grid
                self.num_clicks = (self.num_clicks + 1) % 2
    
        self.fig.canvas.mpl_connect('button_press_event', on_fig_cursor_press)
        
        # Add a button for loading bounding boxes
        def load_bb_on_click():
            print("Load Bounding Boxes")
            root = tk.Tk()
            root.withdraw()
            
            bb_dir = tk.filedialog.askopenfilename(initialdir=os.getcwd(),
                                                           title='Select Bounding Box File')
            
            if not bb_dir:
                pass
            else:
                try:
                    with open(bb_dir, "rb") as input_file:
                         e = cpl.load(input_file)
                         self.bound_box_list = e
                         
                    self.update_plot()
                except:
                    print("Something failed when loading the params file")
                    pass
            
            # update sliders
        self.load_bounding_box_button = tk.Button(self.window, text="Load Bounding Boxes", command=load_bb_on_click)
        self.load_bounding_box_button.place(x=820, y=150, height=40, width=160)
        
        # Add a button for saving bounding boxes
        def save_bb_on_click():
            print("Save Bounding Box")
            root = tk.Tk()
            root.withdraw()
            
            bb_dir_fname = tk.filedialog.asksaveasfilename(confirmoverwrite=False,
                                                                 initialdir=os.getcwd(),
                                                                 title='Save Bounding Box')
            
            if not bb_dir_fname:
                pass
            else:
                try:
                    with open(bb_dir_fname, "wb") as output_file:
                        cpl.dump(self.bound_box_list, output_file)
                    
                except:
                    print("Something failed when saving the params file")
                    pass
            
        self.save_bounding_box_button = tk.Button(self.window, text="Save Bounding Boxes", command=save_bb_on_click)
        self.save_bounding_box_button.place(x=820, y=200, height=40, width=160)
        
        
        # add slider for intensity adjustment        
        def inten_lim_slider_change(event):
            
            if self.vmin_slider.get() > self.vmax_slider.get() - 10:
                self.vmin_slider.set(self.vmax_slider.get() - 10)
            if self.vmax_slider.get() < self.vmin_slider.get() + 10:
                self.vmax_slider.set(self.vmin_slider.get() + 10)
            
            self.vmin_slider.setvar('to', self.vmax_slider.get() - 10)
            self.vmax_slider.setvar('from_', self.vmin_slider.get() + 10)
            
            
            
            self.inten_lim = [self.vmin_slider.get(), self.vmax_slider.get()]
            self.update_plot()
        
        self.vmin_slider = tk.Scale(self.window, label='Min Plot Intensity', 
                                          command=inten_lim_slider_change,
                                          orient='horizontal',
                                          from_=0, to=self.img.max())
        self.vmin_slider.set(self.img.min())
        self.vmin_slider.place(x=820, y=350, height=50, width=160)
        
        self.vmax_slider = tk.Scale(self.window, label='Max Plot Intensity', 
                                           command=inten_lim_slider_change,
                                           orient='horizontal',
                                           from_=0, to=self.img.max())
        self.vmax_slider.set(self.img.max())
        self.vmax_slider.place(x=820, y=400, height=50, width=160)
        
        
        
        # Add a button for loading grid points
        def clear_bounding_boxes_on_click():
            print("Clear Bounding Boxes")
            root = tk.Tk()
            root.withdraw()
            
            self.bound_box_list = []
            self.update_plot()
            
        self.clear_bb_button = tk.Button(self.window, text="Clear Bounding Boxes", command=clear_bounding_boxes_on_click)
        self.clear_bb_button.place(x=820, y=250, height=40, width=160)
        
        
        # Add a button for quitting
        def on_closing(root):
            root.destroy()
            root.quit()            
        self.quit_button = tk.Button(self.window, text="Quit", command=lambda root=self.window:on_closing(root))
        self.quit_button.place(x=820, y=700, height=40, width=160)
        self.window.protocol("WM_DELETE_WINDOW", lambda root=self.window:on_closing(root))
        self.window.mainloop()
    
        
    def update_plot(self):
        # initialize local variables
        bbl = self.bound_box_list
        
        # Add the patch to the Axes
        self.img_ax.cla()
        self.fig.suptitle("Use the left mouse button to select regions of interest")
        self.img_ax.imshow(self.img, cmap='Greys_r', vmin=self.inten_lim[0], vmax=self.inten_lim[1])
        
        # generate rectangle patches 
        for bb in bbl:
            bb_rect = patches.Rectangle((np.min(bb[:, 0]), np.min(bb[:, 1])), 
                                     np.abs(np.diff(bb[:, 0])), np.abs(np.diff(bb[:, 1])), 
                                     linewidth=1, edgecolor='g', facecolor='none')
            self.img_ax.add_patch(bb_rect)
        self.canvas.draw()
    
    def get_bounding_box_list(self):
        return self.bound_box_list
    
    def generate_bounding_box_img_idx(self):
        # initialize local variables
        bbl = self.bound_box_list
        bb_img_idx = []
        
        for bb in bbl:
            row = np.arange(np.min(bb[:, 1]), np.max(bb[:, 1]))
            col = np.arange(np.min(bb[:, 0]), np.max(bb[:, 0]))
            i_row, j_col = np.meshgrid(row, col, indexing='ij')
            bb_idx = np.vstack([i_row.flatten(), j_col.flatten()]).T
            
            bb_img_idx.append(bb_idx)
        
        if len(bb_img_idx):
            bb_img_idx = np.vstack(bb_img_idx)
            bb_img_idx = np.unique(bb_img_idx, axis=1)
            
            bb_img_idx = (bb_img_idx[:, 1], bb_img_idx[:, 0])
        else:
            bb_img_idx = np.where(np.logical_not(np.isnan(self.img)))
        
        return bb_img_idx


#%% ***************************************************************************
# FUNCTION DECLARATION

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

# raw dexela hdf5 to frame-cache shape processing
def hdf5_to_frame_cache_shape(img, key):
    if key == 'ff1':
        img = img[:, ::-1]
        img = np.insert(img, 1, 0, axis=0)
        img = np.insert(img, 1, 0, axis=1)
    elif key == 'ff2':
        img = img[::-1, :]
        img = np.insert(img, 1, 0, axis=0)
        img = np.insert(img, 1, 0, axis=1)
    return img


#%% ***************************************************************************
# CONSTANTS AND PARAMETERS
debug = False 

# experiment directory settings
expt_name = 'miller-881-2'
samp_name = 'ss718-1'
aux_dir = '/nfs/chess/aux/'
raw_dir = '/nfs/chess/raw/2020-2/id3a/%s/%s/' %(expt_name, samp_name)
raw_dir_template = os.path.join(raw_dir, '%d/ff/')
raw_img_template = '%s*.h5'


# hexrd/imageseries settings
img_fmt = 'hdf5' # format of images used in analysis (e.g. 'hdf5' for raw images in .hdf5)
img_path = '/imageseries' # path for hexrd/imageseries
det_keys = ['ff1', 'ff2'] # detector keys for accessing images from directory and image series
scan_start = 1
img_series_len = 1445 # length of images series to use in analysis (ct_scans = 2 frames, full rotation ~ 1445 frames)
img_frame_num = 7 # frame number in image series to analyze
#TODO omega_img_frame = -15 # sample rotation (omega) angle in degrees to define what frame of an image series to analyze
#TODO omega_tol = 0.5 # tolerance of omega angle in degrees to determine whether image series has an image at the defined at the prescribed omega_img_frame


# define dark image subtraction settings (optional)
use_dark_img = False # True = use dark image subtraction in image pre-processing
num_img_for_dark = 700
dark_img_dir = None
dark_img_template = '%s_dark.npz'


# define material and detector files for converting images to polar maps (optional)
use_polar_mapping = False # True = use polar mapping of images, False = use raw images
instrument_fname = '/media/djs522/djs522_nov2020/chess_2020_11/analysis/dexela_nov_2020_ceo2_ruby.yml'
material_fname = '/media/djs522/djs522_nov2020/chess_2020_11/analysis/materials_61_332_36000.h5'
material_key = 'in718'
if instrument_fname is not None:
    instr = load_instrument(instrument_fname)
if material_fname is not None:
    plane_data = load_pdata_hexrd3(material_fname, material_key)
    plane_data.set_exclusions([-1])
eta_pix_size = 0.05 # azimuthal detector angle pixel size in degrees
tth_pix_size = 0.005 # radial detector angle pixel size in degrees
det_eta_ranges = [[-60, 60], [120, 240]] # ranges of detector azimuthal angle ranges in degrees, must be same length as det_keys


#%% ***************************************************************************
# FIRST IMAGE PROCESSING
# - load first image
# - define or load dark image
# - determine whether polar maps or raw

# load first image for preprocessing and defining
#TODO REMOVE COMMENT first_img_series_dir = raw_dir_template %(scan_start) + raw_img_template
first_img_series_dir = '/home/djs522/bin/ff/%s*.h5'

first_img_series_dict = {}
processed_first_img_dict = {}
for i, key in enumerate(det_keys):
    files = []
    for file in glob.glob(first_img_series_dir %(key)):
        files.append(file)
    first_img_series_dict[key] = imageseries.open(files[0], img_fmt, path=img_path)
    
    img = first_img_series_dict[key][img_frame_num]
    
    # TODO: following only used if using raw hdf5 images with hexrd frame-cache detector (to be added later)
    img = hdf5_to_frame_cache_shape(img, key)
    processed_first_img_dict[key] = img


# generate or load dark images
if use_dark_img:
    dark_img_series_dict = {}
    
    if dark_img_dir is None:
        for i, key in enumerate(det_keys):
            dark = imageseries.stats.median(first_img_series_dict[key], nframes=num_img_for_dark)
            
            # TODO: following only used if using raw hdf5 images with hexrd frame-cache detector (to be added later)
            dark = hdf5_to_frame_cache_shape(dark, key)
            
            dark_img_series_dict[key] = dark
            np.save(os.path.join(dark_img_dir, dark_img_template%(key)), dark)
    else:
        for key in det_keys:
            dark_img_series_dict[key] = np.load(os.path.join(dark_img_dir, dark_img_template%(key)))
    
    for i, key in enumerate(det_keys):
        processed_first_img_dict[key] = processed_first_img_dict[key] - dark_img_series_dict[key]

# define polar mapping functions, if necessary
if use_polar_mapping:
    polar_mapping_dict = {}
    temp_img_dict = {}
    for i, key in enumerate(det_keys):
        polar_mapping_dict[key] = xrdutil.PolarView(plane_data, instr,
                                                    eta_min=det_eta_ranges[i][0], eta_max=det_eta_ranges[i][1],
                                                    pixel_size=(tth_pix_size, eta_pix_size))
        
        temp_img_dict[key] = processed_first_img_dict[key]
    
    for i, key in enumerate(det_keys):
        processed_first_img_dict[key] = polar_mapping_dict[key].warp_image(temp_img_dict, pad_with_nans=True, do_interpolation=False)


#%% ***************************************************************************
# DEFINE REGION OF INTEREST
# (y_min, y_max, x_min, x_max)
img_ROI_dict = {}
ROI_message = 'To declare region of interest boxes for each detector \n \
               - Left mouse click and drag cursor to define boxes \n \
               - Press Q to quit once all boxes are defined \n \
               - To use the entire image without ROI, press Q without defining any boxes'
print(ROI_message)

for i, key in enumerate(det_keys):
    bb_widget = roi_selector_widget(processed_first_img_dict[key])
    img_ROI_dict[key] = bb_widget.generate_bounding_box_img_idx()

#smaller = processed_first_img_list[i][bb_widget.generate_bounding_box_img_idx()]


#%% ***************************************************************************
# LOAD STARTING CURRENT IMAGES

current_scans_list = []
current_img_data_list = []

x, dirs, x = os.walk(raw_dir).next()
for folder in dirs:
    scan = folder.split('/')[-1]
    
    if scan not in current_scans_list:
        scan_img_data_dict = {}
        
        if os.path.exists(os.path.join(folder, 'ff')):
            print('Adding Scan: %i' %(scan))
            files = []
            for i, key in enumerate(det_keys):
                for file in glob.glob(raw_dir_template %(scan) + raw_img_template %(key)):
                    files.append(file)
                img_series = imageseries.open(files[0], img_fmt, path=img_path)
                
                if len(img_series) != img_series_len:
                    continue
                
                img = img_series[img_frame_num]
                if use_dark_img:
                    img = img - dark_img_series_dict[key]
                if use_polar_mapping:
                    temp_img_dict[key] = img
                    img = polar_mapping_dict[key].warp_image(temp_img_dict, pad_with_nans=True, do_interpolation=False)
                
                scan_img_data_dict[key] = img[img_ROI_dict[key]] # TODO : This probably could be done below when combining images
            current_scans_list.append(scan)
            current_img_data_list.append(scan_img_data_dict)


#%% ***************************************************************************
# PERFORM PRELIMINARY ANALYSIS

# assemble data matrix with current images for PCA
num_cmpts = 3
num_scans = len(current_scans_list)
num_pixels = 0
for i, key in enumerate(det_keys):
    num_pixels = num_pixels + current_img_data_list[0][key].size

pixel_data_mat = np.zeros([num_scans, num_pixels])
for i, image_dict in enumerate(current_img_data_list):
    combined_images = np.array([])
    for j, key in enumerate(det_keys):
        combined_images = np.hstack([combined_images, image_dict[key].ravel(order='C')]) # C = Row, F = Column
    pixel_data_mat[i, :] = combined_images


# fit and transform PCA matrix
PCA_func = decomposition.PCA(n_components=num_cmpts)
#PCA_func = TSNE(n_components=num_cmpts,learning_rate=300,perplexity = 30,early_exaggeration = 12,init = 'random',  random_state=2019)
#PCA_func = MDS(n_components=num_cmpts, n_init=12, max_iter=10, metric=True, n_jobs=4, random_state=2019)
#PCA_func = Isomap(n_components=num_cmpts, n_jobs=10, n_neighbors=10)
#PCA_func = LocallyLinearEmbedding(n_neighbors=15, n_components=num_cmpts, reg=1e-3, tol=1e-6, max_iter=100)

transformer = Normalizer().fit(pixel_data_mat)
PCA_mat = transformer.transform(pixel_data_mat)
PCs = PCA_func.fit_transform(PCA_mat)
var_ratio = PCA_func.explained_variance_ratio_


#%% ***************************************************************************
# START REAL TIME PROCESSING

x, dirs, x = os.walk(raw_dir).next()
for folder in dirs:
    scan = folder.split('/')[-1]
    
    if scan not in current_scans_list:
        scan_img_data_dict = {}
        
        if os.path.exists(os.path.join(folder, 'ff')):
            print('Adding Scan: %i' %(scan))
            files = []
            for i, key in enumerate(det_keys):
                for file in glob.glob(raw_dir_template %(scan) + raw_img_template %(key)):
                    files.append(file)
                img_series = imageseries.open(files[0], img_fmt, path=img_path)
                
                if len(img_series) != img_series_len:
                    continue
                
                img = img_series[img_frame_num]
                if use_dark_img:
                    img = img - dark_img_series_dict[key]
                if use_polar_mapping:
                    temp_img_dict[key] = img
                    img = polar_mapping_dict[key].warp_image(temp_img_dict, pad_with_nans=True, do_interpolation=False)
                
                scan_img_data_dict[key] = img[img_ROI_dict[key]] # TODO : This probably could be done below when combining images
            current_scans_list.append(scan)
            current_img_data_list.append(scan_img_data_dict)
