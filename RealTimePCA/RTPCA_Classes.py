"""
Created on Sun Dec  4 20:11:23 2022

@author: Dalton Shadle
"""

# -*- coding: utf-8 -*-

# %% ***************************************************************************
# IMPORTS
import os

import glob

import warnings

import pickle

import json

import numpy as np

import pandas as pd

import tkinter as tk

import matplotlib.pyplot as plt

from hexrd import imageseries
from hexrd.imageseries import process
from hexrd import config


# ***************************************************************************
# CLASS DECLARATION
class lodi_experiment():
    def __init__(self,
                 img_stem='',
                 output_dir='',
                 dic_output_json_dir='',
                 dic_output_txt_dir='',
                 dic_output_cols=[],
                 dic_output_data='',
                 lodi_json_dir='',
                 lodi_par_dir='',
                 lodi_par_cols=[],
                 lodi_par_data='',
                 pair_lodi_with_dic=False,
                 first_img_dict={},
                 box_mask_dict={},
                 box_points_dict={},
                 rings_mask_dict={},
                 ring_eta_dict={},
                 use_rings_mask_dict={},
                 curr_loadstep_nums=[],
                 curr_img_path_dict={},
                 curr_img_data_dict={},
                 cfg=config):

        # initialize class variables
        self._raw_img_stem = img_stem
        self._raw_output_dir = output_dir
        self._dic_output_json_dir = dic_output_json_dir
        self._dic_output_txt_dir = dic_output_txt_dir
        self._dic_output_cols = dic_output_cols
        self._dic_output_data = dic_output_data
        self._lodi_json_dir = lodi_json_dir
        self._lodi_par_dir = lodi_par_dir
        self._lodi_par_cols = lodi_par_cols
        self._lodi_par_data = lodi_par_data
        self._pair_lodi_with_dic = pair_lodi_with_dic
        self._first_img_dict = first_img_dict
        self._box_mask_dict = box_mask_dict
        self._box_points_dict = box_points_dict
        self._rings_mask_dict = rings_mask_dict
        self._ring_eta_dict = ring_eta_dict
        self._use_rings_mask_dict = use_rings_mask_dict
        self._curr_loadstep_nums = curr_loadstep_nums
        self._curr_img_path_dict = curr_img_path_dict
        self._curr_img_data_dict = curr_img_data_dict
        self._cfg = cfg
        
        # class constants 
        self.dic_output_prompt_dict = {'Stress (Loading Dir)':0, 'Strain (Loading Dir)':1, 'Load Step Number':2}
        self.lodi_par_prompt_dict = {'Scan Number':0, 'Load Step Number':1, 'Final Load Newtons':2, 'Final Screw':3}
        
    # SETTERS AND GETTERS *****************************************************
    @property
    def raw_img_stem(self):
        return self._raw_img_stem
    @raw_img_stem.setter
    def raw_img_stem(self, img_stem):
        self._raw_img_stem = img_stem
    
    @property
    def raw_output_dir(self):
        return self._raw_output_dir
    @raw_output_dir.setter
    def raw_output_dir(self, output_dir):
        if os.path.exists(output_dir):
            self._raw_output_dir = output_dir
        else:
            raise ValueError("Directory '%s' does not exists" % (output_dir))
    
    @property
    def dic_output_json_dir(self):
        return self._dic_output_json_dir
    @dic_output_json_dir.setter
    def dic_output_json_dir(self, dic_output_json_dir):
        if os.path.exists(dic_output_json_dir):
            self._dic_output_json_dir = dic_output_json_dir
        else:
            raise ValueError("Directory '%s' does not exists" % (dic_output_json_dir))
            
    @property
    def dic_output_txt_dir(self):
        return self._dic_output_txt_dir
    @dic_output_txt_dir.setter
    def dic_output_txt_dir(self, dic_output_txt_dir):
        if os.path.exists(dic_output_txt_dir):
            self._dic_output_txt_dir = dic_output_txt_dir
        else:
            raise ValueError("Directory '%s' does not exists" % (dic_output_txt_dir))
    
    @property
    def dic_output_cols(self):
        return self._dic_output_cols
    @dic_output_cols.setter
    def dic_output_cols(self, dic_output_cols):
        self._dic_output_cols = dic_output_cols
        
    @property
    def dic_output_data(self):
        return self._dic_output_data
    @dic_output_data.setter
    def dic_output_data(self, dic_output_data):
        self._dic_output_data = dic_output_data
    
    @property
    def lodi_json_dir(self):
        return self._lodi_json_dir
    @lodi_json_dir.setter
    def lodi_json_dir(self, lodi_json_dir):
        if os.path.exists(lodi_json_dir):
            self._lodi_json_dir = lodi_json_dir
        else:
            raise ValueError("Directory '%s' does not exists" % (lodi_json_dir))
    
    @property
    def lodi_par_dir(self):
        return self._lodi_par_dir
    @lodi_par_dir.setter
    def lodi_par_dir(self, lodi_par_dir):
        if os.path.exists(lodi_par_dir):
            self._lodi_par_dir = lodi_par_dir
        else:
            raise ValueError("Directory '%s' does not exists" % (lodi_par_dir))
    
    @property
    def lodi_par_cols(self):
        return self._lodi_par_cols
    @lodi_par_cols.setter
    def lodi_par_cols(self, lodi_par_cols):
        self._lodi_par_cols = lodi_par_cols
        
    @property
    def lodi_par_data(self):
        return self._lodi_par_data
    @lodi_par_data.setter
    def lodi_par_data(self, lodi_par_data):
        self._lodi_par_data = lodi_par_data
    
    @property
    def pair_lodi_with_dic(self):
        return self._pair_lodi_with_dic
    @pair_lodi_with_dic.setter
    def pair_lodi_with_dic(self, pair_lodi_with_dic):
        self._pair_lodi_with_dic = pair_lodi_with_dic
    
    @property
    def first_img_dict(self):
        return self._first_img_dict
    @first_img_dict.setter
    def first_img_dict(self, first_img_dict):
        self._first_img_dict = first_img_dict
    
    @property
    def box_mask_dict(self):
        return self._box_mask_dict
    @box_mask_dict.setter
    def box_mask_dict(self, box_mask_dict):
        self._box_mask_dict = box_mask_dict
        
    @property
    def box_points_dict(self):
        return self._box_points_dict
    @box_points_dict.setter
    def box_points_dict(self, box_points_dict):
        self._box_points_dict = box_points_dict
    
    @property
    def rings_mask_dict(self):
        return self._rings_mask_dict
    @rings_mask_dict.setter
    def rings_mask_dict(self, rings_mask_dict):
        self._rings_mask_dict = rings_mask_dict
    
    @property
    def ring_eta_dict(self):
        return self._ring_eta_dict
    @ring_eta_dict.setter
    def ring_eta_dict(self, ring_eta_dict):
        self._ring_eta_dict = ring_eta_dict
    
    @property
    def use_rings_mask_dict(self):
        return self._use_rings_mask_dict
    @use_rings_mask_dict.setter
    def use_rings_mask_dict(self, use_rings_mask_dict):
        self._use_rings_mask_dict = use_rings_mask_dict
    
    @property
    def curr_loadstep_nums(self):
        return self._curr_loadstep_nums
    @curr_loadstep_nums.setter
    def curr_loadstep_nums(self, curr_loadstep_nums):
        self._curr_loadstep_nums = curr_loadstep_nums
    
    @property
    def curr_img_path_dict(self):
        return self._curr_img_path_dict
    @curr_img_path_dict.setter
    def curr_img_path_dict(self, curr_img_path_dict):
        self._curr_img_path_dict = curr_img_path_dict
    
    @property
    def curr_img_data_dict(self):
        return self._curr_img_data_dict
    @curr_img_data_dict.setter
    def curr_img_data_dict(self, curr_img_data_dict):
        self._curr_img_data_dict = curr_img_data_dict
    
    @property
    def cfg(self):
        return self._cfg
    @cfg.setter
    def cfg(self, cfg):
        self._cfg = cfg
        self.init_dicts()
        
    
    # OTHER FUNCITONS *********************************************************
    def det_keys(self):
       return self.cfg.instrument.hedm.detectors.keys()
    
    
    # DICT FUNCITONS **********************************************************
    def init_dicts(self):
        if self.first_img_dict.keys() != self.det_keys():
            self.reset_first_img_dict()
        if self.box_points_dict.keys() != self.det_keys():
            self.reset_box_points_dict()
        if self.use_rings_mask_dict.keys() != self.det_keys():
            self.reset_use_rings_mask_dict()
        if self.box_mask_dict.keys() != self.det_keys():
            self.calc_box_mask_dict()
        if self.rings_mask_dict.keys() != self.det_keys():
            self.calc_rings_mask_dict() 
        if self.curr_img_path_dict.keys() != self.det_keys():
            self.reset_curr_img_path_dict()
        if self.curr_img_data_dict.keys() != self.det_keys():
            self.reset_curr_img_data_dict()
                
        self.dic_output_prompt_dict = {'Stress (Loading Dir)':0, 'Strain (Loading Dir)':1, 'Load Step Number':2}
        self.lodi_par_prompt_dict = {'Scan Number':0, 'Load Step Number':1, 'Final Load Newtons':2, 'Final Screw':3}
        for key in self.det_keys():
            self.lodi_par_prompt_dict[key + ' Image Number'] = len(self.lodi_par_prompt_dict)
        
        
    def reset_first_img_dict(self):
        self.first_img_dict = dict.fromkeys(self.det_keys())
            
    def reset_box_points_dict(self):
        self.box_points_dict = {}
        for det_key in self.det_keys():
            self.box_points_dict[det_key] = np.array([])
    
    def reset_use_rings_mask_dict(self):
        self.use_rings_mask_dict = {}
        for det_key in self.det_keys():
            self.use_rings_mask_dict[det_key] = 0
    
    def reset_curr_img_path_dict(self):
        self.curr_img_path_dict = dict.fromkeys(self.det_keys())
            
    def reset_curr_img_data_dict(self):
        self.curr_img_data_dict = dict.fromkeys(self.det_keys())
    
    def calc_box_mask_dict(self): 
        print('Calculating Box Mask')
        self.box_mask_dict = {}
        for det_key in self.det_keys():
            img_size = [self.cfg.instrument.hedm.detectors[det_key].rows, 
                        self.cfg.instrument.hedm.detectors[det_key].cols]
            if self._box_points_dict[det_key].size == 0 and not self.use_rings_mask_dict[det_key]:
                # no bounding boxes and not using rings, then use whole image (not recommended due to slow computations)
                img_box_mask = np.ones(img_size)
            else:
                img_box_mask = np.zeros(img_size)
                for i in range(self.box_points_dict[det_key].shape[0]):
                    pts = self.box_points_dict[det_key][i, :, :]
                    pts = np.floor(pts).astype(int)
                    # pts = [min_x, min_y; max_x, max_y]
                    # pts = [min_col, min_row; max_col, max_row]
                    img_box_mask[pts[0, 1]:pts[1, 1], pts[0, 0]:pts[1, 0]] = 1
                
            self.box_mask_dict[det_key] = img_box_mask.astype(bool)
    
    def calc_rings_mask_dict(self):
        print('Calculating Rings Mask')
        pd = self.cfg.material.plane_data
        self.rings_mask_dict = {}
        for det_key in self.det_keys():
            panel = self.cfg.instrument.hedm.detectors[det_key]
            panel_tth, panel_eta = panel.pixel_angles()
            panel_eta = np.degrees(panel_eta)
            print("Panel %s: min eta = %0.2f deg, max eta = %0.2f deg" 
                  %(det_key, np.min(panel_eta), np.max(panel_eta)))
            
            panel_eta_mask = np.ones(panel_eta.shape).astype(bool)
            if det_key in self.ring_eta_dict.keys():
                if len(self.ring_eta_dict[det_key]) > 0:
                    panel_eta_mask = np.zeros(panel_eta.shape).astype(bool)
                    for eta_range in self.ring_eta_dict[det_key]:
                        print("eta range: ", eta_range)
                        panel_eta_mask = np.logical_or(panel_eta_mask, 
                                                       ((panel_eta > np.min(eta_range)) & (panel_eta < np.max(eta_range))))
            
            panel_tth_mask = np.zeros(panel_tth.shape).astype(bool)
            for tth in pd.getTThRanges():
                panel_tth_mask = np.logical_or(panel_tth_mask,
                                               ((panel_tth > tth[0]) & (panel_tth < tth[1])))
            
            self.rings_mask_dict[det_key] = np.logical_and(panel_tth_mask, panel_eta_mask)
    
    def total_img_mask_dict(self):
        img_mask_dict = {}
        
        if self.rings_mask_dict.keys() != self.det_keys():
            self.calc_rings_mask_dict()
            
        self.calc_box_mask_dict()
        
        for det_key in self.det_keys():
            if self.use_rings_mask_dict[det_key]:
                img_mask_dict[det_key] = (self.box_mask_dict[det_key] + self.rings_mask_dict[det_key]).astype(bool)
            else:
                img_mask_dict[det_key] = (self.box_mask_dict[det_key]).astype(bool)
        return img_mask_dict
    
    
    # IMAGE FUNCITONS *********************************************************
    def open_first_image(self, first_img_dict=None):
        use_gui = False
        if first_img_dict is not None:
            if first_img_dict.keys() != self.det_keys():
                use_gui = True
            for key in first_img_dict.keys():
                if not os.path.exists(first_img_dict[key]):
                    use_gui = True
            
            if not use_gui:
                print("Using specificied image dict: %s" %(first_img_dict))
        
        if use_gui:
            first_img_dict = {}
            for det_key in self.det_keys():
                root = tk.Tk()
                root.withdraw()
                path = self.raw_img_stem.split('*')[0]

                file_dir = tk.filedialog.askopenfilename(initialdir=path,
                                                         defaultextension=".npz",
                                                         filetypes=[("npz files","*.npz"),
                                                                    ('H5 files','*.h5'),
                                                                    ("All Files", "*.*")],
                                                         title="Select First Image File for %s" % (det_key))

                if not file_dir:
                    quit()
                else:
                    try:
                        first_img_dict[det_key] = file_dir
                    except Exception as e:
                        print(e)
                        self.open_first_image()
        self._first_img_dict = first_img_dict
    
    def load_ims_from_path(self, path, img_process_list=None):
        if path.endswith('.npz'):
            # frame cache
            ims = imageseries.open(path, format='frame-cache')
        elif path.endswith('.h5'):
            # raw
            ims = imageseries.open(path, format='hdf5', path='/imageseries')
        else:
            print("IMAGE NOT LOADED, FILE TYPE NOT SUPPORTED")
            ims = None
        if img_process_list is not None:
            ims = process.ProcessedImageSeries(ims, img_process_list)
            
        return ims
    
    def overwrite_img_list(self, ims_length=2, frane_num_or_img_aggregation_options=None):
        # overwrite img path list
        all_image_paths_dict = {}
        for det_key in self.det_keys():
            det_image_path_stem = self.raw_img_stem %(det_key)
            sort_split = det_image_path_stem.split('*')

            det_files = glob.glob(det_image_path_stem)
            sort_det_files = []
            
            # sorting on last * in image_path_stem as that probably has the scan number
            for f in det_files:
                sort_det_files.append(f.split(sort_split[-2])[1].split(sort_split[-1])[0])
            
            ind = np.argsort(sort_det_files)
            
            all_image_paths_dict[det_key] = np.array(det_files)[ind.tolist()]
            
            # start list at the first images
            s_ind = np.where(all_image_paths_dict[det_key] == self.first_img_dict[det_key])[0].astype(int)
            if len(s_ind) == 0:
                raise ValueError("lodi_experiment.first_img_dict for %s is not in image paths list" %(det_key))
            s_ind = s_ind[0]
            all_image_paths_dict[det_key] = (all_image_paths_dict[det_key][s_ind:]).tolist()
        self.curr_img_path_dict = all_image_paths_dict
        
        # load images as overwrite
        img_data_list_dict = {}
        img_mask = self.total_img_mask_dict()
        for det_key in self.det_keys():
            # !!! TODO: return list of indices actually used
            ims_list = []
            
            for img_path in self.curr_img_path_dict[det_key]:
                ims = self.load_ims_from_path(img_path)
                
                if len(ims) == ims_length:
                    img_data = []
                    for i in range(ims_length):
                        img_data.append(ims[i][img_mask[det_key]].flatten())
                        
                    ims_list.append(np.hstack(img_data))
            
            img_data_list_dict[det_key] = np.array(ims_list)
        
        self.curr_img_data_dict = img_data_list_dict
    
    def update_img_list(self, ims_length=2, frane_num_or_img_aggregation_options=None):
        # get update paths list
        all_image_paths_dict = {}
        update_image_paths_dict = {}
        for det_key in self.det_keys():
            det_image_path_stem = self.raw_img_stem %(det_key)
            sort_split = det_image_path_stem.split('*')

            det_files = glob.glob(det_image_path_stem)
            sort_det_files = []
            
            # sorting on last * in image_path_stem as that probably has the scan number
            for f in det_files:
                sort_det_files.append(f.split(sort_split[-2])[1].split(sort_split[-1])[0])
            
            ind = np.argsort(sort_det_files)
            
            all_image_paths_dict[det_key] = np.array(det_files)[ind.tolist()]
            
            # start list at the first images
            s_ind = np.where(all_image_paths_dict[det_key] == self.first_img_dict[det_key])[0].astype(int)
            if len(s_ind) == 0:
                raise ValueError("lodi_experiment.first_img_dict for %s is not in image paths list" %(det_key))
            s_ind = s_ind[0]
            all_image_paths_dict[det_key] = (all_image_paths_dict[det_key][s_ind:])
            
            # find differences
            update_image_paths_dict[det_key] = np.setdiff1d(all_image_paths_dict[det_key], 
                                                         self.curr_img_path_dict[det_key])
            self.curr_img_path_dict[det_key] = all_image_paths_dict[det_key].tolist()
        
        # load updated images
        img_mask = self.total_img_mask_dict()
        for det_key in self.det_keys():
            # !!! TODO: return list of indices actually used
            ims_list = []
            
            for img_path in update_image_paths_dict[det_key]:
                ims = self.load_ims_from_path(img_path)
                
                if len(ims) == ims_length:
                    img_data = []
                    for i in range(ims_length):
                        img_data.append(ims[i][img_mask[det_key]].flatten())
                        
                    ims_list.append(np.hstack(img_data))
            if len(ims_list) > 0:
                if self.curr_img_data_dict[det_key] is None:
                    self.curr_img_data_dict[det_key] = np.array(ims_list)
                else:
                    self.curr_img_data_dict[det_key] = np.vstack([self.curr_img_data_dict[det_key], np.array(ims_list)])
    
    
    
<<<<<<< HEAD
    def overwrite_img_list_new(self, img_process_dict=None, frane_num_or_img_aggregation_options=None, 
                               start_loadstep_num=0, ignore_loadstep_list=[]):
=======
    def overwrite_img_list_new(self, img_process_dict=None, frane_num_or_img_aggregation_options=None):
>>>>>>> 3b0ecb6565d08a5e078485c2ec88f34e0b61bdf9
        # frane_num_or_img_aggregation_options
        # None = all images
        # int = frame_num
        # max = max over all frames to one image
        # mean = mean over all frames to one image
        # sum = sum over all frames to one image
        
        self.curr_loadstep_nums = []
        
        # TODO: need to consider if not pairing with DIC
        curr_lodi_par_data = self.process_lodi_par_file(self.lodi_par_dir, self.lodi_par_cols)
        curr_lodi_loadstep_nums = curr_lodi_par_data[:, self.lodi_par_prompt_dict['Load Step Number']]
        self.curr_loadstep_nums = curr_lodi_loadstep_nums
        
        if self.pair_lodi_with_dic:
            curr_dic_output_data = self.process_dic_output_file(self.dic_output_txt_dir, self.dic_output_cols)
            curr_dic_loadstep_nums = curr_dic_output_data[:, self.dic_output_prompt_dict['Load Step Number']]
            self.curr_loadstep_nums = np.intersect1d(self.curr_loadstep_nums, curr_dic_loadstep_nums)
            print(curr_lodi_loadstep_nums.shape, curr_dic_loadstep_nums.shape, self.curr_loadstep_nums.shape)
        
<<<<<<< HEAD
        self.curr_loadstep_nums = self.curr_loadstep_nums[self.curr_loadstep_nums >= start_loadstep_num]
        self.curr_loadstep_nums = np.setdiff1d(self.curr_loadstep_nums, ignore_loadstep_list)
        
=======
>>>>>>> 3b0ecb6565d08a5e078485c2ec88f34e0b61bdf9
        # load images
        img_mask = self.total_img_mask_dict()
        for det_key in self.det_keys():
            # !!! TODO: return list of indices actually used
            ims_list = []
            
            for curr_ls in self.curr_loadstep_nums:
                print("Loading det %s loadstep %i" %(det_key, curr_ls))
                # sample_raw_stem = '/nfs/chess/raw/%s/%s/%s/%s' %(beamtime_cycle, beamline_id, exp_name, sample_name)  + '/%i/ff/%s_%06i.h5' 
                curr_lodi_ind = np.where(curr_lodi_par_data[:, self.lodi_par_prompt_dict['Load Step Number']] == curr_ls)[0][0]
                curr_scan = curr_lodi_par_data[curr_lodi_ind, self.lodi_par_prompt_dict['Scan Number']]
                curr_ff_img_num = curr_lodi_par_data[curr_lodi_ind, self.lodi_par_prompt_dict[det_key + ' Image Number']]
                
                img_path = self.raw_img_stem %(curr_scan, det_key, curr_ff_img_num)
                ims = self.load_ims_from_path(img_path, img_process_list=img_process_dict[det_key])
                
                # frane_num_or_img_aggregation_options
                # None = all images
                # list = frame_nums
                # 'max' = max over all frames to one image
                # 'mean' = mean over all frames to one image
                img_data = []
                if frane_num_or_img_aggregation_options is None:
                    for i in range(len(ims)):
                        img_data.append(ims[i][img_mask[det_key]].flatten())
                    img_data = np.hstack(img_data).flatten()
                elif (isinstance(frane_num_or_img_aggregation_options, list) 
<<<<<<< HEAD
                      or isinstance(frane_num_or_img_aggregation_options, np.ndarray)):
=======
                      or isinstance(frane_num_or_img_aggregation_options, np.array)):
>>>>>>> 3b0ecb6565d08a5e078485c2ec88f34e0b61bdf9
                    frame_num_arr = np.array(frane_num_or_img_aggregation_options).astype(int)
                    frame_num_arr = frame_num_arr.flatten()
                    for fn in frame_num_arr:
                        img_data.append(ims[fn][img_mask[det_key]].flatten())
                    img_data = np.hstack(img_data).flatten()
                elif (isinstance(frane_num_or_img_aggregation_options, str) and
                      frane_num_or_img_aggregation_options.lower() == 'max'):
                    for i in range(len(ims)):
                        img_data.append(ims[i][img_mask[det_key]].flatten())
                    img_data = np.vstack(img_data)
                    img_data = np.max(img_data, axis=0).flatten()
                elif (isinstance(frane_num_or_img_aggregation_options, str) and
                      frane_num_or_img_aggregation_options.lower() == 'mean'):
                    for i in range(len(ims)):
                        img_data.append(ims[i][img_mask[det_key]].flatten())
                    img_data = np.vstack(img_data)
                    img_data = np.mean(img_data, axis=0).flatten()
                elif (isinstance(frane_num_or_img_aggregation_options, str) and
                      frane_num_or_img_aggregation_options.lower() == 'sum'):
                    for i in range(len(ims)):
                        img_data.append(ims[i][img_mask[det_key]].flatten())
                    img_data = np.vstack(img_data)
                    img_data = np.sum(img_data, axis=0).flatten()
                else:
                    print('frane_num_or_img_aggregation_options %s is not supported' %(frane_num_or_img_aggregation_options))
                
                ims_list.append(img_data)
            
            self.curr_img_data_dict[det_key] = np.array(ims_list)
        
        # update lodi_par_data, have to deal with multiple lodi at load step
        uni_curr_lodi_ls_nums, uni_curr_lodi_ls_nums_ind = np.unique(curr_lodi_loadstep_nums, return_index=True) 
        curr_lodi_par_ind = np.where(np.in1d(uni_curr_lodi_ls_nums, self.curr_loadstep_nums))[0]
        curr_lodi_par_ind = uni_curr_lodi_ls_nums_ind[curr_lodi_par_ind]
        self.lodi_par_data = curr_lodi_par_data[curr_lodi_par_ind, :]
        
        if self.pair_lodi_with_dic:
            # update dic_output_data, have to deal with multiple dic images at load step
            uni_curr_dic_ls_nums, uni_curr_dic_ls_nums_ind = np.unique(curr_dic_loadstep_nums, return_index=True) 
            curr_dic_output_ind = np.where(np.in1d(uni_curr_dic_ls_nums, self.curr_loadstep_nums))[0]
            curr_dic_output_ind = uni_curr_dic_ls_nums_ind[curr_dic_output_ind]
            self.dic_output_data = curr_dic_output_data[curr_dic_output_ind, :]
        
    
<<<<<<< HEAD
    def update_img_list_new(self, img_process_dict=None, frane_num_or_img_aggregation_options=None,
                            start_loadstep_num=0, ignore_loadstep_list=[]):
=======
    def update_img_list_new(self, img_process_dict=None, frane_num_or_img_aggregation_options=None):
>>>>>>> 3b0ecb6565d08a5e078485c2ec88f34e0b61bdf9
        # frane_num_or_img_aggregation_options
        # None = all images
        # int = frame_num
        # max = max over all frames to one image
        # mean = mean over all frames to one image
        # sum = sum over all frames to one image
        
        #self.curr_loadstep_nums = []
        
        # TODO: need to consider if not pairing with DIC
        curr_lodi_par_data = self.process_lodi_par_file(self.lodi_par_dir, self.lodi_par_cols)
        curr_lodi_loadstep_nums = curr_lodi_par_data[:, self.lodi_par_prompt_dict['Load Step Number']]
        update_loadstep_nums = curr_lodi_loadstep_nums
        
        if self.pair_lodi_with_dic:
            curr_dic_output_data = self.process_dic_output_file(self.dic_output_txt_dir, self.dic_output_cols)
            curr_dic_loadstep_nums = curr_dic_output_data[:, self.dic_output_prompt_dict['Load Step Number']]
            update_loadstep_nums = np.intersect1d(update_loadstep_nums, curr_dic_loadstep_nums)
        
<<<<<<< HEAD
        update_loadstep_nums = update_loadstep_nums[update_loadstep_nums >= start_loadstep_num]
        update_loadstep_nums = np.setdiff1d(update_loadstep_nums, ignore_loadstep_list)
=======
>>>>>>> 3b0ecb6565d08a5e078485c2ec88f34e0b61bdf9
        new_loadstep_nums = np.setdiff1d(update_loadstep_nums, self.curr_loadstep_nums)
        
        # load images
        img_mask = self.total_img_mask_dict()
        for det_key in self.det_keys():
            # !!! TODO: return list of indices actually used
            new_ims_list = []
            
            for new_ls in new_loadstep_nums:
                print("Loading det %s loadstep %i" %(det_key, new_ls))
                # sample_raw_stem = '/nfs/chess/raw/%s/%s/%s/%s' %(beamtime_cycle, beamline_id, exp_name, sample_name)  + '/%i/ff/%s_%06i.h5' 
                curr_lodi_ind = np.where(curr_lodi_par_data[:, self.lodi_par_prompt_dict['Load Step Number']] == new_ls)[0][0]
                curr_scan = curr_lodi_par_data[curr_lodi_ind, self.lodi_par_prompt_dict['Scan Number']]
                curr_ff_img_num = curr_lodi_par_data[curr_lodi_ind, self.lodi_par_prompt_dict[det_key + ' Image Number']]
                
                img_path = self.raw_img_stem %(curr_scan, det_key, curr_ff_img_num)
                ims = self.load_ims_from_path(img_path, img_process_list=img_process_dict[det_key])
                
                # frane_num_or_img_aggregation_options
                # None = all images
                # list = frame_nums
                # 'max' = max over all frames to one image
                # 'mean' = mean over all frames to one image
                img_data = []
                if frane_num_or_img_aggregation_options is None:
                    for i in range(len(ims)):
                        img_data.append(ims[i][img_mask[det_key]].flatten())
                    img_data = np.hstack(img_data).flatten()
                elif (isinstance(frane_num_or_img_aggregation_options, list) 
<<<<<<< HEAD
                      or isinstance(frane_num_or_img_aggregation_options, np.ndarray)):
=======
                      or isinstance(frane_num_or_img_aggregation_options, np.array)):
>>>>>>> 3b0ecb6565d08a5e078485c2ec88f34e0b61bdf9
                    frame_num_arr = np.array(frane_num_or_img_aggregation_options).astype(int)
                    frame_num_arr = frame_num_arr.flatten()
                    for fn in frame_num_arr:
                        img_data.append(ims[fn][img_mask[det_key]].flatten())
                    img_data = np.hstack(img_data).flatten()
                elif (isinstance(frane_num_or_img_aggregation_options, str) and
                      frane_num_or_img_aggregation_options.lower() == 'max'):
                    for i in range(len(ims)):
                        img_data.append(ims[i][img_mask[det_key]].flatten())
                    img_data = np.vstack(img_data)
                    img_data = np.max(img_data, axis=0).flatten()
                elif (isinstance(frane_num_or_img_aggregation_options, str) and
                      frane_num_or_img_aggregation_options.lower() == 'mean'):
                    for i in range(len(ims)):
                        img_data.append(ims[i][img_mask[det_key]].flatten())
                    img_data = np.vstack(img_data)
                    img_data = np.mean(img_data, axis=0).flatten()
                elif (isinstance(frane_num_or_img_aggregation_options, str) and
                      frane_num_or_img_aggregation_options.lower() == 'sum'):
                    for i in range(len(ims)):
                        img_data.append(ims[i][img_mask[det_key]].flatten())
                    img_data = np.vstack(img_data)
                    img_data = np.sum(img_data, axis=0).flatten()
                else:
                    print('frane_num_or_img_aggregation_options %s is not supported' %(frane_num_or_img_aggregation_options))
                
                new_ims_list.append(img_data)
            
            new_ims_list = np.atleast_2d(np.array(new_ims_list))
            if new_ims_list.shape[1] == self.curr_img_data_dict[det_key].shape[1]:
                self.curr_img_data_dict[det_key] = np.vstack([self.curr_img_data_dict[det_key], new_ims_list])
        
        self.curr_loadstep_nums = update_loadstep_nums
        
        # update lodi_par_data, have to deal with multiple lodi at load step
        uni_curr_lodi_ls_nums, uni_curr_lodi_ls_nums_ind = np.unique(curr_lodi_loadstep_nums, return_index=True) 
        curr_lodi_par_ind = np.where(np.in1d(uni_curr_lodi_ls_nums, self.curr_loadstep_nums))[0]
        curr_lodi_par_ind = uni_curr_lodi_ls_nums_ind[curr_lodi_par_ind]
        self.lodi_par_data = curr_lodi_par_data[curr_lodi_par_ind, :]
        
        if self.pair_lodi_with_dic:
            # update dic_output_data, have to deal with multiple dic images at load step
            uni_curr_dic_ls_nums, uni_curr_dic_ls_nums_ind = np.unique(curr_dic_loadstep_nums, return_index=True) 
            curr_dic_output_ind = np.where(np.in1d(uni_curr_dic_ls_nums, self.curr_loadstep_nums))[0]
            curr_dic_output_ind = uni_curr_dic_ls_nums_ind[curr_dic_output_ind]
            self.dic_output_data = curr_dic_output_data[curr_dic_output_ind, :]
        
        
    
    
    
    # PCA FUNCITONS ***********************************************************
    def assemble_data_matrix(self):
        for i, det_key in enumerate(self.det_keys()):
            if i == 0:
                data_matrix = self.curr_img_data_dict[det_key]
            else:
                data_matrix = np.hstack([data_matrix, self.curr_img_data_dict[det_key]])
        return data_matrix
    
    
    # DEBUG FUNCITONS *********************************************************
    def reassemble_image_frame_from_roi(self, frame_num=0):
        reassbmle_frame_dict = {}
        tot_mask = self.total_img_mask_dict()
        for det_key in self.det_keys():
            re_frame = np.zeros(tot_mask[det_key].shape)
            s_ind = int(frame_num * np.sum(tot_mask[det_key]))
            e_ind = int((frame_num + 1) * np.sum(tot_mask[det_key]))
            re_frame[tot_mask[det_key].astype(bool)] = self.curr_img_data_dict[det_key][0, s_ind:e_ind]
            reassbmle_frame_dict[det_key] = re_frame
            
        return reassbmle_frame_dict
    
    def plot_reassemble_image_frame_from_roi(self, frame_num=0):
        re = self.reassemble_image_frame_from_roi(frame_num=frame_num)
        tot_mask = self.total_img_mask_dict()
        
        fig = plt.figure()
        ax = fig.subplots(nrows=1, ncols=len(re))
        
        for i, det_key in enumerate(self.det_keys()):
            ax[i].imshow(re[det_key], vmax=100)
            ax[i].imshow(tot_mask[det_key], cmap='Reds', alpha=0.1)
            
        plt.show()
    
    
    # UTILITY / IO FUNCITONS **************************************************
    def __repr__(self):
        return "lodi_experiment()\n" + self.__str__()

    def __str__(self):
        class_dict = {"raw_img_stem": self._raw_img_stem,
        "output_dir": self._raw_output_dir,
        "dic_output_json_dir": self._dic_output_json_dir,
        "dic_output_txt_dir": self._dic_output_txt_dir,
        "dic_output_cols": self._dic_output_cols,
        "dic_output_data": self._dic_output_data,
        "lodi_json_dir": self._lodi_json_dir,
        "lodi_par_dir": self._lodi_par_dir,
        "lodi_par_cols": self._lodi_par_cols,
        "lodi_par_data": self._lodi_par_data,
        "pair_lodi_with_dic": self._pair_lodi_with_dic,
        "first_img_dict": self._first_img_dict,
        "box_mask_dict": self._box_mask_dict,
        "box_points_dict": self._box_points_dict,
        "rings_mask_dict": self._rings_mask_dict,
        "ring_eta_dict": self._ring_eta_dict,
        "use_rings_mask_dict": self._use_rings_mask_dict,
        "curr_loadstep_nums": self._curr_loadstep_nums,
        "curr_img_data_dict": self._curr_img_data_dict,
        "cfg": self._cfg,
        "dic_output_prompt_dict": self.dic_output_prompt_dict,
        "lodi_par_prompt_dict": self.lodi_par_prompt_dict
        }

        return str(class_dict)
    
    def open_output_dir(self, output_dir=None):
        print("Opening output directory")
        if output_dir is not None and os.path.exists(output_dir):
            print("Using specificied output: %s" %(output_dir))
        else:
            root = tk.Tk()
            root.withdraw()
            
            output_dir = tk.filedialog.askdirectory(initialdir=__file__,
                                                     title='Open Output Directory')
        
        if not output_dir:
            print("Aborted opening output directory")
            pass
        else:
            try:
                self.raw_output_dir = output_dir  
                return self.raw_output_dir
            except Exception as e:
                print(e)
                warnings.warn("Something went wrong when choosing output directory")
                pass
    
    def open_config_from_file(self, config_dir=None):
        print("Open config file")
        if config_dir is not None and os.path.exists(config_dir):
            print("Using specificied config file: %s" %(config_dir))
        else:
            root = tk.Tk()
            root.withdraw()
            
            config_dir = tk.filedialog.askopenfilename(initialdir=self.raw_output_dir,
                                                       title='Select Configuration File')
        if not config_dir:
            print("Aborted opening config file")
            pass
        else:
            try:
                self.cfg = config.open(config_dir)[0]
                return self.cfg
            except Exception as e:
                print(e)
                warnings.warn("Something went wrong when loading configuration file")
                pass
    
    def open_dic_output_file(self, dic_output_json_dir=None, dic_output_txt_dir=None, dic_output_cols=None):
        root = tk.Tk()
        root.withdraw()
        # user selects json file to use
        if dic_output_json_dir is not None and os.path.exists(dic_output_json_dir):
            print("Using specificied json: %s" %(dic_output_json_dir))
        else:
            dic_output_json_dir = tk.filedialog.askopenfilename(initialdir=self.dic_output_json_dir,
                                                        defaultextension=".json",
                                                        filetypes=[("dic output json files", "*.json"),
                                                                   ("All Files", "*.*")],
                                                        title="Select dic_output.json File")
        # user selects par file to use
        if dic_output_txt_dir is not None and os.path.exists(dic_output_txt_dir):
            print("Using specificied txt: %s" %(dic_output_txt_dir))
        else:
            dic_output_txt_dir = tk.filedialog.askopenfilename(initialdir=self.dic_output_txt_dir,
                                                        defaultextension=".txt",
                                                        filetypes=[("dic output txt files", "*.txt"),
                                                                   ("All Files", "*.*")],
                                                        title="Select dic_output.txt File")
                                                        
        if not dic_output_json_dir or not dic_output_txt_dir:
            quit()
        else:
            try:
            #if True:
                self.dic_output_json_dir = dic_output_json_dir
                json_file = open(dic_output_json_dir)
                json_dict = json.load(json_file)
                
                print("\nColumn and Key Pairs for dic_output")
                print(*json_dict.items())
                
                # TODO: Remove these hardcoded list and store as constants for object
                if dic_output_cols is not None and len(dic_output_cols) == len(self.dic_output_prompt_dict):
                    self.dic_output_cols = dic_output_cols
                else:
                    self.dic_output_cols = [None] * len(self.dic_output_prompt_dict)
                    for key in self.dic_output_prompt_dict.keys():
                        prompt = input("\nChose %s column to import. Press q to quit. \n" %(key))
                        if input == "q":
                            break
                        self.dic_output_cols[self.dic_output_prompt_dict[key]] = int(prompt)
                print("DIC output columns selected: ", self.dic_output_cols)
                self.dic_output_txt_dir = dic_output_txt_dir
                self.dic_output_data = self.process_dic_output_file(self.dic_output_txt_dir, self.dic_output_cols)
            except Exception as e:
            #else:
                print(e)
                self.open_dic_output_file(dic_output_json_dir=dic_output_json_dir, dic_output_txt_dir=dic_output_txt_dir)
    
    def process_dic_output_file(self, dic_output_txt_dir, dic_output_cols):
        df = pd.read_csv(dic_output_txt_dir, sep="\s+|\t", header=None, engine='python')
        return np.array(df)[:, dic_output_cols]    
    
    def open_lodi_par_file(self, lodi_json_dir=None, lodi_par_dir=None, lodi_par_cols=None):
        root = tk.Tk()
        root.withdraw()
        # user selects json file to use
        if lodi_json_dir is not None and os.path.exists(lodi_json_dir):
            print("Using specificied json: %s" %(lodi_json_dir))
        else:
            lodi_json_dir = tk.filedialog.askopenfilename(initialdir=self.lodi_json_dir,
                                                          defaultextension=".json",
                                                        filetypes=[("lodi par json files", "*.json"),
                                                                   ("All Files", "*.*")],
                                                        title="Select lodi.json File")
        # user selects par file to use
        if lodi_par_dir is not None and os.path.exists(lodi_par_dir):
            print("Using specificied par: %s" %(lodi_par_dir))
        else:
            lodi_par_dir = tk.filedialog.askopenfilename(initialdir=self.lodi_par_dir,
                                                        defaultextension=".par",
                                                        filetypes=[("lodi par files", "*.par"),
                                                                   ("All Files", "*.*")],
                                                        title="Select lodi.par File")
                                                        
        if not lodi_json_dir or not lodi_par_dir:
            quit()
        else:
            try:
                self.lodi_json_dir = lodi_json_dir
                json_file = open(lodi_json_dir)
                json_dict = json.load(json_file)
                
                print("\nColumn and Key Pairs for lodi par")
                print(*json_dict.items())
                
                # TODO: Remove these hardcoded list and store as constants for object
                if lodi_par_cols is not None and len(lodi_par_cols) == len(self.lodi_par_prompt_dict):
                    self.lodi_par_cols = lodi_par_cols
                else:
                    self.lodi_par_cols = [None] * len(self.lodi_par_prompt_dict)
                    for key in self.lodi_par_prompt_dict.keys():
                        prompt = input("\nChose %s column to import. Press q to quit. \n" %(key))
                        if input == "q":
                            break
                        self.lodi_par_cols[self.lodi_par_prompt_dict[key]] = int(prompt)
                print("lodi par columns selected: ", self.lodi_par_cols)
                self.lodi_par_dir = lodi_par_dir
                self.lodi_par_data = self.process_lodi_par_file(self.lodi_par_dir, self.lodi_par_cols)
            except Exception as e:
                print(e)
                self.open_lodi_par_file(lodi_json_dir=lodi_json_dir, lodi_par_dir=lodi_par_dir)
    
    def process_lodi_par_file(self, lodi_par_file, lodi_par_cols):
        df = pd.read_csv(lodi_par_file, sep=" ", header=None)
        return np.array(df)[:, lodi_par_cols]
    
    def save_mask_dict_to_file(self, mask_dir=None):
        print("Save RealTimePCA Mask File")
        if mask_dir is None:
            root = tk.Tk()
            root.withdraw()
            
            mask_dir = tk.filedialog.asksaveasfilename(confirmoverwrite=False,
                                                       initialdir=self.raw_output_dir,
                                                       title='Save RealTimePCA Mask File')
        
        if not mask_dir:
            pass
        else:
            try:
                with open(mask_dir, "wb") as output_file:
                    pickle.dump([self.box_points_dict, self.use_rings_mask_dict], output_file)    
            except Exception as e:
                print(e)
                warnings.warn("Something went wrong when saving RealTimePCA Box ROI file")
                pass
        
    def load_mask_dict_from_file(self, mask_dir=None):
        print("Load RealTimePCA Mask File")
        if mask_dir is None:
            root = tk.Tk()
            root.withdraw()
            
            mask_dir = tk.filedialog.askopenfilename(initialdir=self.raw_output_dir,
                                                           title='Select RealTimePCA Mask File')
            
        if not mask_dir:
            pass
        else:
            try:
                with open(mask_dir, "rb") as input_file:
                     e = pickle.load(input_file)
                     self.box_points_dict = e[0]
                     self.use_rings_mask_dict = e[1]
            except Exception as e:
                print(e)
                warnings.warn("Something went wrong when loading RealTimePCA Mask file")
                pass
             
    def save_lodi_exp_to_file(self, lodi_exp_dir=None):
        print("Save RealTimePCA LODI Experiment Object")
        if lodi_exp_dir is None:
            root = tk.Tk()
            root.withdraw()
            
            lodi_exp_dir = tk.filedialog.asksaveasfilename(confirmoverwrite=False,
                                                       initialdir=self.raw_output_dir,
                                                       title='Save RealTimePCA LODI Experiment')
        
        if not lodi_exp_dir:
            pass
        else:
            try:
                with open(lodi_exp_dir, "wb") as output_file:
                    pickle.dump(self, output_file, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(e)
                warnings.warn("Something went wrong when saving RealTimePCA LODI Experiment")
                pass
    
    def load_lodi_exp_from_file(self, lodi_exp_dir=None):
        print("Load RealTimePCA LODI Experiment Object")
        if lodi_exp_dir is None:
            root = tk.Tk()
            root.withdraw()
            
            lodi_exp_dir = tk.filedialog.askopenfilename(initialdir=self.raw_output_dir,
                                                           title='Select RealTimePCA LODI Experiment File')
            
        if not lodi_exp_dir:
            pass
        else:
            try:
                with open(lodi_exp_dir, "rb") as input_file:
                     self = pickle.load(input_file)
                     # self.raw_img_stem = e.raw_img_stem
                     # self.raw_output_dir = e.raw_output_dir
                     # self.dic_output_json_dir = e.dic_output_json_dir
                     # self.dic_output_txt_dir = e.dic_output_txt_dir
                     # self.dic_output_cols = e.dic_output_cols
                     # self.lodi_json_dir = e.lodi_json_dir
                     # self.lodi_par_dir = e.lodi_par_dir
                     # self.lodi_par_cols = e.lodi_par_cols
                     # self.pair_lodi_with_dic = e.pair_lodi_with_dic
                     # self.first_img_dict = e.first_img_dict
                     # self.box_mask_dict = e.box_mask_dict
                     # self.box_points_dict = e.box_points_dict
                     # self.rings_mask_dict = e.rings_mask_dict
                     # self.use_rings_mask_dict = e.use_rings_mask_dict
                     # self.curr_img_path_dict = e.curr_img_path_dict
                     # self.curr_img_data_dict = e.curr_img_data_dict
                     # self.cfg = e.cfg
            except Exception as e:
                print(e)
                warnings.warn("Something went wrong when loading RealTimePCA Mask file")
                pass
        
        

