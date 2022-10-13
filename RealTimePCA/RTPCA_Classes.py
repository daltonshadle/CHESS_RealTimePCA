# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# %% ***************************************************************************
# IMPORTS
import os

import glob

import warnings

import pickle

import numpy as np

import tkinter as tk

import matplotlib.pyplot as plt

from hexrd import imageseries
from hexrd import config


# ***************************************************************************
# CLASS DECLARATION
class lodi_experiment():
    def __init__(self,
                 img_stem='',
                 output_dir='',
                 first_img_dict={},
                 box_mask_dict={},
                 box_points_dict={},
                 rings_mask_dict={},
                 use_rings_mask_dict={},
                 curr_img_path_dict={},
                 curr_img_data_dict={},
                 cfg=config):

        # initialize class variables
        self._img_stem = img_stem
        self._output_dir = output_dir
        self._first_img_dict = first_img_dict
        self._box_mask_dict = box_mask_dict
        self._box_points_dict = box_points_dict
        self._rings_mask_dict = rings_mask_dict
        self._use_rings_mask_dict = use_rings_mask_dict
        self._curr_img_path_dict = curr_img_path_dict
        self._curr_img_data_dict = curr_img_data_dict
        self._cfg = cfg
        
    # SETTERS AND GETTERS *****************************************************
    @property
    def img_stem(self):
        return self._img_stem
    @img_stem.setter
    def img_stem(self, img_stem):
        self._img_stem = img_stem
    
    @property
    def output_dir(self):
        return self._output_dir
    @output_dir.setter
    def output_dir(self, output_dir):
        if os.path.exists(output_dir):
            self._output_dir = output_dir
        else:
            raise ValueError("Directory '%s' does not exists" % (output_dir))
    
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
    def use_rings_mask_dict(self):
        return self._use_rings_mask_dict
    @use_rings_mask_dict.setter
    def use_rings_mask_dict(self, use_rings_mask_dict):
        self._use_rings_mask_dict = use_rings_mask_dict
    
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
    
    def is_frame_cache(self):
       return self.first_img_dict[self.det_keys[0]].endswith('.npz')
    
    
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
            
            panel_mask = np.zeros(panel_tth.shape)
            for tth in pd.getTThRanges():
                panel_mask += ((panel_tth > tth[0]) & (panel_tth < tth[1]))
            
            self.rings_mask_dict[det_key] = panel_mask
    
    def total_img_mask_dict(self):
        img_mask_dict = {}
        
        if self.rings_mask_dict.keys() != self.det_keys():
            print(self.rings_mask_dict.keys())
            print(self.det_keys())
            self.calc_rings_mask_dict()
            
        self.calc_box_mask_dict()
        
        for det_key in self.det_keys():
            if self.use_rings_mask_dict[det_key]:
                img_mask_dict[det_key] = (self.box_mask_dict[det_key] + self.rings_mask_dict[det_key]).astype(bool)
            else:
                img_mask_dict[det_key] = (self.box_mask_dict[det_key]).astype(bool)
        return img_mask_dict
    
    
    # IMAGE FUNCITONS *********************************************************
    def open_first_image(self):
        first_img_dict = {}
        for det_key in self.det_keys():
            root = tk.Tk()
            root.withdraw()
            path = self.img_stem.split('*')[0]

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
    
    def get_all_image_paths_dict(self):
        # path = '/home/djs522/additional_sw/RealTimePCA/CHESS_RealTimePCA/example/*%s*.npz' %('ff1')
        
        all_image_paths_dict = {}
        
        for det_key in self.det_keys():
            det_image_path_stem = self.img_stem %(det_key)
            sort_split = det_image_path_stem.split('*')

            det_files = glob.glob(det_image_path_stem)
            sort_det_files = []
            
            # sorting on last * in image_path_stem as that probably has the scan number
            for f in det_files:
                sort_det_files.append(f.split(sort_split[-2])[1].split(sort_split[-1])[0])
            
            ind = np.argsort(sort_det_files)
            
            all_image_paths_dict[det_key] = np.array(det_files)[ind.tolist()].tolist()
        
        # TODO: Need a seperate function for UPDATING paths and data to just append
        self.curr_img_path_dict = all_image_paths_dict
    
    def load_ims_from_path(self, path):
        if self.is_frame_cache:
            # frame cache
            ims = imageseries.open(path, format='frame-cache')
        else:
            # raw
            ims = imageseries.open(path, format='hdf5', path='/imageseries')
        return ims
    
    def load_img_list(self, ims_length=2, frane_num_or_img_aggregation_options=None):
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
        return "pca_matrices()\n" + self.__str__()

    def __str__(self):
        class_dict = {'pca_par_mat': self._pca_par_mat,
                      'box_points_dict': self._box_points_dict}

        return str(class_dict)
    
    def open_output_dir(self):
        print("Opening output directory")
        root = tk.Tk()
        root.withdraw()
        
        output_dir = tk.filedialog.askdirectory(initialdir=__file__,
                                                 title='Open Output Directory')
        
        if not output_dir:
            pass
        else:
            try:
                self.output_dir = output_dir  
                return self.output_dir
            except Exception as e:
                print(e)
                warnings.warn("Something went wrong when schoosing output directory")
                pass
    
    def load_config_from_file(self, config_dir=None):
        print("Load Config File")
        if config_dir is None:
            root = tk.Tk()
            root.withdraw()
            
            config_dir = tk.filedialog.askopenfilename(initialdir=self.output_dir,
                                                       title='Select Configuration File')
        if not config_dir:
            pass
        else:
            try:
                self.cfg = config.open(config_dir)[0]
                return self.cfg
            except Exception as e:
                print(e)
                warnings.warn("Something went wrong when loading configuration file")
                pass
    
    def save_mask_dict_to_file(self, mask_dir=None):
        print("Save RealTimePCA Mask File")
        if mask_dir is None:
            root = tk.Tk()
            root.withdraw()
            
            mask_dir = tk.filedialog.asksaveasfilename(confirmoverwrite=False,
                                                       initialdir=self.output_dir,
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
            
            mask_dir = tk.filedialog.askopenfilename(initialdir=self.output_dir,
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
                                                       initialdir=self.output_dir,
                                                       title='Save RealTimePCA LODI Experiment')
        
        if not lodi_exp_dir:
            pass
        else:
            try:
                with open(lodi_exp_dir, "wb") as output_file:
                    pickle.dump(self, output_file)
            except Exception as e:
                print(e)
                warnings.warn("Something went wrong when saving RealTimePCA LODI Experiment")
                pass
    
    def load_lodi_exp_from_file(self, lodi_exp_dir=None):
        print("Load RealTimePCA LODI Experiment Object")
        if lodi_exp_dir is None:
            root = tk.Tk()
            root.withdraw()
            
            lodi_exp_dir = tk.filedialog.askopenfilename(initialdir=self.output_dir,
                                                           title='Select RealTimePCA LODI Experiment File')
            
        if not lodi_exp_dir:
            pass
        else:
            try:
                with open(lodi_exp_dir, "rb") as input_file:
                     e = pickle.load(input_file)
                     self.img_stem = e.img_stem
                     self.output_dir = e.output_dir
                     self.first_img_dict = e.first_img_dict
                     self.box_mask_dict = e.box_mask_dict
                     self.box_points_dict = e.box_points_dict
                     self.rings_mask_dict = e.rings_mask_dict
                     self.use_rings_mask_dict = e.use_rings_mask_dict
                     self.curr_img_path_dict = e.curr_img_path_dict
                     self.curr_img_data_dict = e.curr_img_data_dict
                     self.cfg = e.cfg
            except Exception as e:
                print(e)
                warnings.warn("Something went wrong when loading RealTimePCA Mask file")
                pass
        
        

