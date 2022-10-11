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

from hexrd import imageseries
from hexrd import config


# ***************************************************************************
# CLASS DECLARATION
class pca_matrices():
    def __init__(self,
                 pca_par_mat=np.array([]),
                 box_points_dict={},
                 ims_list_dict={},
                 pca_matrix=np.array([]),
                 det_keys=['det']):

        # initialize class variables
        self._pca_par_mat = pca_par_mat
        self._box_points_dict = box_points_dict
        self._ims_list_dict = ims_list_dict
        self._pca_matrix = pca_matrix
        self._det_keys = list(det_keys)
        
        if len(self._det_keys) != len(self._box_points_dict):
            self.reset_box_points_dict()

    # setters and getters
    @property
    def pca_par_mat(self):
        return self._pca_par_mat
    @pca_par_mat.setter
    def pca_par_mat(self, pca_par_mat):
        self._pca_par_mat = pca_par_mat

    @property
    def box_points_dict(self):
        return self._box_points_dict
    @box_points_dict.setter
    def box_points_dict(self, box_points_dict):
        self._box_points_dict = box_points_dict

    @property
    def ims_list_dict(self):
        return self._ims_list_dict
    @ims_list_dict.setter
    def ims_list_dict(self, ims_list_dict):
        self._ims_list_dict = ims_list_dict

    @property
    def pca_matrix(self):
        return self._pca_matrix
    @pca_matrix.setter
    def pca_matrix(self, pca_matrix):
        self._pca_matrix = pca_matrix
    
    @property
    def det_keys(self):
        return self._det_keys
    @det_keys.setter
    def det_keys(self, det_keys):
        self._det_keys = det_keys
    
    # other functions
    def reset_box_points_dict(self):
        self._box_points_dict = {}
        for det_key in self.det_keys:
            self._box_points_dict[det_key] = np.array([])
    
    def make_det_image_mask(self, img_size):
        img_mask_dict = {}
        
        for det_key in self.det_keys:
            if self._box_points_dict[det_key].size == 0:
                img_mask = np.ones(img_size)
            else:
                img_mask = np.zeros(img_size)
                for i in range(self._box_points_dict[det_key].shape[0]):
                    pts = self._box_points_dict[det_key][i, :, :]
                    pts = np.floor(pts).astype(int)
                    img_mask[pts[0, 0]:pts[1, 0], pts[0, 1]:pts[1, 1]] = 1
                
            img_mask_dict[det_key] = img_mask
        
        return img_mask_dict
    
    def load_ims_from_path(self, path, is_frame_cache=True):
        if is_frame_cache:
            ims = imageseries.open(path, format='frame-cache')
        else:
            ims = imageseries.open(path, format='raw-image')
        return ims
         
    def load_img_list(self, img_path_list_dict, img_mask_dict=None, ims_length=2, 
                      is_frame_cache=True, frane_num_or_img_aggregation_options=None):
        img_data_list_dict = {}
        
        for det_key in img_path_list_dict.keys():
            # !!! TODO: return list of indices actually used
            ims_list = []
            
            for img_path in img_path_list_dict[det_key]:
                ims = self.load_ims_from_path(img_path, is_frame_cache=is_frame_cache)
                
                if len(ims) == ims_length:
                    img_data = []
                    for i in range(ims_length):
                        if img_mask_dict is not None:
                            img_data.append(ims[i][img_mask_dict[det_key].astype(bool)].flatten())
                        else:
                            img_data.append(ims[i].flatten())
                        
                    ims_list.append(np.hstack(img_data))
            
            img_data_list_dict[det_key] = np.array(ims_list)
        
        self._ims_list_dict = img_data_list_dict
        return img_data_list_dict
        
        
    
    def load_pca_matrices_from_file(self, pca_mats_dir):
        with open(pca_mats_dir, "rb") as input_file:
             e = pickle.load(input_file)
             self._pca_par_mat = e.pca_par_mat
             self._box_points_dict = e.box_points_dict
             self._image_files = e.image_files
             self._pca_matrix = e.pca_matrix
             self._det_keys = e.det_keys
             
    def save_pca_matrices_to_file(self, pca_mats_dir):
        with open(pca_mats_dir, "wb") as output_file:
            pickle.dump(self, output_file)
    
    # str and rep
    def __repr__(self):
        return "pca_matrices()\n" + self.__str__()

    def __str__(self):
        class_dict = {'pca_par_mat': self._pca_par_mat,
                      'box_points_dict': self._box_points_dict}

        return str(class_dict)


class pca_paths():
    def __init__(self,
                 base_dir=os.getcwd(),
                 img_dir=os.getcwd(),
                 first_img_dict={'panel_id': 'image.npz'},
                 is_frame_cache=True, 
                 output_dir=os.getcwd(),
                 output_fname='output.txt',
                 config_fname=None
                 ):

        # /id1a3/ko-3371-a/c103-2-90-ff-1/3/ff/**image**.h5

        # intialize class variables
        self._base_dir = base_dir
        self._img_dir = img_dir

        self._first_img_dict = first_img_dict
        self._is_frame_cache = is_frame_cache

        self._output_dir = output_dir
        self._output_fname = output_fname

        self._config_fname = config_fname
        self._config = config.open(self._config_fname)[0]

    # setters and getters
    @property
    def base_dir(self):
        return self._base_dir

    @base_dir.setter
    def base_dir(self, base_dir):
        if os.path.exists(base_dir):
            self._base_dir = base_dir
        else:
            raise ValueError(
                "Base directory '%s' does not exists" % (base_dir))

    @property
    def img_dir(self):
        return self._img_dir

    @img_dir.setter
    def img_dir(self, img_dir):
        if os.path.exists(img_dir):
            self._img_dir = img_dir
        else:
            raise ValueError("Img directory '%s' does not exists" %
                             (img_dir))

    @property
    def first_img_dict(self):
        return self._first_img_dict

    @first_img_dict.setter
    def first_img_dict(self, first_img_dict):
        for img_key in first_img_dict.keys():
            if not os.path.exists(first_img_dict[img_key]):
                raise ValueError("Img directory '%s' does not exists" %(first_img_dict[img_key]))

        self._first_img_dict = first_img_dict
    
    @property
    def is_frame_cache(self):
        return self._is_frame_cache

    @is_frame_cache.setter
    def is_frame_cache(self, is_frame_cache):
        self._is_frame_cache = is_frame_cache
    
    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, output_dir):
        if os.path.exists(output_dir):
            self._output_dir = output_dir
        else:
            raise ValueError(
                "Output directory '%s' does not exists" % (output_dir))

    @property
    def output_fname(self):
        return self._output_fname

    @output_fname.setter
    def output_fname(self, output_fname):
        if output_fname.endswith('.txt'):
            self._output_fname = output_fname
        else:
            self._output_fname = output_fname + '.txt'

    @property
    def config_fname(self):
        return self._config_fname

    @config_fname.setter
    def config_fname(self, cfg_fname):
        self._config_fname = cfg_fname

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, cfg):
        self._config = cfg

    # other functions
    def open_first_image(self):
        det_keys = self.config.instrument.hedm.detectors.keys()
        first_img_dict = {}
        for det_key in det_keys:
            root = tk.Tk()
            root.withdraw()
            path = self.base_dir

            file_dir = tk.filedialog.askopenfilename(initialdir=path,
                                                     defaultextension=".npz",
                                                     filetypes=[("npz files", "*.npz"),
                                                                ('H5 files',
                                                                 '*.h5'),
                                                                ("All Files", "*.*")],
                                                     title="Select Image File for %s" % (det_key))

            if not file_dir:
                quit()
            else:
                try:
                    first_img_dict[det_key] = file_dir
                    if file_dir.endswith('.npz'):
                        self._is_frame_cache = True
                except Exception as e:
                    print(e)
                    self.open_first_image()
        self._first_img_dict = first_img_dict

    def get_all_image_paths_dict(self, image_path_stem):
        # path = '/home/djs522/additional_sw/RealTimePCA/CHESS_RealTimePCA/example/*%s*.npz' %('ff1')
        
        all_image_paths_dict = {}
        
        det_keys = self.config.instrument.hedm.detectors.keys()
        for det_key in det_keys:
            det_image_path_stem = image_path_stem %(det_key)
            sort_split = det_image_path_stem.split('*')

            det_files = glob.glob(det_image_path_stem)
            sort_det_files = []
            
            # sorting on last * in image_path_stem as that probably has the scan number
            for f in det_files:
                sort_det_files.append(f.split(sort_split[-2])[1].split(sort_split[-1])[0])
            
            ind = np.argsort(sort_det_files)
            
            all_image_paths_dict[det_key] = np.array(det_files)[ind.tolist()].tolist()
        
        return all_image_paths_dict
            
            

    # str and rep
    def __repr__(self):
        return "pca_paths()\n" + self.__str__()

    def __str__(self):
        class_dict = {'base_dir': self._base_dir,
                      'img_dir': self._img_dir,
                      'output_dir': self._output_dir,
                      'output_fname': self._output_fname,
                      'first_img_num': self._first_img_num,
                      'config_fname': self._config_fname,
                      'config': self._config}

        return str(class_dict)
