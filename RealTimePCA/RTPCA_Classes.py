# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

#%% ***************************************************************************
# IMPORTS
import os

import numpy as np

import pickle

import json

import pandas as pd

import tkinter as tk

# ***************************************************************************
# CLASS DECLARATION
class pca_matrices():
    def __init__(self, 
                 pca_par_mat=np.array([]), 
                 box_points=np.array([])):
        
        # initialize class variables
        self._pca_par_mat = pca_par_mat
        self._box_points= box_points
    
    # setters and getters
    @property
    def pca_par_mat(self):
        return self._pca_par_mat
    @pca_par_mat.setter
    def pca_par_mat(self, pca_par_mat):
        self._pca_par_mat = pca_par_mat
    
    @property
    def box_points(self):
        return self._box_points
    @box_points.setter
    def box_points(self, box_points):
        self._box_points = box_points
    
    def calc_bounding_box(self):
        if len(self._box_points.shape) == 2:
            return np.array([[np.min(self._box_points[:, 0]), np.min(self._box_points[:, 1])],
                             [np.max(self._box_points[:, 0]), np.max(self._box_points[:, 1])]])
        else:
            return np.zeros([2, 2])
    
    # str and rep
    def __repr__(self):
        return "pca_matrices()\n" + self.__str__()
    def __str__(self):
        class_dict = {'pca_par_mat' : self._pca_par_mat,
        'box_points' : self._box_points}
        
        return str(class_dict)

class pca_paths():
    def __init__(self, 
                 base_dir=os.getcwd(),
                 img_dir=os.getcwd(),
                 img_fname_template='pca_%06i.tif',
                 output_dir=os.getcwd(),
                 output_fname='output.txt'
                 ):
        
        # intialize class variables
        self._base_dir = base_dir
        self._img_dir = img_dir
        self._img_fname_template = img_fname_template
        self._output_dir = output_dir
        self._output_fname = output_fname
        self._first_img_num = 0
        
    # setters and getters  
    @property
    def base_dir(self):
        return self._base_dir
    @base_dir.setter
    def base_dir(self, base_dir):
        if os.path.exists(base_dir):
            self._base_dir = base_dir
        else:
            raise ValueError("Base directory '%s' does not exists" %(base_dir))
    
    @property
    def img_dir(self):
        return self._img_dir
    @img_dir.setter
    def img_dir(self, img_dir):
        if os.path.exists(img_dir):
            self._img_dir = img_dir
        else:
            raise ValueError("Img directory '%s' does not exists" %(img_dir))
    
    @property
    def img_fname_template(self):
        return self._img_fname_template
    @img_fname_template.setter
    def img_fname_template(self, img_fname_template):
        self._img_fname_template = img_fname_template
    
    @property
    def output_dir(self):
        return self._output_dir
    @output_dir.setter
    def output_dir(self, output_dir):
        if os.path.exists(output_dir):
            self._output_dir = output_dir
        else:
            raise ValueError("Output directory '%s' does not exists" %(output_dir))
    
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
    def first_img_num(self):
        return self._first_img_num
    @first_img_num.setter
    def first_img_num(self, first_img_num):
        self._first_img_num = first_img_num 
    
    def open_first_image(self):
        root = tk.Tk()
        root.withdraw()
        
        first_img_dir = tk.filedialog.askopenfilename(initialdir=self._img_dir, 
                                                    defaultextension='.tiff',
                                                    filetypes=[("TIFF Files", "*.tiff"),
                                                               ("TIFF Files", "*.tif"),
                                                               ("All Files", "*.*")],
                                                    title='Select First pca Image File')
        
        if not first_img_dir:
            quit()
        else:
            try:
                self.img_dir = os.path.dirname(first_img_dir)
                first_img_fname = os.path.basename(first_img_dir)
                self.first_img_num = int((first_img_fname.split('_')[-1]).split('.')[0])
            except Exception as e:
                print(e)
                self.open_first_image()
    
    def get_img_num_dir(self, img_num):
        return os.path.join(self._img_dir, self._img_fname_template %(img_num))
    def get_first_img_dir(self):
        return self.get_img_num_dir(self._first_img_num)
    def get_output_full_dir(self):
        return os.path.join(self._output_dir, self._output_fname)
    
    # str and rep
    def __repr__(self):
        return "pca_paths()\n" + self.__str__()
    def __str__(self):
        class_dict = {'base_dir' : self._base_dir,
        'img_dir' : self._img_dir,
        'img_fname_template' : self._img_fname_template,
        'output_dir' : self._output_dir,
        'output_fname' : self._output_fname,
        'first_img_num' : self._first_img_num}
        
        return str(class_dict)

