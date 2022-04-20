#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 16:02:23 2020

@author: djs522
"""

#%% ***************************************************************************
# IMPORTS
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn.preprocessing import Normalizer

from hexrd import imageseries


#%% ***************************************************************************
# FUNCTION DECLARATION


#%% ***************************************************************************
# CONSTANTS AND PARAMETERS
sync_ct_size = 47792228
slew_ome_size = 1051091300
full_scan_size = 34518329364

debug = True 

expt_name = 'miller-881-2'
samp_name = 'dp718-1'
base_path = '/nfs/chess/raw/current/id3a/%s/%s/' %(expt_name, samp_name)
scan_data_path = os.path.join(base_path, '/%d/ff/')
start_scan = 22
end_scan = 960
dic_par_path = os.path.join(base_path, 'dic.par')


#%% ***************************************************************************
# PREPROCESSING

# read in dic file
dic_mat = np.loadtxt(dic_par_path)
print(dic_mat[0:3, :])

# initalize match array [image_num, load, screw_pos, image_index, scan_num]
dic_scan_match = []

# start walking along scans
for (dir_path, dir_names, filenames) in os.walk(base_path):
    # remove base path
    temp_path = dir_path.split(base_path)[1]
    
    # check if actual ff scan (not snapshot, not base_path, in ff dir not nf)
    if (not 'snapshot' in temp_path) and ('ff' in temp_path):
        scan_num = int(temp_path.split('/')[0])
        
        # check if the scan number is in the right region
        if scan_num >= start_scan:
            # grab the first scan file and size
            fs_file = filenames[0]
            fs_file_stat = os.stat(os.path.join(dir_path, fs_file))
            fs_file_size = fs_file_stat.st_size
            fs_file_mtime = fs_file_stat.st_mtime

            if fs_file_size == sync_ct_size:
                print(scan_num)
                # find where dic_mat[:, 2] - fs_file_mtime = least negative number
                time_diff = dic_mat[:, 2] - fs_file_mtime
                
                # remove positive values
                reduce_dic_mat = dic_mat[time_diff < 0, :]
                reduce_time_diff = time_diff[time_diff < 0]

                # find least negative number
                scan_dic_info = reduce_dic_mat[np.argmax(reduce_time_diff), :]
                
                # create scan dic info item
                temp_scan_dic_list = scan_dic_info[[3, 4, 5]].tolist()
                dic_image_index = np.where((dic_mat == scan_dic_info).all(axis=1))[0][0]
                temp_scan_dic_list.append(dic_image_index)
                temp_scan_dic_list.append(scan_num)
                
                # add to big list
                dic_scan_match.append(temp_scan_dic_list)
        
        # assuming that the files are in order, we can stop early if we are above the end scan
        if scan_num > end_scan:
            break

print('Saving array...')
dic_scan_match_arr = np.array(dic_scan_match)
np.save('sync_ct_scans_2_dic_dp718.npy', dic_scan_match_arr)







                



