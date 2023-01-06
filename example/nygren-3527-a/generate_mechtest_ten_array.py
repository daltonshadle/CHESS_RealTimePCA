#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 12:54:58 2022

@author: djs522
"""

import numpy as np
'''
mechtest_ten						
increment_int	cycle_num	scan_type 	ten_target_abs	ten_speed	mechtest_array_file_id (integer from json)	ndics

scan_type = (ff:1, dic-lodi:2, move_ten:3, dic_only:4)		
mechtest_array_file_id 	= (ff:1, dic-lodi:2, move_ten:0, dic_only:2)	
'''

'''
mechtest_ten						
increment_int	cycle_num	scan_type 	ten_target_abs	ten_speed	mechtest_array_file_id (integer from json)	ndics

scan_type = (ff:1, dic-lodi:2, move_ten:3, dic_only:4)		
mechtest_array_file_id 	= (ff:1, dic-lodi:2, move_ten:0, dic_only:2)	
'''

ten_abs = -23.555
comp_abs = -23.680

#ten_abs_dict = {-1:-23.680, 0:-23.635, 1:-23.555}
umvr_off_yield_compression = 0.005
umvr_off_yield_tension = -0.005
num_lodi_dic_between = 20
ten_rate = 0.004 #mm/s

mechtest_ten_strem = "%i %i %i %0.3f %0.3f %i %i"
#mechtest_ten_strem %(incre, cyc_num, type, ten_abs, ten_speed, array_id, ndic)
# 1% strain to 0% strain

#%% CYCLE TENSION OFF YIELD TO COMPRESSION TIP TO TENSION TIP TO TENSION OFF YIELD
print("#TENSION -> COMPRESSION -> TENSION (%i Total DIC-LODI)" %(num_lodi_dic_between*2+1))
j = 1
disp_ten2comp = np.linspace(ten_abs + umvr_off_yield_tension, comp_abs, num=num_lodi_dic_between)
for i in range(num_lodi_dic_between):
    print(mechtest_ten_strem %(j, 2, 2, disp_ten2comp[i], ten_rate, 2, 1))
    j = j + 1

disp_comp2ten = np.linspace(comp_abs + umvr_off_yield_compression, ten_abs, num=num_lodi_dic_between)
for i in range(num_lodi_dic_between):
    print(mechtest_ten_strem %(j, 2, 2, disp_comp2ten[i], ten_rate, 2, 1))
    j = j + 1

print(mechtest_ten_strem %(j, 2, 2, ten_abs + umvr_off_yield_tension, ten_rate, 2, 1))
j = j + 1


#%% CYCLE COMPRESSION OFF YIELD TO TENSION TIP TO COMPRESSION TIP TO COMPRESSION OFF YIELD
print("#COMPRESSION -> TENSION -> COMPRESSION (%i Total DIC-LODI)" %(num_lodi_dic_between*2+1))
j = 1
disp_comp2ten = np.linspace(comp_abs + umvr_off_yield_compression, ten_abs, num=num_lodi_dic_between)
for i in range(num_lodi_dic_between):
    print(mechtest_ten_strem %(j, 2, 2, disp_comp2ten[i], ten_rate, 2, 1))
    j = j + 1

disp_ten2comp = np.linspace(ten_abs + umvr_off_yield_tension, comp_abs, num=num_lodi_dic_between)
for i in range(num_lodi_dic_between):
    print(mechtest_ten_strem %(j, 2, 2, disp_ten2comp[i], ten_rate, 2, 1))
    j = j + 1

print(mechtest_ten_strem %(j, 2, 2, comp_abs + umvr_off_yield_compression, ten_rate, 2, 1))
j = j + 1