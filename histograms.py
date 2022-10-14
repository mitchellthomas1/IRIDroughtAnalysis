#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 15:21:54 2022

@author: Mitchell
"""

from builddata import aggregate_yearly, return_data, normalize_array, log10_zerosnan
import numpy as np
import matplotlib.pyplot as plt


# dictionary with data array and info about that data
data_dict = return_data() 



# normalize data
normalized_arr = normalize_array(data_dict['imagearray'],
                                  transform = log10_zerosnan)
data_dict['imagearray'] = normalized_arr

#Kiremt:  June-Sept
# Belg: Feb-May

def clip_months(data_dict, month_arr):
    '''
    Parameters
    ----------
    data_dict : dictionary with image array and ymp
    
    month_arr : array-like of ints
        months to select (inclusive)

    Returns
    -------
    sel_months_arr : ndarray
        data_array with select months

    '''
    arr = data_dict['imagearray']
    ymp = np.array(data_dict['ymp']).astype(int)
    sel = np.isin (ymp[:,1] , month_arr)
    sel_months_arr = arr[sel,:,:]
    return sel_months_arr



kiremt_arr = clip_months(data_dict,  np.arange(6,10) )
belg_arr = clip_months(data_dict,   np.arange(2,6) )
whole_season  = clip_months(data_dict,   np.arange(2,10) )

f, (ax1, ax2, ax3 )= plt.subplots(1,3,figsize = (14,4))
ax1.hist(kiremt_arr.flatten(), bins = 60, range = (-3,3))
ax1.set_title('Kiremt (whole roi)')
ax2.hist(belg_arr.flatten(), bins = 60, range = (-3,3))
ax2.set_title('Belg (whole roi)')
ax3.hist(whole_season.flatten(), bins = 60, range = (-3,3))
ax3.set_title('Feb-Sept (whole roi)')

for ax in (ax1, ax2, ax3):
    ax.set_xlabel('log10(rainfall)')
    ax.set_ylim([0,270000])

plt.show()













