#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 13:21:38 2022

Script to calculate mock index values from cluster labels and precipitation array

@author: Mitchell
"""
from builddata import return_data
import numpy as np

'''
steps:
    1. input clusters from kmeans, 
    
'''
def make_date_array(start,end):
    '''
    start and end are mm_dd str
    '''
    s_m, e_m = int(start.split('_')[0]), int(end.split('_')[0])
    s_d, e_d = int(start.split('_')[1]), int(end.split('_')[1])
    date_li = []
    if s_m <= e_m:
        month_range = range(s_m, e_m+1)
    else:
        month_range = list(range(s_m, 13)) + list(range(1, e_m+1))
    for m in month_range:
        if m == s_m:
            for d in range(s_d, 32):
                date_li.append('{:02d}_{:02d}'.format(m, d))
        elif m == e_m:
            for d in range(1, e_d+1):
                date_li.append('{:02d}_{:02d}'.format(m, d))
        else:
            for d in range(1,32):
                date_li.append('{:02d}_{:02d}'.format(m, d))
    return date_li
        
        
                    
        
    

def spatially_average(cluster_labels, data_dict, time_periods):
    '''
    Calculates spatially averaged rainfall by spatial clusters 
    and time periods. Calculates over a m x n size study area.

    Parameters
    ----------
    cluster_labels : m x n array
        DESCRIPTION.
    precip_ts : t x m x n array
        DESCRIPTION.
    time_periods : dict
        Dictionary which maps time period name (str) 
            to a start, end tuple (mm_dd , mm_dd)

    Returns
    -------
    average_dict : TYPE
        DESCRIPTION.

    '''
    
    clusters = np.unique(cluster_labels)
    date_li = ['{}_{}'.format( file.split('_')[0][4:6], file.split('_')[0][6:8]) for file in data_dict['filenames']]
    
    average_dict = {} 
    
        
    for tp_key in time_periods.keys():
        average_dict[tp_key] = {}
        start, end = time_periods[tp_key]
        possible_dates = make_date_array(start,end)
        global temporal_bool
        temporal_bool = np.isin(date_li, possible_dates)
        
        for x in clusters:
            
            global spatial_bool
            spatial_bool = cluster_labels == x
            
            cluster_temporal_arr = data_dict['imagearray'][temporal_bool][:,spatial_bool]
            
            total_sum = np.nansum(cluster_temporal_arr)
            n_years = 41
            yearly_avg = total_sum / n_years
            
            average_dict[tp_key][x] = yearly_avg
            
            
            
    
    
    return average_dict



time_periods = {'Kiremt': ('06_01','08_31'),
                'Belg': ('03_01','05_15')}
data_dict = return_data(start_yr=1981, end_yr = 2021)
cluster_labels = np.ones_like(data_dict['imagearray'][0])
average_dict = spatially_average(cluster_labels, data_dict, time_periods)












