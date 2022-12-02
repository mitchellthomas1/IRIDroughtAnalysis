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
        
        
                    
        
    

def calculate_index(cluster_labels, data_dict, time_periods, yr_range):
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
    yr_range: tuple of ints
        start and end dates (inclusive) for computation

    Returns
    -------
    index_dict : doctionary
        dictionary which maps cluster name to season name to the 20th percentile and 
        trigger years

    '''
    
    clusters = np.unique(cluster_labels)
    date_li = ['{}_{}'.format( file.split('_')[0][4:6], file.split('_')[0][6:8]) for file in data_dict['filenames']]
    
    index_dict = {} 
    
        
    for x in clusters:
        index_dict['Cluster_{}'.format(x)] = {}
        spatial_bool = cluster_labels == x
        cluster_data = data_dict['imagearray'][:, spatial_bool]
        print('cluster data shape', cluster_data.shape)
        
        for tp_key in time_periods.keys():
            index_dict['Cluster_{}'.format(x)][tp_key] = {}
            
            start, end = time_periods[tp_key]

            possible_toy = make_date_array(start,end)
            time_of_yr_bool = np.isin(date_li, possible_toy)
                
            yrs = np.array(list(range(yr_range[0], yr_range[1]+1)))
            yr_sum = np.zeros(yrs.shape)
            for i, yr in enumerate(yrs):
                image_yr_bool = np.array([int(f[0:4]) for f in data_dict['filenames']]) == yr
                assert len(image_yr_bool) == len(time_of_yr_bool)
                final_time_index = image_yr_bool & time_of_yr_bool
                year_toy_arr = cluster_data[final_time_index]
                year_toy_sum = np.nansum(year_toy_arr)
                yr_sum[i] = year_toy_sum
                
            
            #20th percentile
            
            perc_20 = np.percentile(yr_sum, 20)
            below_20 = yrs[np.array(yr_sum) < perc_20]
            print('TP key: ', tp_key, ' Cluster: ', x, ' perc 20: ', perc_20, ' years: ', below_20)
            
            index_dict['Cluster_{}'.format(x)][tp_key]['Perc20(mm)']= perc_20
            index_dict['Cluster_{}'.format(x)][tp_key]['TriggerYears']= below_20
                

    return index_dict



time_periods = {'Kiremt': ('06_01','08_31'),
                'Belg': ('03_01','05_15')}
yr_range = (1981, 2021)
data_dict = return_data(start_yr=yr_range[0], end_yr = yr_range[1])
cluster_labels = np.ones_like(data_dict['imagearray'][0])
 
average_dict = calculate_index(cluster_labels, data_dict, time_periods, yr_range)












