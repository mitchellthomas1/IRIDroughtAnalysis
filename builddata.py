#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 10:35:18 2022

@author: Mitchell
"""

import numpy as np
import os 





path = '/Users/Mitchell/IRI/IRIFall2022/Export/export_files/CHIRPS_EthiopiaAOI/'

image_folder = 'CHIRPS_Pentad_final/'





def return_data():
    lon = np.genfromtxt( path + "AOI_longitude.csv", delimiter=',')
    lat = np.genfromtxt( path + "AOI_latitude.csv", delimiter=',')
    
    files_raw = np.array(os.listdir(path + image_folder))
    files = np.sort( np.delete( files_raw, np.where(files_raw == '.DS_Store')[0][0] ))
    ymp = [ (f.split('_')[1][1:5],  f.split('_')[1].split('m')[1].split('p')[0] ,
                f.split('_')[1].split('m')[1].split('p')[1])  for f in files]
    image_arr = np.array([ np.genfromtxt( path + image_folder + f, delimiter=',') for f in files])
    # image_arr_reshaped = np.transpose(image_arr, (1,2,0))
        
    return {'latitude': lat,
            'longitude':lon,
            'imagearray': image_arr,
            'filenames':files,
            'ymp': ymp}

def log10_zerosnan(m):
    '''
    function which applies the numpy log10 to an array, keeping zeros set to 0
    '''
    o = np.empty_like(m)
    o[:] = np.nan
    return np.log10(m, out=o, where=(m!=0))

def normalize_array(data_array, transform = None):
    '''
    Parameters
    ----------
    data_array : 3-d numpy array 
        specifically a stack of rasters with time, lon (x), lat(y)
        data to get normalized
    
    transform: function(ndarray), (optional) :
        optional transformation function to apply across the data before transofrmation
        i.e. np.log10()
        

    Returns
    -------
    data_norm: a normalized version of that 3-d array, where every "image" in the time dimension is 
    normalized to 
    '''
    
    if transform != None:
        transformed_arr = transform(data_array)
    else:
        transformed_arr = data_array
    mean = np.nanmean(transformed_arr, axis = 0)
    std = np.nanstd(transformed_arr, axis = 0)
    data_norm = (transformed_arr - mean ) / std
    if data_norm.shape != data_array.shape:
        raise (ValueError)
    
    
    
    
    return data_norm


def aggregate_yearly(data_dict):
    #dataset parameters
    years = np.arange(1981,2022)
    months = np.arange(4,12) 
    
    arr = data_dict['imagearray']
    ymp = np.array(data_dict['ymp']).astype(int)
    yearly_sums = np.zeros((len(years), *arr[0].shape))
    for i, y in enumerate(years):
        sel = (ymp[:,0] == y) & np.isin (ymp[:,1] , months)
        year_arr = arr[sel,:,:]
        y_sum = np.sum(year_arr, axis = 0)
        yearly_sums[i, :,:] = y_sum
        
    print('Mean map')
    plt.imshow(np.nanmean(yearly_sums, axis = 0), 
               cmap= 'Blues', interpolation='None')
    plt.show()
    print('Std map')
    plt.imshow(np.nanstd(yearly_sums, axis = 0), 
               cmap= 'bwr', interpolation='None')
    plt.show()
    
    
        
        
    
    


