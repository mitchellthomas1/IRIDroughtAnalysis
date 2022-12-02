#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 10:35:18 2022

@author: Mitchell
"""

import numpy as np
import os 






path = 'Export/export_files/CHIRPS_EthiopiaAOI/'





def return_data(mode = None, spi_window = 0, start_yr =1981, end_yr=2022):
    '''
    

    Parameters
    ----------
    mode : str, optional
        set as 'spi' if using standardized precipitation index. The default is None.
    spi_window : int, optional
        Window for spi computation if mode == 'spi'. The default is None.

    Returns
    -------
    dict
        DESCRIPTION.

    '''
    if mode == 'spi':
        image_folder = 'CHIRPS_SPI_{}pentads/'.format(spi_window)
        print('returning data with Standardized Precipitation Index as TRUE')
    else:
    
        image_folder = 'CHIRPS_Pentad_final/'

        
    
    lon = np.genfromtxt( path + "AOI_longitude.csv", delimiter=',')
    lat = np.genfromtxt( path + "AOI_latitude.csv", delimiter=',')
    
    files_raw = np.array(os.listdir(path + image_folder))
    if '.DS_Store' in files_raw:    
        files = np.delete( files_raw, np.where(files_raw == '.DS_Store')[0][0] )
    else:
        files = files_raw
    files = np.sort(files)
    years = list(range(start_yr, end_yr+1))
    files = [x for x in files if int(x[0:4]) in years]
    
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

def divide_by_max(m):
    max_val = np.nanmax(m)
    return m / max_val

def normalize_array(data_array, transform = None, axis = 0):
    '''
    Parameters
    ----------
    data_array : 3-d numpy array 
        specifically a stack of rasters with time, lon (x), lat(y)
        data to get normalized
    
    transform: function(ndarray), (optional) :
        optional transformation function to apply across the data before transofrmation
        i.e. np.log10()
        
    axis: int or None, (optional) - default 0:
        aggregate mean and standard deviation over this axis
        select 0 to do a pixel-wise normalization
        select None to do a dataset-wide normalization
        

    Returns
    -------
    data_norm: a normalized version of that 3-d array, where every "image" in the time dimension is 
    normalized to 
    '''
    
    if transform != None:
        transformed_arr = transform(data_array)
    else:
        transformed_arr = data_array
    mean = np.nanmean(transformed_arr, axis = axis)
    std = np.nanstd(transformed_arr, axis = axis)
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
        y_sum = np.nansum(year_arr, axis = 0)
        yearly_sums[i, :,:] = y_sum
    return yearly_sums
        
    # print('Mean map')
    # plt.imshow(np.nanmean(yearly_sums, axis = 0), 
    #            cmap= 'Blues', interpolation='None')
    # plt.show()
    # print('Std map')
    # plt.imshow(np.nanstd(yearly_sums, axis = 0), 
    #            cmap= 'bwr', interpolation='None')
    # plt.show()
    
    
        
def average_monthly(data_dict):
    ia_shape = data_dict['imagearray'].shape
    ymp = data_dict['ymp']
    y_m = np.array(['{}_{}'.format(x[0],x[1]) for x in ymp])
    # make unique array in same order
    y_m_unique = []
    for j in y_m:
        # print(j)
        if j not in y_m_unique:
            y_m_unique.append(j)
    
    monthly_arr = np.zeros((len(y_m_unique), *ia_shape[1:3]))
    print('test shape', monthly_arr.shape)
    for i in range(len(y_m_unique)):
        # print(i)
        select_ym = y_m == y_m_unique[i]
        select_images= data_dict['imagearray'][select_ym]
        mean = np.nanmean(select_images, axis = 0)
        monthly_arr[i,:,:] = mean

        
        
        
    
    
    return {'imagearray': monthly_arr, 
            'year_month': y_m_unique}
    
def filter_by_month(data_dict, months):
    '''
    
    filters data dict by month
    
    Parameters
    ----------
    data_dict : data dict created by return_data

    months : array type
        months to include in analysis

    Returns
    -------
    updated_data_dict:
        data dict with only selected months' data

    '''
    ymp = np.array(data_dict['ymp'])[:,1].astype(int)
    selected_months = np.isin(ymp, months)
    updated_data_dict = {}
    updated_data_dict['latitude'] = data_dict['latitude']
    updated_data_dict['longitude'] = data_dict['longitude']
    updated_data_dict['imagearray'] = data_dict['imagearray'][selected_months,:,:]
    updated_data_dict['ymp'] = np.array(data_dict['ymp'])[selected_months]
    updated_data_dict['filenames'] = np.array(data_dict['filenames'])[selected_months]
    
    return updated_data_dict
    
    
    
    


