#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:36:38 2022

@author: Mitchell
"""

import numpy as np
import scipy.stats as stats
from builddata import return_data
np.random.seed(42)


def fit_gamma(precip_arr):
    alpha_arr = np.zeros_like(precip_arr[0,:,:])
    loc_arr = np.zeros_like(precip_arr[0,:,:])
    beta_arr = np.zeros_like(precip_arr[0,:,:])
    for y in range(alpha_arr.shape[0]):
        print(y , '/', alpha_arr.shape[0] )
        for x in range(precip_arr.shape[1]):
            pixel_ts = precip_arr[:,y,x]
            # print(pixel_ts.shape)
            fit_alpha, fit_loc, fit_beta = stats.gamma.fit(pixel_ts)
            alpha_arr[y,x] = fit_alpha
            loc_arr[y,x] = fit_loc
            beta_arr[y,x] = fit_beta
            
    return alpha_arr, loc_arr, beta_arr

# data_dict = return_data()
# alpha_arr, loc_arr, beta_arr = fit_gamma(data_dict['imagearray'])


# np.savetxt('spigammafits_'+ 'alpha' +'.csv', alpha_arr, delimiter=",")
# np.savetxt('spigammafits_'+ 'loc' +'.csv', loc_arr, delimiter=",")
# np.savetxt('spigammafits_'+ 'beta' +'.csv', beta_arr, delimiter=",")
    
def export_gamma():
    data_dict = return_data()
    alpha_arr, loc_arr, beta_arr = fit_gamma(data_dict['imagearray'])
    
    np.savetxt('spigammafits_'+ 'alpha' +'.csv', alpha_arr, delimiter=",")
    np.savetxt('spigammafits_'+ 'loc' +'.csv', loc_arr, delimiter=",")
    np.savetxt('spigammafits_'+ 'beta' +'.csv', beta_arr, delimiter=",")

def compute_spi(data_dict, agg_window = 4):
    '''
    

    Parameters
    ----------
    precip_arr : 3-d numpy array in shape time,  y, x
        array of spatial precipitation values over a given number of time steps
        
    agg_window: int 
        window of aggregation in each direction (in pentads)
        default is 4 pentads in each direction

    Returns
    -------
    None.

    '''
    # change this!
    # x_dim, y_dim = 0,1
    precip_arr = data_dict['imagearray']
    # fit and normalize from gamma dist
    a, l, s = stats.gamma.fit(np.random.choice(precip_arr.flatten(), 500000) )
    cdf_vals = stats.gamma.cdf(precip_arr, a, loc = l, scale = s)
    norm_vals = stats.norm.ppf(cdf_vals)
    assert norm_vals.shape == precip_arr.shape
    
    
    ymp = data_dict['ymp']
    # ymp_2000 = [x for x in ymp if x[0] == '2000']
    #month, pentad
    mp = ['{}_{}'.format(k[1], k[2]) for k in ymp]
    
    # path = 'spi/spigammafits_'
    # alpha_arr = np.genfromtxt( path+ 'alpha.csv', delimiter=',') 
    # loc_arr = np.genfromtxt( path+ 'loc.csv', delimiter=',') 
    # beta_arr = np.genfromtxt( path+ 'beta.csv', delimiter=',') 
    
    
    
    spi_arr = np.zeros_like(precip_arr)
    spi_arr[:agg_window,:,:], spi_arr[-agg_window:,:,:] = np.nan,  np.nan
    for t in range(precip_arr.shape[0])[agg_window: -agg_window]:
        if t %100 == 0:
            print(t)
        
        # normalize
        
        i_start, i_stop = t - agg_window ,  t + agg_window + 1
        select_mps = mp[i_start : i_stop]
        select_images = np.isin(mp, select_mps)
        test_coll = precip_arr[select_images, :, :]
        

        
        mean_i = np.nanmean(test_coll, axis = 0)
        
        std_i = np.nanstd(test_coll, axis = 0)
        
        spi_i = (precip_arr[t, :,:] - mean_i )/ std_i

        spi_arr[t,:,:] = spi_i
        
        
    
    
    
    
    return spi_arr


def save_spi(data_dict, agg_window = 4):
    path = 'Export/export_files/CHIRPS_EthiopiaAOI/CHIRPS_SPI_{}pentads/'.format(agg_window)
    spi_arr = compute_spi(data_dict, agg_window = agg_window)
    for i in range(spi_arr.shape[0]):
        if i % 100 == 0:
            print(i)
        filename = data_dict['filenames'][i].split('.')[0] + '_SPI_{}pentads.csv'
        image_arr = spi_arr[i]
        np.savetxt(path +filename, image_arr, delimiter=",")
        
# save_spi(data_dict, agg_window = 4)