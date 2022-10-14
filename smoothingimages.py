#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 16:13:01 2022

@author: Mitchell
"""


from scipy.ndimage import gaussian_filter as gf
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from builddata import aggregate_yearly, return_data, normalize_array, log10_zerosnan
import numpy as np
import matplotlib.pyplot as plt


#  do it the way it is being done in GEE:
#     take log of data
#     sum log of data over each year
#     normalize by each pixel's mean and std
    
    
# dictionary with data array and info about that data
data_dict = return_data() 

# take log
data_dict['imagearray'] = log10_zerosnan(data_dict['imagearray'])

#sum over each year
yearly_sum = aggregate_yearly(data_dict)

# normalize, transform is None because we already normalized
normalized_arr  = normalize_array(yearly_sum)

smoothed_arr = np.zeros_like(normalized_arr)
for i in range(normalized_arr.shape[0]):
    smoothed_arr[i,:,:] = gf(normalized_arr[i,:,:], 2)

# plt.imshow(normalized_arr[0,:,:], cmap='gray',vmin = -2 , vmax =0)


# # # normalize data
# normalized_arr = normalize_array(data_dict['imagearray'],
#                                   transform = log10_zerosnan)
# normalized_arr[np.isnan(normalized_arr)] = 0

# note! lat is w and lon is h
d, w, h = original_shape = tuple(normalized_arr.shape)
assert d == 41

X = np.reshape(normalized_arr, (w * h, d))

for i,y in enumerate(np.arange(1981,2022)):
    plt.imshow(normalized_arr[i,:,:])
    plt.title(y)
    plt.show()
    plt.clf()
    
    smoothed = gf(normalized_arr[i,:,:], 2)
    plt.imshow(smoothed)
    plt.title(str(y) + 'smoothed')
    plt.show()
    plt.clf()
    
    
    
    
    