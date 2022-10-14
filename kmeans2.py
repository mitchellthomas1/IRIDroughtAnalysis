#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 12:06:22 2022

@author: Mitchell
"""

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from builddata import aggregate_yearly, return_data, normalize_array, log10_zerosnan
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter as gf


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

#transpose - to get x,y,t
normalized_arr_t = normalized_arr.transpose(1,2,0)

# test smoothing
smoothed_arr = np.zeros_like(normalized_arr_t)
for i in range(normalized_arr_t.shape[2]):
    smoothed_arr[:,:,i] = gf(normalized_arr_t[:,:,i], 2)

# plt.imshow(normalized_arr[0,:,:], cmap='gray',vmin = -2 , vmax =0)


# # # normalize data
# normalized_arr = normalize_array(data_dict['imagearray'],
#                                   transform = log10_zerosnan)
# normalized_arr[np.isnan(normalized_arr)] = 0





# note! lat is w and lon is h
w, h, d = original_shape = tuple(smoothed_arr.shape)
assert d == 41

X = np.reshape(smoothed_arr, (w * h, d))


# declare 
kmeans = KMeans(n_clusters=5, # Number of clusters to form
        init='k-means++', # Method for initialization: 
          # ‘k-means++’ : selects initial cluster centroids using sampling based on an empirical probability distribution of the points’ contribution to the overall inertia. This technique speeds up convergence, and is theoretically proven to be -optimal. See the description of n_init for more details.
        n_init=20, # Number of time the k-means algorithm will be run with different centroid seeds. Result is best output in terms of inertia
        max_iter=1000, # Maximum number of iterations of the k-means algorithm for a single run.
        tol=0.0001, #Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.
        verbose=0, # Verbosity mode
        random_state=0, 
        copy_x=True, 
        algorithm='lloyd'# K-means algorithm to use. The classical EM-style algorithm is "lloyd"
        )
    
# # fit algorithm
labels = kmeans.fit_predict(X)

x_min, x_max = data_dict['longitude'].min(), data_dict['longitude'].max()
y_min, y_max = data_dict['latitude'].min(), data_dict['latitude'].max()

extent = [x_min , x_max, y_min , y_max]

im = plt.imshow(labels.reshape(w,h) ,
            cmap= 'tab20', interpolation='None', extent = extent)

# get the colors of the values, according to the 
# colormap used by imshow
values = np.unique(labels)
colors = [ im.cmap(im.norm(value)) for value in values]
# create a patch (proxy artist) for every color 
patches = [ mpatches.Patch(color=colors[i], label="Cluster {l}".format(l=values[i] +1) ) for i in range(len(values)) ]
# put those patched as legend-handles into the legend
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

plt.show()



## see what each cluster describes over time

f1, axs = plt.subplots(5,1,figsize(9,14))
for i, ax in enumerate(axs):
    ts = kmeans.cluster_centers_[i]








