#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 12:06:22 2022

@author: Mitchell
"""

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from builddata import aggregate_yearly, return_data, normalize_array,divide_by_max
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as gf
from spi import compute_spi
from sklearn.mixture import GaussianMixture


#  do it the way it is being done in GEE:
#     take log of data
#     sum log of data over each year
#     normalize by each pixel's mean and std



def MSE(clusterer, X):
    '''
    Parameters
    ----------
    clusterer : sklearn.cluster
        a fitted sklearn cluster object
    X : nd numpy array
        nd numpy array used to train the sklearn clusterer

    Returns
    -------
    mean squared error 

    '''
    labels = clusterer.labels_
    cluster_centers = clusterer.cluster_centers_    
    pred = cluster_centers[labels]
    MSE = np.nanmean(np.square(pred - X))
    return MSE


def intercluster_var(clusterer, X):
    labels = clusterer.labels_
    cluster_centers = clusterer.cluster_centers_    
    pred = cluster_centers[labels]
    global_avg  = np.nanmean(pred)
    assert (pred - global_avg).shape == pred.shape
    SSB = np.nanmean(np.square(pred - global_avg))
    
    return SSB
    
# # dictionary with data array and info about that data

def prepare_data(data_dict,method):
    '''
    Function to run kmeans over 

    Parameters
    ----------
    data_dict : dictionary
        dictionary output from return_data() in builddata.py.
    method : str
        Mode of aggregation and data transformation. 
        Either 'yearlysum', alldata', 'spi', or 'customdatadict'.

    Returns
    -------
    X: ndarray
        prepareddataarray
    shape: tuple
        shape tuple of format (w, h, d)

    '''

    
    if method == 'yearlysum':
        # take log
        data_dict['imagearray'] = np.log(data_dict['imagearray'] + 1)
        
        #sum over each year
        yearly_sum = aggregate_yearly(data_dict)
        
        # normalize, transform is None because we already normalized
        normalized_arr  = normalize_array(yearly_sum)
        
        # test smoothing
        smoothed_arr = np.zeros_like(normalized_arr)
        for i in range(normalized_arr.shape[0]):
            smoothed_arr[i,:,:] = gf(normalized_arr[i,:,:], 2)
        
        #transpose - to get x,y,t
        arr_t = smoothed_arr.transpose(1,2,0)
        
        
            
        # note! lat is w and lon is h
        w, h, d = original_shape = tuple(arr_t.shape)
        assert d == 41 
        
        X = np.reshape(arr_t, (w * h, d))
            
    elif method == 'alldata':
 
        normalized_arr = normalize_array(data_dict['imagearray'] , transform = divide_by_max, axis = None)
        #transpose - to get x,y,t
        normalized_arr_t = normalized_arr.transpose(1,2,0)
        # note! lat is w and lon is h
        w, h, d = original_shape = tuple(normalized_arr_t.shape)
        assert d == 3000
        
        X = np.reshape(normalized_arr_t, (w * h, d))
    
    elif method == 'spi':
        
        aggregation_window = 4
        arr_t = data_dict['imagearray'].transpose(1,2,0)
        spi_arr = compute_spi( arr_t, agg_window = aggregation_window )[:,:,aggregation_window:-aggregation_window]
        w, h, d = original_shape = tuple(spi_arr.shape)
        assert d == 3000 - (2* aggregation_window)
        X = np.reshape(spi_arr, (w * h, d))
        
    elif method == 'customdatadict':
        image_arr = data_dict['imagearray'].transpose(1,2,0)
        w, h, d = original_shape = tuple(image_arr.shape)
        X = np.reshape(image_arr, (w * h, d))
        
        
    else:
        raise(ValueError('Invalid method specified'))
        
        
    return X, (w,h,d)

def run_kmeans(X,k):
    '''
    Function to run kmeans over 

    Parameters
    ----------
    X: ndarray
        input data array from prepare data array
    data_dict : dictionary
        dictionary output from return_data() in builddata.py.
    k : int
        number of clusters
    method : str
        Mode of aggregation and data transformation. 
        Either 'yearlysum', alldata', 'spi', or 'customdatadict'.
    plot_map : boolean, optional
        Whether or not to plot cluster map. The default is True.
    print_metrics : boolean, optional
        Whether or not to print accuracy metrics. The default is True.
    plot_cluster_ts : boolean, optional
        Whether or not to plot time series of cluster centers. The default is False.
    cmap : matplotlib colomrap
        colormap for plotting clusters

    Returns
    -------
    kmeans
        fitted clusterer
    metrics
        tuple of accuracy metrics (SSW, SSB, SST)

    '''
  
    # declare 
    kmeans = KMeans(n_clusters=k, # Number of clusters to form
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

    return kmeans
    

def run_gmm(X,k):
    '''
    Function to run kmeans over 

    Parameters
    ----------
    X: ndarray
        input data array from prepare data array
    k : int
        number of clusters
    shape_tuple: tuple

    Returns
    -------
    gmm: fitted gaussian mixture model

    '''
    gmm = GaussianMixture(n_components = k, init_params = 'k-means++' )
    gmm.fit(X)
    return gmm

# data_dict = return_data(mode = 'spi', spi_window = 4)
# run_kmeans(data_dict,5, 'customdatadict', plot_map = True, print_metrics = True, plot_cluster_ts = False)


