#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 17:54:44 2022

@author: Mitchell
"""

from builddata import return_data
from kmeans2 import run_kmeans, MSE, intercluster_var
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics 

test = 'yearly'



if test == 'yearly':
    data_dict = return_data()
    method = 'yearlysum'
elif test == 'alldata':
    data_dict = return_data()
    method = 'alldata'
    
elif test == 'spi':
    method = 'customdatadict'
    spi_window = 4
    data_dict = return_data(mode = 'spi', spi_window = spi_window)
    for key in ['imagearray', 'filenames', 'ymp']:
        data_dict[key] = data_dict[key][spi_window:-spi_window]

    
else:
    raise(ValueError())
    
    
    

# k_arr =  [2] + [x**2 for x in range(2,10)]
# k_arr = [2**x for x in range(1,10)]
k_arr = list(range(2,41,1))
print('Test type: '+ test, ' over ', k_arr)
# mean squared error minimize
ssw_arr = []
ssb_arr = []
# minimize
davies_bouldin_arr = []
#maximize
calinski_harabasz_arr= []
# The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters
silhouette_arr = []


for k in k_arr:
    print(k)
    kmeans, X = run_kmeans(data_dict,k, method, plot_map = False, print_metrics = False, plot_cluster_ts = False)
    
    ssw_arr.append(MSE(kmeans,X))
    ssb_arr.append(intercluster_var(kmeans, X))
    davies_bouldin_arr.append(metrics.davies_bouldin_score(X, kmeans.labels_))
    calinski_harabasz_arr.append(metrics.calinski_harabasz_score(X,kmeans.labels_))
    silhouette_arr.append(metrics.silhouette_score(X, kmeans.labels_))



f, ((ax1, ax2),(ax3, ax4) )= plt.subplots(2,2,figsize = (8,6))

ts = [ssw_arr ,davies_bouldin_arr ,calinski_harabasz_arr,silhouette_arr]
ts_titles = ['SSW','Davies Bouldin','Calinksi Harabasz','Silhouette']
axes = [ax1,ax2,ax3,ax4]
for i in range(len(axes)):
    axes[i].plot(k_arr, ts[i])
    axes[i].set_xlabel('k')
    axes[i].set_ylabel(ts_titles[i])
    axes[i].set_title(ts_titles[i])
    
plt.tight_layout()
plt.show()

# ax1.plot(k_arr, ssw_arr)
# ax1.set_xlabel('k')
# ax1.set_ylabel('SSW')
# ax1.set_title('Sum of Squares Within (SSW) vs Clusters (k)')

# ax2.plot(k_arr, ssb_arr)
# ax2.set_xlabel('k')
# ax2.set_ylabel('SSB')
# ax2.set_title('Sum of Squares Between (SSB) vs Clusters (k)')
# plt.show()


    
    
    
    