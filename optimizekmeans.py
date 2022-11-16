#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 17:54:44 2022

@author: Mitchell
"""

from builddata import return_data, average_monthly, filter_by_month
from kmeans2 import run_kmeans, run_gmm, prepare_data, MSE, intercluster_var
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics 







test = 'spi'



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
elif test == 'spi_monthly':
    method = 'customdatadict'
    spi_window = 4
    data_dict = return_data(mode = 'spi', spi_window = spi_window)
    for key in ['imagearray', 'filenames', 'ymp']:
        data_dict[key] = data_dict[key][spi_window:-spi_window]
    #set new dummy data dict to pass through
    data_dict = average_monthly(data_dict)

    
else:
    raise(ValueError())
    
# filter by rainy season
select_months = [2,3,4,5,6,7,8,9] #Feb to Sept
data_dict = filter_by_month(data_dict, select_months)
    

# k_arr =  [2] + [x**2 for x in range(2,10)]
# k_arr = [2**x for x in range(2,8)]
k_arr = list(range(2,42, 4))
print('Test type: '+ test, ' over ', k_arr)

# criteria
aic_arr = []
bic_arr = []

# ssw_arr = []
# ssb_arr = []

# # minimize
# davies_bouldin_arr = []
# #maximize
# calinski_harabasz_arr= []
# # The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters
# silhouette_arr = []

X, shape_tuple = prepare_data(data_dict,method)
for k in k_arr:
    print(k)
    
    # kmeans = run_kmeans(X,k,data_dict, shape_tuple, plot_map = False, cmap = None)
    gmm = run_gmm(X, k) 
    bic, aic = gmm.bic(X), gmm.aic(X)
    
    
    
    
    # ssw_arr.append(MSE(kmeans,X))
    # ssb_arr.append(intercluster_var(kmeans, X))
    aic_arr.append(aic)
    bic_arr.append(bic)
    # davies_bouldin_arr.append(metrics.davies_bouldin_score(X, kmeans.labels_))
    # calinski_harabasz_arr.append(metrics.calinski_harabasz_score(X,kmeans.labels_))
    # silhouette_arr.append(metrics.silhouette_score(X, kmeans.labels_))



# f, ((ax1, ax2,ax3),(ax4, ax5, ax6) )= plt.subplots(2,3,figsize = (10,6))

# ts = [ssw_arr ,aic_arr, bic_arr, davies_bouldin_arr ,calinski_harabasz_arr,silhouette_arr]
# ts_titles = ['SSW','AIC','BIC', 'Davies Bouldin','Calinksi Harabasz','Silhouette']
# axes = [ax1,ax2,ax3,ax4, ax5, ax6]

    
# plt.tight_layout()
# plt.show()

# ax1.plot(k_arr, ssw_arr)
# ax1.set_xlabel('k')
# ax1.set_ylabel('SSW')
# ax1.set_title('Sum of Squares Within (SSW) vs Clusters (k)')

# ax2.plot(k_arr, ssb_arr)
# ax2.set_xlabel('k')
# ax2.set_ylabel('SSB')
# ax2.set_title('Sum of Squares Between (SSB) vs Clusters (k)')
# plt.show()




# ------ ONLY AIC and BIC --------
ts = [aic_arr, bic_arr]
ts_titles = ['AIC','BIC']
f, axes = plt.subplots(1,2,figsize = (7,3.5))
for i in range(len(axes)):
    axes[i].plot(k_arr, ts[i])
    axes[i].set_xlabel('k')
    axes[i].set_ylabel(ts_titles[i])
    axes[i].set_title(ts_titles[i])

    
    
    