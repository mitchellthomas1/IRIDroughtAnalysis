#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 12:06:22 2022

@author: Mitchell
"""

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from builddata import aggregate_yearly, return_data, normalize_array, log10_zerosnan
import numpy as np
import matplotlib.pyplot as plt


# dictionary with data array and info about that data
data_dict = return_data() 



# # normalize data
normalized_arr = normalize_array(data_dict['imagearray'],
                                  transform = log10_zerosnan)
normalized_arr[np.isnan(normalized_arr)] = 0
X = normalized_arr
# X = aggregate_yearly(data_dict)
X_2d = X.reshape((-1,X.shape[0]))



# declare 
kmeans = KMeans(n_clusters=4, # Number of clusters to form
       init='k-means++', # Method for initialization: 
          # ‘k-means++’ : selects initial cluster centroids using sampling based on an empirical probability distribution of the points’ contribution to the overall inertia. This technique speeds up convergence, and is theoretically proven to be -optimal. See the description of n_init for more details.
       n_init=20, # Number of time the k-means algorithm will be run with different centroid seeds. Result is best output in terms of inertia
       max_iter=1000, # Maximum number of iterations of the k-means algorithm for a single run.
       tol=0.00001, #Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.
       verbose=0, # Verbosity mode
       random_state=None, 
       copy_x=True, 
       algorithm='lloyd'# K-means algorithm to use. The classical EM-style algorithm is "lloyd"
       )
    


# fit algorithm
clusters = kmeans.fit_predict(X_2d).reshape(X.shape[1:3])
# clusters = kmeans.labels_.reshape(X.shape[1:3])
plt.imshow( clusters,
          cmap= 'tab20', interpolation='None')
plt.show()
# prediction_raw = kmeans.predict(X_2d)
# prediction = prediction_raw.reshape(X.shape[1:3])

# # plt.imshow(prediction)








