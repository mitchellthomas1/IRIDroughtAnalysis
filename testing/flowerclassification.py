#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 12:52:02 2022

@author: Mitchell
"""

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


flower_img = Image.open('flowerimage.jpeg')
flower_arr = np.asarray(flower_img) / 255.


w, h, d = original_shape = tuple(flower_arr.shape)
assert d  == 3

X = np.reshape(flower_arr, (w * h, d))

# X_01 = flower_arr / 255.0
# X_norm = (X_01 - np.nanmean(X_01) ) / np.nanstd(X_01)
# X = X_norm.transpose(2,0,1)
# X_2d = X.reshape((-1,X.shape[0]))



# declare 
kmeans = KMeans(n_clusters=3, # Number of clusters to form
        init='k-means++', # Method for initialization: 
          # ‘k-means++’ : selects initial cluster centroids using sampling based on an empirical probability distribution of the points’ contribution to the overall inertia. This technique speeds up convergence, and is theoretically proven to be -optimal. See the description of n_init for more details.
        n_init=20, # Number of time the k-means algorithm will be run with different centroid seeds. Result is best output in terms of inertia
        max_iter=1000, # Maximum number of iterations of the k-means algorithm for a single run.
        tol=0.0001, #Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.
        verbose=0, # Verbosity mode
        random_state=None, 
        copy_x=True, 
        algorithm='lloyd'# K-means algorithm to use. The classical EM-style algorithm is "lloyd"
        )
    


# # fit algorithm
labels = kmeans.fit_predict(X)
# segmented = kcenters[labels.flatten()].reshape((flower_arr.shape))

# def recreate_image(codebook, labels, w, h):
#     """Recreate the (compressed) image from the code book & labels"""
#     return codebook[labels].reshape(w, h, -1)
# final_image = recreate_image(kmeans.cluster_centers_, 
#                           labels, w, h)
# plt.imshow(final_image[:,:,1],
#             cmap= 'tab20', interpolation='None')
# # clusters = kmeans.labels_.reshape(X.shape[1:3])
plt.imshow( labels.reshape(w,h),
            cmap= 'tab20', interpolation='None')
plt.show()
# # prediction_raw = kmeans.predict(X_2d)
# # prediction = prediction_raw.reshape(X.shape[1:3])

# # # plt.imshow(prediction)







