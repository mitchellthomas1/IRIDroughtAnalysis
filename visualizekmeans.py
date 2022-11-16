#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 17:54:44 2022

@author: Mitchell
"""


from builddata import return_data, average_monthly, filter_by_month
from kmeans2 import run_kmeans, run_gmm, prepare_data
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches





# kmeans parameters
k = 4
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
    
select_months = [2,3,4,5,6,7,8,9] #Feb to Sept
data_dict = filter_by_month(data_dict, select_months)

X, shape_tuple = prepare_data(data_dict,method)
w,h,d = shape_tuple
kmeans = run_kmeans(X,k)
gmm = run_gmm(X, k) 

# colormap
cmap = ListedColormap(['tab:red','tab:blue','tab:green','tab:orange'])

def plot_map(cmap = cmap):
    def get_extent(data_dict):
        x_min, x_max = data_dict['longitude'].min(), data_dict['longitude'].max()
        y_min, y_max = data_dict['latitude'].min(), data_dict['latitude'].max()
        extent = [x_min , x_max, y_min , y_max]
        return extent
    # labels = kmeans.labels_
    labels = gmm.predict(X)
    im = plt.imshow(labels.reshape(w,h) ,
                cmap= cmap, 
                interpolation='None', extent = get_extent(data_dict))
    # get the colors of the values, according to the 
    # colormap used by imshow
    values = np.unique(labels)
    colors = [ im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label="Cluster {l}".format(l=values[i] +1) ) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    
    plt.show()

plot_map()


def yearly_cluster_bar_chart():
    #group by cluster
    cluster_centers = kmeans.cluster_centers_
    years = np.arange(1981,2022)
    datetimes =  [dt.datetime(x[0], x[1], x[2] ) for x in data_dict['ymp'].astype(int)]
    date_index = pd.DatetimeIndex(datetimes)
    cluster_df = pd.DataFrame(data = np.transpose(cluster_centers), index=date_index, columns = ['Cluster{}'.format(x) for x in range(1,k+1)])
    
    yearly_mean = cluster_df.resample('AS').mean()
    yearly_sum = cluster_df.resample('AS').sum()
    yearly_10 = cluster_df.resample('AS').apply(lambda a : np.percentile(a, 10))
    
    fig,ax = plt.subplots(1,1,figsize = (14,4))
    yearly_mean.plot.bar(ax =ax , alpha = 1, cmap = cmap)
    ax.set_xticklabels(np.arange(1981,2023))
    # ax.set_ylim([-1,-0.4])
    ax.set_ylabel('spi')
    ax.set_title('Mean spi of cluster center over each year')
    plt.show()
    


#old cluster ts code
    
    # if plot_cluster_ts == True:
    #     ## see what each cluster describes over time
    #     if method == 'yearlysum':
    #         x = np.arange(1981, 2022)
    #     else:
    #         x = np.arange(0,3000)
    #     f1, axs = plt.subplots(k,1,figsize = (3*k,11), sharex = True)
    #     for i, ax in enumerate(axs):
            
    #         ts = kmeans.cluster_centers_[i]
    #         ax.plot(x,ts, label = 'Cluster ' + str(i+1), color = colors[i], lw = 4)
    #         ax.set_ylabel('$z_{cluster}$')
    #     f1.legend(loc = 5)
    #     axs[0].set_title('Cluster centers over time , k = {}'.format(k))
    #     axs[-1].set_xlabel('Year')
    #     plt.show()
        
        
        
    #     mean_cluster_center = np.mean(kmeans.cluster_centers_, axis = 0)
    #     std_cluster_center = np.std(kmeans.cluster_centers_, axis = 0)
        
    #     f2, axs = plt.subplots(k,1,figsize = (3*k,11), sharex = True)
    #     for i, ax in enumerate(axs):
    #         ts = (kmeans.cluster_centers_[i] - mean_cluster_center ) 
    #         ax.plot(x,ts, label = 'Cluster ' + str(i+1), color = colors[i], lw = 4)
    #         ax.set_ylabel('$z_{cluster} - z_{mean}$')
    #     f2.legend(loc = 5)
    #     axs[0].set_title('(Cluster centers) - (mean cluster center) over time, k = {}'.format(k))
    #     axs[-1].set_xlabel('Year')
    #     plt.show()
        
    
    
    