#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 11:20:51 2022

@author: Mitchell
"""

import ee
import geemap
import matplotlib.pyplot as plt
import matplotlib as mp
import numpy as np
ee.Initialize()

# number of clusters
k = 4

#DATE RANGE
start, end = '2000-01-01','2022-01-01'
#ROI
projectArea = ee.Geometry.Rectangle([39.82485, 10.02040 ,36.90537  , 12.39272 ])

def build_collection():
    # Build the collection
    # terra = ee.ImageCollection("MODIS/061/MOD09A1")
    # aqua = ee.ImageCollection("MODIS/061/MYD09A1")
    aqua = ee.ImageCollection("MODIS/061/MYD13A2").select('NDVI')
    terra = ee.ImageCollection("MODIS/061/MOD13A2").select('NDVI')
    merged = terra.merge(aqua).sort('system:time_start')
    filtered = merged.filterDate(start,end).filterBounds(projectArea)
    clip_func = lambda i : i.clip(projectArea)
    unmask_val = 0
    unmask_func = lambda i : i.unmask(unmask_val)
    ndvi_coll  = filtered.map(clip_func).map(unmask_func)
    # get the ndvi collection
    # def get_ndvi(image):
    #     return image.normalizedDifference(['sur_refl_b02','sur_refl_b01'])
    # ndvi_coll = filtered.map(get_ndvi)
    ndvi_image = ndvi_coll.toBands()
    return ndvi_image
    

ndvi_image = build_collection()
kmeans = ee.Algorithms.Image.Segmentation.KMeans(ndvi_image, k)
    
# image_arr = geemap.ee_to_numpy(ndvi_image.select(0), region=projectArea, default_value=-9999)
proj = ndvi_image.select(0).projection()
kmeans_proj = kmeans.reproject(crs = proj.crs(), scale = proj.nominalScale())
image_arr = geemap.ee_to_numpy(kmeans, region=projectArea) #, default_value=-9999)


##### visualize ######
def plot_map(arr, cmap, extent):
    im = plt.imshow(arr,
                cmap= cmap, 
                interpolation='None', extent = extent)
    # get the colors of the values, according to the colormap
    values = np.unique(arr)
    colors = [ im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mp.patches.Patch(color=colors[i], label="Cluster {l}".format(l=values[i] +1) ) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.show()
cmap =  mp.colors.ListedColormap(['tab:red','tab:blue','tab:green','tab:orange'])
extent = [36.90537, 39.82485, 10.02040 ,  12.39272 ]
plot_map(image_arr, cmap,extent)

