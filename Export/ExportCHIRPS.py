#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:15:49 2022

@author: Mitchell

extract pixels as numpy, build numpy array of CHIRPS history

"""

#

import ee
import geemap
import numpy as np
import matplotlib.pyplot as plt
import time
import os

ee.Initialize()

CHIRPSPentad = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD")

path = 'export_files/CHIRPS_EthiopiaAOI/'
image_folder = 'CHIRPS_Pentad_working/'

aoi = ee.Geometry.Polygon(
        [[[39.82485, 12.39272],
          [39.82485, 10.0204],
          [36.90537, 10.0204],
          [36.90537, 12.39272]]], None, False)

start_from_last_finished = True


def makeBandLabel(img):
    month = ee.Number(img.get('month')).int().format()
    pentad = ee.Number(img.get('pentad')).int().format()
    year = ee.Number(img.get('year')).int().format()
    label = ee.String('y').cat(year).cat('m').cat(month).cat('p').cat(pentad).cat('_Precipitation')
    # index = year.cat(month).cat(pentad)
    return img.rename([label])

chirpsFilteredLabeled = CHIRPSPentad.filterBounds(aoi).map(makeBandLabel)
test = chirpsFilteredLabeled
if start_from_last_finished == True:
    finished_indices = np.sort([f.split('_')[0] for f in os.listdir(path + image_folder)])
    finished_indices = [x for x in finished_indices if (x != 'AOI' and x != '.DS')]
    try:
        last_finished = chirpsFilteredLabeled.filter(
            ee.Filter.eq('system:index', finished_indices[-1])).first()
        chirpsFilteredLabeled = chirpsFilteredLabeled.filterDate(
                last_finished.get('system:time_start'), '2030-01-01')
    except IndexError:
        print('There is no last image to start from. Set start_from_last_finished to False')
        

chirpsExportImage = chirpsFilteredLabeled.toBands()
bandNames = chirpsExportImage.bandNames().getInfo()
ts = time.time()
ts_arr = []
for i, b in enumerate(bandNames):

    precipImage = chirpsExportImage.select(b)
    chirps_arr = geemap.ee_to_numpy(precipImage, region=aoi)[:,:,0]

    np.savetxt(path +image_folder + b + '.csv', chirps_arr, delimiter=",")
    ts_arr.append(time.time() - ts)
    if (i % 100 == 0 ) or (i < 100):
        print(i) #, ': ', precipImage.bandNames().get(0).getInfo())
        print( 'time remaining: ' , round(np.mean(ts_arr) * (len(bandNames) -i)/60 , 0), ' minutes')
    
    ts =time.time()
    
# lonlatimage = ee.Image.pixelLonLat().reproject(**{'crs': 'EPSG:4326',
#                       'scale': 5565.974622603162})
# for b in ['longitude','latitude']:
#     lonlatexport =  geemap.ee_to_numpy(lonlatimage.select(b), region=aoi)[:,:,0]
#     np.savetxt(path + 'AOI_'+b + '.csv', lonlatexport, delimiter=",")
     

# ts = time.time()
# numpy_chirps = geemap.ee_to_numpy(chirpsFilteredLabeled.first(), region=aoi)
# te = time.time()
# print('time for one band = ', te - ts)
# print('time for 3000 bands = ', ((te-ts)*3000) / 60 , 'minutes')


# # Scale the data to [0, 255] to show as an RGB image.
# # Adapted from https://bit.ly/2XlmQY8. Credits to Justin Braaten
# rgb_img_test = (255 * ((rgb_img[:, :, 0:3] - 100) / 3500)).astype('uint8')
# plt.imshow(rgb_img_test)
# plt.show()

