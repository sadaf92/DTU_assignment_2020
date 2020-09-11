# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 10:53:32 2020

@author: farkh
"""

import numpy as np
from geoarray import GeoArray
import affine
import pandas as pd
import random
import sys
import os
import gdal
import glob


def align_data(prox_pth, orth_pth):
    """ Get the data (sentinel-2 pixels) aligned with proximal images.
    
    prox_pth : directory to the proximal csv file
    orth_pth : directory to the orthophoto tif file
    """
    geoArr = GeoArray(os.path.join(orth_pth[:-5], 'split_band1.tif'))
    tgt_gt = geoArr.geotransform
    
    total_points = pd.read_csv(prox_pth)
    geo_coord = tuple(np.array([total_points['utmx'], total_points['utmy']]))
    x, y = geo_coord[0], geo_coord[1]
    # Satellite
    forward_transform = affine.Affine.from_gdal(*tgt_gt)
    reverse_transform = ~forward_transform
    
    px, py = reverse_transform * (x, y)
    px, py = px + 0.5, py + 0.5
    
    px = px.astype(int)-1
    py = py.astype(int)-1
    
    bands = []
    for raster in glob.glob(orth_pth):
        raster_i = gdal.Open(raster)
        band = raster_i.GetRasterBand(1).ReadAsArray()
        bands.append(band)
    
    binary_mask = np.zeros_like(band)
    binary_mask[py, px] = 1
    
    Bands_after_mask = []
    for band_j in range(len(bands)):
        Bands_after_mask.append(bands[band_j]*binary_mask)

    Bands = list([Bands_after_mask[0],Bands_after_mask[1],Bands_after_mask[2],Bands_after_mask[3]])
    Band = np.moveaxis(Bands, 0, -1)

    # Ground-based
    p_yx = np.swapaxes(np.array([py,px]), 1, 0)
    unq_, ind_, count_ = np.unique(p_yx, axis = 0, return_counts = True, return_inverse = True)
    
    all_points = []
    bands_4 = []
    for i in range(len(unq_)):
        indx = np.where(ind_ == i)
        # get proximal responses of grass
        if len(indx)>1:
            indx = random.sample(indx, 1)
        all_points.append(np.vstack([np.array(total_points['Grass_ratio_with_mask'].loc[indx]),
                                     total_points['color'].loc[indx]]).T)
        # get orthophoto bands (RGB-NIR)
        bands_4.append(Band[tuple(unq_[i])[0], tuple(unq_[i])[1], :])
    