import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
 

import numpy as np
import cv2
from library.maps.map_utils import _gkern
from library.hafting_spatial_maps import HaftingRateMap, SpatialSpikeTrain2D
# from library.spatial_spike_train import SpatialSpikeTrain2D
from library.maps.map_utils import disk_mask
from skimage.measure import block_reduce

def custom_flat_disk_mask(rate_map):
    masked_rate_map = disk_mask(rate_map)
    masked_rate_map.data[masked_rate_map.mask] = 0
    return  masked_rate_map.data

def _downsample(img, downsample_factor):
    downsampled = block_reduce(img, downsample_factor) 
    return downsampled

# Taken from https://stackoverflow.com/questions/59144828/opencv-getting-all-blob-pixels
#public
def map_blobs(spatial_map: SpatialSpikeTrain2D | HaftingRateMap, nofilter=False, **kwargs):

    '''
        Segments and labels firing fields in ratemap.

        Params:
            ratemap (np.ndarray):
                Array encoding neuron spike events in 2D space based on where
                the subject walked during experiment.

        Returns:
            tuple:
                image, n_labels, labels, centroids
            --------
            image (np.ndarray):
                Semi-processed image used for blob detection
            n_labels (np.ndarray):
                Array of blob numbers / ID's
            labels (np.ndarray):
                Segmented ratemap with each blob labelled
            centroids (np.ndarray):
                Array of coordinates for each blobs weighted centroid.
            field_sizes (list):
                List of size of each field as a percentage of map coverage
    '''

    if 'smoothing_factor' in kwargs:
        smoothing_factor = kwargs['smoothing_factor']
    else:
        smoothing_factor = spatial_map.session_metadata.session_object.smoothing_factor

    if 'ratemap_size' in kwargs:
        ratemap_size = kwargs['ratemap_size']
        if isinstance(spatial_map, HaftingRateMap):
            ratemap, _ = spatial_map.get_rate_map(smoothing_factor, new_size=ratemap_size)
        elif isinstance(spatial_map, SpatialSpikeTrain2D):
            ratemap, _ = spatial_map.get_map('rate').get_rate_map(smoothing_factor, new_size=ratemap_size)
        else:
            ratemap = spatial_map
    else:
        if isinstance(spatial_map, HaftingRateMap):
            ratemap, _ = spatial_map.get_rate_map(smoothing_factor)
        elif isinstance(spatial_map, SpatialSpikeTrain2D):
            ratemap, _ = spatial_map.get_map('rate').get_rate_map(smoothing_factor)
        else:
            ratemap = spatial_map

    if 'downsample' in kwargs:
        if kwargs['downsample'] == True:
            ratemap = _downsample(ratemap, kwargs['downsample_factor'])

    if 'cylinder' in kwargs:
        cylinder = kwargs['cylinder']

        if len(ratemap[ratemap != ratemap]) > 0:
            # already disk masked just repalce nan with 0
            ratemap[ratemap != ratemap] = 0
        else:
            if cylinder:
                ratemap = custom_flat_disk_mask(ratemap)

    

    # Kernel size
    kernlen = int(smoothing_factor*8)
    # Standard deviation size
    std = int(0.2*kernlen)

    # Create kernel for convolutional smoothing
    # kernel = _gkern(26, 3)
    kernel = _gkern(kernlen,std)

    ratemap_copy = np.copy(ratemap)

    # Compute a 'low_noise' threshold where anything below the 10th percentile activity is removed
    low_noise = np.mean(ratemap_copy[ratemap_copy <= np.percentile(ratemap_copy, 20)])
    ratemap_copy[ratemap_copy <= np.percentile(ratemap_copy, 80)] = low_noise
    # threshold = 0.2 * np.max(ratemap_copy)
    # ratemap_copy[ratemap_copy <= threshold] = 0

    # Initial segmentation into blobs
    image = np.array(ratemap_copy * 255, dtype = np.uint8)
    thresh, blobs = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blobs, connectivity=4)

    if nofilter == False:
        # Filter through each blob, and remove any blob smaller than some threshold.
        for i in range(1, n_labels):
            num_pix = len(np.where(labels==i)[0])
            if num_pix <= (blobs.size * 0.01):
                blobs[np.where(labels==i)] = 0

        # Once smaller blobs are removed, re-smooth, and re-normalize image
        image[np.where(blobs == 0)] = 0
        image = image / max(image.flatten())
        image = cv2.filter2D(image,-1,kernel)
        image = image / max(image.flatten())
        image_2 = np.array(image * 255, dtype = np.uint8)
    else:
        image_2 = image

    # Second round of segmentation to acquire more clean and accurate blobs from pre-preocessed image
    thresh, blobs = cv2.threshold(image_2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blobs, connectivity=4)

    # Flip centroids to follow (x,y) convention
    if len(centroids) > 0:
        centroids = centroids[1:]
        centroids = np.fliplr(centroids)

    field_sizes = []
    for i in range(1, n_labels):
        field_sizes.append(( len(np.where(labels==i)[0]) / len(image_2.flatten()) ) * 100)

    map_blobs_dict = {'image': image, 'n_labels': n_labels, 'centroids': centroids, 'field_sizes': field_sizes}

    if isinstance(spatial_map, HaftingRateMap):
        spatial_map.spatial_spike_train.add_map_to_stats('map_blobs', map_blobs_dict)
    elif isinstance(spatial_map, SpatialSpikeTrain2D):
        spatial_map.add_map_to_stats('map_blobs', map_blobs_dict)

    return image, n_labels, labels, centroids, field_sizes

