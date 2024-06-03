import os
import sys

from library import spatial

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)


import numpy as np
from library.hafting_spatial_maps import HaftingRateMap, SpatialSpikeTrain2D
# from library.spatial_spike_train import SpatialSpikeTrain2D


def binary_map(spatial_map: HaftingRateMap | SpatialSpikeTrain2D, percentile=75, **kwargs):

    '''
        Produces a binary map of place fields from a ratemap based on [1].

        "A place field was defined as an area of nine or more (5 × 5 cm) adjacent bins with firing rates exceeding 20% of the peak firing rate of the rate map." (p 1843)

        [1] C. B. Alme, C. Miao, K. Jezek, A. Treves, E. I. Moser, and M.-B. Moser, “Place cells in the hippocampus: Eleven maps for eleven rooms,” Proceedings of the National Academy of Sciences, vol. 111, no. 52, pp. 18428–18435, Dec. 2014, doi: 10.1073/pnas.1421056111.


        Params:
            ratemap (np.ndarray):
                Array encoding neuron spike events in 2D space based on where
                the subject walked during experiment.

        Returns:
            np.ndarray:
                binary_map
    '''

    if 'smoothing_factor' in kwargs:
        smoothing_factor = kwargs['smoothing_factor']
    else:
        smoothing_factor = spatial_map.session_metadata.session_object.smoothing_factor

    if 'use_map_directly' in kwargs:
        if kwargs['use_map_directly']:
            ratemap = spatial_map
    else:
        if isinstance(spatial_map, HaftingRateMap):
            ratemap, _ = spatial_map.get_rate_map(smoothing_factor)
        elif isinstance(spatial_map, SpatialSpikeTrain2D):
            ratemap, _ = spatial_map.get_map('rate').get_rate_map(smoothing_factor)

    binary_map = np.zeros(ratemap.shape)
    # percentile = 75 
    binary_map[  ratemap >= np.percentile(ratemap.flatten(), percentile)  ] = 1

    # binary_map_copy = np.copy(ratemap)
    # binary_map = np.zeros(ratemap.shape)
    # binary_map[  binary_map_copy >= np.percentile(binary_map_copy.flatten(), 75)  ] = 1
    # binary_map[  binary_map_copy < np.percentile(binary_map_copy.flatten(), 75)  ] = 0  

    if isinstance(spatial_map, HaftingRateMap):
        spatial_map.spatial_spike_train.add_map_to_stats('binary', binary_map)
    elif isinstance(spatial_map, SpatialSpikeTrain2D):
        spatial_map.add_map_to_stats('binary', binary_map)

    return binary_map


