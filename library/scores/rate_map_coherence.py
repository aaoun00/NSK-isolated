""""""""""""""""""""""""""" From Opexebo https://pypi.org/project/opexebo/ """""""""""""""""""""""""""

"""Provide function for calculating coherence of a Rate Map"""
# Hack to speed up the borked Astropy configuration search
import os
import pathlib
from library import spatial

# from library.spatial_spike_train import SpatialSpikeTrain2D
from library.hafting_spatial_maps import HaftingRateMap, SpatialSpikeTrain2D
# from library.spatial_spike_train import SpatialSpikeTrain2D
os.environ["HOMESHARE"] = str(pathlib.Path.home())

import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Suppress the ConfigurationMissingWarning that Astropy triggers
    from astropy.convolution import convolve


def rate_map_coherence(spatial_map: SpatialSpikeTrain2D | HaftingRateMap, **kwargs):
    '''
    Calculate coherence of a rate map

    Coherence is calculated based on RU Muller, JL Kubie "The firing of hippocampal place
    cells predicts the future position of freely moving rats", Journal of Neuroscience, 1 December 1989,
    9(12):4101-4110. The paper doesn't provide information about how to deal with border values
    which do not have 8 well-defined neighbours. This function uses zero-padding technique.

    Parameters
    ----------
    rate_map_raw: np.ma.MaskedArray
        Non-smoothed rate map: n x m array where cell value is the firing rate,
        masked at locations with low occupancy

    Returns
    -------
    coherence: float
        see relevant literature (above)

    Notes
    -----
    BNT.+analyses.coherence(map)
        
    Copyright (C) 2019 by Simon Ball
    '''

    if 'smoothing_factor' in kwargs:
        smoothing_factor = kwargs['smoothing_factor']
    else:
        smoothing_factor = spatial_map.session_metadata.session_object.smoothing_factor

    if isinstance(spatial_map, HaftingRateMap):
        _, rate_map_raw = spatial_map.get_rate_map(smoothing_factor)
    elif isinstance(spatial_map, SpatialSpikeTrain2D):
        _, rate_map_raw = spatial_map.get_map('rate').get_rate_map(smoothing_factor)
    else:
        rate_map_raw = spatial_map


    kernel = np.array([[0.125, 0.125, 0.125],
                      [0.125, 0,     0.125],
                      [0.125, 0.125, 0.125]])

    avg_map = convolve(rate_map_raw, kernel, 'fill', fill_value=0)
    avg_map = avg_map.ravel()
    avg_map = np.nan_to_num(avg_map)

    rmap = np.copy(rate_map_raw)
    rmap = rmap.ravel()
    rmap = np.nan_to_num(rmap)

    coherence = np.corrcoef(avg_map, rmap)[0,1]

    return coherence