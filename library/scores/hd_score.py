import numpy as  np
import matplotlib.pyplot as plt
import math
import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
 
from library.maps import spatial_tuning_curve

from library.hafting_spatial_maps import SpatialSpikeTrain2D

def _moving_sum(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window:]

def _get_rolling_sum(array_in, window):
    if window > (len(array_in) / 3) - 1:
        print('Window for head-direction histogram is too big, HD plot cannot be made.')
    inner_part_result = _moving_sum(array_in, window)
    edges = np.append(array_in[-2 * window:], array_in[: 2 * window])
    edges_result = _moving_sum(edges, window)
    end = edges_result[window:math.floor(len(edges_result)/2)]
    beginning = edges_result[math.floor(len(edges_result)/2):-window]
    array_out = np.hstack((beginning, inner_part_result, end))
    return array_out

# called by batch_process module only
# def hd_score(angles, window_size=23):
def hd_score(spatial_spike_train: SpatialSpikeTrain2D, **kwargs):

    # if 'smoothing_factor' in kwargs:
    #     smoothing_factor = kwargs['smoothing_factor']
    # else:
    #     smoothing_factor = spatial_spike_train.session_metadata.session_object.smoothing_factor

    if 'window_size' in kwargs:
        window_size = kwargs['window_size']
    else:
        window_size = 23

    if spatial_spike_train.get_map('spatial_tuning') is None:
        spatial_tuning_data = spatial_tuning_curve(spatial_spike_train)
        angles = spatial_spike_train.get_map('spatial_tuning')['spike_angles']
    angles = spatial_tuning_data = spatial_spike_train.get_map('spatial_tuning')['spike_angles']

    angles = angles[~np.isnan(angles)]
    theta = np.linspace(0, 2*np.pi, 361)  # x axis

    # IF THIS IS SLOW TRY NP.HISTOGRAM INSTEAD OF PLT.HIST
    binned_hd, _, _ = plt.hist(angles, theta)
    smooth_hd = _get_rolling_sum(binned_hd, window=window_size)
    plt.close()
    return smooth_hd
