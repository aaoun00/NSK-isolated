import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
 

import numpy as np
import cv2
from library.maps.map_utils import _compute_resize_ratio, _interpolate_matrix, _gkern

def spike_map(pos_x: np.ndarray, pos_y: np.ndarray, pos_t: np.ndarray,
                arena_size: tuple, spike_x: np.ndarray, spike_y: np.ndarray,
                kernlen: int, std: int, interp_size=(64,64)) -> np.ndarray:



    # Min and max dimensions of arena for scaling
    min_x = min(pos_x)
    max_x = max(pos_x)
    min_y = min(pos_y)
    max_y = max(pos_y)

    arena_size = (abs(max_y - min_y), abs(max_x - min_x)) # (height, width)

    # Resize ratio
    row_resize, column_resize = _compute_resize_ratio(arena_size)

    # Instantiate empty maps
    spike_map_raw = np.zeros((row_resize,column_resize))

    # Load spike data and set up arrays to map spike timestamps to subject position
    row_values = np.linspace(max_x, min_x, row_resize)
    column_values = np.linspace(min_y,max_y, column_resize)

    # Generate raw spike map
    for i in range(len(spike_x)):
        row_index = np.abs(row_values - spike_y[i]).argmin()
        column_index = np.abs(column_values - spike_x[i]).argmin()
        spike_map_raw[row_index][column_index] += 1

    # Remove low spike counts from spike map (20th percentile)
    #spike_map_raw[spike_map_raw <= np.percentile(spike_map_raw, 20)] = 0

    # Smooth spike map (must happen before resizing)
    spike_map_smooth = cv2.filter2D(spike_map_raw,-1,_gkern(kernlen, std))

    # Resize maps
    spike_map_smooth = _interpolate_matrix(spike_map_smooth, new_size=interp_size,  cv2_interpolation_method=cv2.INTER_NEAREST)
    spike_map_smooth = spike_map_smooth/max(spike_map_smooth.flatten())
    spike_map_raw = _interpolate_matrix(spike_map_raw, new_size=interp_size, cv2_interpolation_method=cv2.INTER_NEAREST)



    return spike_map_smooth, spike_map_raw
