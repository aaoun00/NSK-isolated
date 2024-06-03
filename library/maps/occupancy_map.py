import os
import sys


PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.spatial import Position2D

import cv2
import numpy as np
from scipy import signal
# from library.maps.map_utils import _compute_resize_ratio, _interpolate_matrix, _gkern

def occupancy_map(position: Position2D, kernlen=None, std=None, interp_size=(64,64)) -> np.ndarray:

    '''
        Computes the position, or occupancy map, which is a 2D numpy array
        enconding subjects position over entire experiment.

        Params:
            pos_x, pos_y and pos_t (np.ndarray):
                Arrays of the subjects x and y coordinates, as
                well as timestamp array.
            arena_size (tuple):
                Arena dimensions (width (x), height (y) in meters)
            resolution (float):
                Resolution of occupancy map (in meters)
            kernlen, std : kernel size and standard deviation (i.e 'spread') for convolutional smoothing
                of 2D map

            Returns:
                np.ndarray: occ_map_smoothed, occ_map_raw, coverage_map
    '''
    pos_x, pos_y, pos_t, arena_size = position.x, position.y, position.t, (position.arena_height, position.arena_width)

    min_x = min(pos_x)
    max_x = max(pos_x)
    min_y = min(pos_y)
    max_y = max(pos_y)

    arena_size = (abs(max_y-min_y), abs(max_x - min_x)) # (height, width)

    # Resize ratio
    row_resize, column_resize = _compute_resize_ratio(arena_size)

    # Initialize empty map
    occ_map_raw = np.zeros((row_resize,column_resize))
    coverage_map = np.zeros((row_resize,column_resize))
    row_values = np.linspace(max_y,min_y,row_resize)
    column_values = np.linspace(min_x,max_x,column_resize)

    # Generate the raw occupancy map
    for i in range(1, len(pos_t)):

        row_index = np.abs(row_values - pos_y[i]).argmin()
        column_index = np.abs(column_values - pos_x[i]).argmin()
        occ_map_raw[row_index][column_index] += pos_t[i] - pos_t[i-1]
        coverage_map[row_index][column_index] = 1

    # Normalize and smooth with scaling facotr
    occ_map_normalized = occ_map_raw / pos_t[-1]
    occ_map_smoothed = cv2.filter2D(occ_map_normalized,-1,_gkern(kernlen,std))

    # dilate coverage map
    kernel = np.ones((2,2))
    coverage_map = cv2.dilate(coverage_map, kernel, iterations=1)

    # Resize maps
    occ_map_raw = _interpolate_matrix(occ_map_raw, new_size=interp_size, cv2_interpolation_method=cv2.INTER_NEAREST)
    occ_map_smoothed = _interpolate_matrix(occ_map_smoothed, new_size=interp_size, cv2_interpolation_method=cv2.INTER_NEAREST)
    occ_map_smoothed = occ_map_smoothed/max(occ_map_smoothed.flatten())

    return occ_map_smoothed, occ_map_raw, coverage_map

def _interpolate_matrix(matrix, new_size=(256,256), cv2_interpolation_method=cv2.INTER_NEAREST):
    '''
        Interpolate a matrix using cv2.INTER_LANCZOS4.
    '''
    return cv2.resize(matrix, dsize=new_size,
                      interpolation=cv2_interpolation_method)

def _compute_resize_ratio(arena_size: tuple) -> tuple:

    '''
        Computes resize ratio which is later used to shape all ratemaps
        and ocupancy maps to have the same shape.

        Params:
            arena_size (tuple): (arena_height, arena_width)
                Dimensions of arena. Arena is assumed to be square/rectangular

        Returns:
            Tuple: (row_resize, column_resize)
            --------
            row_resize (int):
                Number of rows to resize to
            column_resize (int):
                Number of columns to resize to
    '''

    # Each maps largest dimension is always set to 64
    # base_resolution = 16
    base_resolution = 64
    resize_ratio = arena_size[0] / arena_size[1] # height/width

    # If width is smaller than height, set height resize to 64 and row to less
    if resize_ratio > 1:
      row_resize = int(np.ceil(base_resolution*(1/resize_ratio)))
      column_resize = base_resolution

    # If length is smaller than width, set width resize to 64 and height to less
    elif resize_ratio < 1:
        row_resize = base_resolution
        column_resize = int(np.ceil(base_resolution*(resize_ratio)))

    # If the arena is perfectly square, set both side resizes to 64
    else:
        row_resize = base_resolution
        column_resize = base_resolution
    # print('HERE')
    # print(row_resize, column_resize)
    return row_resize, column_resize

def _gkern(kernlen: int, std: int) -> np.ndarray:

    '''
        Returns a 2D Gaussian kernel array.

        Params:
            kernlen, std (int):
                Kernel length and standard deviation

        Returns:
            np.ndarray:
                gkern2d
    '''

    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

