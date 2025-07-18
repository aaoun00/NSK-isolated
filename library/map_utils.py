import os, sys
import numpy as np
import multiprocessing as mp
import functools
import itertools
import cv2
# from numba import jit, njit
import matplotlib.pyplot as plt
from scipy import signal

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.spatial import Position2D


def _speed_bins(lower_speed: float, higher_speed: float, pos_v: np.ndarray,
               pos_x: np.ndarray, pos_y: np.ndarray, pos_t: np.ndarray) -> tuple:

    '''
        Selectively filters position values of subject travelling between
        specific speed limits.

        Params:
            lower_speed (float):
                Lower speed bound (cm/s)
            higher_speed (float):
                Higher speed bound (cm/s)
            pos_v (np.ndarray):
                Array holding speed values of subject
            pos_x, pos_y, pos_t (np.ndarray):
                X, Y coordinate tracking of subject and timestamps

        Returns:
            Tuple: new_pos_x, new_pos_y, new_pos_t
            --------
            new_pos_x, new_pos_y, new_pos_t (np.ndarray):
                speed filtered x,y coordinates and timestamps
    '''

    # Initialize empty array that will only be populated with speed values within
    # specified bounds
    choose_array = []

    # Iterate and select speeds
    for index, element in enumerate(pos_v):
        if element > lower_speed and element < higher_speed:
            choose_array.append(index)

    # construct new x,y and t arrays
    new_pos_x = np.asarray([ float(pos_x[i]) for i in choose_array])
    new_pos_y = np.asarray([ float(pos_y[i]) for i in choose_array])
    new_pos_t = np.asarray([ float(pos_t[i]) for i in choose_array])

    return new_pos_x, new_pos_y, new_pos_t


def _speed2D(x, y, t):
    """calculates an averaged/smoothed speed"""

    N = len(x)
    v = np.zeros((N, 1))

    for index in range(1, N-1):
        v[index] = np.sqrt((x[index + 1] - x[index - 1]) ** 2 + (y[index + 1] - y[index - 1]) ** 2) / (
        t[index + 1] - t[index - 1])

    v[0] = v[1]
    v[-1] = v[-2]

    return v


def _compute_unmasked_ratemap(occpancy_map, spike_map):
        return spike_map/occpancy_map

def _interpolate_matrix(matrix, new_size=(256,256), cv2_interpolation_method=cv2.INTER_NEAREST):
    '''
        Interpolate a matrix using cv2.INTER_LANCZOS4.
    '''
    return cv2.resize(matrix, dsize=new_size,
                      interpolation=cv2_interpolation_method)

def _gaussian2D_pdf(sigma, x, y):
    '''
    Parameters:
        sigma: the standard deviation of the gaussian kernel
        x: the x-coordinate with respect to the center of the distribution
        y: the y-coordinate with respect to the distribution
    Returns:
        z: the 2D PDF value at (x,y)
    '''
    z = 1/(2*np.pi*(sigma**2))*np.exp(-(x**2 + y**2)/(2*(sigma**2)))
    return z

def _get_point_pdfs(rel_pos):
    pdfs = []
    for rel_xi, rel_yi in rel_pos:
        pdfs.append(_gaussian2D_pdf(1, rel_xi, rel_yi))
    return np.array(pdfs)

def _sum_point_pdfs(pdfs):
    return np.sum(pdfs)

def _spike_pdf(spike_x, spike_y, smoothing_factor, point):

    x, y = point

    rel_x = (spike_x - x)/smoothing_factor

    rel_y = (spike_y - y)/smoothing_factor

    rel_coords = ((rel_xi, rel_yi) for rel_xi, rel_yi in zip(rel_x, rel_y))

    pdfs = _get_point_pdfs(rel_coords)

    estimate = _sum_point_pdfs(pdfs)

    return estimate

def _integrate_pos_pdfs(pdfs, pos_time):
    return np.trapz(y=np.array(pdfs).T, x=np.array(pos_time).T)

def _pos_pdf(pos_x: np.ndarray, pos_y: np.ndarray, pos_time: np.ndarray, smoothing_factor, point: tuple, ):

        x, y = point

        rel_x = (pos_x - x)/smoothing_factor

        rel_y = (pos_y - y)/smoothing_factor

        rel_coords = ((rel_xi, rel_yi) for rel_xi, rel_yi in zip(rel_x, rel_y))

        pdfs = _get_point_pdfs(rel_coords)

        estimate = _integrate_pos_pdfs(pdfs, pos_time)

        return float(estimate)

def _mask_points_far_from_curve(mask_threshold, curve_x, curve_y, point):
    '''
    Parameters:
        point (numeric): the point to calculate the distance to the curve
        curve (iterable): the curve to calculate the distance to the point
    Returns:
        distance (numeric): the distance between the point and the curve

    This is very inefficient --- O(base_resolution^2 * curve_length) --- any way to speed this up?
    '''
    point_x, point_y = point
    distance = min(np.sqrt((point_x - curve_x)**2 + (point_y - curve_y)**2))
    if distance > mask_threshold:
        return 1
    else:
        return 0

    return distance

def save_map(occupancy_map, title, units_label, file_name, directory):
    '''
    Parameters:
        map: the map to save
        filename: the filename to save the map as
    '''
    # print(np.max(occupancy_map))
    f, ax = plt.subplots(figsize=(7, 7))
    m = ax.imshow(occupancy_map, cmap="jet", interpolation="nearest")
    cbar = f.colorbar(m, ax=ax, shrink=0.8)
    cbar.set_label(units_label, rotation=270, labelpad=20)
    ax.set_title(title, fontsize=16, pad=20)
    ax.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.savefig(directory + '/' + file_name + '.png')
    plt.close('all')

def _compute_resize_ratio(arena_size: tuple, base_resolution=64) -> tuple:

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
    # base_resolution = 64
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

    gkern1d = signal.windows.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def _temp_occupancy_map(position: Position2D, smoothing_factor, interp_size=(64,64), useMinMaxPos=False) -> np.ndarray:

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
    pos_x, pos_y, pos_t = position.x, position.y, position.t

    if isinstance(position, Position2D):
        arena_size = (position.arena_height, position.arena_width)
    else:
        arena_size = position.arena_size

    if useMinMaxPos:
        min_x = min(pos_x)
        max_x = max(pos_x)
        min_y = min(pos_y)
        max_y = max(pos_y)
    else:
        min_x = [-arena_size[1]/2]
        max_x = [arena_size[1]/2] # width 
        min_y = [-arena_size[0]/2]
        max_y = [arena_size[0]/2] # height

    # arena_size = (abs(max_y-min_y), abs(max_x - min_x)) # (height, width)

    # print('search here')
    # print(arena_size, min_x, max_x, min_y, max_y)
    # print(pos_x, pos_y)

    # Resize ratio
    # row_resize, column_resize = _compute_resize_ratio(arena_size, base_resolution=interp_size[0])
    # print(interp_size)
    row_resize, column_resize = interp_size[0], interp_size[1]

    # Initialize empty map
    occ_map_raw = np.zeros((row_resize,column_resize))
    coverage_map = np.zeros((row_resize,column_resize))
    row_values = np.linspace(max_y,min_y,row_resize)
    column_values = np.linspace(min_x,max_x,column_resize)

    # print('Generating raw occupancy map...')
    # Generate the raw occupancy map
    for i in range(1, len(pos_t)):

        row_index = np.abs(row_values - pos_y[i]).argmin()
        column_index = np.abs(column_values - pos_x[i]).argmin()
        occ_map_raw[row_index][column_index] += pos_t[i] - pos_t[i-1]
        coverage_map[row_index][column_index] = 1

   # Kernel size
    kernlen = int(smoothing_factor*8)
    # Standard deviation size
    std = int(0.2*kernlen)

    # print('Smoothing raw occupancy map...')
    # Normalize and smooth with scaling facotr
    occ_map_normalized = occ_map_raw / pos_t[-1]
    occ_map_smoothed = cv2.filter2D(occ_map_normalized,-1,_gkern(kernlen,std))

    # print('Dilating coverage map...')
    # dilate coverage map
    kernel = np.ones((2,2))
    coverage_map = cv2.dilate(coverage_map, kernel, iterations=1)

    # Resize maps
    # occ_map_raw = _interpolate_matrix(occ_map_raw, new_size=interp_size, cv2_interpolation_method=cv2.INTER_NEAREST)
    # occ_map_smoothed = _interpolate_matrix(occ_map_smoothed, new_size=interp_size,  cv2_interpolation_method=cv2.INTER_NEAREST)
    occ_map_smoothed = occ_map_smoothed/max(occ_map_smoothed.flatten())
    # print('SHAPE IS HERE: ', occ_map_smoothed.shape, coverage_map.shape, occ_map_raw.shape)
    return occ_map_smoothed, occ_map_raw, coverage_map

def _temp_spike_map(pos_x: np.ndarray, pos_y: np.ndarray, pos_t: np.ndarray,
                arena_size: tuple, spike_x: np.ndarray, spike_y: np.ndarray,
                smoothing_factor: int, interp_size=(64,64)) -> np.ndarray:

    # Kernel size
    kernlen = int(smoothing_factor*8)
    # Standard deviation size
    std = int(0.2*kernlen)

    # kerneln = 32
    # std = int(0.2*kernlen)

    # TESTING
    # kernelen relative to ratemap (e.g. ratemap=(64,64), try smtg like (32,32) or other stable ration)
    # std play around with based on kernlen
    # play with ratios, see how scales


    # Min and max dimensions of arena for scaling
    # min_x = min(pos_x)
    # max_x = max(pos_x)
    # min_y = min(pos_y)
    # max_y = max(pos_y)
    min_x = 0
    max_x = arena_size[1] # width 
    min_y = 0
    max_y = arena_size[0] # height

    # arena_size = (abs(max_y - min_y), abs(max_x - min_x)) # (height, width)

    # Resize ratio
    # row_resize, column_resize = _compute_resize_ratio(arena_size, base_resolution=interp_size[0])
    row_resize, column_resize = interp_size[0], interp_size[1]

    # Instantiate empty maps
    spike_map_raw = np.zeros((row_resize,column_resize))
    
    # Load spike data and set up arrays to map spike timestamps to subject position
    row_values = np.linspace(max_x, min_x, row_resize)
    column_values = np.linspace(min_y,max_y, column_resize)

    # print('Generating raw spike map...')
    # Generate raw spike map
    for i in range(len(spike_x)):
        row_index = np.abs(row_values - spike_y[i]).argmin()
        column_index = np.abs(column_values - spike_x[i]).argmin()
        spike_map_raw[row_index][column_index] += 1

    # Remove low spike counts from spike map (20th percentile)
    #spike_map_raw[spike_map_raw <= np.percentile(spike_map_raw, 20)] = 0

    # print('Smoothing spike map...')
    # Smooth spike map (must happen before resizing)
    spike_map_smooth = cv2.filter2D(spike_map_raw,-1,_gkern(kernlen, std))

    # Resize maps
    # spike_map_smooth = _interpolate_matrix(spike_map_smooth, new_size=interp_size, cv2_interpolation_method=cv2.INTER_NEAREST)
    spike_map_smooth = spike_map_smooth/max(spike_map_smooth.flatten())
    # spike_map_raw = _interpolate_matrix(spike_map_raw, new_size=interp_size,  cv2_interpolation_method=cv2.INTER_NEAREST)



    return spike_map_smooth, spike_map_raw
   
def _temp_spike_map_new(pos_x, pos_y, pos_t, arena_size, spike_x, spike_y, smoothing_factor, interp_size=(64, 64), useMinMaxPos=False):
    kernlen = int(smoothing_factor * 8)
    std = int(0.2 * kernlen)

    if useMinMaxPos:
        min_x, max_x = np.min(pos_x), np.max(pos_x)
        min_y, max_y = np.min(pos_y), np.max(pos_y)
        # arena_height = abs(max_y - min_y)
        # arena_width = abs(max_x - min_x)
        # arena_size = (abs(max_y - min_y), abs(max_x - min_x)) # (height, width)
    else:
        min_x = [-arena_size[1]/2]
        max_x = [arena_size[1]/2] # width 
        min_y = [-arena_size[0]/2]
        max_y = [arena_size[0]/2] # height
    # print('search here')
    # print(arena_height, arena_width)
    # print(arena_size, min_x, max_x, min_y, max_y)
    # print('a')
    # print(pos_x, pos_y)
    # stop()


    row_resize, column_resize = interp_size[0], interp_size[1]
    spike_map_raw = np.zeros((row_resize, column_resize))
    row_values = np.linspace(max_x, min_x, row_resize)
    column_values = np.linspace(min_y, max_y, column_resize)
    row_index = np.abs(row_values[:, np.newaxis] - spike_y).argmin(axis=0)
    column_index = np.abs(column_values[:, np.newaxis] - spike_x).argmin(axis=0)
    # np.add.at(spike_map_raw, (row_index, column_index), 1)
    np.add.at(spike_map_raw, (row_index, column_index), np.ones_like(row_index))
    spike_map_smooth = cv2.filter2D(spike_map_raw, -1, _gkern(kernlen, std))
    spike_map_smooth = spike_map_smooth / np.max(spike_map_smooth)
    return spike_map_smooth, spike_map_raw

