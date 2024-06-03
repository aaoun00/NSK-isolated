from PIL import Image
import cv2
import numpy as np 
from scipy import signal

def _interpolate_matrix(matrix, new_size=(256,256), cv2_interpolation_method=cv2.INTER_NEAREST):
    '''
        Interpolate a matrix using cv2.INTER_LANCZOS4.
    '''
    return cv2.resize(matrix, dsize=new_size,
                      interpolation=cv2_interpolation_method)

def disk_mask(matrix):
        y_segments, x_segments = matrix.shape

        y_center, x_center = (y_segments-1)/2, (x_segments-1)/2

        mask_r = min(x_segments, y_segments)/2

        mask_y,mask_x = np.ogrid[-y_center:y_segments-y_center, -x_center:x_segments-x_center]
        mask = mask_x**2 + mask_y**2 > mask_r**2

        masked_matrix = np.ma.array(matrix, mask=mask)

        return masked_matrix

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

    return row_resize, column_resize

def _resize_numpy2D(array: np.ndarray, x: int, y: int) -> np.ndarray:

    '''
        Resizes a numpy array.

        Params:
            array (numpy.ndarray):
                Numpy array to be resized
            x (int):
                Resizing row number (length)
            y (int):
                Resizing column number (width)

        Returns:
            array (numpy.ndarray): Resized array with new dimensions (array.shape = (x,y))
    '''

    array = Image.fromarray(array)
    array = array.resize((x,y))
    array = np.array(array)

    return array

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

