import numpy as np
#from opexebo.analysis import place_field, border_score
import math
import os,sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

# from library.spatial_spike_train import SpatialSpikeTrain2D
from library.hafting_spatial_maps import SpatialSpikeTrain2D
# from library.opexebo import opexebo_tuning_curve, angular_occupancy
import library.opexebo.defaults as default
import library.opexebo.errors as err
from library.lib_utils import bin_width_to_bin_number
from library.spatial import accumulate_spatial

def _smooth(array: np.ndarray, window: int) -> np.ndarray:

    '''
        Smooths an array using sliding wsindow approach

        Params:
            array (np.ndarray):
                Array to be smoothed
            window (int):
                Number of points to be smoothed at a time as a sliding window

        Returns:
            np.ndarray:
                smoothed_array
    '''

    # Initialize empty array
    smoothed_array = np.zeros((len(array), 1))

    # Iterate over array using sliding window and compute averages
    for i in range(len(array) - window + 1):
        current_average = sum(array[i:i+window]) / window
        smoothed_array[i:i+window] = current_average

    return smoothed_array

def _get_head_direction(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    '''
        Will compute the head direction angle of subject over experiemnt.
        Params:
            x, y (np.ndarray):
                Arrays of x and y coordinates.

        Returns:
            np.ndarray:
                angles
            --------
            angles:
                Array of head direction angles in radians, reflecting what heading the subject
                during the course of the sesison.
    '''

    last_point = [0,0]  # Keep track of the last x,y point
    last_angle = 0      # Keep track of the most previous computed angle
    angles = []         # Will accumulate angles as they are computed

    # Iterate over the x and y points
    for i in range(len(x)):

        # Grab the current point
        current_point = [float(x[i]), float(y[i])]

        # If the last point is the same as the current point
        if (last_point[0] == current_point[0]) and (last_point[1] == current_point[1]):
            # Retain the same angle
            angle = last_angle

        else:
            # Compute the arctan (i.e the angle formed by the line defined by both points
            # and the horizontal axis [range -180 to +180])
            # Uses the formula arctan( (y2-y1) / (x2-x1))
            Y = current_point[1] - last_point[1]
            X = current_point[0] - last_point[0]
            angle = math.atan2(Y,X) * (180/np.pi) # Convert to degrees

        # Append angle value to list
        angles.append(angle)

        # Update new angle and last_point values
        last_point[0] = current_point[0]
        last_point[1] = current_point[1]
        last_angle = angle

    # Scale angles between 0 to 360 rather than -180 to 180
    angles = np.array(angles)
    angles = (angles + 360) % 360

    # Convert to radians
    angles = angles * (np.pi/180)

    return angles

# def spatial_tuning_curve(x: np.ndarray, y: np.ndarray, t: np.ndarray, spike_times: np.ndarray, smoothing: int) -> tuple:
def spatial_tuning_curve(spatial_spike_train: SpatialSpikeTrain2D) -> tuple:

    '''
        Compute a polar plot of the average directional firing of a neuron.

        Params:
            x, y, t (np.ndarray):
                Arrays of x and y coordinates, and timestamps
            spike_times (np.ndarray):
                Timestamps of when spike events occured
            smoothing (int):
                Smoothing factor for angle data

        Returns:
            tuple: tuned_data, spike_angles, ang_occ, bin_array
            --------
            tuned_data (np.ndarray):
                Tuning curve
            spike_angles (np.ndarray):
                Angles at which spike occured
            ang_occ (np.ndarray):
                Histogram of occupanices within bins of angles
            bin_array (np.ndarray):
                Bins of angles (360 split into 36 bins of width 10 degrees)
    '''
    spike_times = np.array(spatial_spike_train.spike_times)
    smoothing = spatial_spike_train.session_metadata.session_object.smoothing_factor
    t = np.array(spatial_spike_train.t)
    x = np.array(spatial_spike_train.x)
    y = np.array(spatial_spike_train.y)

    # Compute head direction angles
    hd_angles = _get_head_direction(x, y)

    # Split angle range (0 to 360) into angle bins
    bin_array = np.linspace(0,2*np.pi,36)

    # Compute histogram of occupanices in each bin
    ang_occ = _angular_occupancy(t.flatten(), hd_angles.flatten(), bin_width=10)

    # Extract spike angles (i.e angles at which spikes occured)
    spike_angles = []
    for i in range(len(spike_times)):
        index = np.abs(t - spike_times[i]).argmin()
        spike_angles.append(hd_angles[index])

    spike_angles = np.array(spike_angles)
    spike_angles = spike_angles.flatten()

    # Compute tuning curve and smooth
    tuned_data = _opexebo_tuning_curve(ang_occ[0], spike_angles, bin_width=10)
    tuned_data_masked = np.copy(tuned_data)
    bin_array = bin_array
    tuned_data[tuned_data == np.nan] = 0
    tuned_data = _smooth(tuned_data,smoothing)

    dir_dict = {'tuned_data': tuned_data, 'spike_angles': spike_angles, 'ang_occ': ang_occ, 'bin_array': bin_array}

    spatial_spike_train.add_map_to_stats('spatial_tuning', dir_dict)

    return tuned_data, spike_angles, ang_occ, bin_array

""""""""""""""""""""""""""" From Opexebo https://pypi.org/project/opexebo/ """""""""""""""""""""""""""

def _opexebo_tuning_curve(angular_occupancy, spike_angles, **kwargs):
    """Analogous to a RateMap - i.e. mapping spike activity to spatial position
    map spike rate as a function of angle

    Parameters
    ----------
    angular_occupancy : np.ma.MaskedArray
        unsmoothed histogram of time spent at each angular range
        Nx1 array, covering the range [0, 2pi] radians
        Masked at angles of zero occupancy
    spike_angles : np.ndarray
        Mx1 array, where the m'th value is the angle of the animal (in radians)
        associated with the m'th spike
    kwargs
        bin_width : float
            width of histogram bin in DEGREES
            Must match that used in calculating angular_occupancy
            In the case of a non-exact divisor of 360 deg, the bin size will be 
            shrunk to yield an integer bin number. 


    Returns
    -------
    tuning_curve : np.ma.MaskedArray
        unsmoothed array of firing rate as a function of angle
        Nx1 array

    Notes
    --------
    BNT.+analyses.turningcurve

    Copyright (C) 2019 by Simon Ball

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
    """

    occ_ndim = angular_occupancy.ndim
    spk_ndim = spike_angles.ndim
    if occ_ndim != 1:
        raise ValueError("angular_occupancy must be a 1D array. You provided a"\
                         " %d dimensional array" % occ_ndim)
    if spk_ndim != 1:
        raise ValueError("spike_angles must be a 1D array. You provided a %d"\
                         " dimensional array" % spk_ndim)
    if np.nanmax(spike_angles) > 2*np.pi:
        raise Warning("Angles higher than 2pi detected. Please check that your"\
                      " spike_angle array is in radians. If it is in degrees,"\
                      " you can convert with 'np.radians(array)'")

    bin_width = kwargs.get("bin_width", default.bin_angle) # in degrees
    num_bins = bin_width_to_bin_number(360., bin_width) # This is for validation ONLY, the value num_bins here is not passed onwards
    if num_bins != angular_occupancy.size:
        raise ValueError("Keyword 'bin_width' must match the value used to"\
                         " generate angular_occupancy")
    #UNITS!!!
    # bin_width and arena_size need to be in the same units. 
    # As it happens, I hardcoded arena_size as 2pi -> convert bin_width to radians
    # limits and spike_angles need to be in the same units
    
    bin_width = np.radians(bin_width)

    spike_histogram, bin_edges = accumulate_spatial(spike_angles,
                arena_size=2*np.pi, limits=(0, 2*np.pi), bin_width=bin_width)

    tuning_curve = spike_histogram / (angular_occupancy + np.spacing(1))

    return tuning_curve

def _angular_occupancy(time, angle, **kwargs):
    '''
    Calculate angular occupancy from tracking angle and kwargs over (0,2*pi)

    Parameters
    ----------
    time : numpy.ndarray
        time stamps of angles in seconds
    angle : numpy array
        Head angle in radians
        Nx1 array
    bin_width : float, optional
        Width of histogram bin in degrees

    Returns
    -------
    masked_histogram : numpy masked array
        Angular histogram, masked at angles at which the animal was never 
        observed. A mask value of True means that the animal never occupied
        that angle. 
    coverage : float
        Fraction of the bins that the animal visited. In range [0, 1]
    bin_edges : list-like
        x, or (x, y), where x, y are 1d np.ndarrays
        Here x, y correspond to the output histogram
    
    Notes
    --------
    Copyright (C) 2019 by Simon Ball, Horst Obenhaus

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
    '''
    if time.ndim != 1:
        raise err.ArgumentError("time must be provided as a 1D array. You provided %d"\
                         " dimensions" % time.ndim)
    if angle.ndim != 1:
        raise err.ArgumentError("angle must be provided as a 1D array. You provided %d"\
                         " dimensions" % angle.ndim)
    if time.size != angle.size:
        raise err.ArgumentError("Arrays 'time' and 'angle' must have the same number"\
                         f" of elements. You provided {time.size} and {angle.size}")
    if time.size == 0:
        raise err.ArgumentError("Zero length array provided when data expected")
    if np.nanmax(angle) > 2*np.pi:
        raise Warning("Angles greater than 2pi detected. Please check that your"\
                      " angle array is in radians. If it is in degrees, you can"\
                      " convert with 'np.radians(array)'")

    bin_width = kwargs.get('bin_width', default.bin_angle)
    bin_width = np.radians(bin_width)
    arena_size = 2*np.pi
    limits = (0, arena_size)

    angle_histogram, bin_edges = accumulate_spatial(angle, bin_width=bin_width, 
                                                arena_size=arena_size, limits=limits)
    masked_angle_histogram = np.ma.masked_where(angle_histogram==0, angle_histogram)
    
    # masked_angle_histogram is in units of frames. It needs to be converted to units of seconds
    frame_duration = np.mean(np.diff(time))
    masked_angle_seconds = masked_angle_histogram * frame_duration
    
    # Calculate the fractional coverage based on locations where the histogram
    # is zero. If all locations are  non-zero, then coverage is 1.0
    coverage = np.count_nonzero(angle_histogram) / masked_angle_seconds.size
    
    return masked_angle_seconds, coverage, bin_edges
