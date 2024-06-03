import numpy as np
import os, sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.spatial import Position2D

def filter_pos_by_speed(position_object: Position2D ,lower_speed: float, higher_speed: float) -> tuple:

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
    pos_x = position_object.x 
    pos_y = position_object.y 
    pos_t = position_object.t
    
    for var in [pos_x, pos_y, pos_t]:
        assert np.array(var).all() != None, str(var) + ' is None, cannot proceed'

    pos_v = _compute_speed(pos_x, pos_y, pos_t)

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

    position_object.stats_dict['speed_filter'] = {}
    position_object.stats_dict['speed_filter']['lower_bound'] = lower_speed
    position_object.stats_dict['speed_filter']['upper_bound'] = higher_speed
    position_object.stats_dict['speed_filter']['new_x'] = new_pos_x
    position_object.stats_dict['speed_filter']['new_y'] = new_pos_y
    position_object.stats_dict['speed_filter']['new_t'] = new_pos_t

    return new_pos_x, new_pos_y, new_pos_t

def _compute_speed(x, y, t):
    """calculates an averaged/smoothed speed"""

    N = len(x)
    v = np.zeros((N, 1))

    for index in range(1, N-1):
        v[index] = np.sqrt((x[index + 1] - x[index - 1]) ** 2 + (y[index + 1] - y[index - 1]) ** 2) / (
        t[index + 1] - t[index - 1])

    v[0] = v[1]
    v[-1] = v[-2]

    return v