import numpy as np

class Position2D():
    def __init__(self, input_dict, **kwargs):
        # self.subject = subject
        # self.limb = limb # e.g. head
        self._input_dict = input_dict

        self.t, self.x, self.y, self.arena_height, self.arena_width, self.session_metadata = self._read_input_dict()
        self.time_index = self.t

        if 'session_metadata' in kwargs:
            if self.session_metadata != None: 
                print('Ses metadata is in the input dict and init fxn, init fnx will override')
            self.session_metadata = kwargs['session_metadata']

        self._input_dict = input_dict
        self.stats_dict = {}

        self.arena_size = (self.arena_height, self.arena_width)

    def _read_input_dict(self):
        t, x, y, arena_height, arena_width, session_metadata = None, None, None, None, None, None

        for key in self._input_dict:
            if key == 't' or key == 'time':
                t = self._input_dict['t']
            elif key == 'rate' and 'x' in self._input_dict:
                t = np.arange(0, len(self._input_dict['x']) / self._input_dict['rate'], 1 / self._input_dict['rate'])
            elif key == 'x':
                x = self._input_dict['x']
            elif key == 'y':
                y = self._input_dict['y']
            elif key == 'session_metadata':
                session_metadata = self._input_dict['session_metadata']
            elif key == 'arena_height':
                arena_height = self._input_dict['arena_height']
            elif key == 'arena_width':
                arena_width = self._input_dict['arena_width']

        return t, x, y, arena_height, arena_width, session_metadata
            

"""
    def speed_from_locations(location) -> np.ndarray:
        '''calculates an averaged/smoothed speed'''

        x = location.x
        y = location.y
        t = location.t

        N = len(x)
        v = np.zeros((N, 1))

        _speed_formula = lambda index, x, y, t: np.sqrt((x[index + 1] - x[index - 1]) ** 2 + (y[index + 1] - y[index - 1]) ** 2) / (t[index + 1] - t[index - 1])

        v = np.ndarray(map(_speed_formula, range(1, N-1), x, y, t))

        '''
        for index in range(1, N-1):
            v[index] = np.sqrt((x[index + 1] - x[index - 1]) ** 2 + (y[index + 1] - y[index - 1]) ** 2) / (
            t[index + 1] - t[index - 1])
        '''

        v[0] = v[1]
        v[-1] = v[-2]
        v = v.flatten()

        kernel_size = 12
        kernel = np.ones(kernel_size) / kernel_size
        v_convolved = np.convolve(v, kernel, mode='same')

        return v_convolved

    def filter_pos_by_speed(self, lower_speed: float, higher_speed: float, speed2d: np.ndarray, pos_x: np.ndarray, pos_y: np.ndarray, pos_t: np.ndarray) -> tuple:
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
"""
