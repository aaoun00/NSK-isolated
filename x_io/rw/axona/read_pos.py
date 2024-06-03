"""
This module contains methods for reading and writing to the .pos file format, which stores position data.
"""

# Internal Dependencies

# A+ Grade Dependencies
import numpy as np
import scipy 
import json, struct

# A Grade Dependencies

# Other Dependencies


def _rem_bad_track(x, y, t, threshold):
    """function [x,y,t] = _rem_bad_track(x,y,t,treshold)

    % Indexes to position samples that are to be removed
   """

    remInd = []
    diffx = np.diff(x, axis=0)
    diffy = np.diff(y, axis=0)
    diffR = np.sqrt(diffx ** 2 + diffy ** 2)

    # the MATLAB works fine without NaNs, if there are Nan's just set them to threshold they will be removed later
    diffR[np.isnan(diffR)] = threshold # setting the nan values to threshold
    ind = np.where((diffR > threshold))[0]

    if len(ind) == 0:  # no bad samples to remove
        return x, y, t

    if ind[-1] == len(x):
        offset = 2
    else:
        offset = 1

    for index in range(len(ind) - offset):
        if ind[index + 1] == ind[index] + 1:
            # A single sample position jump, tracker jumps out one sample and
            # then jumps back to path on the next sample. Remove bad sample.
            remInd.append(ind[index] + 1)
        else:
            ''' Not a single jump. 2 possibilities:
             1. TrackerMetadata jumps out, and stay out at the same place for several
             samples and then jumps back.
             2. TrackerMetadata just has a small jump before path continues as normal,
             unknown reason for this. In latter case the samples are left
             untouched'''
            idx = np.where(x[ind[index] + 1:ind[index + 1] + 1 + 1] == x[ind[index] + 1])[0]
            if len(idx) == len(x[ind[index] + 1:ind[index + 1] + 1 + 1]):
                remInd.extend(
                    list(range(ind[index] + 1, ind[index + 1] + 1 + 1)))  # have that extra since range goes to end-1

    # keep_ind = [val for val in range(len(x)) if val not in remInd]
    keep_ind = np.setdiff1d(np.arange(len(x)), remInd)

    x = x[keep_ind]
    y = y[keep_ind]
    t = t[keep_ind]

    return x.reshape((len(x), 1)), y.reshape((len(y), 1)), t.reshape((len(t), 1))

def _find_center(NE, NW, SW, SE):
    """Finds the center point (x,y) of the position boundaries"""

    x = np.mean([np.amax([NE[0], SE[0]]), np.amin([NW[0], SW[0]])])
    y = np.mean([np.amax([NW[1], NE[1]]), np.amin([SW[1], SE[1]])])
    return np.array([x, y])

def _center_box(posx, posy):
    # must remove Nans first because the np.amin will return nan if there is a nan
    posx = posx[~np.isnan(posx)]  # removes NaNs
    posy = posy[~np.isnan(posy)]  # remove Nans

    NE = np.array([np.amax(posx), np.amax(posy)])
    NW = np.array([np.amin(posx), np.amax(posy)])
    SW = np.array([np.amin(posx), np.amin(posy)])
    SE = np.array([np.amax(posx), np.amin(posy)])

    return _find_center(NE, NW, SW, SE)

def _fix_timestamps(post):
    first = post[0]
    N = len(post)
    uniquePost = np.unique(post)

    if len(uniquePost) != N:
        didFix = True
        numZeros = 0
        # find the number of zeros at the end of the file

        while True:
            if post[-1 - numZeros] == 0:
                numZeros += 1
            else:
                break
        last = first + (N-1-numZeros)*0.02
        fixedPost = np.arange(first, last+0.02, 0.02)
        fixedPost = fixedPost.reshape((len(fixedPost), 1))

    else:
        didFix = False
        fixedPost = []

    return didFix, fixedPost

def _arena_config(posx, posy, ppm, center, flip_y=True):
    """
    :param posx:
    :param posy:
    :param arena:
    :param conversion:
    :param center:
    :param flip_y: bool value that will determine if you want to flip y or not. When recording on Intan we inverted the
    positions due to the camera position. However in the virtualmaze you might not want to flip y values.
    :return:
    """
    print('PPM HERE: ', ppm)
    center = center
    conversion = ppm

    posx = 100 * (posx - center[0]) / conversion
    
    if flip_y:
        # flip the y axis
        posy = 100 * (-posy + center[1]) / conversion
    else:
        posy = 100 * (posy + center[1]) / conversion

    return posx, posy

def _remove_nan(posx, posy, post):
    """Remove any NaNs from the end of the array"""
    remove_nan = True
    while remove_nan:
        if np.isnan(posx[-1]):
            posx = posx[:-1]
            posy = posy[:-1]
            post = post[:-1]
        else:
            remove_nan = False
    return posx, posy, post

def _get_position(pos_fpath, ppm=None, method='', flip_y=True):
    """
    _get_position function:
    ---------------------------------------------
    variables:
    -pos_fpath: the full path (C:/example/session.pos)

    output:
    t: column numpy array of the time stamps
    x: a column array of the x-values (in pixels)
    y: a column array of the y-values (in pixels)
    """
    with open(pos_fpath, 'rb+') as f:  # opening the .pos file
        headers = ''  # initializing the header string
        for line in f:  # reads line by line to read the header of the file
            # print(line)
            if 'data_start' in str(line):  # if it reads data_start that means the header has ended
                headers += 'data_start'
                break  # break out of for loop once header has finished
            elif 'duration' in str(line):
                headers += line.decode(encoding='UTF-8')
            elif 'num_pos_samples' in str(line):
                num_pos_samples = int(line.decode(encoding='UTF-8')[len('num_pos_samples '):])
                headers += line.decode(encoding='UTF-8')
            elif 'bytes_per_timestamp' in str(line):
                bytes_per_timestamp = int(line.decode(encoding='UTF-8')[len('bytes_per_timestamp '):])
                headers += line.decode(encoding='UTF-8')
            elif 'bytes_per_coord' in str(line):
                bytes_per_coord = int(line.decode(encoding='UTF-8')[len('bytes_per_coord '):])
                headers += line.decode(encoding='UTF-8')
            elif 'timebase' in str(line):
                timebase = (line.decode(encoding='UTF-8')[len('timebase '):]).split(' ')[0]
                headers += line.decode(encoding='UTF-8')
            elif 'pixels_per_metre' in str(line):
                new_ppm = float(line.decode(encoding='UTF-8')[len('pixels_per_metre '):])
                headers += line.decode(encoding='UTF-8')
                if ppm is None:
                    print('DECODING PPM FROM FILE')
                    ppm = new_ppm
                else:
                    print('USING PPM YOU SET IN SETTINGS')
            elif 'min_x' in str(line) and 'window' not in str(line):
                min_x = int(line.decode(encoding='UTF-8')[len('min_x '):])
                headers += line.decode(encoding='UTF-8')
            elif 'max_x' in str(line) and 'window' not in str(line):
                max_x = int(line.decode(encoding='UTF-8')[len('max_x '):])
                headers += line.decode(encoding='UTF-8')
            elif 'min_y' in str(line) and 'window' not in str(line):
                min_y = int(line.decode(encoding='UTF-8')[len('min_y '):])
                headers += line.decode(encoding='UTF-8')
            elif 'max_y' in str(line) and 'window' not in str(line):
                max_y = int(line.decode(encoding='UTF-8')[len('max_y '):])
                headers += line.decode(encoding='UTF-8')
            elif 'pos_format' in str(line):
                headers += line.decode(encoding='UTF-8')
                if 't,x1,y1,x2,y2,numpix1,numpix2' in str(line):
                    two_spot = True
                else:
                    two_spot = False
                    print('The position format is unrecognized!')

            elif 'sample_rate' in str(line):
                sample_rate = float(line.decode(encoding='UTF-8').split(' ')[1])
                headers += line.decode(encoding='UTF-8')

            else:
                headers += line.decode(encoding='UTF-8')

    assert ppm is not None, 'PPM must be in position file or settings dictionary to proceed'

    if two_spot:
        '''Run when two spot mode is on, (one_spot has the same format so it will also run here)'''
        with open(pos_fpath, 'rb+') as f:
            '''get_pos for one_spot'''
            pos_data = f.read()  # all the position data values (including header)
            pos_data = pos_data[len(headers):-12]  # removes the header values

            byte_string = 'i8h'

            pos_data = np.asarray(struct.unpack('>%s' % (num_pos_samples * byte_string), pos_data))
            pos_data = pos_data.astype(float).reshape((num_pos_samples, 9))  # there are 8 words and 1 time sample

        x = pos_data[:, 1]
        y = pos_data[:, 2]
        t = pos_data[:, 0]

        x = x.reshape((len(x), 1))
        y = y.reshape((len(y), 1))
        t = t.reshape((len(t), 1))

        if method == 'raw':
            return x, y, t, sample_rate

        t = np.divide(t, np.float64(timebase))  # converting the frame number from Axona to the time value

        # values that are NaN are set to 1023 in Axona's system, replace these values by NaN's

        x[np.where(x == 1023)] = np.nan
        y[np.where(y == 1023)] = np.nan

        didFix, fixedPost = _fix_timestamps(t)

        if didFix:
            t = fixedPost

        t = t - t[0]

        x, y = _arena_config(x, y, ppm, center=_center_box(x,y), flip_y=flip_y)

        # remove any NaNs at the end of the file
        x, y, t = _remove_nan(x, y, t)

    else:
        print("Haven't made any code for this part yet.")

    return x.reshape((len(x), 1)), y.reshape((len(y), 1)), t.reshape((len(t), 1)), sample_rate, ppm


def grab_position_data(pos_path: str, ppm=None, override_arena_size=None) -> tuple:

    '''
        Extracts position data from .pos file

        Params:
            pos_path (str):
                Directory of where the position file is stored
            ppm (float):
                Pixel per meter value

        Returns:
            Tuple: pos_x,pos_y,pos_t,(pos_x_width,pos_y_width)
            --------
            pos_x, pos_y, pos_t (np.ndarray):
                Array of x, y coordinates, and timestamps
            pos_x_width (float):
                max - min x coordinate value (arena width)
            pos_y_width (float)
                max - min y coordinate value (arena length)
    '''

    pos_data = _get_position(pos_path, ppm=ppm)

    # Correcting pos_t data in case of bad position file
    new_pos_t = np.copy(pos_data[2])
    if len(new_pos_t) < len(pos_data[0]):
        while len(new_pos_t) != len(pos_data[0]):
            new_pos_t = np.append(new_pos_t, float(new_pos_t[-1] + 0.02))
    elif len(new_pos_t) > len(pos_data[0]):
        while len(new_pos_t) != len(pos_data[0]):
            new_pos_t = np.delete(new_pos_t, -1)

    Fs_pos = pos_data[3]

    file_ppm = pos_data[-1]

    assert file_ppm is not None, 'PPM must be in position file or settings dictionary to proceed'

    pos_x = pos_data[0]
    pos_y = pos_data[1]
    pos_t = new_pos_t

    # Rescale coordinate values with respect to a center point
    # (i.e arena center = origin (0,0))
    center = _center_box(pos_x, pos_y)
    pos_x = pos_x - center[0]
    pos_y = pos_y - center[1]

    # Correct for bad tracking
    pos_data_corrected = _rem_bad_track(pos_x, pos_y, pos_t, 2)
    pos_x = pos_data_corrected[0]
    pos_y = pos_data_corrected[1]
    pos_t = pos_data_corrected[2]

    # Remove NaN values
    nonNanValues = np.where(np.isnan(pos_x) == False)[0]
    pos_t = pos_t[nonNanValues]
    pos_x = pos_x[nonNanValues]
    pos_y = pos_y[nonNanValues]

    # Smooth data using boxcar convolution
    B = np.ones((int(np.ceil(0.4 * Fs_pos)), 1)) / np.ceil(0.4 * Fs_pos)
    pos_x = scipy.ndimage.convolve(pos_x, B, mode='nearest')
    pos_y = scipy.ndimage.convolve(pos_y, B, mode='nearest')

    pos_x_width = max(pos_x) - min(pos_x)
    pos_y_width = max(pos_y) - min(pos_y)

    if override_arena_size is not None:
        # override_arena_size is (height, width)
        pos_x_width = override_arena_size[1] # width
        pos_y_width = override_arena_size[0] # height

    return {"t": pos_t, "x": pos_x, "y": pos_y, "arena_width":pos_x_width, "arena_height":pos_y_width, "sample_rate":pos_data[3], "ppm":file_ppm}