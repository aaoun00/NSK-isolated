import numpy as np
from scipy import signal
import os 
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.ephys import EphysSeries

def custom_cheby1(ephys_series: EphysSeries, N, Rp, Wp, Ws=None, filtresponse='bandpass', analog_value=False, showresponse=0):
    data, Fs = ephys_series.signal, ephys_series.sample_rate[0]

    nyquist = Fs/2

    if filtresponse == 'bandpass':
        Wn = [Wp/nyquist, Ws/nyquist]
    else:
        Wn = [Wp/nyquist]

    b, a = signal.cheby1(N, Rp, Wn, 'bandpass', analog=analog_value)

    if len(data) > 0:
        if len(data.shape) > 1:
            #print('Filtering multidimensional array!')
            filtered_data = np.zeros((data.shape[0], data.shape[1]))
            filtered_data = signal.filtfilt(b, a, data, axis=1)
            #for channel_num in range(0, data.shape[0]):
            #    # filtered_data[channel_num,:] = signal.lfilter(b, a, data[channel_num,:])
            #    filtered_data[channel_num, :] = signal.filtfilt(b, a, data[channel_num, :])
        else:
            # filtered_data = signal.lfilter(b, a, data)
            filtered_data = signal.filtfilt(b, a, data)

    return filtered_data

