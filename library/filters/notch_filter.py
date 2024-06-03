from scipy import signal
import numpy as np
import os 
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.ephys import EphysSeries

def notch_filt(ephys_series: EphysSeries, band=10, freq=60, ripple=1, order=2, filter_type='butter', analog_filt=False):
    '''# Required input defintions are as follows;
    # time:   Time between samples
    # band:   The bandwidth around the centerline freqency that you wish to filter
    # freq:   The centerline frequency to be filtered
    # ripple: The maximum passband ripple that is allowed in db
    # order:  The filter order.  For FIR notch filters this is best set to 2 or 3,
    #         IIR filters are best suited for high values of order.  This algorithm
    #         is hard coded to FIR filters
    # filter_type: 'butter', 'bessel', 'cheby1', 'cheby2', 'ellip'
    # data:         the data to be filtered'''

    data, Fs = ephys_series.signal, ephys_series.sample_rate[0]

    cutoff = freq
    nyq = Fs / 2.0
    low = freq - band / 2.0
    high = freq + band / 2.0
    low = low / nyq
    high = high / nyq
    b, a = signal.iirfilter(order, [low, high], rp=ripple, btype='bandstop', analog=analog_filt, ftype=filter_type)

    filtered_data = np.array([])

    if len(data) != 0:
        if len(data.shape) > 1:  # lfilter is one dimensional so we need to perform for loop on multi-dimensional array
            # filtered_data = np.zeros((data.shape[0], data.shape[1]))
            filtered_data = signal.filtfilt(b, a, data, axis=1)
            # for channel_num in range(0, data.shape[0]):
            # filtered_data[channel_num,:] =scipy.signal.lfilter(b, a, data[channel_num,:])
            #   filtered_data[channel_num, :] =scipy.signal.filtfilt(b, a, data[channel_num, :])
        else:
            # filtered_data =scipy.signal.lfilter(b, a, data)
            filtered_data = signal.filtfilt(b, a, data)

    return filtered_data
