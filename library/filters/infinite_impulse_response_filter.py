import numpy as np
from scipy import signal
import os 
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.ephys import EphysSeries

def iirfilt(ephys_series: EphysSeries, bandtype, Wp, Ws=[], order=3, analog_val=False, automatic=0, Rp=3, As=60, filttype='butter'):
    '''Designs butterworth filter:
    Data is the data that you want filtered
    Fs is the sampling frequency (in Hz)
    Ws and Wp are stop and pass frequencies respectively (in Hz)

    Passband (Wp) : This is the frequency range which we desire to let the signal through with minimal attenuation.
    Stopband (Ws) : This is the frequency range which the signal should be attenuated.

    Digital: Ws is the normalized stop frequency where 1 is the nyquist freq (pi radians/sample in digital)
             Wp is the normalized pass frequency

    Analog: Ws is the stop frequency in (rads/sec)
            Wp is the pass frequency in (rads/sec)

    Analog is false as default, automatic being one has Python select the order for you. pass_atten is the minimal attenuation
    the pass band, stop_atten is the minimal attenuation in the stop band. Fs is the sample frequency of the signal in Hz.

    Rp = 0.1      # passband maximum loss (gpass)
    As = 60 stoppand min attenuation (gstop)


    filttype : str, optional
        The type of IIR filter to design:
        Butterworth : ‘butter’
        Chebyshev I : ‘cheby1’
        Chebyshev II : ‘cheby2’
        Cauer/elliptic: ‘ellip’
        Bessel/Thomson: ‘bessel’

    bandtype : {‘bandpass’, ‘lowpass’, ‘highpass’, ‘bandstop’}, optional
    '''

    data = ephys_series.signal
    Fs = ephys_series.sample_rate[0]

    b, a = get_a_b(bandtype, Fs, Wp, Ws, order=order, Rp=Rp, As=As, analog_val=analog_val, filttype=filttype, automatic=automatic)

    assert len(data) != 0

    if len(data.shape) > 1:
        filtered_data = np.zeros((data.shape[0], data.shape[1]))
        filtered_data = signal.filtfilt(b, a, data, axis=1)
    else:
        filtered_data = signal.filtfilt(b, a, data)

    return filtered_data

def get_a_b(bandtype, Fs, Wp, Ws, order=3, Rp=3, As=60, analog_val=False, filttype='butter', automatic=0):

    stop_amp = 1.5
    stop_amp2 = 1.4

    if not analog_val:  # need to convert the Ws and Wp to have the units of pi radians/sample
        # this is for digital filters
        if bandtype in ['low', 'high']:
            Wp = Wp / (Fs / 2)  # converting to fraction of nyquist frequency

            Ws = Wp * stop_amp

        elif bandtype == 'band':
            Wp = Wp / (Fs / 2)  # converting to fraction of nyquist frequency
            Wp2 = Wp / stop_amp2

            Ws = Ws / (Fs / 2)  # converting to fraction of nyquist frequency
            Ws2 = Ws * stop_amp2

    else:  # need to convert the Ws and Wp to have the units of radians/sec
        # this is for analog filters
        if bandtype in ['low', 'high']:
            Wp = 2 * np.pi * Wp

            Ws = Wp * stop_amp

        elif bandtype == 'band':
            Wp = 2 * np.pi * Wp
            Wp2 = Wp / stop_amp2

            Ws = 2 * np.pi * Ws
            Ws2 = Ws * stop_amp2

    if automatic == 1:
        if bandtype in ['low', 'high']:
            b, a = signal.iirdesign(wp=Wp, ws=Ws, gpass=Rp, gstop=As, analog=analog_val, ftype=filttype)
        elif bandtype == 'band':
            b, a = signal.iirdesign(wp=[Wp, Ws], ws=[Wp2, Ws2], gpass=Rp, gstop=As, analog=analog_val, ftype=filttype)
    else:
        if bandtype in ['low', 'high']:
            if filttype == 'cheby1' or 'cheby2' or 'ellip':
                b, a = signal.iirfilter(order, Wp, rp=Rp, rs=As, btype=bandtype, analog=analog_val, ftype=filttype)
            else:
                b, a = signal.iirfilter(order, Wp, btype=bandtype, analog=analog_val, ftype=filttype)
        elif bandtype == 'band':
            if filttype == 'cheby1' or 'cheby2' or 'ellip':
                b, a = signal.iirfilter(order, [Wp, Ws], rp=Rp, rs=As, btype=bandtype, analog=analog_val,
                                        ftype=filttype)
            else:
                b, a = signal.iirfilter(order, [Wp, Ws], btype=bandtype, analog=analog_val, ftype=filttype)

    return b, a