"""
Functions that take a single waveform as an input and return a feature.
"""

import numpy as np
from operator import add
from scipy.signal import savgol_filter

def waveform_features(waveform, time_step, channel):
    """
    Calculate the features of a waveform.
    Parameters
    ----------
    waveform : array_like
        The waveform to be analyzed.

    Returns
    -------
    feature vector
        A vector of features of the waveform.

    References:
    ----------
    Caro-Martín, Carmen Rocío, José M. Delgado-García, Agnès Gruart, and R. Sánchez-Campusano. “Spike Sorting Based on Shape, Phase, and Distribution Features, and K-TOPS Clustering with Validity and Error Indices.” Scientific Reports 8, no. 1 (December 12, 2018): 17796. https://doi.org/10.1038/s41598-018-35491-4.
    """
    # get domains)
    t = time_index(waveform, time_step)
    d_waveform = derivative(waveform, time_step)
    d2_waveform = derivative2(waveform, time_step)
    # get morphological points
    p1, p2, p3, p4, p5, p6 = morphological_points(t, waveform, d_waveform, d2_waveform, time_step)

    # FEATURE EXTRACTION
    fd = dict() # feature dictionary

    # get peak amplitude for source attribution
    try:
        fd[f"{channel}peak_amplitude"] = p3.v
    except:
        fd[f"{channel}peak_amplitude"] = 0
    # SHAPE FEATURES
    # waveform duration of the first derivative (FD) of the action potential (AP)
    try:
        fd[f"{channel}f1"] = p5.t - p1.t
    except:
        fd[f"{channel}f1"] = 0
    # peak to valley amplitude of the FD of the AP
    try:
        fd[f"{channel}f2"] = p4.dv - p2.dv
    except:
        fd[f"{channel}f2"] = 0
    # valley to valley amplitude of the FD of the AP
    try:
        fd[f"{channel}f3"] = p6.dv - p2.dv
    except:
        fd[f"{channel}f3"] = 0
    # integral of the spike slice in the waveform, normalized for time
    # NOTE: This feature is NOT in the original paper
    # in the original paper, f4 is the correlation between
    # the waveform and a reference waveform (we don't use reference waveforms).
    try:
        fd[f"{channel}f4"] = area_under_curve(waveform[p1.i:p5.i], time_step)/(p5.t - p1.t)
    except:
        fd[f"{channel}f4"] = 0
    # logarithm of the positve deflection of the FD of the AP
    # NOTE: This feature is NOT in the original paper
    # in the original paper, f5 is the logarithm of the term below
    # However, their definition did not generalize to excidatory
    # neurons, where the principal peak comes before the big trough.
    try:
        fd[f"{channel}f5"] = symmetric_logarithm((p4.dv - p2.dv) / (p4.t - p2.t))
    except:
        fd[f"{channel}f5"] = 0
    # negative deflection of the FD of the AP
    try:
        fd[f"{channel}f6"] = (p6.dv - p4.dv) / (p6.t - p4.t)
    except:
        fd[f"{channel}f6"] = 0
    # logarithm of the slope among valleys of the FD of the AP
    try:
        fd["f7"] = symmetric_logarithm((p6.dv - p2.dv) / (p6.t - p2.t))
    except:
        fd["f7"] = 0

    # root mean square of the pre-event amplitude of the FD of the AP
    # NOTE: This feature is MODIFIED from the original paper
    # in the original paper, f8 is the root mean square of the pre-event amplitude of the FD of the AP
    # However, their definition did not generalize to excidatory
    # neurons, where the first extremum before the principal peak could
    # be the boundary of the pre-event amplitude.
    # We use the first extremum of the first derivative as the cutoff
    # when the first voltage domain extremum is the boundary.
    try:
        fd[f"{channel}f8"] = np.sqrt(np.mean([x**2 for x in d_waveform[:p1.i+1]]))
    except:
        fd[f"{channel}f8"] = 0

    # # negative slope ratio of the FD of the AP
    # # print((p2.dv - p1.dv), (p2.t - p1.t), (p3.dv - p2.dv), (p3.t - p2.t))
    try:
        fd[f"{channel}f9"] = ((p2.dv - p1.dv)/(p2.t - p1.t))/((p3.dv - p2.dv)/(p3.t - p2.t))
    except:
        fd[f"{channel}f9"] = 0
    # # postive slope ratio of the FD of the AP
    try:
        fd[f"{channel}f10"] = ((p4.dv - p3.dv)/(p4.t - p3.t))/((p5.dv - p4.dv)/(p5.t - p4.t))
    except:
        fd[f"{channel}f10"] = 0
    # peak to valley ratio of the action potential
    try:
        fd[f"{channel}f11"] = p2.dv / p4.dv
    except:
        fd[f"{channel}f11"] = 0
    # PHASE FEATURES
    # amplitude of the FD of the AP relating to p1
    try:
        fd[f"{channel}f12"] = p1.dv
    except:
        fd[f"{channel}f12"] = 0
    # amplitude of the FD of the AP relating to p3
    try:
        fd[f"{channel}f13"] = p3.dv
    except:
        fd[f"{channel}f13"] = 0
    # amplitude of the FD of the AP relating to p4
    try:
        fd[f"{channel}f14"] = p4.dv
    except:
        fd[f"{channel}f14"] = 0
    # amplitude of the FD of the AP relating to p6
    try:
        fd[f"{channel}f15"] = p5.dv
    except:
        fd[f"{channel}f15"] = 0
    # amplitude of the FD of the AP relating to p6
    try:
        fd[f"{channel}f16"] = p6.dv
    except:
        fd[f"{channel}f16"] = 0
    # amplitude of the second derivative (SD) of the AP relating to p1
    try:
        fd[f"{channel}f17"] = p1.d2v
    except:
        fd[f"{channel}f17"] = 0
    # amplitude of the SD of the AP relating to p3
    try:
        fd[f"{channel}f18"] = p3.d2v
    except:
        fd[f"{channel}f18"] = 0
    # amplitude of the SD of the AP relating to p5
    try:
        fd[f"{channel}f19"] = p5.d2v
    except:
        fd[f"{channel}f19"] = 0

    # inter-quartile range of the FD of the AP
    fd[f"{channel}f20"] = inter_quartile_range(d_waveform)
    # inter-quartile range of the SD of the AP
    fd[f"{channel}f21"] = inter_quartile_range(d2_waveform)
    # kurtosis coefficient of the FD of the AP
    fd[f"{channel}f22"] = kurtosis(d_waveform)
    # skew / Fisher assymetry of the FD of the AP
    fd[f"{channel}f23"] = skew(d_waveform)
    # skew / Fisher assymetry of the SD of the AP
    fd[f"{channel}f24"] = skew(d2_waveform)

    for key, value in fd.items():
        if value == np.inf or value == -np.inf or np.isnan(value):
            fd[key] = 0

    return fd

def morphological_points(time_index, waveform, d_waveform, d2_waveform, time_step):
    """
    Find the key morphological points in a waveform.

    Parameters
    ----------
    waveform : array_like
        The waveform to be analyzed.
    d_waveform : array_like
        The first derivative of the waveform to be analyzed.
    d2_waveform : array_like
        The second derivative of the waveform to be analyzed.

    Returns
    -------
    p1, p2, p3, p4, p5, p6 : tuple
        The key morphological points in the waveform.

    References:
    ----------
    Caro-Martín, Carmen Rocío, José M. Delgado-García, Agnès Gruart, and R. Sánchez-Campusano. “Spike Sorting Based on Shape, Phase, and Distribution Features, and K-TOPS Clustering with Validity and Error Indices.” Scientific Reports 8, no. 1 (December 12, 2018): 17796. https://doi.org/10.1038/s41598-018-35491-4.
    """

    waveform_point = lambda i: Point(i, time_index, waveform, d_waveform, d2_waveform)

    # get morphological points in the voltage domain
    voltage_peaks = peaks(waveform)
    voltage_troughs = [0] + troughs(waveform) + [len(waveform) - 1]
    voltage_peak_values = [waveform[i] for i in voltage_peaks]
    voltage_trough_values = [waveform[i] for i in voltage_troughs]
    try:
        x = np.argmax(voltage_peak_values)
        # find principal voltage peak
        p3 = waveform_point(voltage_peaks[x])
    except:
        p3 = None
        # get pre-spike trough
    try:
        p1 = waveform_point(max(filter(lambda i: i < p3.i, voltage_troughs)))
    except:
        p1 = None
        # get refractory trough
    try:
        p5 = waveform_point(min(filter(lambda i: i > p3.i, voltage_troughs)))
    except:
        p5 = None
        # get refractory peak index (discard after use)
    try:
        rp = waveform_point(min(filter(lambda i: i > p5.i, [0] + voltage_peaks + [len(waveform) - 1])))
    except:
        rp = None
    # get morphological points in the rate domain
    def steepest_point_in_region(start, end):
        rate_extrema_indexes = local_extrema(d_waveform[start.i:end.i])
        r = lambda start, end: list(filter(lambda i: i > start.i and i < end.i, rate_extrema_indexes))
        v = lambda indexes: [abs(d_waveform[i]) for i in indexes]
        indexes = r(start, end)
        if len(indexes) > 0:
            x = np.argmax(v(indexes))
        else:
            return waveform_point(int(np.median([start.i, end.i])))
        return waveform_point(indexes[x])
    # get steepest point between pre-spike trough and principal peak
    try:
        p2 = steepest_point_in_region(p1, p3)
    except:
        p2 = None
    # get steepest point between principal peak and refractory trough
    try:
        p4 = steepest_point_in_region(p3, p5)
    except:
        p4 = None
    # get steepest point between refractory trough and refractory peak
    try:
        p6 = steepest_point_in_region(p5, rp)
    except:
        p6 = None

    return p1, p2, p3, p4, p5, p6

class Point(object):
    def __init__(self, index, time_index, values, d_values, d2_values):
        assert index >= 0 and index < len(values)
        assert int(index) == index, "index must be an integer"
        self.i = int(index)
        self.t = float(time_index[index])
        self.v = float(values[index])
        self.dv = float(d_values[index])
        self.d2v = float(d2_values[index])

# ============================================================================ #

# utility functions for extracting features from a waveform signal
def symmetric_logarithm(x):
    return sign(x) * log(1 + abs(x))

def inter_quartile_range(data):
    """
    Calculate the interquartile range of data.

    Parameters
    ----------
    data : array_like
        The data to be analyzed.

    Returns
    -------
    float
        The interquartile range of data.
    """
    return np.percentile(data, 75) - np.percentile(data, 25)

def skew(data):
    """
    Calculate the skewness of data.

    Parameters
    ----------
    data : array_like
        Data to be analyzed.

    Returns
    -------
    float
        The skewness (Fisher Asymmetry) of data.
    """
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)
    return np.sum((data - mean)**3) / (n * std**3)

def kurtosis(data):
    """
    Calculate the kurtosis of data.

    Parameters
    ----------
    data : array_like
        The data to be analyzed.

    Returns
    -------
    float
        The kurtosis of data.
    """
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)
    return np.sum((data - mean)**4) / (n * std**4)

def area_under_curve(waveform, time_step):
    return np.trapz(waveform, dx=time_step)

def filter_indexes(extrema_indexes, start, end):
    # get the indexes of the extrema in the region
    return list(filter(lambda i: start <= i <= end, extrema_indexes))

def local_extrema(t):
    """
    Find the local extrema in a time series.
    """
    # get the indexes of the extrema
    _is_peak = lambda i: t[i] > t[i - 1] and t[i] >= t[i + 1] and t[i] > np.quantile(t, 0.6)
    _is_trough = lambda i: t[i] < t[i - 1] and t[i] <= t[i + 1] and t[i] < np.quantile(t, 0.4)
    _is_extrema = lambda i: _is_peak(i) or _is_trough(i)
    return list(filter(_is_extrema, range(1, len(t) - 1)))

def peaks(t):
    _is_peak = lambda i: t[i] > t[i - 1] and t[i] >= t[i + 1] and t[i] > 0
    return list(filter(_is_peak, range(1, len(t) - 1)))

def troughs(t):
    _is_trough = lambda i: t[i] < t[i - 1] and t[i] <= t[i + 1] and t[i] < 0
    return list(filter(_is_trough, range(1, len(t) - 1)))

def zero_crossings(timeseries, time_step):
    """
    Find the indexes of the points at or near zero crossings in a waveform.

    This function is not 100% precise. It finds the indexes of the points
    at or near zero crossings in a waveform. It does not find the exact
    zero crossing points.

    We generally only care about the max and min values at zero crossings,
    so this function serves its purpose in this context.

    Parameters
    ----------
    waveform : array_like
        The waveform to be analyzed.
    time_step : float
        The time between samples in the waveform.

    Returns
    -------
    array_like
        The indexes of the zeroes in the waveform.
    """
    return list(np.where(derivative(np.sign(timeseries), 1))[0])

# functions for getting waveform domains.

def derivative2(waveform, time_step):
    """
    Calculate the second derivative of a waveform.
    """
    return derivative(derivative(waveform, time_step), time_step)

def derivative(waveform, time_step):
    """
    Calculate the derivative of a waveform.
    """
    differential = lambda i: _differential(i, waveform, time_step)
    return list(map(differential, range(len(waveform))))

def _differential(i, waveform, time_step):
    if i == 0:
        return (waveform[i+1] - waveform[i]) / time_step
    elif i == len(waveform)-1:
        return (waveform[i] - waveform[i-1]) / time_step
    else:
        try:
            return float((waveform[i+1] - waveform[i-1]) / (2 * time_step))
        except IndexError:
            raise IndexError(f"Index {i} out of range (0,{len(waveform)})")

def time_index(waveform, time_step):
    """
    Return the time index for a waveform.
    """
    return list(map(lambda i: float(i * time_step), range(len(waveform))))
    return duration / len(waveform)