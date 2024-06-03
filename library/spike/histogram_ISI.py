import os
import sys
import numpy as np
import matplotlib.pyplot as plt

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.spikes import SpikeCluster, SpikeTrain
from library.ensemble_space import Cell
from library.cluster import create_features

def MatlabNumSeq(start, stop, step, exclude=True):
    """In Matlab you can type:
    start:step:stop and easily create a numerical sequence
    if exclude is true it will exclude any values greater than the stop value
    """

    '''np.arange(start, stop, step) works good most of the time
    However, if the step (stop-start)/step is an integer, then the sequence
    will stop early'''

    seq = np.arange(start, stop + step, step)

    if exclude:
        if seq[-1] > stop:
            seq = seq[:-1]

    return seq


# Currentl;y 'unit' == cell id is passed in, when refactored to take in animal: Animal(), can use agg.sorted_events to get cell data, or agg labels, + add agg FD
def histogram_ISI(spike_class : Cell | SpikeCluster | SpikeTrain, maxisi=0.01, isi_histo=[0, 1000, 10], req_burst_spikes=2):
    """Performs analysis on the data for a tetrode and unit (cell)
    ts- spike times of the unit (cell)
    cluster_mat - cluster_labels
    isi_histo - a 1x3 matrix [minimum histogram value in ms, maximum value in ms, width in ms]
    FD = features used for cluster quality
    """

    ts = spike_class.event_times

    if type(ts) == list:
        ts = np.array(ts)

    ts = ts.flatten()

    n_spikes = len(ts)

    if n_spikes < 2:
        print('There are only %d spikes in this cell, skipping analysis!' % n_spikes)
        
    # --------- Interspike Interval Analysis ---------------------- #

    ISI = np.multiply(1000, np.diff(ts))  # ISI in milliseconds

    ISI_min = np.min(ISI)
    ISI_max = np.amax(ISI)
    ISI_mean = np.mean(ISI)  # average ISI, milliseconds
    ISI_median = np.median(ISI)
    ISI_std = np.std(ISI)  # standard deviation of ISI, milliseconds
    ISI_cv = ISI_mean / ISI_std  # coef. of variation of ISI, unitless

    # we will go from 0 to 1 second by 10 ms

    isi_histo_min, isi_histo_max, isi_histo_width = isi_histo
    binedges = MatlabNumSeq(isi_histo_min, isi_histo_max, isi_histo_width, exclude=False)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    n, bins, patches = ax.hist(ISI, binedges, histtype='bar')
    # ax.set_xlim([0, 1000])  # limiting the x-axis between 0 and 1000 ms
    # ax.set_title('T%sC%s: Interspike Interval Histogram' % (tet_num, unit))
    ax.set_xlabel("ISI(ms)")
    ax.set_ylabel("Counts(#)")

    ISI_dict = {'min': ISI_min, 'max': ISI_max, 'median': ISI_median, 'mean': ISI_mean, 'std': ISI_std, 'cv':
        ISI_cv, 'n': n, 'bins': bins, 'fig': fig}

    spike_class.stats_dict['spike']['ISI_dict'] = ISI_dict

    return ISI_dict