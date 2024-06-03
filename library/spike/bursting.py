import os
import sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.spikes import SpikeCluster, SpikeTrain
from library.ensemble_space import Cell

def find_burst(spike_class: Cell | SpikeTrain | SpikeCluster, maxisi=0.01, req_burst_spikes=2):
    """This function is used to calculate the percentage of bursting

    inputs:
    ts- the spike times
    maxisi- the maximum inter spike interval"""

    ts = spike_class.event_times

    if type(ts) == list:
        ts = np.asarray(ts)

    if len(ts) < 2:
        # cannot perform this calculation with less than 2 spike times
        return [], []

    ts = ts.flatten()

    bursts = []
    singlespikes = []

    isi = np.diff(ts)  # calculating the inter spike intervals
    n = len(ts)

    if isi[0] <= maxisi:
        bursts = [0]
    else:
        singlespikes = [0]

    indices = np.where(isi > maxisi)[0]

    max_index = len(isi) - 1

    for i in indices:
        if (i + 1) > max_index:
            break

        if isi[i + 1] <= maxisi:
            bursts.append(i + 1)
        elif isi[i + 1] > maxisi:
            singlespikes.append(i + 1)

    if isi[n - 2] > maxisi:
        singlespikes.append(n - 1)

    # get spikes that belong to the potential bursts

    total_n = np.arange(len(ts.flatten()))

    burst_spikes = np.setdiff1d(total_n, singlespikes)  # these are all indices belonging to the bursts

    if len(burst_spikes) != 0:
        burst_spikes = _find_consec(burst_spikes)

        # check if the bursts have the required amount of spikes

        burst_spikes = [burst for burst in burst_spikes if len(burst) >= req_burst_spikes]

        bursts = [burst[0] for burst in burst_spikes]

        # flatten the burst spikes
        burst_spikes = [spike for burst in burst_spikes for spike in burst]

        singlespikes = np.setdiff1d(total_n, np.asarray(burst_spikes)).tolist()
    else:

        bursts = np.array([])
        singlespikes = total_n

    # use the new burst spikes to calculate everything else as single spikes

    # return np.asarray(bursts), np.asarray(singlespikes)
    bursting = 100 * len(bursts) / (len(bursts) + len(singlespikes))

    bursts_n_spikes_avg = _avg_spike_burst(ts, bursts, singlespikes)  # num of spikes on avg per burst

    spike_class.stats_dict['spike']['bursting'] = bursting
    spike_class.stats_dict['spike']['bursts_n_spikes_avg'] = bursts_n_spikes_avg

    return bursting, bursts_n_spikes_avg



def _find_consec(data):
    '''finds the consecutive numbers and outputs as a list'''
    consecutive_values = []  # a list for the output
    current_consecutive = [data[0]]

    if len(data) == 1:
        return [[data[0]]]

    for index in range(1, len(data)):

        if data[index] == data[index - 1] + 1:
            current_consecutive.append(data[index])

            if index == len(data) - 1:
                consecutive_values.append(current_consecutive)

        else:
            consecutive_values.append(current_consecutive)
            current_consecutive = [data[index]]

            if index == len(data) - 1:
                consecutive_values.append(current_consecutive)
    return consecutive_values

import numpy as np

def _avg_spike_burst(ts, bursts, singleSpikes):
    """calculates the average spikes per burst"""
    total_n = np.arange(len(ts.flatten()))
    burst_spikes = np.setdiff1d(total_n, singleSpikes)  # these are all indices belonging to the bursts

    # divide by total number of burst events to return the output

    if len(bursts) != 0:
        return len(burst_spikes) / len(bursts)
    else:
        return np.NaN
