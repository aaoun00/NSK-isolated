from datetime import datetime, timedelta
from random import sample
import numpy as np

def make_hms_index_from_rate(start_time, sample_length, sample_rate):
    """
    Creates a time index for a sample of length sample_length at sample_rate
    starting at start_time.

    Output is in hours, minutes, seconds (HMS)
    """
    if type(start_time) == str:
        start_time = datetime.strptime(start_time, '%H:%M:%S')
    time_index = [start_time]
    for i in range(1,sample_length):
        time_index.append(time_index[-1] + timedelta(seconds=1/sample_rate))
    str_time_index = [time.strftime('%H:%M:%S.%f') for time in time_index]
    return str_time_index

def make_seconds_index_from_rate(sample_length, sample_rate):
    """
    Same as above but output is in seconds, start_time is automatically 0
    Can think of this as doing all times - start_time so we have 0,0.02,0.04... array etc..
    """
    start_time = 0
    dt = 1/sample_rate
    time = np.arange(start_time, int(sample_length), dt)
    return time


def make_1D_binary_spikes(size=100):
    spike_train = np.random.randint(2, size=size)

    return list(spike_train)

def make_2D_binary_spikes(count=20, size=100):
    spike_trains = np.zeros((count, size))

    for i in range(count):
        spike_trains[i] = np.random.randint(2, size=size).tolist()

    return spike_trains.tolist()

def make_1D_timestamps(T=2, dt=0.02):
    time = np.arange(0,T,dt)

    spk_count = np.random.choice(len(time), size=1)
    while spk_count <= 10:
        spk_count = np.random.choice(len(time), size=1)
    spk_time = np.random.choice(time, size=spk_count, replace=False).tolist()

    return spk_time

def make_2D_timestamps(count=20, T=2, dt=0.02):
    time = np.arange(0,T,dt)
    # spk_times = np.zeros((count, len(time)))
    spk_times = []

    for i in range(count):
        spk_count = np.random.choice(len(time), size=1)
        spk_times.append(np.random.choice(time, size=spk_count, replace=False).tolist())

    return spk_times

def make_waveforms(channel_count, spike_count, samples_per_wave):
    waveforms = np.zeros((channel_count, spike_count, samples_per_wave))

    for i in range(channel_count):
        for j in range(samples_per_wave):
            waveforms[i,:,j] = np.random.randint(-20,20,size=spike_count).tolist() + sum(np.random.sample(size=3))

    return waveforms.tolist()

def make_clusters(timestamps, cluster_count):
    event_labels = []
    for i in range(len(timestamps)):
        idx = np.random.choice(cluster_count, size=1)[0]
        event_labels.append(int(idx))
    return event_labels

