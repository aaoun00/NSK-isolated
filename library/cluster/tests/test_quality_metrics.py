
import os
import sys
import numpy as np
from datetime import datetime

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.cluster.quality_metrics import _mahal, L_ratio, isolation_distance
from core.core_utils import make_1D_timestamps, make_waveforms, make_clusters
from core.spikes import SpikeCluster
from library.batch_space import SpikeClusterBatch
from library.cluster.features import create_features
from library.study_space import Session

def make_spike_cluster_batch():
    event_times = make_1D_timestamps()
    ch_count = 4
    samples_per_wave = 50
    waveforms = make_waveforms(ch_count, len(event_times), samples_per_wave)
    cluster_count = 10

    event_labels = make_clusters(event_times, cluster_count)

    T = 2
    dt = .02

    input_dict1 = {}
    input_dict1['duration'] = int(T)
    input_dict1['sample_rate'] = float(1 / dt)
    input_dict1['event_times'] = event_times
    input_dict1['event_labels'] = event_labels
    input_dict1['datetime'] = datetime(1,1,1)


    for i in range(ch_count):
        key = 'channel_' + str(i+1)
        input_dict1[key] = waveforms[i]

    ses = Session()
    spike_cluster_batch = ses.make_class(SpikeClusterBatch, input_dict1)

    return spike_cluster_batch

def make_spike_cluster():
    event_times = make_1D_timestamps()
    ch_count = 8
    samples_per_wave = 50
    waveforms = make_waveforms(ch_count, len(event_times), samples_per_wave)

    T = 2
    dt = .02
    idx = np.random.choice(len(event_times), size=1)[0]

    input_dict1 = {}
    input_dict1['duration'] = int(T)
    input_dict1['sample_rate'] = float(1 / dt)
    input_dict1['event_times'] = event_times
    input_dict1['cluster_label'] = int(idx + 1)
    input_dict1['datetime'] = datetime(1,1,1)


    for i in range(ch_count):
        key = 'channel_' + str(i+1)
        input_dict1[key] = waveforms[i]

    ses = Session()
    spike_cluster = ses.make_class(SpikeCluster, input_dict1)

    return spike_cluster



def test_isolation_distance():
    spike_cluster = make_spike_cluster()
    spike_cluster_batch = make_spike_cluster_batch()

    iso_dist = isolation_distance(spike_cluster)
    assert type(iso_dist) == float

    iso_dist = isolation_distance(spike_cluster_batch)
    assert type(iso_dist) == float

def test_L_ratio():
    spike_cluster = make_spike_cluster()
    spike_cluster_batch = make_spike_cluster_batch()

    L, ratio, df = L_ratio(spike_cluster)

    assert type(L) == np.float64
    assert type(ratio) == np.float64 
    assert type(df) == int 

    L, ratio, df = L_ratio(spike_cluster_batch)

    assert type(L) == np.float64
    assert type(ratio) == np.float64 
    assert type(df) == int 

def test__mahal():
    spike_cluster = make_spike_cluster()
    spike_cluster_batch = make_spike_cluster_batch()

    FD = create_features(spike_cluster)
    FD_batch = create_features(spike_cluster_batch)

    d = _mahal(FD, FD[spike_cluster.cluster_labels, :])
    assert type(d) == np.ndarray

    d = _mahal(FD_batch, FD_batch[spike_cluster_batch.cluster_labels, :])
    assert type(d) == np.ndarray

if __name__ == '__main__':
    test__mahal()
    test_isolation_distance()
    test_L_ratio()