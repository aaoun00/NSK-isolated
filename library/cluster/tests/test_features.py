import os
import sys
import numpy as np
from datetime import datetime

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.cluster.features import _wave_PCA, feature_energy, feature_wave_PCX, create_features
from core.core_utils import make_1D_timestamps, make_waveforms, make_clusters
from core.spikes import SpikeCluster, Spike
from library.batch_space import SpikeClusterBatch
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

def make_spike():
    T = 2
    dt = .02
    event_times = make_1D_timestamps()
    event_labels = make_clusters(event_times, 5)
    ch_count = 4
    samples_per_wave = 50
    waveforms = make_waveforms(ch_count, len(event_times), samples_per_wave)
    ch1, ch2 = waveforms[0], waveforms[1]
    cell_waves = [ch1[0], ch2[0]]
    idx = np.random.choice(len(event_times), size=1)[0]
    ses = Session()
    cluster = SpikeCluster({'duration': int(T), 'sample_rate': float(1/dt), 'event_times': event_times, 'cluster_label': event_labels[0], 'channel_1': waveforms[0], 'channel_2': waveforms[1]}, session_metadata=ses.session_metadata)
    spike = Spike(event_times[0], int(idx+1), cell_waves, cluster)

    return spike


def test__wave_PCA():
    cv = np.eye(5)

    pc, rpc, ev, rev = _wave_PCA(cv)

    assert type(pc) == np.ndarray
    assert type(rpc) == np.ndarray
    assert type(ev) == np.ndarray
    assert type(rev) == np.ndarray

def test_feature_wave_PCX():
    spike_cluster = make_spike_cluster()
    spike_cluster_batch = make_spike_cluster_batch()
    spike = make_spike()
    
    wavePCData = feature_wave_PCX(spike_cluster)
    wavePCData_batch = feature_wave_PCX(spike_cluster_batch)
    wavePCData_spike = feature_wave_PCX(spike)
    assert type(wavePCData) == np.ndarray
    assert type(wavePCData_batch) == np.ndarray
    assert type(wavePCData_spike) == np.ndarray

def test_create_features():
    spike_cluster = make_spike_cluster()
    FD = create_features(spike_cluster)
    assert type(FD) == np.ndarray

    spike_cluster_batch = make_spike_cluster_batch()
    FD = create_features(spike_cluster_batch)
    assert type(FD) == np.ndarray

    spike = make_spike()
    FD = create_features(spike)
    assert type(FD) == np.ndarray

def test_feature_energy():
    spike_cluster = make_spike_cluster()
    spike_cluster_batch = make_spike_cluster_batch()
    spike = make_spike()

    E = feature_energy(spike_cluster)
    assert type(E) == np.ndarray

    E = feature_energy(spike_cluster_batch)
    assert type(E) == np.ndarray

    E = feature_energy(spike)
    assert type(E) == np.ndarray

if __name__ == '__main__':
    test_feature_wave_PCX()
    test_feature_energy()
    test__wave_PCA()
    test_create_features()




