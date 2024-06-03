import os
import sys
import numpy as np
from library.batch_space import SpikeClusterBatch
from datetime import datetime

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.study_space import Session

from core.spikes import (
    SpikeTrain,
    Spike,
    SpikeCluster,
)

from core.core_utils import *

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
data_dir = os.path.join(parent_dir, 'neuroscikit_test_data')
test_cut_file_path = os.path.join(data_dir, 'axona/20140815-behavior2-90_1.cut')
test_tetrode_file_path = os.path.join(data_dir, 'axona/20140815-behavior2-90.1')

np.random.seed(0)

############################
# NOT CALLED
# def test_spike_keys():
#     spike_keys = SpikeKeys()

#     spike_train_init_keys = spike_keys.get_spike_train_init_keys()

#     assert type(spike_train_init_keys) == list
#     for i in range(len(spike_train_init_keys)):
#         assert type(spike_train_init_keys[i]) == str
# NOT CALLED
# def test_spike_types():
#     spike_types = SpikeTypes()
#     spike_keys = SpikeKeys()

#     spike_train_init_keys = spike_keys.get_spike_train_init_keys()

#     input_dict = spike_types.format_keys(spike_train_init_keys)
#     keys = list(input_dict.keys())
#     type_counter = [0,0,0]
#     for i in range(len(keys)):
#         if type(input_dict[keys[i]]) == int:
#             type_counter[0] += 1
#         if type(input_dict[keys[i]]) == float:
#             type_counter[1] += 1
#         if type(input_dict[keys[i]]) == list:
#             type_counter[2] += 1

#     assert type(input_dict) == dict
#     assert sum(type_counter) == 4
#     assert type_counter[-1] == 2
# NOT CALLED
############################

def test_spike_train_class():
    event_times = make_1D_timestamps()

    T = 2
    dt = .02

    input_dict1 = {}
    input_dict1['duration'] = int(T)
    input_dict1['sample_rate'] = float(1 / dt)
    input_dict1['events_binary'] = []
    input_dict1['event_times'] = event_times
    input_dict1['datetime'] = datetime(1,1,1)

    ses = Session()
    spike_train1 = ses.make_class(SpikeTrain, input_dict1)

    rate1 = spike_train1.get_event_rate()
    events_binary1 = spike_train1.get_binary()

    assert type(rate1) == float
    assert type(spike_train1.events_binary) == list
    assert type(spike_train1.event_times) == list
    assert type(spike_train1.event_labels) == list

    events_binary2 = make_1D_binary_spikes()

    input_dict2 = {}
    input_dict2['duration'] = int(T)
    input_dict2['sample_rate'] = float(1 / dt)
    input_dict2['events_binary'] = events_binary2
    input_dict2['event_times'] = []
    input_dict2['datetime'] = datetime(1,1,1)

    ses = Session()
    spike_train2 = ses.make_class(SpikeTrain, input_dict2)

    rate2 = spike_train2.get_event_rate()
    event_times2 = spike_train2.get_event_times()

    assert type(rate2) == float
    assert type(spike_train2.events_binary) == list
    assert type(spike_train2.event_times) == list
    assert type(spike_train2.event_labels) == list

def test_spike_object_class():
    event_times = make_1D_timestamps()
    ch_count = 8
    samples_per_wave = 50
    waveforms = make_waveforms(ch_count, len(event_times), samples_per_wave)
    event_labels = make_clusters(event_times, 5)

    T = 2
    dt = .02
    idx = np.random.choice(len(event_times), size=1)[0]

    input_dict1 = {}
    input_dict1['duration'] = int(T)
    input_dict1['sample_rate'] = float(1 / dt)
    input_dict1['spike_time'] = event_times[idx]
    input_dict1['cluster_label'] = int(idx + 1)
    input_dict1['datetime'] = datetime(1,1,1)

    wf = []
    for i in range(ch_count):
        key = 'channel_' + str(i+1)
        input_dict1[key] = waveforms[i][idx]
        wf.append(waveforms[i][idx])

    ses = Session()
    cluster = SpikeCluster({'duration': int(T), 'sample_rate': float(1/dt), 'event_times': event_times, 'cluster_label': int(idx+1), 'channel_1':waveforms[0]}, session_metadata=ses.session_metadata)

    spike_object = Spike(input_dict1['spike_time'], input_dict1['cluster_label'], wf, cluster)

    # label = spike_object.get_cluster_label()
    chan, _ = spike_object.get_peak_signal()
    waveform = spike_object.get_signal(chan)

    # assert type(label) == int
    assert type(spike_object.spike_time) == float
    assert type(chan) == int
    assert chan <= ch_count
    assert chan > 0
    assert type(waveform) == list
    assert type(waveform[0]) == float
    assert len(waveform) == samples_per_wave
    assert len(waveforms) == ch_count

def test_spike_cluster_class():
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

    all_channel_waveforms = spike_cluster.get_all_channel_waveforms()
    rate = spike_cluster.get_cluster_firing_rate()
    label = spike_cluster.get_cluster_label()
    spk_count = spike_cluster.get_cluster_spike_count()
    single_channel_waveform = spike_cluster.get_single_channel_waveforms(4)
    spike_objects = spike_cluster.get_spike_object_instances()

    assert type(spike_objects) == list
    assert isinstance(spike_objects[0], Spike)
    assert type(single_channel_waveform) == list
    assert type(single_channel_waveform[0]) == list
    assert type(single_channel_waveform[0][0]) == float
    assert len(all_channel_waveforms) == ch_count
    assert len(single_channel_waveform) == len(event_times)

    assert type(label) == int
    assert type(spk_count) == int
    assert type(rate) == float
    assert spk_count == len(event_times)




# def test_spike_cluster_batch_class():
#     event_times = make_1D_timestamps()
#     ch_count = 8
#     samples_per_wave = 50
#     waveforms = make_waveforms(ch_count, len(event_times), samples_per_wave)
#     cluster_count = 10

#     event_labels = make_clusters(event_times, cluster_count)

#     T = 2
#     dt = .02

#     input_dict1 = {}
#     input_dict1['duration'] = int(T)
#     input_dict1['sample_rate'] = float(1 / dt)
#     input_dict1['event_times'] = event_times
#     input_dict1['event_labels'] = event_labels


#     for i in range(ch_count):
#         key = 'channel_' + str(i+1)
#         input_dict1[key] = waveforms[i]

#     spike_cluster_batch = SpikeClusterBatch(input_dict1)

#     all_channel_waveforms = spike_cluster_batch.get_all_channel_waveforms()
#     rate = spike_cluster_batch.get_single_cluster_firing_rate(event_labels[0])
#     labels = spike_cluster_batch.get_cluster_labels()
#     unique_labels = spike_cluster_batch.get_unique_cluster_labels()
#     # spk_count = spike_cluster_batch.get_cluster_spike_count()
#     single_channel_waveform = spike_cluster_batch.get_single_channel_waveforms(event_labels[0])
#     # spike_objects = spike_cluster_batch.get_spike_object_instances()
#     rates = spike_cluster_batch.get_all_cluster_firing_rates()
#     spike_clusters = spike_cluster_batch.get_spike_cluster_instances()
#     count, cluster_event_times, cluster_waveforms = spike_cluster_batch.get_single_spike_cluster_instance(event_labels[0])
#     single_cluster_spike_objects = spike_cluster_batch.get_single_spike_cluster_objects(event_labels[0])
#     cluster_spike_objects = spike_cluster_batch.get_spike_cluster_objects()

#     ids = np.where(np.array(event_labels) == event_labels[0])[0]
#     assert np.array(cluster_event_times).all() == np.array(event_times)[ids].all()
#     assert np.array(cluster_waveforms).all() == np.array(waveforms[2]).all()
#     assert len(unique_labels) <= cluster_count
#     assert type(rates) == list
#     assert rate in rates
#     assert type(single_cluster_spike_objects) == list
#     assert isinstance(single_cluster_spike_objects[0], Spike)
#     assert type(cluster_spike_objects) == list
#     assert isinstance(cluster_spike_objects[0], list)
#     assert isinstance(cluster_spike_objects[0][0], Spike)
#     assert type(spike_clusters) == list
#     assert isinstance(spike_clusters[0], SpikeCluster)
#     assert type(single_channel_waveform) == list
#     assert type(single_channel_waveform[0]) == list
#     assert type(single_channel_waveform[0][0]) == float
#     assert len(all_channel_waveforms) == ch_count
#     assert len(single_channel_waveform) == len(event_times)
#     assert type(labels) == list
#     assert type(rate) == float

# def test_spike_train_batch_class():
#     event_times = make_2D_timestamps()

#     T = 2
#     dt = .02

#     input_dict1 = {}
#     input_dict1['duration'] = int(T)
#     input_dict1['sample_rate'] = float(1 / dt)
#     input_dict1['events_binary'] = []
#     input_dict1['event_times'] = event_times

#     spike_train1 = SpikeTrainBatch(input_dict1)
#     rate1 = spike_train1.get_average_event_rate()
#     rate_list1 = spike_train1.get_indiv_event_rate()
#     spike_train1.get_binary()
#     instances1 = spike_train1.get_spike_train_instances()

#     assert type(rate1) == float
#     assert type(rate_list1) == list
#     assert type(spike_train1.events_binary) == list
#     assert type(spike_train1.event_times) == list
#     assert type(spike_train1.event_labels) == list
#     assert type(spike_train1.events_binary[0]) == list
#     assert type(spike_train1.event_times[0]) == list
#     assert isinstance(instances1[0], SpikeTrain) == True

#     events_binary2 = make_2D_binary_spikes()

#     input_dict2 = {}
#     input_dict2['duration'] = int(T)
#     input_dict2['sample_rate'] = float(1 / dt)
#     input_dict2['events_binary'] = events_binary2
#     input_dict2['event_times'] = []

#     spike_train2 = SpikeTrainBatch(input_dict2)

#     spike_train2.get_event_times()
#     rate2 = spike_train2.get_average_event_rate()
#     rate_list2 = spike_train2.get_indiv_event_rate()
#     instances2 = spike_train2.get_spike_train_instances()

#     assert type(rate2) == float
#     assert type(rate_list2) == list
#     assert type(spike_train2.events_binary) == list
#     assert type(spike_train2.event_times) == list
#     assert type(spike_train2.event_labels) == list
#     assert type(spike_train2.events_binary[0]) == list
#     assert type(spike_train2.event_times[0]) == list
#     assert isinstance(instances2[0], SpikeTrain) == True



if __name__ == '__main__':
    # test_spike_keys()
    # test_spike_types()
    test_spike_train_class()
    # test_spike_train_batch_class()
    test_spike_object_class()
    test_spike_cluster_class()
    # test_spike_cluster_batch_class()
    print('we good')
