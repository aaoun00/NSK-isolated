import os, sys

# from prototypes.wave_form_sorter.sort_cell_spike_times import sort_cell_spike_times

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.core_utils import make_seconds_index_from_rate
from core.spikes import *
import numpy as np
from library.workspace import Workspace

class SpikeTrainBatch(Workspace):
    """
    Class to hold collection of 1D spike trains
    """
    def __init__(self, input_dict, **kwargs):
        self._input_dict = input_dict

        self.duration, self.sample_rate, self.events_binary, self.event_times, self.session_metadata = self._read_input_dict()

        if 'session_metadata' in kwargs:
            if self.session_metadata != None:
                print('Ses metadata is in the input dict and init fxn, init fnx will override')
            self.session_metadata = kwargs['session_metadata']

        # self.time_index = make_seconds_index_from_rate(self.duration, self.sample_rate)
        self.time_index = self.session_metadata.session_object.time_index

        # assert ((len(events_binary) == 0) and (len(event_times) == 0)) != True, "No spike data provided"

        self.units = max(len(self.events_binary), len(self.event_times))

        self.event_labels = []
        self._event_rate = None
        self._spike_train_instances = []
        self.dir_names = self.session_metadata.dir_names
        self.stats_dict = self._init_stats_dict()
        # self.good_label_ids = None

    def set_sorted_label_ids(self, sorted_ids):
        self.good_label_ids = sorted_ids

    def _read_input_dict(self):
        duration = None
        sample_rate = None
        if 'duration' in  self._input_dict:
            duration = self._input_dict['duration']
        if 'sample_rate' in  self._input_dict:
            sample_rate = self._input_dict['sample_rate']

        events_binary = self._input_dict['events_binary']
        assert type(events_binary) == list, 'Binary spikes are not a list, check inputs'
        if len(events_binary) > 0:
            assert type(events_binary[0]) == list, 'Binary spikes are not nested lists (2D), check inputs are not 1D'
            spike_data_present = True

        event_times = self._input_dict['event_times']
        assert type(event_times) == list, 'Spike times are not a list, check inputs'
        if len(event_times) > 0:
            assert type(event_times[0]) == list, 'Spike times are not nested lists (2D), check inputs are not 1D'
            spike_data_present = True
        assert spike_data_present == True, 'No spike times or binary spikes provided'

        session_metadata = None
        if 'session_metadata' in self._input_dict:
            session_metadata = self._input_dict['session_metadata']

        return duration, sample_rate, events_binary, event_times, session_metadata

    def _make_spike_train_instance(self):
        # arr to collect SpikeTrain() instances
        instances = []

        # Both are 2d arrays so can do len() to iterate thru number of cells
        for i in range(self.units):
            input_dict = {}
            input_dict['duration'] = self.duration
            input_dict['sample_rate'] = self.sample_rate
            if len(self.events_binary) > 0:
                input_dict['events_binary'] = self.events_binary[i]
            else:
                input_dict['events_binary'] = []
            if len(self.event_times) > 0:
                input_dict['event_times'] = self.event_times[i]
            else:
                input_dict['event_times'] = []
            input_dict['session_metadata'] = self.session_metadata
            instances.append(SpikeTrain(input_dict))

        self._spike_train_instances = instances

    def get_spike_train_instances(self):
        if len(self._spike_train_instances) == 0:
            self._make_spike_train_instance()
        return self._spike_train_instances

    def get_indiv_event_rate(self):
        self.get_spike_train_instances()
        # event_rates = []
        # for i in range(len(self._spike_train_instances)):
        #     event_rates.append(self._spike_train_instances[i].get_event_rate())
        event_rates = list(map(lambda x: x.get_event_rate(), self._spike_train_instances))
        return event_rates

    def get_average_event_rate(self):
        self.get_spike_train_instances()
        event_rates = self.get_indiv_event_rate()
        event_rates = np.array(event_rates)
        ids = np.where(event_rates != None)[0]
        return sum(event_rates[ids]) / len(event_rates[ids])

    def _set_binary(self):
        for i in range(len(self._spike_train_instances)):
            self.events_binary.append(self._spike_train_instances[i].get_binary())
        return self.events_binary

    def _set_event_times(self):
        for spike_train in self._spike_train_instances:
            self.event_times.append(spike_train.get_event_times())
        return self.event_times

    # get 2d spike inputs as binary
    def get_binary(self):
        if len(self.events_binary) == 0:
            self.get_spike_train_instances()
            self._set_binary()
        else:
            return self.events_binary

    # get 2d spike inputs as time_index
    def get_event_times(self):
        if len(self.event_times) == 0:
            self.get_spike_train_instances()
            self._set_event_times()
        else:
            return self.event_times

    def _init_stats_dict(self):
        stats_dict = {}

        for dir in self.dir_names:
            if dir != 'tests' and 'cache' not in dir:
                stats_dict[dir] = {}

        return stats_dict

class SpikeClusterBatch(Workspace):
    """
    Class to batch process SpikeClusters. Can pass in unorganized set of 1D spike times + cluster labels
    will create a collection of spike clusters with a collection of spikie objects in each cluster
    """
    def __init__(self, input_dict, **kwargs):
        self._input_dict = input_dict

        if 'session_metadata' in kwargs:
            self.session_metadata = kwargs['session_metadata']
        else:
            self.session_metadata = None

        duration, sample_rate, cluster_labels, event_times, waveforms, waveform_sample_rate, spikeparam = self._read_input_dict()

        assert type(cluster_labels) == list, 'Cluster labels missing'
        assert type(cluster_labels[0]) == int, 'Cluster labels missing'
        assert len(event_times) > 0, 'Spike times missing'
        assert len(waveforms) <= 8 and len(waveforms) > 0, 'Cannot have fewer than 0 or more than 8 channels'

        # self.time_index = make_seconds_index_from_rate(duration, sample_rate)
        self.time_index = self.session_metadata.session_object.time_index

        self.event_times = event_times
        self.spikeparam = spikeparam
        self.waveform_sample_rate = waveform_sample_rate
        self.cluster_labels = cluster_labels
        self.duration = duration
        self.sample_rate = sample_rate
        self.spike_clusters = []
        self.spike_objects = []
        self.waveforms = waveforms
        self.good_label_ids = None
        self.dir_names = self.session_metadata.dir_names
        self.stats_dict = self._init_stats_dict()
        # self._adjusted_labels = []

        # unique = self.get_unique_cluster_labels()
        # if len(unique) < int(max(unique) + 1):
        #     self._format_cluster_count()


    # # e.g. if cluster labels is [1,5,2,4] ==> [0,3,1,2]
    # def _format_cluster_count(self):
    #     unique = self.get_unique_cluster_labels()
    #     count = len(unique)
    #     adjusted_labels = [i for i in range(count)]
    #     for i in range(len(self.cluster_labels)):
    #         for j in range(len(unique)):
    #             if self.cluster_labels[i] == unique[j]:
    #                 self.cluster_labels[i] = adjusted_labels[j]
    #     self._adjusted_labels = adjusted_labels

    def set_sorted_label_ids(self, sorted_ids):
        self.good_label_ids = sorted_ids

    def _read_input_dict(self):
        duration = None
        sample_rate = None
        spikeparam = None
        if 'duration' in  self._input_dict:
            duration = self._input_dict['duration']
        if 'sample_rate' in  self._input_dict:
            sample_rate = self._input_dict['sample_rate']
        if 'waveform_sample_rate' in self._input_dict:
            waveform_sample_rate = self._input_dict['waveform_sample_rate']
        if 'spikeparam' in self._input_dict:
            spikeparam = self._input_dict['spikeparam']

        # events_binary = self._input_dict['events_binary']
        cluster_labels = self._input_dict['event_labels']
        # assert type(events_binary) == list, 'Binary spikes are not a list, check inputs'
        # if len(events_binary) > 0:
            # spike_data_present = True
        event_times = self._input_dict['event_times']
        assert type(event_times) == list, 'Spike times are not a list, check inputs'
        # if len(event_times) > 0:
        #     spike_data_present = True
        # assert spike_data_present == True, 'No spike times or binary spikes provided'
        waveforms = self._extract_waveforms()
        return duration, sample_rate, cluster_labels, event_times, waveforms, waveform_sample_rate, spikeparam

    # uses InputKeys() to get channel bychannel waveforms
    def _extract_waveforms(self):
        input_keys = InputKeys()
        channel_keys = input_keys.get_channel_keys()
        waveforms = {}
        for i in range(len(channel_keys)):
            if channel_keys[i] in self._input_dict.keys():
                # waveforms.append(self._input_dict[channel_keys[i]])
                waveforms[channel_keys[i]] = self._input_dict[channel_keys[i]]
        return waveforms

    # returns all channel waveforms across all spike times
    def get_all_channel_waveforms(self):
        all_channels = []
        for i in range(len(self.waveforms)):
            all_channels.append(self.waveforms['channel_'+str(i+1)])
        return all_channels

    # returns specific channel waveforms across all spike times
    def get_single_channel_waveforms(self, id):
        assert id in [1,2,3,4,5,6,7,8], 'Channel number must be from 1 to 8'
        return self.waveforms['channel_'+str(id)]

    # returns uniqe cluster labels (id of clusters)
    def get_unique_cluster_labels(self):
        return np.unique(self.cluster_labels)

    def get_cluster_labels(self):
        return self.cluster_labels

    def get_single_cluster_firing_rate(self, cluster_id):
        assert cluster_id in self.cluster_labels, 'Invalid cluster ID'
        T = self.time_index[1] - self.time_index[0]
        count, _, _ = self.get_single_spike_cluster_instance(cluster_id)
        rate = float(count / T)
        return rate

    # firing rate list, 1 per cluster
    def get_all_cluster_firing_rates(self):
        rates = []
        unique = self.get_unique_cluster_labels()
        for i in unique:
            rates.append(self.get_single_cluster_firing_rate(i))
        return rates

    # # Get specific SpikeCluster() instance
    # def get_single_spike_cluster_instance(self, cluster_id):
    #     assert cluster_id in self.cluster_labels, 'Invalid cluster ID'
    #     count = 0
    #     clusterevent_times = []
    #     cluster_waveforms = []
    #     for i in range(len(self.event_times)):
    #         if self.cluster_labels[i] == cluster_id:
    #             count += 1
    #             clusterevent_times.append(self.event_times[i])
    #             waveforms_by_channel = []
    #             for j in range(len(self.waveforms)):
    #                 waveforms_by_channel.append(self.waveforms['channel_'+str(j+1)][i])
    #             cluster_waveforms.append(waveforms_by_channel)

    #     assert len(cluster_waveforms[0]) > 0 and len(cluster_waveforms[0]) <= 8
    #     assert len(cluster_waveforms) == count

    #     return count, clusterevent_times, cluster_waveforms

    def get_single_spike_cluster_instance(self, cluster_id):
        assert cluster_id in self.cluster_labels, 'Invalid cluster ID'

        isInCluster = list(map(lambda x: x == cluster_id, self.cluster_labels))

        if type(self.event_times) == list:
            if type(self.event_times[0]) == list:
                events_mask = list(map(lambda y, x: x[0] * int(y) - 1, isInCluster, self.event_times))
            else:
                events_mask = list(map(lambda y, x: x * int(y) - 1, isInCluster, self.event_times))
        else:
            self.event_times = self.event_times.squeeze()
            events_mask = list(map(lambda y, x: x[0] * int(y) - 1, isInCluster, self.event_times))

        clusterevent_times = list(filter(lambda x: x != -1, events_mask))

        cluster_waveforms = []
        for i in range(len(self.waveforms)):
            chan = self.waveforms['channel_'+str(i+1)]
            cluster_mask = list(map(lambda y, x: x if (int(y) == 1) else None, isInCluster, chan))
            cluster_chan = list(filter(lambda x: x != None, cluster_mask))
            cluster_waveforms.append(cluster_chan)

        return len(clusterevent_times), clusterevent_times, cluster_waveforms





    # List of SpikeCluster() instances
    def get_spike_cluster_instances(self):
        assert self.good_label_ids is not None, 'Call study.make_animals() to sort your cells and populate self.good_label_ids, no need to make spike clusters for noise'
        if len(self.spike_clusters) == 0:
            self._make_spike_cluster_instances()
            return self.spike_clusters
        else:
            return self.spike_clusters

    # Spike objects (Spike() + waveforms) for one cluster (list)
    def get_single_spike_cluster_objects(self, cluster_id):
        self.spike_clusters = self.get_spike_cluster_instances()
        unique = self.get_unique_cluster_labels()
        for i in range(len(unique)):
            if unique[i] == cluster_id:
                spike_cluster = self.spike_clusters[i]
        spike_object_instances = spike_cluster.get_spike_object_instances()
        return spike_object_instances

    # Spike objects across all clusters (list of list)
    def get_spike_cluster_objects(self):
        instances = []
        unique = self.get_unique_cluster_labels()
        for i in range(len(unique)):
            if i in self.good_label_ids:
                spike_cluster = self.spike_clusters[i]
                instances.append(spike_cluster.get_spike_object_instances())
        return instances

    # Laze eval, called with get_spike_cluster_instances
    def _make_spike_cluster_instances(self):
        # arr to collect SpikeTrain() instances
        instances = []
        # labelled = []

        session_constant = {'session_metadata': self.session_metadata, 'duration': self.duration, 'sample_rate': self.sample_rate}
        # Both are 2d arrays so can do len() to iterate thru number of cells
        for i in self.get_unique_cluster_labels():
            if i in self.good_label_ids:
                # if self.cluster_labels[i] not in labelled:
                input_dict = {}
                input_dict['cluster_label'] = int(i)
                if len(self.event_times) > 0:
                    _, clusterevent_times, _ = self.get_single_spike_cluster_instance(i)
                    input_dict['event_times'] = clusterevent_times
                else:
                    input_dict['event_times'] = []
                for key in session_constant:
                    input_dict[key] = session_constant[key]
                idx = np.where(self.cluster_labels == i)[0]
                idx = idx[idx <= len(self.event_times)-1]
                for j in range(len(self.waveforms)):
                    key = 'channel_' + str(j+1)
                    input_dict[key] = np.asarray(self.waveforms[key])[idx]
                input_dict['waveform_sample_rate'] = self.waveform_sample_rate
                instances.append(SpikeCluster(input_dict))
                    # labelled.append(self.cluster_labels[i])
        # assert len(labelled) == max(self.cluster_labels)
        self.spike_clusters = instances

    def _init_stats_dict(self):
        stats_dict = {}

        for dir in self.dir_names:
            if dir != 'tests' and 'cache' not in dir:
                stats_dict[dir] = {}

        return stats_dict

