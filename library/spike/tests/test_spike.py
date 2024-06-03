import os
import sys
import wave
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.core_utils import make_seconds_index_from_rate
from core.core_utils import make_1D_timestamps, make_waveforms, make_clusters
from library.spike import sort_spikes_by_cell, find_burst, histogram_ISI
from library.cluster import create_features
from library.batch_space import SpikeClusterBatch
from core.spikes import SpikeCluster
from library.ensemble_space import Cell
from library.lib_test_utils import make_spike_cluster, make_spike_cluster_batch, make_cell

def test_sort_cell_spike_times():
  
    cluster_batch = make_spike_cluster_batch()
    good_sorted_cells, good_sorted_wavforms, good_clusters, good_label_ids = sort_spikes_by_cell(cluster_batch)

    assert type(good_sorted_cells) == list 
    assert type(good_sorted_wavforms) == list
    assert type(good_clusters) == list 
    # assert isinstance(good_clusters[0], SpikeCluster) 
    assert type(good_label_ids) == list or type(good_label_ids) == np.ndarray

def test_find_burst():
    cluster_batch = make_spike_cluster_batch()
    cluster = make_spike_cluster()
    cell = make_cell()

    bursting, bursts_n_spikes_avg = find_burst(cluster_batch)
    assert 'bursting' in cluster_batch.stats_dict['spike'] and 'bursts_n_spikes_avg' in cluster_batch.stats_dict['spike']
    assert cluster_batch.stats_dict['spike']['bursts_n_spikes_avg'] == bursts_n_spikes_avg

    bursting, bursts_n_spikes_avg = find_burst(cluster)
    assert 'bursting' in cluster.stats_dict['spike'] and 'bursts_n_spikes_avg' in cluster.stats_dict['spike']
    assert cluster.stats_dict['spike']['bursts_n_spikes_avg'] == bursts_n_spikes_avg

    bursting, bursts_n_spikes_avg = find_burst(cell)
    assert 'bursting' in cell.stats_dict['spike'] and 'bursts_n_spikes_avg' in cell.stats_dict['spike']
    assert cell.stats_dict['spike']['bursts_n_spikes_avg'] == bursts_n_spikes_avg

def test_histogram_ISI():
    cluster_batch = make_spike_cluster_batch()
    cluster = make_spike_cluster()
    cell = make_cell()

    ISI_dict = histogram_ISI(cluster_batch)
    assert 'ISI_dict' in cluster_batch.stats_dict['spike'] 
    assert cluster_batch.stats_dict['spike']['ISI_dict'] == ISI_dict

    ISI_dict = histogram_ISI(cluster)
    assert 'ISI_dict' in cluster.stats_dict['spike']
    assert cluster.stats_dict['spike']['ISI_dict'] == ISI_dict

    ISI_dict = histogram_ISI(cell)
    assert 'ISI_dict' in cell.stats_dict['spike']
    assert cell.stats_dict['spike']['ISI_dict'] == ISI_dict



if __name__ == '__main__':
    test_sort_cell_spike_times()
    test_find_burst()
    test_histogram_ISI()


