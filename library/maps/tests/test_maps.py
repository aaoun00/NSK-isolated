import os
import sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.maps import autocorrelation, filter_pos_by_speed, firing_rate_vs_time, map_blobs, spatial_tuning_curve, binary_map
from core.core_utils import make_seconds_index_from_rate, make_1D_timestamps
from library.lib_test_utils import make_2D_arena, make_cell, make_spike_cluster, make_spike_cluster_batch, make_spatial_spike_train
from library.study_space import Session
from core.spikes import SpikeTrain
from core.spatial import Position2D
# from library.spatial_spike_train import SpatialSpikeTrain2D
from library.hafting_spatial_maps import HaftingRateMap, SpatialSpikeTrain2D

def test_autocorrelation():
    spatial_spike_train, session_metadata = make_spatial_spike_train()

    autocorr = autocorrelation(spatial_spike_train)

    assert type(autocorr) == np.ndarray
    assert 'autocorr' in spatial_spike_train.stats_dict

    hafting_rate = HaftingRateMap(spatial_spike_train)

    autocorr = autocorrelation(hafting_rate)

    assert type(autocorr) == np.ndarray
    assert 'autocorr' in spatial_spike_train.stats_dict

def test_filter_pos_by_speed():

    T = 2
    dt = .02
    pos_t = make_seconds_index_from_rate(T, 1/dt)

    pos_x, pos_y = make_2D_arena(len(pos_t))

    position_object = Position2D({'x': pos_x, 'y': pos_y, 't': pos_t})

    new_pos_x, new_pos_y, new_pos_t = filter_pos_by_speed(position_object, 0.1, 0.9)

    assert len(new_pos_x) == len(new_pos_y)
    assert len(new_pos_y) == len(new_pos_t)
    assert 'speed_filter' in position_object.stats_dict

def test_firing_rate_vs_time():
    cluster_batch = make_spike_cluster_batch()
    cluster = make_spike_cluster()
    cell = make_cell()

    rate, firing_time = firing_rate_vs_time(cell, 100)
    assert type(rate) == np.ndarray 
    assert type(firing_time) == list

    rate, firing_time = firing_rate_vs_time(cluster, 100)
    assert type(rate) == np.ndarray 
    assert type(firing_time) == list

    rate, firing_time = firing_rate_vs_time(cluster_batch, 100)
    assert type(rate) == np.ndarray 
    assert type(firing_time) == list

    assert 'rate_vector' in cell.stats_dict
    assert 'rate_vector' in cluster.stats_dict
    assert 'rate_vector' in cluster_batch.stats_dict

def test_map_blobs():
    spatial_spike_train, session_metadata = make_spatial_spike_train()

    image, n_labels, labels, centroids, field_sizes = map_blobs(spatial_spike_train)

    assert type(centroids) == np.ndarray
    assert 'map_blobs' in spatial_spike_train.stats_dict
    assert type(spatial_spike_train.get_map('map_blobs')) == dict
    assert len(image) == len(labels)
    assert len(np.unique(labels)) == n_labels
    assert len(centroids) == len(field_sizes)

    
    hafting_rate = HaftingRateMap(spatial_spike_train)
    image, n_labels, labels, centroids, field_sizes = map_blobs(hafting_rate)

    assert type(centroids) == np.ndarray
    assert 'map_blobs' in hafting_rate.spatial_spike_train.stats_dict
    assert len(image) == len(labels)
    assert len(np.unique(labels)) == n_labels
    assert len(centroids) == len(field_sizes)

def test_spatial_tuning_curve():
    spatial_spike_train, session_metadata = make_spatial_spike_train()

    tuned_data, spike_angles, ang_occ, bin_array = spatial_tuning_curve(spatial_spike_train)

    assert type(tuned_data) == np.ndarray
    assert type(spike_angles) == np.ndarray
    assert type(ang_occ) == tuple
    assert type(bin_array) == np.ndarray
    assert 'spatial_tuning' in spatial_spike_train.stats_dict
    assert type(spatial_spike_train.get_map('spatial_tuning')) == dict

def test_binary_map():
    spatial_spike_train, session_metadata = make_spatial_spike_train()

    binmap = binary_map(spatial_spike_train)

    assert type(binmap) == np.ndarray

    hafting_rate = HaftingRateMap(spatial_spike_train)

    binmap = binary_map(hafting_rate)

    assert type(binmap) == np.ndarray


# def test_spike_pos():

#     T = 2
#     dt = .02
#     pos_t = make_seconds_index_from_rate(T, 1/dt)

#     smoothing_factor = 5
#     # Kernel size
#     kernlen = int(smoothing_factor*8)
#     # Standard deviation size
#     std = int(0.2*kernlen)
#     arena_size = (1,1)

#     spk_times = make_1D_timestamps()
#     pos_x, pos_y = make_2D_arena(len(pos_t))

#     spikex, spikey, spiket, _ = spike_pos(spk_times, pos_x, pos_y, pos_t, pos_t, False, False)

#     assert len(spikex) == len(spikey)
#     assert len(spikex) == len(spiket)

# def test_spike_map():

#     T = 2
#     dt = .02
#     pos_t = make_seconds_index_from_rate(T, 1/dt)

#     smoothing_factor = 5
#     # Kernel size
#     kernlen = int(smoothing_factor*8)
#     # Standard deviation size
#     std = int(0.2*kernlen)
#     arena_size = (1,1)

#     spk_times = make_1D_timestamps()
#     pos_x, pos_y = make_2D_arena(len(pos_t))

#     spikex, spikey, spiket, _ = spike_pos(spk_times, pos_x, pos_y, pos_t, pos_t, False, False)

#     spike_map_smooth, spike_map_raw = spike_map(pos_x, pos_y, pos_y, arena_size, spikex, spikey, kernlen, std)

#     assert len(spike_map_smooth) == len(spike_map_raw)

# def test_occupancy_map():
#     T = 2
#     dt = .02
#     pos_t = make_seconds_index_from_rate(T, 1/dt)

#     smoothing_factor = 5
#     # Kernel size
#     kernlen = int(smoothing_factor*8)
#     # Standard deviation size
#     std = int(0.2*kernlen)
#     arena_size = (1,1)

#     pos_x, pos_y = make_2D_arena(len(pos_t))

#     occ_map_smoothed, occ_map_raw, coverage_map = occupancy_map(pos_x, pos_y, pos_t, arena_size, kernlen, std)

#     assert len(occ_map_smoothed) == len(occ_map_raw)
#     assert type(coverage_map) == np.ndarray
    
# def test_rate_map():  

#     T = 2
#     dt = .02
#     pos_t = make_seconds_index_from_rate(T, 1/dt)

#     smoothing_factor = 5
#     # Kernel size
#     kernlen = int(smoothing_factor*8)
#     # Standard deviation size
#     std = int(0.2*kernlen)
#     arena_size = (1,1)

#     spk_times = make_1D_timestamps()
#     pos_x, pos_y = make_2D_arena(len(pos_t))

#     spikex, spikey, spiket, _ = spike_pos(spk_times, pos_x, pos_y, pos_t, pos_t, False, False)

#     rate_map_smooth, rate_map_raw = rate_map(pos_x, pos_y, pos_t, arena_size, spikex, spikey, kernlen, std)

#     assert type(rate_map_smooth) == np.ndarray 
#     assert type(rate_map_raw) == np.ndarray
#     assert rate_map_smooth.shape == rate_map_raw.shape


if __name__ == '__main__':
    # test_spike_pos()
    # test_spike_map()
    # test_rate_map()
    test_autocorrelation()
    test_filter_pos_by_speed()
    test_firing_rate_vs_time()
    test_map_blobs()
    # test_occupancy_map()
    test_spatial_tuning_curve()
    # test_binary_map()

