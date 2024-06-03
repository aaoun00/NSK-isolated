import os
import sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.lib_test_utils import make_2D_arena, make_spatial_spike_train
from library.hafting_spatial_maps import HaftingOccupancyMap, HaftingRateMap, HaftingSpikeMap, SpatialSpikeTrain2D
from library.scores import border_score, grid_score, hd_score, rate_map_coherence, speed_score
from library.lib_test_utils import make_2D_arena
from core.core_utils import make_seconds_index_from_rate
from core.core_utils import make_1D_timestamps
from library.scores.rate_map_stats import rate_map_stats
from library.shuffle_spikes import shuffle_spikes


def test_border_score():
    spatial_spike_train, session_metadata = make_spatial_spike_train()

    bscore = border_score(spatial_spike_train)

    assert type(bscore) == tuple

def test_grid_score():
    spatial_spike_train, session_metadata = make_spatial_spike_train()

    gscore = grid_score(spatial_spike_train)

    assert type(gscore) == np.float64 or type(gscore) == float

def test_hd_score():
    spatial_spike_train, session_metadata = make_spatial_spike_train()

    smooth_hd = hd_score(spatial_spike_train)

    assert type(smooth_hd) == np.ndarray

def test_rate_map_coherence():
    spatial_spike_train, session_metadata = make_spatial_spike_train()
    coherence = rate_map_coherence(spatial_spike_train)

    assert type(coherence) == np.float64

def test_rate_map_stats():
    spatial_spike_train, session_metadata = make_spatial_spike_train()
    ratemap = HaftingRateMap(spatial_spike_train)
    occmap = HaftingOccupancyMap(spatial_spike_train)
    spkmap = HaftingSpikeMap(spatial_spike_train)
    spatial_spike_train.add_map_to_stats('rate', ratemap)
    spatial_spike_train.add_map_to_stats('occupancy', occmap)
    spatial_spike_train.add_map_to_stats('spike', spkmap)
    map_stats = rate_map_stats(spatial_spike_train)

    assert type(map_stats) == dict
    assert 'spatial_information_rate' in map_stats

def test_shuffle_spikes():
    T = 100
    dt = .1

    event_times = make_1D_timestamps(T, dt)
    t = make_seconds_index_from_rate(T, 1/dt)
    x, y = make_2D_arena(count=len(t))

    shuffled_spikes = shuffle_spikes(np.array(event_times), np.array(x), np.array(y), np.array(t))

    assert type(shuffled_spikes) == list

def test_speed_score():
    spatial_spike_train, session_metadata = make_spatial_spike_train()

    scores, bounds = speed_score(spatial_spike_train) 

    assert type(scores) == dict
    assert type(bounds) == tuple 

# def test_shuffle_spikes():
#     T = 10
#     dt = .01
#     pos_t = make_seconds_index_from_rate(T, 1/dt)

#     spk_times = make_1D_timestamps(T=T, dt=dt)

#     pos_x, pos_y = make_2D_arena(len(pos_t))
#     shuffled_spikes = shuffle_spikes(np.array(spk_times), np.array(pos_x), np.array(pos_y), pos_t)

#     assert len(shuffled_spikes) == len(spk_times)
#     assert type(shuffled_spikes) == np.ndarray

if __name__ == '__main__':
    test_border_score()
    test_grid_score()
    test_hd_score()
    # test_shuffle_spikes()


