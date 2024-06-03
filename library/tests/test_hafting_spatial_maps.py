import os
import sys
from datetime import datetime
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.hafting_spatial_maps import HaftingSpikeMap, HaftingOccupancyMap, HaftingRateMap, SpatialSpikeTrain2D
from library.study_space import Session
from core.core_utils import make_1D_timestamps, make_seconds_index_from_rate
from library.lib_test_utils import make_2D_arena
# from library.spatial_spike_train import SpatialSpikeTrain2D
from core.spikes import SpikeTrain
from core.spatial import Position2D
from core.spatial import Position2D
from core.subjects import SessionMetadata
from PIL import Image
from matplotlib import cm

def test_spatial_spike_train():
    spike_times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ses = Session()
    spike_train = ses.make_class(SpikeTrain, {'event_times': spike_times, 'duration': 1, 'sample_rate': 50, 'session_metadata': ses.session_metadata,  'datetime': datetime(1,1,1)})
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    t = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    location = Position2D({'x': x, 'y': y, 't': t})

    spatial_spike_train = SpatialSpikeTrain2D({'spike_train':spike_train, 'position': location})

    assert len(spatial_spike_train.x) == len(spatial_spike_train.y) == len(spatial_spike_train.spike_times)
    assert spatial_spike_train.x == x
    assert spatial_spike_train.y == y
    assert spatial_spike_train.spike_times == spike_times

def test_hafting_occupancy_map():

    T = 2
    dt = .02

    event_times = make_1D_timestamps(T, dt)
    t = make_seconds_index_from_rate(T, 1/dt)
    x, y = make_2D_arena(count=len(t))

    pos_dict = {'x': x, 'y': t, 't': t, 'arena_height': max(y) - min(y), 'arena_width': max(x) - min(x)}

    spike_dict = {}
    spike_dict['duration'] = int(T)
    spike_dict['sample_rate'] = float(1 / dt)
    spike_dict['events_binary'] = []
    spike_dict['event_times'] = event_times
    spike_dict['datetime'] = datetime(1,1,1)

    session = Session()
    session_metadata = session.session_metadata

    spike_train = session.make_class(SpikeTrain, spike_dict)
    position = session.make_class(Position2D, pos_dict)


    # spatial_spike_train = SpatialSpikeTrain2D(spike_train, position)
    spatial_spike_train = session.make_class(SpatialSpikeTrain2D, {'spike_train': spike_train, 'position': position})

    hafting_occupancy = HaftingOccupancyMap(spatial_spike_train)

    occ_map, _, _ = hafting_occupancy.get_occupancy_map(smoothing_factor=3)

    assert isinstance(hafting_occupancy, HaftingOccupancyMap)
    assert hafting_occupancy.x.all() == pos_dict['x'].all() 
    assert hafting_occupancy.map_data.all() == occ_map.all()

    assert 'spatial_spike_train' in session_metadata.session_object.get_spike_data()
    assert isinstance(session_metadata.session_object.session_data.data['spatial_spike_train'], SpatialSpikeTrain2D)
    assert session_metadata.session_object.session_data.data['spatial_spike_train'] == spatial_spike_train
    assert 'occupancy' in spatial_spike_train.stats_dict

def test_hafting_spike_map():

    T = 2
    dt = .02

    event_times = make_1D_timestamps(T, dt)
    t = make_seconds_index_from_rate(T, 1/dt)
    x, y = make_2D_arena(count=len(t))

    pos_dict = {'x': x, 'y': t, 't': t, 'arena_height': max(y) - min(y), 'arena_width': max(x) - min(x)}

    spike_dict = {}
    spike_dict['duration'] = int(T)
    spike_dict['sample_rate'] = float(1 / dt)
    spike_dict['events_binary'] = []
    spike_dict['event_times'] = event_times
    spike_dict['datetime'] = datetime(1,1,1)

    session = Session()
    session_metadata = session.session_metadata

    spike_train = session.make_class(SpikeTrain, spike_dict)
    position = session.make_class(Position2D, pos_dict)


    # spatial_spike_train = SpatialSpikeTrain2D(spike_train, position)
    spatial_spike_train = session.make_class(SpatialSpikeTrain2D, {'spike_train': spike_train, 'position': position})

    hafting_spike = HaftingSpikeMap(spatial_spike_train)

    spike_map, _ = hafting_spike.get_spike_map(smoothing_factor=3)

    assert isinstance(hafting_spike, HaftingSpikeMap)
    assert hafting_spike.map_data.all() == spike_map.all()

    assert 'spatial_spike_train' in session_metadata.session_object.get_spike_data()
    assert isinstance(session_metadata.session_object.session_data.data['spatial_spike_train'], SpatialSpikeTrain2D)
    assert session_metadata.session_object.session_data.data['spatial_spike_train'] == spatial_spike_train
    assert 'spike' in spatial_spike_train.stats_dict

def test_hafting_rate_map():

    T = 2
    dt = .02

    event_times = make_1D_timestamps(T, dt)
    t = make_seconds_index_from_rate(T, 1/dt)
    x, y = make_2D_arena(count=len(t))

    pos_dict = {'x': x, 'y': t, 't': t, 'arena_height': max(y) - min(y), 'arena_width': max(x) - min(x)}

    spike_dict = {}
    spike_dict['duration'] = int(T)
    spike_dict['sample_rate'] = float(1 / dt)
    spike_dict['events_binary'] = []
    spike_dict['event_times'] = event_times
    spike_dict['datetime'] = datetime(1,1,1)

    session = Session()
    session_metadata = session.session_metadata

    spike_train = session.make_class(SpikeTrain, spike_dict)
    position = session.make_class(Position2D, pos_dict)

    # spatial_spike_train = SpatialSpikeTrain2D(spike_train, position)
    spatial_spike_train = session.make_class(SpatialSpikeTrain2D, {'spike_train': spike_train, 'position': position})

    hafting_occupancy = HaftingOccupancyMap(spatial_spike_train)
    occ_map = hafting_occupancy.get_occupancy_map(smoothing_factor=3)
    # spatial_spike_train.add_map_to_stats('occupancy', hafting_occupancy)

    hafting_spike = HaftingSpikeMap(spatial_spike_train)
    spike_map = hafting_spike.get_spike_map(smoothing_factor=3)
    # spatial_spike_train.add_map_to_stats('spike', hafting_spike)

    hafting_rate = HaftingRateMap(spatial_spike_train)

    rate_map, _ = hafting_rate.get_rate_map(smoothing_factor=3)

    colored_ratemap = Image.fromarray(np.uint8(cm.jet(rate_map)*255))
    colored_ratemap.save('test.png')

    assert isinstance(hafting_rate, HaftingRateMap)
    assert type(hafting_rate.map_data) == np.ma.core.MaskedArray
   
    assert 'spatial_spike_train' in session_metadata.session_object.get_spike_data()
    assert isinstance(session_metadata.session_object.session_data.data['spatial_spike_train'], SpatialSpikeTrain2D)
    assert session_metadata.session_object.session_data.data['spatial_spike_train'] == spatial_spike_train
    assert 'rate' in spatial_spike_train.stats_dict



