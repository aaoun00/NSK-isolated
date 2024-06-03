# import os
# import sys
# import numpy as np

# PROJECT_PATH = os.getcwd()
# sys.path.append(PROJECT_PATH)

# from core.spikes import SpikeTrain
# from core.spatial import Position2D
# from library.spatial_spike_train import SpatialSpikeTrain2D


# def test_spatial_spike_train():
#     spike_times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#     spike_train = SpikeTrain({'event_times': spike_times, 'duration': 1, 'sample_rate': 50})
#     x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#     y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#     t = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#     location = Position2D({'x': x, 'y': y, 't': t})

#     spatial_spike_train = SpatialSpikeTrain2D({'spike_train':spike_train, 'position': location})

#     assert len(spatial_spike_train.x) == len(spatial_spike_train.y) == len(spatial_spike_train.spike_times)
#     assert spatial_spike_train.x == x
#     assert spatial_spike_train.y == y
#     assert spatial_spike_train.spike_times == spike_times