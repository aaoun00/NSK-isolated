import os
import sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.spatial import Position2D
from library.spatial import accumulate_spatial, fit_ellipse, place_field
from library.lib_test_utils import make_2D_arena, make_spatial_spike_train
from library.hafting_spatial_maps import HaftingRateMap, SpatialSpikeTrain2D

def test_accumulate_spatial():
    spatial_spike_train, _ = make_spatial_spike_train()

    hist, edges = accumulate_spatial(spatial_spike_train)

    assert type(hist) == np.ndarray
    assert type(edges) == list

    x, y = make_2D_arena()

    position = Position2D({'x': x, 'y': y, 'arena_height': max(y) - min(y), 'arena_width': max(x) - min(x)})

    hist, edges = accumulate_spatial(position)

    assert type(hist) == np.ndarray
    assert type(edges) == list

def test_fit_ellipse():
    x, y = make_2D_arena()

    output = fit_ellipse(x, y)

    for val in output:
        assert type(val) == np.float64

def test_place_field():
    spatial_spike_train, _ = make_spatial_spike_train()

    fields, fields_map = place_field(spatial_spike_train)

    assert type(fields) == list 
    assert type(fields_map) == np.ma.MaskedArray

    spatial_spike_train, _ = make_spatial_spike_train()
    rate_obj = HaftingRateMap(spatial_spike_train)

    fields, fields_map = place_field(rate_obj)

    assert type(fields) == list 
    assert type(fields_map) == np.ma.MaskedArray

