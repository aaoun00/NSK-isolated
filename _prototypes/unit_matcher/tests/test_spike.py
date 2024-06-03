import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

import pytest
import _prototypes.unit_matcher.tests.read as read

from _prototypes.unit_matcher.spike import waveform_level_features
from library.lib_test_utils import make_spike_cluster

# spike = read.spike
time_step = read.time_step

def test_waveform_level_features():
    cluster = make_spike_cluster()
    spike = cluster.get_spike_object_instances()[0]
    features = waveform_level_features(spike, time_step)
    assert features is not None
    assert type(features) == dict
    for key, value in features.items():
        assert type(key) == str
        assert type(value) == float or type(value) == int
        assert value is not None
        assert value != float('inf')
        assert value != float('-inf')
        assert value != float('nan')

@pytest.mark.skip(reason="Not implemented yet")
def test_localize_source():
    pass

@pytest.mark.skip(reason="Not implemented yet")
def test_spike_features():
    pass