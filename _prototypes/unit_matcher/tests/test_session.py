import os
import sys
import numpy as np
import pytest

prototype_dir = os.getcwd()
sys.path.append(prototype_dir)
parent_dir = os.path.dirname(prototype_dir)

from library.study_space import Session
from _prototypes.unit_matcher.read_axona import read_sequential_sessions
from _prototypes.unit_matcher.session import compare_sessions, compute_distances, guess_remaining_matches, extract_full_matches

data_dir = parent_dir + r'\neuroscikit_test_data\single_sequential'

animal = {'animal_id': 'id', 'species': 'mouse', 'sex': 'F', 'age': 1, 'weight': 1, 'genotype': 'type', 'animal_notes': 'notes'}
devices = {'axona_led_tracker': True, 'implant': True}
implant = {'implant_id': 'id', 'implant_type': 'tetrode', 'implant_geometry': 'square', 'wire_length': 25, 'wire_length_units': 'um', 'implant_units': 'uV'}

session_settings = {'channel_count': 4, 'animal': animal, 'devices': devices, 'implant': implant}

settings_dict = {'ppm': 511, 'session':session_settings, 'smoothing_factor': 3, 'useMatchedCut': False}

session1, session2 = read_sequential_sessions(data_dir, settings_dict)


def test_compute_MSD_distances():
    distances, pairs = compute_distances(session1.get_spike_data()['spike_cluster'], session2.get_spike_data()['spike_cluster'], method='MSD')
    assert type(distances) == np.ndarray
    assert type(pairs) == np.ndarray
    assert len(distances) == len(pairs)
    assert len(np.unique(session1.get_cell_data()['cell_ensemble'].get_label_ids())) == len(distances)
    assert len(np.unique(session2.get_cell_data()['cell_ensemble'].get_label_ids())) == len(distances.T)

def test_compute_JSD_distances():
    distances, pairs = compute_distances(session1.get_spike_data()['spike_cluster'], session2.get_spike_data()['spike_cluster'], method='JSD')

    assert type(distances) == np.ndarray
    assert type(pairs) == np.ndarray
    assert len(distances) == len(pairs)
    assert len(np.unique(session1.get_cell_data()['cell_ensemble'].get_label_ids())) == len(distances)
    assert len(np.unique(session2.get_cell_data()['cell_ensemble'].get_label_ids())) == len(distances.T)

def test_extract_full_matches():
    # distances, pairs = compute_distances(session1.get_spike_data()['spike_cluster'], session2.get_spike_data()['spike_cluster'])

    distances = np.array([[5,1,5], [5,1,5], [5,5,2]])
    pairs = np.array([[[0,0],[0,1],[0,2]], [[1,0],[1,1],[1,2]], [[2,0], [2,1], [2,2]]])

    full_matches, full_match_distances, remaining_distances, remaining_pairs = extract_full_matches(distances, pairs)

    assert type(full_matches) == list
    assert type(full_match_distances) == list
    assert type(remaining_distances) == np.ndarray
    assert type(remaining_pairs) == np.ndarray

    assert [0,1] == list(full_matches[0])
    assert [2,2] == list(full_matches[1])

def test_guess_remaining_matches():
    distances, pairs = compute_distances(session1.get_spike_data()['spike_cluster'], session2.get_spike_data()['spike_cluster'])

    full_matches, full_match_distances, remaining_distances, remaining_pairs = extract_full_matches(distances, pairs)

    remaining_matches, remaining_match_distances, unmatched_2, unmatched_1 = guess_remaining_matches(remaining_distances, remaining_pairs)

    assert type(remaining_matches) == list
    assert type(remaining_match_distances) == list
    assert type(unmatched_1) == list
    assert len(full_matches) *2 + len(remaining_matches) *2 + len(unmatched_2) == sum(distances.shape)

def test_compare_sessions():
    matches, match_distances, unmatched_2, unmatched_1  = compare_sessions(session1, session2)
    assert type(matches) == np.ndarray
    assert type(match_distances) == np.ndarray
    assert type(unmatched_1) == list
    assert type(unmatched_2) == list

    # assert len(np.unique(session2.get_cell_data()['cell_ensemble'].get_label_ids())) == len(map_dict)