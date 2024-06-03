import os
import sys

prototype_dir = os.getcwd()
sys.path.append(prototype_dir)
parent_dir = os.path.dirname(prototype_dir)

from library.study_space import Session
from _prototypes.unit_matcher.read_axona import read_sequential_sessions, temp_read_cut

data_dir = parent_dir + r'\neuroscikit_test_data\single_sequential_2'

animal = {'animal_id': 'id', 'species': 'mouse', 'sex': 'F', 'age': 1, 'weight': 1, 'genotype': 'type', 'animal_notes': 'notes'}
devices = {'axona_led_tracker': True, 'implant': True}
implant = {'implant_id': 'id', 'implant_type': 'tetrode', 'implant_geometry': 'square', 'wire_length': 25, 'wire_length_units': 'um', 'implant_units': 'uV'}

session_settings = {'channel_count': 4, 'animal': animal, 'devices': devices, 'implant': implant}

settings_dict = {'ppm': 511, 'session': session_settings, 'smoothing_factor': 3, 'useMatchedCut': False}


def test_read_sequential_sessions():
    session1, session2 = read_sequential_sessions(data_dir, settings_dict)

    assert isinstance(session1, Session)
    assert isinstance(session2, Session)

def test_temp_read_cut():
    cut_file = r'C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit_test_data\single_sequential\1-13_20210621-34-50x50cm-1500um-Test1_3.cut'
    with open(cut_file, 'r') as open_cut_file:
        cut_data, header_data =  temp_read_cut(open_cut_file)

    assert type(cut_data) == list
    assert type(header_data) == list
    assert type(cut_data[0]) == int