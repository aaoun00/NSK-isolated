import os
import sys


prototype_dir = os.getcwd()
sys.path.append(prototype_dir)
parent_dir = os.path.dirname(prototype_dir)

from library.study_space import Session
from _prototypes.unit_matcher.read_axona import read_sequential_sessions, temp_read_cut
from _prototypes.unit_matcher.write_axona import format_new_cut_file_name, apply_remapping, write_cut, get_current_cut_data

data_dir = parent_dir + r'\neuroscikit_test_data\single_sequential'

animal = {'animal_id': 'id', 'species': 'mouse', 'sex': 'F', 'age': 1, 'weight': 1, 'genotype': 'type', 'animal_notes': 'notes'}
devices = {'axona_led_tracker': True, 'implant': True}
implant = {'implant_id': 'id', 'implant_type': 'tetrode', 'implant_geometry': 'square', 'wire_length': 25, 'wire_length_units': 'um', 'implant_units': 'uV'}

session_settings = {'channel_count': 4, 'animal': animal, 'devices': devices, 'implant': implant}

settings_dict = {'ppm': 511, 'session': session_settings, 'smoothing_factor': 3}


def test_format_new_cut_file_name():
    old_path = r'C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit_test_data\single_sequential\1-13_20210621-34-50x50cm-1500um-Test1_3.cut'
    new_path = format_new_cut_file_name(old_path)

    assert new_path == r'C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit_test_data\single_sequential\1-13_20210621-34-50x50cm-1500um-Test1_3_matched.cut'


def test_apply_remapping():
    cut_file = r'C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit_test_data\single_sequential\1-13_20210621-34-50x50cm-1500um-Test1_3.cut'
    map_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11}
    with open(cut_file, 'r') as open_cut_file:
        cut_data, header_data =  temp_read_cut(open_cut_file)
    new_cut_data = apply_remapping(cut_data, map_dict)

    assert new_cut_data == cut_data
    # assert

def test_write_cut():
    cut_file = r'C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit_test_data\single_sequential\1-13_20210621-34-50x50cm-1500um-Test1_3.cut'
    map_dict = {0:0, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:0, 8:0, 9:0, 10:0, 11:0}
    with open(cut_file, 'r') as open_cut_file:
        cut_data, header_data =  temp_read_cut(open_cut_file)
    new_cut_data = apply_remapping(cut_data, map_dict)

    cut_file = r'C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit_test_data\single_sequential\1-13_20210621-34-50x50cm-1500um-Test1_3_matched.cut'
    write_cut(cut_file, new_cut_data, header_data)

    with open(cut_file, 'r') as open_cut_file:
        cut_data, header_data = temp_read_cut(open_cut_file)

    assert new_cut_data == cut_data

def test_get_current_cut_data():
    cut_file = r'C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit_test_data\single_sequential\1-13_20210621-34-50x50cm-1500um-Test1_3.cut'
    cut_data, header_data = get_current_cut_data(cut_file)

    assert type(cut_data) == list
    assert type(header_data) == list