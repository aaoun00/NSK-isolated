
#TODO: set up gin ssh key for https://gin.g-node.org/

#TODO: load test data to the gin repository

#TODO: figure out how to read the data directly from the internet.

# eventually replace this with urllib
# access to the data online

from csv import DictReader
import os
import sys
from turtle import pos


PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

 
from x_io.rw.axona.read_pos import grab_position_data

from x_io.rw.axona.batch_read import (
    make_session,
    make_study,
)

from library.study_space import *
from library.ensemble_space import CellPopulation
from core.subjects import AnimalMetadata
from core.instruments import TrackerMetadata, DevicesMetadata, ImplantMetadata

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
data_dir = os.path.join(parent_dir, 'neuroscikit_test_data/test_dir')
cut_file = os.path.join(data_dir, '20140815-behavior2-90_1.cut')
tet_file = os.path.join(data_dir, '20140815-behavior2-90.1')
pos_file = os.path.join(data_dir, '20140815-behavior2-90.pos')

animal = {'animal_id': 'id', 'species': 'mouse', 'sex': 'F', 'age': 1, 'weight': 1, 'genotype': 'type', 'animal_notes': 'notes'}
devices = {'axona_led_tracker': True, 'implant': True}
implant = {'implant_id': 'id', 'implant_type': 'tetrode', 'implant_geometry': 'square', 'wire_length': 25, 'wire_length_units': 'um', 'implant_units': 'uV'}

session_settings = {'channel_count': 4, 'animal': animal, 'devices': devices, 'implant': implant}


settings_dict = {'ppm': 511, 'session':  session_settings, 'smoothing_factor': 3, 'useMatchedCut': False}
study = make_study([data_dir], settings_dict)

session = study.sessions[0]
# session = make_session(cut_file, tet_file, pos_file, session_settings, settings_dict['ppm'])

def test_animal():
    animal_instance = Animal({'session_1': session, 'session_2': session})
    
    assert type(animal_instance.ensembles) == dict
    assert type(animal_instance.sessions) == dict
    assert isinstance(animal_instance.population, CellPopulation)
    animal_instance.add_session(session)

    assert len(animal_instance.ensembles) == 3
    assert len(animal_instance.sessions) == 3

def test_session():

    assert isinstance(session.get_animal_metadata(), AnimalMetadata)
    assert isinstance(session.get_devices_metadata()['axona_led_tracker'], TrackerMetadata)
    assert isinstance(session.get_devices_metadata()['implant'], ImplantMetadata)

def test_study():

    assert len(study.sessions) == 1
    assert isinstance(study.sessions[0], Session)

    