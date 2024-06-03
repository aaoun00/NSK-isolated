
#TODO: set up gin ssh key for https://gin.g-node.org/

#TODO: load test data to the gin repository

#TODO: figure out how to read the data directly from the internet.

# eventually replace this with urllib
# access to the data online

import os
import sys


PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

 
from x_io.rw.axona.read_pos import grab_position_data

from x_io.rw.axona.batch_read import (
    make_session,
    make_study,
)

from core.subjects import AnimalMetadata, StudyMetadata, SessionMetadata
from core.instruments import TrackerMetadata, DevicesMetadata, ImplantMetadata

from x_io.rw.axona.batch_read import make_session

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


settings_dict = {'ppm': 511, 'session':  session_settings}

def test_animal_metadata():
    animal_metadata = AnimalMetadata(animal)

    assert isinstance(animal_metadata, AnimalMetadata)
    assert animal_metadata.animal_id == animal['animal_id']
    assert animal_metadata.species == animal['species']
    assert animal_metadata.weight == animal['weight']

def test_session_metadata():
    animal_metadata = AnimalMetadata(animal)
    tracker_metadata = TrackerMetadata(implant)
    implant_metadata = ImplantMetadata({'x': [0], 'y': 0})
    devices_dict = {'axona_led_tracker': tracker_metadata, 'implant': implant_metadata}
    devices_metadata = DevicesMetadata(devices_dict)
    session_metadata = SessionMetadata({'animal': animal_metadata, 'devices': devices_metadata})

    assert isinstance(session_metadata, SessionMetadata)
    assert isinstance(session_metadata.metadata['devices'].devices_dict['axona_led_tracker'], TrackerMetadata)
    assert isinstance(session_metadata.metadata['devices'].devices_dict['implant'], ImplantMetadata)
    assert isinstance(session_metadata.metadata['animal'], AnimalMetadata)

##### TO IMPLEMENT

def test_study_metadata():
    pass
    