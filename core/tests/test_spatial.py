
#TODO: set up gin ssh key for https://gin.g-node.org/

#TODO: load test data to the gin repository

#TODO: figure out how to read the data directly from the internet.

# eventually replace this with urllib
# access to the data online
import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
 
from core.spatial import Position2D
from library.study_space import Session

from x_io.rw.axona.read_pos import (
    grab_position_data,
)

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

def test_position2d():
    pos_dict = grab_position_data(pos_file, ppm=settings_dict['ppm'])

    session = Session()

    position_object = session.make_class(Position2D, pos_dict)

    assert isinstance(position_object, Position2D)
    assert position_object.x.all() == pos_dict['x'].all()
    assert position_object.t.all() == pos_dict['t'].all()
