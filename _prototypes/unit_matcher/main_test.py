# Outside imports
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Set necessary paths / make project path = ...../neuroscikit/
unit_matcher_path = os.getcwd()
# prototype_path = os.path.abspath(os.path.join(unit_matcher_path, os.pardir))
# project_path = os.path.abspath(os.path.join(prototype_path, os.pardir))
# lab_path = os.path.abspath(os.path.join(project_path, os.pardir))
sys.path.append(unit_matcher_path)
os.chdir(unit_matcher_path)

# Internal imports

# Read write modules
from x_io.rw.axona.batch_read import make_study
from _prototypes.unit_matcher.read_axona import read_sequential_sessions, temp_read_cut
from _prototypes.unit_matcher.write_axona import format_new_cut_file_name

# Unit matching modules
from _prototypes.unit_matcher.main import format_cut, run_unit_matcher, map_unit_matches_first_session, map_unit_matches_sequential_session
from _prototypes.unit_matcher.session import compare_sessions
from _prototypes.unit_matcher.waveform import time_index, derivative, derivative2, morphological_points