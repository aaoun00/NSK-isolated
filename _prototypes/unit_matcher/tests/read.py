import os
import sys
import numpy as np

# Set up paths
# (assumes neuroscikit_test_data folder is next to neuroscikit folder)

prototype_dir = os.getcwd()

parent_dir = os.path.dirname(prototype_dir)
sys.path.append(parent_dir)

data_dir = parent_dir + r'\neuroscikit_test_data\sequential_axona_sessions'

# Import read modules

from x_io.rw.axona.read_tetrode_and_cut import (
    _read_cut,
    _format_spikes
)

# Test data we are using has two sets of sequential sessions --> extract

files = os.listdir(data_dir)

test1_34 = []
test1_35 = []
test2_34 = []
test2_35 = []

for f in files:
    if 'Test1' in f and '34' in f:
        test1_34.append(f)
    elif 'Test1' in f and '35' in f:
        test1_35.append(f)
    elif 'Test2' in f and '34' in f:
        test2_34.append(f)
    elif 'Test2' in f and '35' in f:
        test2_35.append(f)

# Get tet and cut files from inside folders

# session1 = test1_34
# session2 = test2_34

session1 = test1_35
session2 = test2_35

assert len(session1) == len(session2)

session1_tets = []
session2_tets = []

for i in range(len(session1)):
    if 'cut' in session1[i]:
        session1_cut = session1[i]
    if 'cut' in session2[i]:
        session2_cut = session2[i]
    file_session_1 = session1[i]
    file_session_2 = session2[i]
    out1 = file_session_1.split('.')[-1]
    out2 = file_session_2.split('.')[-1]
    if out1.isnumeric() and 'clu' not in file_session_1:
        session1_tets.append(session1[i])
    if out2.isnumeric() and 'clu' not in file_session_2:
        session2_tets.append(session2[i])

session1_cut_path = os.path.join(data_dir, session1_cut)
session1_tet_path = os.path.join(data_dir, session1_tets[0])

session2_cut_path = os.path.join(data_dir, session2_cut)
session2_tet_path = os.path.join(data_dir, session2_tets[0])


# read data from cut and tet files

with open(session1_cut_path, 'r') as cut_file1, open(session1_tet_path, 'rb') as tetrode_file1:
    cut_data1 = _read_cut(cut_file1)
    ts1, channel_dict1, spikeparam1 = _format_spikes(tetrode_file1)
    # ts, channel_dict spikeparam

with open(session2_cut_path, 'r') as cut_file2, open(session2_tet_path, 'rb') as tetrode_file2:
    cut_data2 = _read_cut(cut_file2)
    ts2, channel_dict2, spikeparam2 = _format_spikes(tetrode_file2)
    # ts, channel_dict spikeparam

# Make dictionaries for core classes

sample_length1 =  spikeparam1['duration']
sample_rate1 = spikeparam1['sample_rate']

session_dict1 = {
    'spike_times': ts1.squeeze().tolist(),
    'cluster_labels': cut_data1
}
session_dict1.update(channel_dict1)

sample_length2 =  spikeparam2['duration']
sample_rate2 = spikeparam2['sample_rate']

session_dict2 = {
    'spike_times': ts2.squeeze().tolist(),
    'cluster_labels': cut_data2,
}
session_dict2.update(channel_dict2)

assert sample_length1 == sample_length2
assert sample_rate1 == sample_rate2

study_dict = {
    'sample_length': sample_length1,
    'sample_rate': sample_rate1,
    'animal_ids': []
}

animal_dict = {
    'id': '0',
}

animal_dict[0] = session_dict1
animal_dict[1] = session_dict2

# ============================================================================ #

# get stuff for waveform module
waveform = session_dict1['ch4'][750]
sample_rate = sample_rate1 # in Hz
time_step = 10e3/sample_rate # in mliliseconds

# ============================================================================ #

# get stuff for spike module
spike = {}
for channel in range(1,5):
    spike[f'ch{channel}'] = session_dict1[f'ch{channel}'][750]

# ============================================================================ #

# get stuff for unit module


