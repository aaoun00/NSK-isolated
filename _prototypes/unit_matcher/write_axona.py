import os, sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from _prototypes.unit_matcher.read_axona import temp_read_cut

def write_cut(cut_file, cut_data, header_data):
    """This function takes a cut file and updates the list of the neuron numbers in the file."""
    extract_cut = False
    with open(cut_file, 'w') as open_cut_file:
        for line in header_data:
            open_cut_file.writelines(line)
            if 'Exact_cut' in line:  # finding the beginning of the cut values
                extract_cut = True
            if extract_cut: # start write to cut file
                open_cut_file.write(" ".join(map(str, cut_data)))
    open_cut_file.close()
    print('Wrote matched cut file ' + str(cut_file))

def format_new_cut_file_name(old_path):
    fp_split = old_path.split('.cut')
    new_fp = fp_split[0] + r'_matched.cut' 
    assert new_fp != old_path

    return new_fp

def get_current_cut_data(cut_session):
    with open(cut_session, 'r') as open_cut_file:
        cut_data, header_data =  temp_read_cut(open_cut_file)
    return cut_data, header_data

def apply_remapping(cut_data, map_dict: dict):
    """
    Input is a dictionary mapping change in label from session 1 to session 2
    If label is None, unit has no match in other sessions

    Session 2 unit labels will be changed to match session 1 unit labels

    e.g. {5:4, 6:2, 7:1}

    If unit in session 1 is unmatched in session 2, value of key will be None e.g. {2: None, 1:None} and no change will be made to units in session 2

    If unit in session 2 is unmatchd in session 1, key of value will be 'None' {'None': 2, 'None': 5}

    Only ever write to cut file 2


    """
    all_labels = np.unique(cut_data)
    for label in all_labels:
        if label not in map_dict:
            map_dict[label] = 0
    new_cut_data = list(map(map_dict.get, cut_data))
    # print(new_cut_data)
    # idx = np.where(new_cut_data != new_cut_data)[0]
    return new_cut_data
