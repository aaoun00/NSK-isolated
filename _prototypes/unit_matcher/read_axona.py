import os, sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.study_space import Session
from x_io.rw.axona.batch_read import make_study

def read_sequential_sessions(dir1, settings_dict: dict, dir2=None):
    """
    Can input one folder with both sessions inside or two folders (one for each session)
    """
    if dir2 == None:
        assert os.path.isdir(dir1), 'input path is not a folder'
        study = make_study(dir1, settings_dict)
    else:
        assert os.path.isdir(dir1) and os.path.isdir(dir2), 'input paths are not folders'
        study = make_study([dir1, dir2], settings_dict)

    study.make_animals()
    animals = study.animals
    assert len(animals) == 1
    assert len(animals[0].sessions) == 2
    session_1  = animals[0].sessions['session_1']
    session_2  = animals[0].sessions['session_2']

    return session_1, session_2 


def temp_read_cut(cut_file):
    """This function takes a pre-opened cut file and updates the list of the neuron numbers in the file."""
    cut_values = None
    extract_cut = False
    header_data = []
    for line in cut_file:
        if not extract_cut:
            header_data.append(line)
        if 'Exact_cut' in line:  # finding the beginning of the cut values
            extract_cut = True
        if extract_cut:  # read all the cut values
            cut_values = str(cut_file.readlines())
            for string_val in ['\\n', ',', "'", '[', ']']:  # removing non base10 integer values
                cut_values = cut_values.replace(string_val, '')
            cut_values = [int(val) for val in cut_values.split()]
        else:
            continue
    if cut_values is None:
        raise ValueError('Either file does not exist or there are no cut values found in cut file.')
        # cut_values = np.asarray(cut_values)
    return cut_values, header_data
