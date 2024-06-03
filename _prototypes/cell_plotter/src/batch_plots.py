import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import re

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from _prototypes.cell_plotter.src.plots import plot_cell_waveform, plot_cell_rate_map

def check_disk_arena(path):
    variations = [r'cylinder', r'round', r'circle']
    var_bool = []
    true_var = None
    for var in variations:
        if re.search(var, path) is not None:
            var_bool.append(True)
            true_var = var
        else:
            var_bool.append(False)
    # if re.search(r'cylinder', path) is not None or re.search(r'round', path) is not None:
    if np.array(var_bool).any() == True:
        cylinder = True
    else:
        cylinder = False

    return cylinder, true_var

def batch_plots(study, settings_dict, data_dir, output_dir=None):

    if output_dir is None:
        output_path = data_dir + '/output/'
    else:
        output_path = output_dir + '/output/'
        
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for animal in study.animals:

        if settings_dict['outputStructure'] == 'nested' or settings_dict['outputStructure'] == 'sequential':
            output_path = data_dir + '/output/' + str(animal.animal_id) + '/'
        elif settings_dict['outputStructure'] =='single':
            split = str(animal.animal_id).split('tet')
            animal_id = split[0]
            tet_id = split[-1]
            output_path = data_dir + '/output/' 
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        

        for session_key in animal.sessions:
            session = animal.sessions[session_key]

            if settings_dict['outputStructure'] == 'nested' or settings_dict['outputStructure'] == 'sequential':
                output_path = data_dir + '/output/' + str(animal.animal_id) + '/' + str(session_key) + '/'
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

            for cell in session.get_cell_data()['cell_ensemble'].cells:

                if settings_dict['plotCellWaveforms']:
                    
                    if settings_dict['outputStructure'] == 'nested' or settings_dict['outputStructure'] == 'sequential':
                        output_path = data_dir + '/output/' + str(animal.animal_id) + '/' + str(session_key) + '/waveforms/'
                    elif settings_dict['outputStructure'] == 'single':
                        output_path = data_dir + '/output/waveforms/'
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    plot_cell_waveform(cell, output_path)

                if settings_dict['plotCellRatemap']:

                    path = session.session_metadata.file_paths['tet'].lower()
                    # Check if cylinder
                    cylinder, _ = check_disk_arena(path)

                    if settings_dict['outputStructure'] == 'nested' or settings_dict['outputStructure'] == 'sequential':
                        output_path = data_dir + '/output/' + str(animal.animal_id) + '/' + str(session_key) + '/ratemaps/'
                    elif settings_dict['outputStructure'] == 'single':
                        output_path = data_dir + '/output/ratemaps/'
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    plot_cell_rate_map(cell, cylinder, output_path)

