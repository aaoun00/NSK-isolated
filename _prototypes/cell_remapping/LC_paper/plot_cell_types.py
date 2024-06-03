import os, sys
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import time
import traceback
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, mannwhitneyu, wilcoxon, ttest_rel, ttest_ind
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import ColorConverter
import re
import matplotlib.gridspec as gridspec


PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
print(PROJECT_PATH)

from library.study_space import Session, Study, SpatialSpikeTrain2D
from x_io.rw.axona.batch_read import make_study
from scripts.batch_map.LEC_naming import LEC_naming_format, extract_name_lec
from _prototypes.cell_remapping.src.stats import get_max_matched_cell_count
from _prototypes.cell_remapping.src.utils import check_disk_arena, read_data_from_fname
from library.maps.map_utils import _interpolate_matrix, disk_mask


def flat_disk_mask(rate_map):
    masked_rate_map = disk_mask(rate_map)
    masked_rate_map.data[masked_rate_map.mask] = np.nan
    return  masked_rate_map.data


def main(study, settings_dict, output_path):

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

            
    for animal in study.animals:

        tetrode = animal.animal_id.split('_tet')[-1]

        settings_dict['single_tet'] = int(tetrode)
        settings_dict['allowed_sessions'] = ['session_1', 'session_2', 'session_3']

        max_matched_cell_count = get_max_matched_cell_count(animal)

        for k in range(int(max_matched_cell_count)):

            cell_label = k + 1
            print('Cell ' + str(cell_label))

            if settings_dict['ses_limit'] is None:
                ses_limit = len(list(animal.sessions.keys()))
            else:
                if settings_dict['ses_limit'] >= len(list(animal.sessions.keys())):
                    ses_limit = len(list(animal.sessions.keys()))
                else:
                    ses_limit = settings_dict['ses_limit']

            ses_ratemaps = []
            # ses_ratemap_objs = []
            # ses_ratemaps_raw = []
            ses_ratemap_cells = []
            # angles = []

                   
          
            # for every session
            # for i in range(len(list(animal.sessions.keys()))):
            for i in range(ses_limit):
                # seskey = 'session_' + str(i+1)
                seskey = list(animal.sessions.keys())[i]
                print(seskey)
                ses = animal.sessions[seskey]
                path = ses.session_metadata.file_paths['tet']
                fname = path.split('/')[-1].split('.')[0]


                if settings_dict['disk_arena']: 
                    cylinder = True
                else:
                    cylinder, _ = check_disk_arena(fname)
                    
                stim, depth, name, date = read_data_from_fname(fname, settings_dict['naming_type'], settings_dict['type'])

                aid = name 

                for cell in ses.get_cell_data()['cell_ensemble'].cells:
                    cell_id = int(cell.cluster.cluster_label)

                    if cell_id == cell_label:
                        save_path = output_path + '/' + str(aid) + '_' + str(date) + '_' + str(tetrode) + '_' + str(cell_id)

                        pos_obj = ses.get_position_data()['position']

                    
                        spatial_spike_train = ses.make_class(SpatialSpikeTrain2D, 
                        {   'cell': cell, 'position': pos_obj, 'speed_bounds': (settings_dict['speed_lowerbound'], settings_dict['speed_upperbound'])})   

                            
                        rate_obj = spatial_spike_train.get_map('rate')
                        rate_map, rate_map_raw = rate_obj.get_rate_map(new_size=settings_dict['ratemap_dims'][0])

                        if cylinder:
                            rate_map = flat_disk_mask(rate_map)
                            # rate_map_raw = flat_disk_mask(rate_map_raw)
                                                
                        ses_ratemaps.append(rate_map) 
                        # ses_ratemaps_raw.append(rate_map_raw)
                        # ses_ratemap_objs.append(rate_obj)
                        ses_ratemap_cells.append(cell)

                        fig = plt.figure(figsize=(4 * len(ses_ratemaps),4))
                        gs_main = gridspec.GridSpec(1, len(ses_ratemaps))
                                

            for i, ratemap in enumerate(ses_ratemaps):
                print('plotting ratemap: ' + str(i))
                gs_sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[i], height_ratios=[12,2]) 
                    
                ax = fig.add_subplot(gs_sub[0])
                img = ax.imshow(ratemap, cmap='jet', aspect='equal')

                firing_rate = len(ses_ratemap_cells[i].event_times) / ses_ratemap_cells[i].event_times[-1] 
                firing_rate = round(firing_rate,2)
                rate_title = str(firing_rate) + ' Hz' 
                ax.set_title(rate_title, fontweight='bold')

            
                fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

                ax2 = fig.add_subplot(gs_sub[1])
                waveforms = ses_ratemap_cells[i].signal
                for ch_id in range(4):
                    if ch_id != len(waveforms):
                        ch = waveforms[:,ch_id,:]
                        idx = np.random.choice(len(ch), size=200)
                        waves = ch[idx, :]
                        avg_wave = np.mean(ch, axis=0)

                        ax2.plot(np.arange(int(50*ch_id+5*ch_id),int(50*ch_id+5*ch_id+50),1), ch[idx,:].T, c='grey')
                        ax2.plot(np.arange(int(50*ch_id+5*ch_id),int(50*ch_id+5*ch_id+50),1), avg_wave, c='k', lw=2)
                                
                        ax2.set_xlim([-25,200])
                        ax2.spines['top'].set_visible(False)
                        ax2.spines['bottom'].set_visible(False)
                        ax2.spines['right'].set_visible(False)
                        ax2.spines['left'].set_visible(False)

                ax2.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False)
                        

                ax2.tick_params(
                    axis='y',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False)

                ax2.set_xticks([])
                ax2.set_yticks([])
                            
            title = 'animal: ' + str(aid) + ', date: ' + str(date) + ', depth: ' + str(depth) + ', tetrode: ' + str(tetrode) + ', unit: ' + str(cell_id)

            fig.suptitle(title)
            fig.tight_layout()

            fig.savefig(save_path , dpi=360)
            plt.close()
            print(save_path)  


if __name__ == '__main__':

    STUDY_SETTINGS = {

        'ppm': 485,  # EDIT HERE

        'smoothing_factor': 3, # EDIT HERE

        'useMatchedCut': True,  # EDIT HERE
    }

    # Switch devices to True/False based on what is used in the acquisition (to be extended for more devices in future)
    device_settings = {'axona_led_tracker': True, 'implant': True} 
    # Make sure implant metadata is correct, change if not, AT THE MINIMUM leave implant_type: tetrode
    implant_settings = {'implant_type': 'tetrode', 'implant_geometry': 'square', 'wire_length': 25, 'wire_length_units': 'um', 'implant_units': 'uV'}
    # WE ASSUME DEVICE AND IMPLANT SETTINGS ARE CONSISTENCE ACROSS SESSIONS
    # Set channel count + add device/implant settings
    SESSION_SETTINGS = {
        'channel_count': 4, # EDIT HERE, default is 4, you can change to other number but code will check how many tetrode files are present and set that to channel copunt regardless
        'devices': device_settings, # EDIT HERE
        'implant': implant_settings, # EDIT HERE
    }
    STUDY_SETTINGS['session'] = SESSION_SETTINGS
    settings_dict = STUDY_SETTINGS

    settings_dict['speed_lowerbound'] = 0 
    settings_dict['speed_upperbound'] = 99
    settings_dict['ratemap_dims'] = (32,32)
    settings_dict['disk_arena'] = True
    settings_dict['naming_type'] = 'LC'
    settings_dict['type'] = 'object'
    settings_dict['arena_size'] = None
    settings_dict['ses_limit'] = None
    settings_dict['hasObject'] = False

    start_time = time.time()
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(parent=root,title='Please select a data directory.')
    output_folder_path = filedialog.askdirectory(parent=root,title='Please select an output folder.')

    subdirs = np.sort([ f.path for f in os.scandir(folder_path) if f.is_dir() ])
    # sys.stdout = open(r'C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit\_prototypes\cell_remapping\testlogRadha.txt', 'w')
    for subdir in subdirs:
        try:
            study = make_study(subdir,settings_dict=settings_dict)
            study.make_animals()

            main(study, settings_dict, output_folder_path)
     
        except Exception:
            print(traceback.format_exc())
            print('DID NOT WORK FOR DIRECTORY ' + str(subdir))
    print('COMPLETED ALL FOLDERS')
    print('Total run time: ' + str(time.time() - start_time))
