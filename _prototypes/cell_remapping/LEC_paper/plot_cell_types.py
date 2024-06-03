import os, sys
import numpy as np
import pandas as pd
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
from _prototypes.cell_remapping.src.masks import make_object_ratemap, check_disk_arena
from library.maps.map_utils import _interpolate_matrix, disk_mask


def flat_disk_mask(rate_map):
    masked_rate_map = disk_mask(rate_map)
    masked_rate_map.data[masked_rate_map.mask] = np.nan
    return  masked_rate_map.data


def _check_single_format(filename, format, fxn):
    if re.match(str(format), str(filename)) is not None:
        return fxn(filename)

def _find_subdir(folder_list,aid,date):
    """
    Finds the subdirectory of a given animal and date
    """
    matched_folders = []
    for folder in folder_list:
        if date in str(folder) and aid in str(folder):
            matched_folders.append(folder)
            # if aid in str(folder.split('/')[-2]):
            #     return folder

    # check folder has a file with .set extension
    for folder in matched_folders:
        for fle in os.listdir(folder):
            if '.set' in fle:
                return folder 


def main(dict_path, output_folder_path, folder_list, settings_dict):
    ctype_dict = pd.read_pickle(dict_path)
    failed = []
    # for ctype in ['trace']:
    for ctype in ctype_dict:
        print(ctype)

        output_path = output_folder_path + '/' + str(ctype)
        
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        for group in ctype_dict[ctype]:
            print(group)

            prev_aid = None 
            prev_date = None
            prev_tetrode = None 
            prev_cell_id = None

            animal_cell_list =  ctype_dict[ctype][group]
            
            for animal in animal_cell_list:
                # print(animal)
                aid = animal[1]
                date = animal[3]
                tetrode = animal[4]
                cell_id = animal[5]

                save_path = output_path + '/' + str(aid) + '_' + str(date) + '_' + str(tetrode) + '_' + str(cell_id)
                

                settings_dict['single_tet'] = int(tetrode)
                settings_dict['allowed_sessions'] = ['session_1', 'session_2', 'session_3']
                
                if True:
                    if aid == prev_aid and date == prev_date:
                        assert tetrode != prev_tetrode or cell_id != prev_cell_id, 'Duplicate cell found'
                    else:
                        subdir = _find_subdir(folder_list,aid,date)
                        if subdir is not None:
                            print(subdir)
                            sub_study = make_study(subdir,settings_dict=settings_dict)
                            sub_study.make_animals()  
                            print('made study')    
                            print( sub_study.animals)

                            animal_obj = sub_study.animals[0]

                            ses_ratemaps = []
                            # ses_ratemap_objs = []
                            ses_ratemaps_raw = []
                            ses_ratemap_cells = []
                            angles = []

                            for ses_id in animal_obj.sessions:
                                print(ses_id)
                                session = animal_obj.sessions[ses_id]
                                pos_obj = session.get_position_data()['position']

                                tet_path = session.session_metadata.file_paths['tet']
                                fname = tet_path.split('/')[-1].split('.')[0]

                                if settings_dict['disk_arena']: 
                                    cylinder = True
                                else:
                                    cylinder, _ = check_disk_arena(fname)
                                    if not cylinder:
                                        print('Not cylinder for {}'.format(fname))
                                        
                                if settings_dict['naming_type'] == 'LEC':
                                    group, name = extract_name_lec(fname)
                                    formats = LEC_naming_format[group][name][settings_dict['type']]

                                for format in list(formats.keys()):
                                    checked = _check_single_format(fname, format, formats[format])
                                    if checked is not None:
                                        break
                                    else:
                                        continue
                                
                                angle, depth, name, date = checked
                                angles.append(angle)

                                for cell in session.get_cell_data()['cell_ensemble'].cells:  
                                    if int(cell.cluster.cluster_label) == int(cell_id):
                                        print(cell_id)

                                        spatial_spike_train = session.make_class(SpatialSpikeTrain2D, 
                                    {   'cell': cell, 'position': pos_obj, 'speed_bounds': (settings_dict['speed_lowerbound'], settings_dict['speed_upperbound'])})   

                        
                                        rate_obj = spatial_spike_train.get_map('rate')
                                        rate_map, rate_map_raw = rate_obj.get_rate_map(new_size=settings_dict['ratemap_dims'][0])

                                        if cylinder:
                                            curr = flat_disk_mask(rate_map)
                                            
                                        ses_ratemaps.append(rate_map) 
                                        ses_ratemaps_raw.append(rate_map_raw)
                                        # ses_ratemap_objs.append(rate_obj)
                                        ses_ratemap_cells.append(cell)

                            fig = plt.figure(figsize=(4 * len(ses_ratemaps),4))
                            gs_main = gridspec.GridSpec(1, len(ses_ratemaps))
                            

                            for i, ratemap in enumerate(ses_ratemaps):
                                angle = angles[i]
                                print('plotting ratemap: ' + str(i))
                                gs_sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[i], height_ratios=[12,2]) 

                                # ax = fig.add_subplot(1,len(ses_ratemaps),i+1)
                                ax = fig.add_subplot(gs_sub[0])
                                img = ax.imshow(ratemap, cmap='jet', aspect='equal')

                                # firing_rate = np.sum(ses_ratemaps_raw[i][~np.isnan(ses_ratemaps_raw[i])] / len(ses_ratemaps_raw[i][~np.isnan(ses_ratemaps_raw[i])]))
                                firing_rate = len(ses_ratemap_cells[i].event_times) / ses_ratemap_cells[i].event_times[-1] 
                                firing_rate = round(firing_rate,2)
                                # firing_rate2 = round(firing_rate2,2)
                                # angle_title = 'Angle: ' + str(angle) + ', Rate: ' + str(firing_rate) + ' Hz'
                                rate_title = str(firing_rate) + ' Hz' 
                                # + ', ' + str(firing_rate2) + ' Hz'
                                ax.set_title(rate_title, fontweight='bold')

                                if angle != 'NO':
                                    angle = int(angle)

                                    _, obj_loc = make_object_ratemap(angle, new_size=settings_dict['ratemap_dims'][0])
                                    
                                    if angle == 0:
                                        obj_loc['x'] += .5
                                        obj_loc['y'] += 2
                                    elif angle == 90:
                                        obj_loc['y'] += .5
                                        obj_loc['x'] -= 2
                                    elif angle == 180:
                                        obj_loc['x'] -= .5
                                        obj_loc['y'] -= 2
                                    elif angle == 270:
                                        obj_loc['y'] -= .5
                                        obj_loc['x'] += 2
                                    ax.plot(obj_loc['x'], obj_loc['y'], 'k', marker='o', markersize=20)

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
                            
                            title = str(ctype) + ' cell - animal: ' + str(aid) + ', date: ' + str(date) + ', depth: ' + str(depth) + ', tetrode: ' + str(tetrode) + ', unit: ' + str(cell_id)

                            # fig.suptitle(str(ctype) + ' cell ' + str(animal), fontweight='bold')
                            fig.suptitle(title)
                            fig.tight_layout()

                            # plt.show()
                            # stop()
                            # save to output folder
                            fig.savefig(save_path , dpi=360)
                            plt.close()
                            print(save_path)  
                            # stop()

                            prev_date = date
                            prev_cell_id = cell_id
                            prev_aid = aid 
                            prev_tetrode = tetrode
                # except:
                #     failed.append(animal)

    return failed


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
    settings_dict['naming_type'] = 'LEC'
    settings_dict['type'] = 'object'

    folder_path = r"C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit_test_data\NON_sample"

    output_folder_path = r"C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit_test_data\NON_sample\output"

    dict_path = r"C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit\_prototypes\cell_remapping\LEC_cell_types.pkl"

    # list all folders in folder_path at any level  
    folder_list = [x[0] for x in os.walk(folder_path)]
    folder_list = [x.replace('\\', '/') for x in folder_list]


    failed = main(dict_path, output_folder_path, folder_list, settings_dict)

    # print('Failed IDs')

    # print(failed)