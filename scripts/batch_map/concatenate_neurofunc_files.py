import tkinter as tk
from tkinter import filedialog
import time
import os
import sys
import pickle
import pandas as pd
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from x_io.rw.axona.batch_read import make_study

pd.options.mode.chained_assignment = None

def concatenate_neurofunc(data_dir, settings):

    isFirst = True

    for file in os.listdir(data_dir):
        if '.xlsx' in file:
            print(file)
            fp = os.path.join(data_dir, file)

            xls = pd.read_excel(fp, sheet_name=None)
            df = xls.pop('Summary')

            if isFirst:

                concat_df = df

                isFirst = False
            else:
                if set(df.columns) != set(concat_df.columns):
                    min_col_count = min(len(df.columns),len(concat_df.columns))
                    max_col_count = max(len(df.columns),len(concat_df.columns))
                    less_col_df = np.argmin([len(df.columns),len(concat_df.columns)])
                    more_col_df = np.argmax([len(df.columns),len(concat_df.columns)])
                    df_to_extend = [df,concat_df][less_col_df]
                    df_to_use = [df,concat_df][more_col_df]

                    for i in range(max_col_count - min_col_count):
                        col_name = list(df_to_use.columns)[min_col_count + i]
                        assert col_name not in df_to_extend
                        df_to_extend[col_name] = ""

                concat_df = pd.concat((concat_df, df))

    if settings['output_file_name'] is None:
        save_path = data_dir + '/combined.xlsx'
    else:
        save_path = data_dir + '/' + settings['output_file_name'] + '.xlsx'

    with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
        concat_df.to_excel(writer, sheet_name="Summary", index=True)
        
    print('saved at ' + save_path)


if __name__ == '__main__':

    animal = {'animal_id': '001', 'species': 'mouse', 'sex': 'F', 'age': 1, 'weight': 1, 'genotype': 'type', 'animal_notes': 'notes'}
    devices = {'axona_led_tracker': True, 'implant': True}
    implant = {'implant_id': '001', 'implant_type': 'tetrode', 'implant_geometry': 'square', 'wire_length': 25, 'wire_length_units': 'um', 'implant_units': 'uV'}

    session_settings = {'channel_count': 4, 'animal': animal, 'devices': devices, 'implant': implant}

    """ FOR YOU TO EDIT """
    settings = {'ppm': 511, 'session':  session_settings, 'smoothing_factor': 3, 'useMatchedCut': True}
    """ FOR YOU TO EDIT """

    settings['output_file_name'] = None
    # do not include file extension or '/'
    # will go directly under data folder you select when prompted

    """ FOR YOU TO EDIT """

    start_time = time.time()
    root = tk.Tk()
    root.withdraw()
    data_dir = filedialog.askdirectory(parent=root,title='Please select a data directory.')    

    concatenate_neurofunc(data_dir, settings)