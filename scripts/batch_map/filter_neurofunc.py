import tkinter as tk
from tkinter import filedialog
import time
import os
import sys
import pickle
import pandas as pd
import numpy as np
import re

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

# project specific naming conventions
from _prototypes.cell_remapping.src.MEC_naming import MEC_naming_format, extract_name_mec
from _prototypes.cell_remapping.src.LC_naming import LC_naming_format, extract_name_lc
from scripts.batch_map.LEC_naming import LEC_naming_format, extract_name_lec

pd.options.mode.chained_assignment = None


def _check_single_format(filename, format, fxn):
    print(filename, format, fxn)
    if re.match(str(format), str(filename)) is not None:
        return fxn(filename)
    
def _get_ses_neuron_count(ses, df):
    return len(df[df['Session'] == ses])

def _get_ses_row_positions(ses,df):
    return df[df['Session'] == ses].index.to_numpy()

def _isValidRow(row, settings):
    valid = True
    if row['spike_count'] < settings['spike_count_lower'] or row['spike_count'] > settings['spike_count_upper']:
        valid = False
    if row['iso_dist'] < settings['iso_dist']:
        valid = False
    if row['ISI_mean'] < settings['ISI_mean']:
        valid = False
    if row['firing_rate'] > settings['firing_rate']:
        valid = False

    return valid

def _add_single_row(pos, valid_ses_dict, valid_ses, df, headers, settings, included_df, excluded_df):
    row = df.loc[pos]

    row['Name'] = valid_ses_dict[valid_ses]['Name']
    row['Depth'] = int(valid_ses_dict[valid_ses]['Depth'])
    row['Date'] = int(valid_ses_dict[valid_ses]['Date'])
    row[settings['type']] = valid_ses_dict[valid_ses][settings['type']]

    row = row[headers]
    if _isValidRow(row, settings):
        included_df.loc[len(included_df.index)] = row
    else:
        excluded_df.loc[len(excluded_df.index)] = row

    return included_df, excluded_df

def _add_valid_sessions(valid_ses, valid_ses_dict, df, headers, settings, included_df, excluded_df):
    row_positions = _get_ses_row_positions(valid_ses, df)
    list(map(lambda pos: _add_single_row(pos, valid_ses_dict, valid_ses, df, headers, settings, included_df, excluded_df), row_positions))
    return included_df, excluded_df

def _visit_depth_date(depth, date, date_depth_dict, visited_depths, valid_ses_dict, ses_dict, df):

    ses_on_same_day = date_depth_dict[date][depth]

    neuron_counts = list(map(lambda x: _get_ses_neuron_count(x, df), ses_on_same_day))
    chosen_ses = ses_on_same_day[np.argmax(neuron_counts)]

    if depth not in visited_depths or (visited_depths[depth] % 7 == 0 and visited_depths[depth] != 0):
        visited_depths[depth] = 0 
        valid_ses_dict[chosen_ses] = ses_dict[chosen_ses]
    else:
        visited_depths[depth] += 1
            
    return visited_depths, valid_ses_dict

def _iterate_depths(date, date_depth_dict, visited_depths, valid_ses_dict, ses_dict, df):
    list(map(lambda depth: _visit_depth_date(depth, date, date_depth_dict, visited_depths, valid_ses_dict,ses_dict,  df), date_depth_dict[date]))
    return visited_depths, valid_ses_dict


def filter_neurofunc_unmatched_object(file_dir, settings):
    xls = pd.read_excel(file_dir, sheet_name=None)

    try:
        df = xls.pop('Summary')
    except:
        df = xls.pop('Sheet1')

    headers = list(df.columns)
    if 'Unnamed: 0' == headers[0]:
        headers = headers[1:]
    if settings['type'] == 'Object':
        headers.insert(3, 'Object')
    elif settings['type'] == 'Odor':
        headers.insert(3, 'Odor')
    headers.insert(3, 'Depth')
    headers.insert(3, 'Date')
    headers.insert(3, 'Name')

    included_df = pd.DataFrame(columns=headers)
    excluded_df = pd.DataFrame(columns=headers)

    all_sessions = df['Session'].to_numpy()
    unique_sessions = np.unique(all_sessions)

    ses_dict = {}
    # ses_order = []
    date_order = []
    date_depth_dict = {}

    # for i, row in df.iterrows():
    for i, ses in iter(enumerate(unique_sessions)):

        # ses = row['Session']
        ses_dict[ses] = {}

        # if '-CAGE-' in ses:
        #     continue

        fname = ses

        if settings['naming_type'] == 'LEC':
            group, name = extract_name_lec(fname)
            formats = LEC_naming_format[group][name][settings['type']]
        elif settings['naming_type'] == 'MEC':
            name = extract_name_mec(fname)
            formats = MEC_naming_format
        elif settings['naming_type'] == 'LC':
            name = extract_name_lc(fname)
            formats = LC_naming_format

        for format in list(formats.keys()):
            checked = _check_single_format(fname, format, formats[format])
            if checked is not None:
                break
            else:
                continue
                
        stim, depth, name, date = checked

        for format in list(formats.keys()):
            checked = _check_single_format(ses, format, formats[format])
            if checked is not None:
                break
            else:
                continue

        print(checked)

        assert len(checked) == 4 
        assert not isinstance(checked[0], type(None)) and not isinstance(checked[0], list)
        stim, depth, name, date = checked

        ses_dict[ses]['Name'] = name
        ses_dict[ses]['Depth'] = depth 
        ses_dict[ses]['Date'] = date
        ses_dict[ses][settings['type']] = stim

        date = pd.to_datetime(date,format='%Y%m%d').date()

        if date not in date_depth_dict:
            date_depth_dict[date] = {}
        
        if depth not in date_depth_dict[date]:
            date_depth_dict[date][depth] = []

        date_order.append(date)
        date_depth_dict[date][depth].append(ses)

    date_order = np.sort(date_order)

    visited_depths = {}
    valid_ses_dict = {}

    list(map(lambda date: _iterate_depths(date, date_depth_dict, visited_depths, valid_ses_dict, ses_dict, df), date_order))

    list(map(lambda valid_ses: _add_valid_sessions(valid_ses, valid_ses_dict, df, headers, settings, included_df, excluded_df), list(valid_ses_dict.keys())))
 

    save_path = file_dir.split('.xlsx')[0] + '_filtered.xlsx'
    with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
        included_df = included_df.sort_values(['Session', 'Name', 'Depth', 'Date', 'Tetrode', 'Cell ID'])
        included_df.to_excel(writer, sheet_name="Included", index=True)

        excluded_df = excluded_df.sort_values(['Session', 'Name', 'Depth', 'Date', 'Tetrode', 'Cell ID'])
        excluded_df.to_excel(writer, sheet_name="Excluded", index=True)
        
    print('saved at ' + save_path)

if __name__ == '__main__':

    settings = {
        'type': 'object', # 'odor
        'spike_count_lower': 100,
        'spike_count_upper': 30000,
        'iso_dist': 7.5,
        'ISI_mean': 1,
        'firing_rate': 80,
        'naming_type': 'LEC'
    }

    start_time = time.time()
    root = tk.Tk()
    root.withdraw()
    file_dir = filedialog.askopenfilename(parent=root,title='Please select the csv file.')

    filter_neurofunc_unmatched_object(file_dir, settings)

