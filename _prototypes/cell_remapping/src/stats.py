import os, sys
import copy

# data manipulation
import numpy as np
import pandas as pd

# file saving/loading/naming
from openpyxl import load_workbook
import re

# stats
from scipy import stats, ndimage

# settings based
from skimage.measure import block_reduce

# unused/relevant code commented out
from scipy.spatial.distance import cdist
from pyemd import emd
import itertools
import matplotlib.pyplot as plt
import cv2
import xlsxwriter

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

# from neuroscikit 
from library.hafting_spatial_maps import SpatialSpikeTrain2D
from _prototypes.cell_remapping.src.rate_map_plots import plot_obj_remapping, plot_regular_remapping, plot_fields_remapping, plot_shuffled_regular_remapping, plot_matched_sesssion_waveforms
from _prototypes.cell_remapping.src.wasserstein_distance import sliced_wasserstein, single_point_wasserstein, pot_sliced_wasserstein, compute_centroid_remapping, _get_ratemap_bucket_midpoints
from _prototypes.cell_remapping.src.masks import make_object_ratemap, flat_disk_mask, generate_grid, _sample_grid
from library.maps import map_blobs

# unused/commented out
from scripts.batch_map.batch_map import batch_map 
from library.shuffle_spikes import shuffle_spikes

# SETTINGS FILE
from _prototypes.cell_remapping.src.settings import obj_output, centroid_output, tasks, session_comp_categories, regular_output, context_output, variations, temporal_output, context_temporal_output

# project specific naming conventions
from _prototypes.cell_remapping.src.MEC_naming import MEC_naming_format, extract_name_mec
from _prototypes.cell_remapping.src.LC_naming import LC_naming_format, extract_name_lc
from scripts.batch_map.LEC_naming import LEC_naming_format, extract_name_lec



def compute_rate_change(prev_spatial_spike_train, curr_spatial_spike_train):
    prev_duration = prev_spatial_spike_train.session_metadata.session_object.get_spike_data()['spike_cluster'].duration
    curr_duration = curr_spatial_spike_train.session_metadata.session_object.get_spike_data()['spike_cluster'].duration
    curr_spike_times = curr_spatial_spike_train.spike_times
    prev_spike_times = prev_spatial_spike_train.spike_times
    curr_fr_rate = len(curr_spike_times) / prev_duration
    prev_fr_rate = len(prev_spike_times) / curr_duration
    fr_rate_ratio = curr_fr_rate / prev_fr_rate
    fr_rate_change = curr_fr_rate - prev_fr_rate

    return fr_rate_ratio, fr_rate_change

def get_max_matched_cell_count(animal):
    try:
        max_matched_cell_count = max(list(map(lambda x: max(animal.sessions[x].get_cell_data()['cell_ensemble'].get_label_ids()), animal.sessions)))
    except:
        max_matched_cell_count = 0
        for x in animal.sessions:
            cell_label_ids = animal.sessions[x].get_cell_data()['cell_ensemble'].get_label_ids() 
            nmb_matched = len(cell_label_ids)
            if nmb_matched > max_matched_cell_count:
                max_matched_cell_count = nmb_matched
    return max_matched_cell_count
        
def compute_cumulative_blob_stats(label_map, rate_map):
    val_r, val_c = np.where(label_map == 1)
    cumulative_coverage = len(np.where(label_map != 0)[0])/len(np.where(~np.isnan(rate_map))[0])
    cumulative_area = len(np.where(label_map == 1)[0])
    cumulative_rate = np.sum(rate_map[val_r, val_c])
    return cumulative_coverage, cumulative_area, cumulative_rate
          
def compute_modified_zscore(x, ref_dist):
    # Compute the median of the data
    median = np.median(ref_dist)

    # Compute the Median Absolute Deviation (MAD)
    mad = stats.median_abs_deviation(ref_dist, scale='normal')

    # Compute the Modified Z-score, adjusting for division by zero
    modified_zscore = 0.6745 * (x - median) / (mad if mad else 1)

    return modified_zscore, median, mad   

                        
def get_ref_ratio_stats(ref_rate_ratio_dist, fr_rate_ratio):
    fr_ratio_mean = np.mean(ref_rate_ratio_dist)
    fr_ratio_std = np.std(ref_rate_ratio_dist)
    fr_ratio_z = (fr_rate_ratio - fr_ratio_mean) / (fr_ratio_std)
    return fr_ratio_mean, fr_ratio_std, fr_ratio_z
                        
def get_ref_change_stats(ref_rate_change_dist, fr_rate_change):
    fr_change_mean = np.mean(ref_rate_change_dist)
    fr_change_std = np.std(ref_rate_change_dist)
    fr_change_z = (fr_rate_change - fr_change_mean) / (fr_change_std)
    return fr_change_mean, fr_change_std, fr_change_z
                    
def get_vector_from_map(map_to_use, arena_size, y, x, obj_y, obj_x, mode):
    height_bucket_midpoints, width_bucket_midpoints = _get_ratemap_bucket_midpoints(arena_size, y, x)
                                        
    if mode == 'field':
        
        centroids = map_to_use
        r, c = centroids[0]
        r = height_bucket_midpoints[int(np.round(r))]
        c = width_bucket_midpoints[int(np.round(c))]
    elif mode == 'whole':
        # find row col of peak firing rate bin
        r, c = np.where(map_to_use == np.nanmax(map_to_use))
        r = height_bucket_midpoints[r[0]]
        c = width_bucket_midpoints[c[0]]
    elif mode == 'spike_density':
        # avg across all points
        r = np.mean(map_to_use[0])
        c = np.mean(map_to_use[1])
    elif mode == 'binary':
        r = np.mean(map_to_use[:,0])
        c = np.mean(map_to_use[:,1])
        r = height_bucket_midpoints[int(np.round(r))]
        c = width_bucket_midpoints[int(np.round(c))]
    elif mode =='centroid':
        r, c = map_to_use
        r = height_bucket_midpoints[int(np.round(r))]
        c = width_bucket_midpoints[int(np.round(c))]
                            

    mag = np.linalg.norm(np.array((obj_y, obj_x)) - np.array((r, c)))
    pt1 = (obj_y, obj_x)
    pt2 = (r, c)
    angle = np.degrees(np.math.atan2(np.linalg.det([pt1,pt2]),np.dot(pt1,pt2)))

    return mag, angle, pt1, pt2


def get_reference_dist_stats(empirical, ref_dist):
    ref_dist_mean = np.mean(ref_dist)
    ref_dist_std = np.std(ref_dist)
    spike_dens_z_score = (empirical - ref_dist_mean) / (ref_dist_std)
    spike_dens_mod_z_score, median, mad = compute_modified_zscore(empirical, ref_dist)
    return  ref_dist_mean, ref_dist_std, spike_dens_z_score, spike_dens_mod_z_score, median, mad


def get_rate_stats(prev_pts, prev_t, curr_pts, curr_t, prev_duration=None, curr_duration=None):    
    use_duration = False
    if prev_duration is not None or curr_duration is not None:
        assert prev_duration is not None and curr_duration is not None, 'Must provide both durations or neither'
        use_duration = True
    if not use_duration:
        prev_fr_rate = len(prev_pts) / (prev_t[-1] - prev_t[0])
        curr_fr_rate = len(curr_pts) / (curr_t[-1] - curr_t[0])
    else:
        prev_fr_rate = len(prev_pts) / prev_duration
        curr_fr_rate = len(curr_pts) / curr_duration
    fr_rate_ratio = curr_fr_rate / prev_fr_rate
    fr_rate_change = curr_fr_rate - prev_fr_rate
    return prev_fr_rate, curr_fr_rate, fr_rate_ratio, fr_rate_change



# move to remapping or stats
def wasserstein_quantile(true_distance, random_distances):
    random_samples = len(random_distances)
    ecdf = stats.cumfreq(random_distances, numbins=random_samples)
    cumulative_counts = ecdf.cumcount / np.max(ecdf.cumcount)
    quantile = np.interp(true_distance, ecdf.lowerlimit + np.linspace(0, ecdf.binsize*ecdf.cumcount.size, ecdf.cumcount.size),cumulative_counts)
    return quantile
