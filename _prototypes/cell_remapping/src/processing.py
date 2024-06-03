import os, sys
import numpy as np
import pandas as pd
import re

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

# from neuroscikit 
from library.hafting_spatial_maps import SpatialSpikeTrain2D
from _prototypes.cell_remapping.src.rate_map_plots import plot_obj_remapping, plot_regular_remapping, plot_fields_remapping, plot_shuffled_regular_remapping, plot_matched_sesssion_waveforms
from _prototypes.cell_remapping.src.wasserstein_distance import pot_sliced_wasserstein, _get_ratemap_bucket_midpoints, single_point_wasserstein, compute_temporal_emd
from _prototypes.cell_remapping.src.masks import make_object_ratemap, flat_disk_mask, generate_grid, _sample_grid
from library.maps import map_blobs
from _prototypes.cell_remapping.src.utils import check_disk_arena, _sort_filter_centroids_by_field_size, _get_valid_weight_bins, _get_valid_midpoints, _get_spk_pts, scale_points, _downsample
from _prototypes.cell_remapping.src.stats import compute_rate_change
from _prototypes.cell_remapping.src.settings import session_comp_categories



def _aggregate_cell_info(animal, settings):
    ratemap_size = settings['ratemap_dims'][0]
    max_centroid_count = 0
    blobs_dict = {}
    # type_dict = {}
    # spatial_obj_dict = {}
    shuffled_ratemap_dict = {}
    shuffled_sample_dict = {}
    ratemap_session_groups = {}
    # for animal in study.animals:

    # get largest possible cell id
    for x in animal.sessions:
        print(animal.sessions[x].session_metadata.file_paths['cut'])
        print(animal.sessions[x].get_cell_data()['cell_ensemble'].get_label_ids())

    # max_matched_cell_count = len(animal.sessions[sorted(list(animal.sessions.keys()))[-1]].get_cell_data()['cell_ensemble'].cells)
    try:
        max_matched_cell_count = max(list(map(lambda x: max(animal.sessions[x].get_cell_data()['cell_ensemble'].get_label_ids()), animal.sessions)))
    except:
        max_matched_cell_count = 0
        for x in animal.sessions:
            cell_label_ids = animal.sessions[x].get_cell_data()['cell_ensemble'].get_label_ids() 
            nmb_matched = len(cell_label_ids)
            if nmb_matched > max_matched_cell_count:
                max_matched_cell_count = nmb_matched
    
    for k in range(int(max_matched_cell_count)):
        cell_label = k + 1
        prev_field_size_len = None 
        print('search2')
        print(animal.animal_id, max_matched_cell_count, animal.sessions.keys())
        for i in range(len(list(animal.sessions.keys()))):
            seskey = list(animal.sessions.keys())[i]
            # seskey = 'session_' + str(i+1)
            seskey_id = seskey.split('_')[-1]
            if seskey_id not in ratemap_session_groups:
                ratemap_session_groups[seskey_id] = {}
            ses = animal.sessions[seskey]
            ensemble = ses.get_cell_data()['cell_ensemble']
            pos_obj = ses.get_position_data()['position']
            path = ses.session_metadata.file_paths['tet']
            fname = path.split('/')[-1].split('.')[0]
            # Check if cylinder
            if settings['disk_arena']: 
                cylinder = True
            else:
                cylinder, _ = check_disk_arena(fname)
            if cell_label in ensemble.get_cell_label_dict():
                cell = ensemble.get_cell_by_id(cell_label)
                # if 'spatial_spike_train' in cell.stats_dict['cell_stats']:
                #     spatial_spike_train = cell.stats_dict['cell_stats']['spatial_spike_train']
                # else:
                #     spatial_spike_train = ses.make_class(SpatialSpikeTrain2D, {'cell': cell, 'position': ses.get_position_data()['position']})
                
                # spatial_spike_train = ses.make_class(SpatialSpikeTrain2D, {'cell': cell, 'position': pos_obj})
                spatial_spike_train = ses.make_class(SpatialSpikeTrain2D, {'cell': cell, 'position': pos_obj, 'speed_bounds': (settings['speed_lowerbound'], settings['speed_upperbound'])})

                cell.stats_dict['spatial_spike_train'] = spatial_spike_train
                rate_map_obj = spatial_spike_train.get_map('rate')
                
                if settings['normalizeRate']:
                    rate_map, _ = rate_map_obj.get_rate_map(new_size = settings['ratemap_dims'][0])
                else:
                    _, rate_map = rate_map_obj.get_rate_map(new_size = settings['ratemap_dims'][0])
                assert rate_map.shape == (settings['ratemap_dims'][0], settings['ratemap_dims'][1]), 'Wrong ratemap shape {} vs settings shape {}'.format(rate_map.shape, (settings['ratemap_dims'][0], settings['ratemap_dims'][1]))
                
                if settings['downsample']:
                    rate_map = _downsample(rate_map, settings['downsample_factor'])
                if cylinder:
                    rate_map = flat_disk_mask(rate_map)
                id = str(animal.animal_id) + '_' + str(seskey) + '_' + str(cell.cluster.cluster_label)
                assert id not in blobs_dict, 'Duplicate cell id ' + str(id)

                ratemap_session_groups[seskey_id][cell_label] = [rate_map, spatial_spike_train]
                    
                if settings['hasObject'] or settings['runFields']:
                    image, n_labels, labels, centroids, field_sizes = map_blobs(spatial_spike_train, ratemap_size=ratemap_size, cylinder=cylinder, downsample=settings['downsample'], downsample_factor=settings['downsample_factor'])
                    if len(np.unique(labels)) == 1:
                        image, n_labels, labels, centroids, field_sizes = map_blobs(spatial_spike_train, ratemap_size=ratemap_size, cylinder=cylinder, downsample=settings['downsample'], downsample_factor=settings['downsample_factor'], nofilter=True)                        

                    labels, centroids, field_sizes = _sort_filter_centroids_by_field_size(rate_map, field_sizes, labels, centroids, spatial_spike_train.arena_size)
                    blobs_dict[id] = [image, n_labels, labels, centroids, field_sizes]
                    if prev_field_size_len is not None:
                        max_centroid_count = max(max_centroid_count, len(field_sizes) * prev_field_size_len)
                    else:
                        prev_field_size_len = len(field_sizes)
                # if (settings['runRegular'] and 'whole' in settings['rate_scores']) or (settings['runUniqueGroups'] and 'whole' in settings['unique_rate_scores']):
                #     print('drawing shuffled samples')
                #     row, col = np.where(~np.isnan(rate_map))
                #     # shuffled_samples = list(map(lambda x: _single_shuffled_sample(spatial_spike_train, settings), np.arange(settings['n_repeats'])))
                #     norm, raw = spatial_spike_train.get_map('rate').get_rate_map(new_size = settings['ratemap_dims'][0], shuffle=True, n_repeats=settings['n_repeats'])
                #     print(norm.shape, raw.shape, rate_map.shape)
                #     shuffled_samples = list(map(lambda x, y: _single_shuffled_sample(x, y, settings), norm, raw))
                #     print(np.array(shuffled_samples).shape)
                #     if cylinder:
                #         shuffled_samples = list(map(lambda x: flat_disk_mask(x), shuffled_samples))
                #     print('turning into valid weights')
                #     shuffled_ratemaps = list(map(lambda sample: np.array(list(map(lambda x, y: sample[x,y], row, col))), shuffled_samples))
                #     shuffled_ratemap_dict[id] = shuffled_ratemaps
                #     shuffled_sample_dict[id] = shuffled_samples[np.random.randint(0, settings['n_repeats'])]
                # spatial_obj_dict[id] = spatial_spike_train
                # type_dict[id] = []
            

    return max_centroid_count, blobs_dict, shuffled_ratemap_dict, shuffled_sample_dict, ratemap_session_groups


def aggregate_map_blobs(study, settings):
    animal_cell_info = {}
    animal_cell_ratemaps = {}
    for animal in study.animals:
        max_centroid_count, blobs_dict, _, _, ratemap_session_groups = _aggregate_cell_info(animal, settings)
        animal_id = animal.animal_id.split('_')[0]
        if animal_id not in animal_cell_ratemaps:
            animal_cell_ratemaps[animal_id] = {}
        animal_cell_info[animal.animal_id] = [max_centroid_count, blobs_dict]
        for ses_id, cell_info in ratemap_session_groups.items():
            if ses_id not in animal_cell_ratemaps[animal_id]:
                animal_cell_ratemaps[animal_id][ses_id] = {}
            for cell_label in cell_info:
                tetrode = animal.animal_id.split('tet')[-1]
                ky = str(tetrode) + '_' + str(cell_label)
                animal_cell_ratemaps[animal_id][ses_id][ky] = cell_info[cell_label]
    
    return animal_cell_info, animal_cell_ratemaps
    
def _animal_update_dict(animal_ref_dist, visited, animal_id):
    # if not visited add to dict + track visited 
    if animal_id not in animal_ref_dist:
        animal_ref_dist[animal_id] = {}
        visited[animal_id] = {}
    
    return animal_ref_dist, visited

def _session_update_dict(animal_ref_dist, visited, animal_id, ses_comp):
    # dictionary for different reference distributions
    if ses_comp not in animal_ref_dist[animal_id]:
        animal_ref_dist[animal_id][ses_comp] = {}
        visited[animal_id][ses_comp] = []
        animal_ref_dist[animal_id][ses_comp]['ref_whole'] = []
        animal_ref_dist[animal_id][ses_comp]['ref_spike_density'] = []
        animal_ref_dist[animal_id][ses_comp]['ref_temporal'] = []
        animal_ref_dist[animal_id][ses_comp]['ref_rate_ratio'] = []
        animal_ref_dist[animal_id][ses_comp]['ref_rate_change'] = []
        animal_ref_dist[animal_id][ses_comp]['ref_weights'] = []
    return animal_ref_dist, visited
                        


def collect_shuffled_ratemaps(animal_cell_ratemaps, settings):
    prev_ses_id = None
    animal_ref_dist = {}
    visited = {}
    
    for animal_id in animal_cell_ratemaps:

        animal_ref_dist, visited = _animal_update_dict(animal_ref_dist, visited, animal_id)

        # check if cell map to cell map remapping is in settings
        if settings['runRegular'] or settings['runTemporal']:
            for ses_id in animal_cell_ratemaps[animal_id]:
                if prev_ses_id is not None:
                        
                    # cell rate maps for session 1 and session 2
                    ses_1 = animal_cell_ratemaps[animal_id][prev_ses_id]
                    ses_2 = animal_cell_ratemaps[animal_id][ses_id]
                        
                    # dict id
                    ses_comp = str(prev_ses_id) + '_' + str(ses_id)

                    animal_ref_dist, visited = _session_update_dict(animal_ref_dist, visited, animal_id, ses_comp)

                    for cell_label_1 in ses_1:
                        for cell_label_2 in ses_2:
                            lbl_pair = [cell_label_1, cell_label_2]

                            if cell_label_1 != cell_label_2 and list(np.sort(lbl_pair)) not in visited[animal_id][ses_comp]: # DO NOT COMPARE SAME CELL, PONLY MAKIGN REF DIST
                                    
                                # print('comparing cell {} to cell {}'.format(cell_label_1, cell_label_2))
                                visited[animal_id][ses_comp].append(lbl_pair)
                                source_map = ses_1[cell_label_1][0]
                                target_map = ses_2[cell_label_2][0]
                                prev_spatial_spike_train = ses_1[cell_label_1][1]
                                curr_spatial_spike_train = ses_2[cell_label_2][1]
                                y, x = source_map.shape
                                assert source_map.shape == target_map.shape, "source and target maps are not the same shape"
                                    
                                # whole map (binned) and spike density (raw position) - map to map
                                if settings['runRegular']:

                                    # get fr rate valid bins
                                    source_weights, row_prev, col_prev = _get_valid_weight_bins(source_map)
                                    target_weights, row_curr, col_curr = _get_valid_weight_bins(target_map)
                                        
                                    prev_height_bucket_midpoints, prev_width_bucket_midpoints, coord_buckets_prev = _get_valid_midpoints(prev_spatial_spike_train.arena_size, y, x, row_prev, col_prev)
                                    curr_height_bucket_midpoints, curr_width_bucket_midpoints, coord_buckets_curr = _get_valid_midpoints(curr_spatial_spike_train.arena_size, y, x, row_curr, col_curr)
                                        
                                    prev_pts = _get_spk_pts(prev_spatial_spike_train)
                                    curr_pts = _get_spk_pts(curr_spatial_spike_train)

                                    # normalized fr rate
                                    source_weights = source_weights / np.sum(source_weights)
                                    target_weights = target_weights / np.sum(target_weights)
                                        
                                    if settings['normalizePos']:
                                        prev_height_bucket_midpoints = (prev_height_bucket_midpoints - np.min(prev_height_bucket_midpoints)) / (np.max(prev_height_bucket_midpoints) - np.min(prev_height_bucket_midpoints))
                                        prev_width_bucket_midpoints = (prev_width_bucket_midpoints - np.min(prev_width_bucket_midpoints)) / (np.max(prev_width_bucket_midpoints) - np.min(prev_width_bucket_midpoints))
                                        curr_height_bucket_midpoints = (curr_height_bucket_midpoints - np.min(curr_height_bucket_midpoints)) / (np.max(curr_height_bucket_midpoints) - np.min(curr_height_bucket_midpoints))
                                        curr_width_bucket_midpoints = (curr_width_bucket_midpoints - np.min(curr_width_bucket_midpoints)) / (np.max(curr_width_bucket_midpoints) - np.min(curr_width_bucket_midpoints))

                                    if settings['normalizePos']:
                                        curr_pts = scale_points(curr_pts)
                                        prev_pts = scale_points(prev_pts)

                                    # spike density
                                    spike_dens_wass = pot_sliced_wasserstein(prev_pts, curr_pts, n_projections=settings['n_projections'])
                                    animal_ref_dist[animal_id][ses_comp]['ref_spike_density'].append(spike_dens_wass)
                                        
                                    # whole map
                                    wass = pot_sliced_wasserstein(coord_buckets_prev, coord_buckets_curr, source_weights, target_weights, n_projections=settings['n_projections'])
                                    animal_ref_dist[animal_id][ses_comp]['ref_whole'].append(wass)
                                    animal_ref_dist[animal_id][ses_comp]['ref_weights'].append([source_weights, target_weights])

                                # temporal emd - map to map
                                if settings['runTemporal']:
                                    prev_spike_times = np.array(prev_spatial_spike_train.spike_times, dtype=np.float32)
                                    curr_spike_times = np.array(curr_spatial_spike_train.spike_times, dtype=np.float32)

                                    if settings['normalizeTime']:
                                        prev_spike_times = (prev_spike_times - np.min(prev_spike_times)) / (np.max(prev_spike_times) - np.min(prev_spike_times))
                                        curr_spike_times = (curr_spike_times - np.min(curr_spike_times)) / (np.max(curr_spike_times) - np.min(curr_spike_times))
                                        
                                    observed_emd = compute_temporal_emd(prev_spike_times, curr_spike_times, settings['temporal_bin_size'], settings['end_time'])
                                    animal_ref_dist[animal_id][ses_comp]['ref_temporal'].append(observed_emd)
                                    
                                fr_rate_ratio, fr_rate_change = compute_rate_change(prev_spatial_spike_train, curr_spatial_spike_train)

                                animal_ref_dist[animal_id][ses_comp]['ref_rate_ratio'].append(fr_rate_ratio)
                                animal_ref_dist[animal_id][ses_comp]['ref_rate_change'].append(fr_rate_change)
                            else:
                                pass
                                # if cell_label_1 == cell_label_2:
                                    # print('SAME CELL, omitting from ref dist {} {}'.format(cell_label_1, cell_label_2))
                                # else:
                                    # print('CELL PAIR ALREADY VISITED, e.g tet1_cell1 + tet2_cell2 == tet2_cell2 + tet1_cell1')
                prev_ses_id = ses_id

        # custom case for non sequential sessions, repetition of code above with chosen session pairs
        # not used, but also will override animal ref dist from 'regular' if both are True
        if settings['runUniqueGroups'] or settings['runUniqueOnlyTemporal']:
                
            for categ in session_comp_categories:

                prev_ses_id_categs = None 

                for ses_id_categs in session_comp_categories[categ]:

                    if prev_ses_id_categs is not None:
                        ses_1 = animal_cell_ratemaps[animal_id][prev_ses_id_categs]
                        ses_2 = animal_cell_ratemaps[animal_id][ses_id_categs]
                            
                        ses_comp = str(prev_ses_id_categs) + '_' + str(ses_id_categs)

                        if ses_comp not in animal_ref_dist[animal_id]:
                            animal_ref_dist[animal_id][ses_comp] = {}
                            animal_ref_dist[animal_id][ses_comp]['ref_whole'] = []
                            animal_ref_dist[animal_id][ses_comp]['ref_spike_density'] = []
                            animal_ref_dist[animal_id][ses_comp]['ref_temporal'] = []
                            animal_ref_dist[animal_id][ses_comp]['ref_rate_ratio'] = []
                            animal_ref_dist[animal_id][ses_comp]['ref_rate_change'] = []

                        for cell_label_1 in ses_1:
                            for cell_label_2 in ses_2:
                                if cell_label_1 != cell_label_2: # DO NOT COMPARE SAME CELL, PONLY MAKIGN REF DIST
                                    source_map = ses_1[cell_label_1][0]
                                    target_map = ses_2[cell_label_2][0]
                                    prev_spatial_spike_train = ses_1[cell_label_1][1]
                                    curr_spatial_spike_train = ses_2[cell_label_2][1]
                                    if settings['runUniqueGroups']:
                                        row_prev, col_prev = np.where(~np.isnan(source_map))
                                        source_weights = np.array(list(map(lambda x, y: source_map[x,y], row_prev, col_prev)))
                                        row_curr, col_curr = np.where(~np.isnan(target_map))
                                        target_weights = np.array(list(map(lambda x, y: target_map[x,y], row_curr, col_curr)))

                                        prev_height_bucket_midpoints, prev_width_bucket_midpoints = _get_ratemap_bucket_midpoints(prev_spatial_spike_train.arena_size, y, x)
                                        curr_height_bucket_midpoints, curr_width_bucket_midpoints = _get_ratemap_bucket_midpoints(curr_spatial_spike_train.arena_size, y, x)
                                                        
                                        prev_height_bucket_midpoints = prev_height_bucket_midpoints[row_prev]
                                        prev_width_bucket_midpoints = prev_width_bucket_midpoints[col_prev]
                                        curr_height_bucket_midpoints = curr_height_bucket_midpoints[row_curr]
                                        curr_width_bucket_midpoints = curr_width_bucket_midpoints[col_curr]

                                        source_weights = source_weights / np.sum(source_weights)
                                        target_weights = target_weights / np.sum(target_weights)
                                            
                                        coord_buckets_curr = np.array(list(map(lambda x, y: [curr_height_bucket_midpoints[x],curr_width_bucket_midpoints[y]], row_curr, col_curr)))
                                        coord_buckets_prev = np.array(list(map(lambda x, y: [prev_height_bucket_midpoints[x],prev_width_bucket_midpoints[y]], row_prev, col_prev)))
                                            
                                        if settings['normalizePos']:
                                            curr_pts = scale_points(curr_pts)
                                            prev_pts = scale_points(prev_pts)

                                        spike_dens_wass = pot_sliced_wasserstein(prev_pts, curr_pts, n_projections=settings['n_projections'])
                                        animal_ref_dist[animal_id][ses_comp]['ref_spike_density'].append(spike_dens_wass)
                                        wass = pot_sliced_wasserstein(coord_buckets_prev, coord_buckets_curr, source_weights, target_weights, n_projections=settings['n_projections'])
                                        animal_ref_dist[animal_id][ses_comp]['ref_whole'].append(wass)

                                    if settings['runUniqueOnlyTemporal']:
                                        prev_spike_times = prev_spatial_spike_train.spike_times
                                        curr_spike_times = curr_spatial_spike_train.spike_times
                                        observed_emd = compute_temporal_emd(prev_spike_times, curr_spike_times, settings['temporal_bin_size'], settings['end_time'])
                                        animal_ref_dist[animal_id][ses_comp]['ref_temporal'].append(observed_emd)
                                        prev_duration = prev_spatial_spike_train.session_metadata.session_object.get_spike_data()['spike_cluster'].duration
                                        curr_duration = curr_spatial_spike_train.session_metadata.session_object.get_spike_data()['spike_cluster'].duration

                                        curr_fr_rate = len(curr_spike_times) / prev_duration
                                        prev_fr_rate = len(prev_spike_times) / curr_duration
                                        fr_rate_ratio = curr_fr_rate / prev_fr_rate
                                        fr_rate_change = curr_fr_rate - prev_fr_rate
                                        animal_ref_dist[animal_id][ses_comp]['ref_rate_ratio'].append(fr_rate_ratio)
                                        animal_ref_dist[animal_id][ses_comp]['ref_rate_change'].append(fr_rate_change)
                                else:
                                    # print('SAME CELL, omitting from ref dist {} {}'.format(cell_label_1, cell_label_2))
                                    pass

                    prev_ses_id_categs = ses_id_categs
        
    return animal_ref_dist



def get_rate_map(rate_map_obj, ratemap_dims,normalizeRate):
    if normalizeRate:
        rate_map, _ = rate_map_obj.get_rate_map(new_size = ratemap_dims[0])
    else:
        _, rate_map = rate_map_obj.get_rate_map(new_size = ratemap_dims[0])

    assert rate_map.shape == (ratemap_dims[0], ratemap_dims[1]), 'Wrong ratemap shape {} vs settings shape {}'.format(rate_map.shape, (ratemap_dims[0], ratemap_dims[1]))

    return rate_map
                
def make_obj_map_dict(variations, settings, cylinder, disk_ids):
    obj_map_dict = {}
    for var in variations:

        if settings['downsample']:
            object_ratemap, object_pos = make_object_ratemap(var, new_size=int(settings['ratemap_dims'][0]/settings['downsample_factor']))
        else:
            object_ratemap, object_pos = make_object_ratemap(var, new_size=settings['ratemap_dims'][0])

        if cylinder:
            object_ratemap = flat_disk_mask(object_ratemap)

        obj_map_dict[var] = [object_ratemap, object_pos, disk_ids]
                            
    return obj_map_dict
    
# UNUSED?
def _single_shuffled_sample(norm, raw, settings):
    # norm, raw = spatial_spike_train.get_map('rate').get_rate_map(new_size = settings['ratemap_dims'][0], shuffle=True)
    if settings['normalizeRate']:
        rate_map = norm
        # rate_map = rate_map / np.sum(rate_map)
    else:
        rate_map = raw   

    # check no nans
    assert np.isnan(rate_map).any() == False, "rate map contains nans pre downsampling"

    if settings['downsample']:
        rate_map = _downsample(rate_map, settings['downsample_factor'])

    # check no nans
    assert np.isnan(rate_map).any() == False, "rate map contains nans post downsampling"
    

    return rate_map 
