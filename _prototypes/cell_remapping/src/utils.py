import os, sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from _prototypes.cell_remapping.src.MEC_naming import MEC_naming_format, extract_name_mec
from _prototypes.cell_remapping.src.LC_naming import LC_naming_format, extract_name_lc
from scripts.batch_map.LEC_naming import LEC_naming_format, extract_name_lec

from skimage.measure import block_reduce
from scipy import stats, ndimage

import numpy as np
import re


# utils
# def _sort_filter_centroids_by_field_size(prev, curr, field_sizes_source, field_sizes_target, blobs_map_source, blobs_map_target, centroids_prev, centroids_curr, arena_size):
def _sort_filter_centroids_by_field_size(rate_map, field_sizes, blobs_map, centroids, arena_size):
    height, width = arena_size
    y, x = rate_map.shape
    heightStep = height/y
    widthStep = width/x

    bin_area = heightStep * widthStep

    if type(bin_area) == list:
        assert len(bin_area) == 1
        bin_area = bin_area[0]

    # find not nan in prev/curr 
    row, col = np.where(np.isnan(rate_map))
    blobs_map[row, col] = 0

    # find bins for each label
    pks = []
    avgs = []
    lbls = []
    for k in np.unique(blobs_map):
        if k != 0:
            row, col = np.where(blobs_map == k)
            
            field = rate_map[row, col]

            # pk_rate = np.max(field)

            # get rate at centroid 
            c_row, c_col = centroids[k-1]
            # convert from decimals to bin by rounding to nearest int
            c_row = int(np.round(c_row))
            c_col = int(np.round(c_col))

            # pk_rate = rate_map[c_row, c_col]  
            # pk_rate = np.max(field)

            avg_rate = np.mean(field)
            pk_rate = np.sum(field)

            pks.append(pk_rate)
            avgs.append(avg_rate)
            lbls.append(k-1)
    
    pks = np.array(pks)
    ids = np.argsort(-pks)
    sort_idx = np.array(lbls)[ids]

    source_labels = np.zeros(blobs_map.shape)
    map_dict = {}
    # source_centroids = []
    # source_field_sizes = []
    all_filtered_out = True
    largest_label_id = None
    largest_field_area = 0
    for k in np.unique(blobs_map):
        if k != 0:
            row, col = np.where(blobs_map == k)
            field_area = (len(row) + len(col)) * bin_area
            if field_area > largest_field_area:
                    largest_label_id = k
                    largest_field_area = field_area

            if field_area > 22.5:
                idx_to_move_to = np.where(sort_idx == k-1)[0][0]
                # map_dict[k] = sort_idx_source[k-1] + 1
                map_dict[k] = idx_to_move_to + 1
                # source_centroids.append(centroids[k-1])
                all_filtered_out = False
            else:
                print('Blob filtered out with size less than 22.5 cm^2')
                map_dict[k] = 0
                # remove from sort_idx
                sort_idx = sort_idx[np.where(sort_idx != k-1)[0]]
        else:
            map_dict[k] = 0
    source_labels = np.vectorize(map_dict.get)(blobs_map)

    if len(sort_idx) > 0:
        source_centroids = np.asarray(centroids)[sort_idx]
        source_field_sizes = np.asarray(field_sizes)[sort_idx]
    else:
        source_centroids = np.asarray(centroids)
        source_field_sizes = np.asarray(field_sizes)

    if all_filtered_out:
        assert largest_label_id is not None
        map_dict[largest_label_id] = 1
        source_labels = np.vectorize(map_dict.get)(blobs_map)
        source_centroids = [np.asarray(centroids)[largest_label_id-1]]
        source_field_sizes = [np.asarray(field_sizes)[largest_label_id-1]]

    # print('blobshere')
    # print('all_filtered_out: ' + str(all_filtered_out))
    # print(pks, ids, lbls, sort_idx)
    # print(np.unique(source_labels), len(source_centroids), len(source_field_sizes))
    # print(map_dict, np.unique(blobs_map))
    assert len(np.unique(source_labels)) - 1 == len(source_centroids) == len(source_field_sizes)

    return source_labels, source_centroids, source_field_sizes

# utils?
def _fill_centroid_dict(centroid_dict, max_centroid_count, wass_to_add, centroid_pairs, centroid_coords, source_map, source_labels, source_field_sizes, target_map, target_labels, target_field_sizes):
    # field_wass, centroid_wass, binary_wass = wass_args
    # to_add = wass_args
    for n in range(max_centroid_count):
        wass_key = 'c_wass_'+str(n+1)
        id_key = 'c_ids_'+str(n+1)
        vector_key = 'c_vector_'+str(n+1)
        coverage_key = 'c_coverage_'+str(n+1)
        area_key = 'c_area_'+str(n+1)
        rate_key = 'c_rate_'+str(n+1)

        if wass_key not in centroid_dict:
            centroid_dict[wass_key] = []
            centroid_dict[id_key] = []
            centroid_dict[coverage_key] = []
            centroid_dict[area_key] = []
            centroid_dict[rate_key] = []
            centroid_dict[vector_key] = []

        if n < len(wass_to_add):
            centroid_dict[wass_key].append(wass_to_add[n])
            centroid_dict[id_key].append(centroid_pairs[n])

            labels_curr = np.copy(source_labels)
            labels_curr[np.isnan(source_map)] = 0
            val_r, val_c = np.where(labels_curr == centroid_pairs[n][0])
            # take only field label = 1 = largest field = main
            # source_field_coverage = source_field_sizes[centroid_pairs[n][0]-1]
            source_field_coverage = len(np.where(labels_curr == centroid_pairs[n][0])[0])/len(np.where(~np.isnan(source_labels))[0])
            source_field_area = len(np.where(labels_curr == centroid_pairs[n][0])[0])
            source_field_rate = np.sum(source_map[val_r, val_c])

            labels_curr = np.copy(target_labels)
            labels_curr[np.isnan(target_map)] = 0
            val_r, val_c = np.where(labels_curr == centroid_pairs[n][1])
            # take only field label = 1 = largest field = main
            # target_field_coverage = target_field_sizes[centroid_pairs[n][1]-1]
            target_field_coverage = len(np.where(labels_curr == centroid_pairs[n][1])[0])/len(np.where(~np.isnan(target_labels))[0])
            target_field_area = len(np.where(labels_curr == centroid_pairs[n][1])[0])
            target_field_rate = np.sum(target_map[val_r, val_c])

            source_r, source_c = centroid_coords[n][0]
            target_r, target_c = centroid_coords[n][1]
            mag = np.linalg.norm(np.array((source_r, source_c)) - np.array((target_r, target_c)))
            pt1 = (source_r, source_c)
            pt2 = (target_r, target_c)
            angle = np.degrees(np.math.atan2(np.linalg.det([pt1,pt2]),np.dot(pt1,pt2)))

            
            centroid_dict[coverage_key].append([source_field_coverage, target_field_coverage])
            centroid_dict[area_key].append([source_field_area, target_field_area])
            centroid_dict[rate_key].append([source_field_rate, target_field_rate])
            centroid_dict[vector_key].append([pt1, pt2, mag, angle])
            
        else:
            centroid_dict[wass_key].append(np.nan)
            centroid_dict[id_key].append([np.nan])
            centroid_dict[coverage_key].append([np.nan, np.nan])
            centroid_dict[area_key].append([np.nan, np.nan])
            centroid_dict[rate_key].append([np.nan, np.nan])
            centroid_dict[vector_key].append([np.nan, np.nan, np.nan, np.nan])

        # if wass_key not in centroid_dict:
        #     centroid_dict[wass_key] = []
        #     centroid_dict[id_key] = []

        # if n < len(centroid_wass):
        #     centroid_dict[wass_key].append([field_wass[n], centroid_wass[n], binary_wass[n]])
        #     centroid_dict[id_key].append(centroid_pairs[n])
        # else:
        #     centroid_dict[wass_key].append([0,0,0])
        #     centroid_dict[id_key].append([0,0])

    return centroid_dict




def _get_ratemap_bucket_midpoints(arena_size, y, x):
    """
    Helper function to create array of height and width bucket midpoints
    
    Takes in arena dimensions (tuple) and y and x (64,64) dims of ratemap
    """
    arena_height, arena_width = arena_size

    if isinstance(arena_height, list):
        if len(arena_height) > 0:
            arena_height = arena_height[0]
    if isinstance(arena_width, list):
        if len(arena_width) > 0:
            arena_width = arena_width[0]

    # this is the step size between each bucket, so 0 to height step is first bucket, height_step to height_step*2 is next and so one
    height_step = arena_height/x
    width_step = arena_width/y

    # convert height/width to arrayswith 64 bins, this gets us our buckets
    height = np.arange(0,arena_height, height_step)
    width = np.arange(0,arena_width, width_step)

    # because they are buckets, i figured I will use the midpoint of the pocket when computing euclidean distances
    height_bucket_midpoints = height + height_step/2
    width_bucket_midpoints = width + width_step/2

    return height_bucket_midpoints, width_bucket_midpoints



def list_to_array(*lst):
    r""" Convert a list if in numpy format """
    if len(lst) > 1:
        return [np.array(a) if isinstance(a, list) else a for a in lst]
    else:
        return np.array(lst[0]) if isinstance(lst[0], list) else lst[0]


def read_data_from_fname(fname, naming_type, typ):
    if naming_type == 'LEC':
        group, name = extract_name_lec(fname)
        formats = LEC_naming_format[group][name][typ]
    elif naming_type == 'MEC':
        name = extract_name_mec(fname)
        formats = MEC_naming_format
    elif naming_type == 'LC':
        name = extract_name_lc(fname)
        formats = LC_naming_format

    for format in list(formats.keys()):
        checked = _check_single_format(fname, format, formats[format])
        if checked is not None:
            break
        else:
            continue
                    
    stim, depth, name, date = checked

    return stim, depth, name, date
        

def _check_object_coords(object_pos, height_bucket_midpoints, width_bucket_midpoints):
    if isinstance(object_pos, dict):
        obj_x = width_bucket_midpoints[object_pos['x']]
        obj_y = height_bucket_midpoints[object_pos['y']]
    else:
        obj_y = height_bucket_midpoints[object_pos[0]]
        obj_x = width_bucket_midpoints[object_pos[1]] 

    return obj_x, obj_y

def check_disk_arena(path):
    variations = [r'cylinder', r'round', r'circle', r'CYLINDER', r'ROUND', r'CIRCLE', r'Cylinder', r'Round', r'Circle']
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
    # print(cylinder, true_var, path)
    return cylinder, true_var


def check_object_location(stim, hasObject):
    if hasObject:
        object_location = stim 
        if object_location != 'NO' and '_' not in object_location:
            object_location = int(object_location)
    else:
        object_location = None
    return object_location

def check_cylinder(fname, isDisk):
    # if disk arena is true in settings, use that, otherwise check if disk arena
    if isDisk: 
        cylinder = True
    else:
        cylinder, _ = check_disk_arena(fname)
        if not cylinder:
            print('Not cylinder for {}'.format(fname))
    
    return cylinder

# move to utils                    
def _check_single_format(filename, format, fxn):
    """ Check if filename matches recorded format from custon naming convention ffiles"""
    # print(filename, format, fxn)
    if re.match(str(format), str(filename)) is not None:
        return fxn(filename)


# move to stats?
def _downsample(img, downsample_factor):
    downsampled = block_reduce(img, downsample_factor) 
    return downsampled
    

# make all field labels = 1 in copy (to compute statistics on)
def _copy_labels(labels, curr):   
    labels_copy = np.copy(labels)
    labels_copy[np.isnan(curr)] = 0
    # get c_count before making all labels = 1 (all = 1 in copy for cumulative stats)
    c_count = len(np.unique(labels_copy)) - 1
    labels_copy[labels_copy != 0] = 1
    return labels_copy, c_count
                        
def _get_spk_pts(spatial_spike_train):
    spike_pos_x, spike_pos_y, spike_pos_t = spatial_spike_train.spike_x, spatial_spike_train.spike_y, spatial_spike_train.new_spike_times
    pts = np.array([spike_pos_x, spike_pos_y]).T
    return pts

def _get_valid_midpoints(arena_size, y, x, row, col):
    height_bucket_midpoints, width_bucket_midpoints = _get_ratemap_bucket_midpoints(arena_size, y, x)
    height_bucket_midpoints = height_bucket_midpoints[row]
    width_bucket_midpoints = width_bucket_midpoints[col]
    # set up (x,y) pairs of bucket midpoints
    coord_buckets = np.array(list(map(lambda x, y: [height_bucket_midpoints[x],width_bucket_midpoints[y]], row, col)))
    
    return height_bucket_midpoints, width_bucket_midpoints, coord_buckets
                                        
# valid fr rate bins
def _get_valid_weight_bins(ratemap):
    row, col = np.where(~np.isnan(ratemap))
    weights = np.array(list(map(lambda x, y: ratemap[x,y], row, col)))
    return weights, row, col
          

# same as above
def scale_points(pts):
    # Separate the x and y coordinates
    curr_spike_pos_x = pts[:, 0]
    curr_spike_pos_y = pts[:, 1]

    # Compute the minimum and maximum values for x and y coordinates
    min_x = np.min(curr_spike_pos_x)
    max_x = np.max(curr_spike_pos_x)
    min_y = np.min(curr_spike_pos_y)
    max_y = np.max(curr_spike_pos_y)

    # Perform Min-Max scaling separately for x and y coordinates
    scaled_x = (curr_spike_pos_x - min_x) / (max_x - min_x)
    scaled_y = (curr_spike_pos_y - min_y) / (max_y - min_y)

    # Combine the scaled x and y coordinates
    scaled_pts = np.column_stack((scaled_x, scaled_y))

    return scaled_pts

                        
# THIS IS TO ROTATE ON IN A PAIR NOT BOTH
def _check_rotate_evening(pth, pts, ratemap, isRotate,rotate_angle=None):
    """ ONLY CURR (target session, 2nd in a pair, = evening for this custom setting )"""
    if isRotate:
        if 'evening' in pth.lower() or 'rotated' in pth.lower():
            ratemap = ndimage.rotate(ratemap, rotate_angle)
            if rotate_angle == 90:
                pts = np.array([pts[:,1], -pts[:,0]]).T
                print('rotating 90 degrees for {}'.format(pth))
            else:
                raise ValueError('Rotation angle not supported {}'.format(rotate_angle))
        else:
            print('not rotating for {}'.format(pth))
    return ratemap, pts
                        



# check if used
def _fill_cell_type_stats(inp_dict, prev_cell, curr_cell):
    inp_dict['information'].append([prev_cell.stats_dict['cell_stats']['spatial_information_content'],curr_cell.stats_dict['cell_stats']['spatial_information_content'],curr_cell.stats_dict['cell_stats']['spatial_information_content']-prev_cell.stats_dict['cell_stats']['spatial_information_content']])
    inp_dict['grid_score'].append([prev_cell.stats_dict['cell_stats']['grid_score'],curr_cell.stats_dict['cell_stats']['grid_score'],curr_cell.stats_dict['cell_stats']['grid_score']-prev_cell.stats_dict['cell_stats']['grid_score']])
    inp_dict['b_top'].append([prev_cell.stats_dict['cell_stats']['b_score_top'],curr_cell.stats_dict['cell_stats']['b_score_top'],curr_cell.stats_dict['cell_stats']['b_score_top']-prev_cell.stats_dict['cell_stats']['b_score_top']])
    inp_dict['b_bottom'].append([prev_cell.stats_dict['cell_stats']['b_score_bottom'],curr_cell.stats_dict['cell_stats']['b_score_bottom'],curr_cell.stats_dict['cell_stats']['b_score_bottom']-prev_cell.stats_dict['cell_stats']['b_score_bottom']])
    inp_dict['b_right'].append([prev_cell.stats_dict['cell_stats']['b_score_right'],curr_cell.stats_dict['cell_stats']['b_score_right'],curr_cell.stats_dict['cell_stats']['b_score_right']-prev_cell.stats_dict['cell_stats']['b_score_right']])
    inp_dict['b_left'].append([prev_cell.stats_dict['cell_stats']['b_score_left'],curr_cell.stats_dict['cell_stats']['b_score_left'],curr_cell.stats_dict['cell_stats']['b_score_left']-prev_cell.stats_dict['cell_stats']['b_score_left']])
                            
    # inp_dict['speed_score'].append([prev_cell.stats_dict['cell_stats']['speed_score'],curr_cell.stats_dict['cell_stats']['speed_score'],curr_cell.stats_dict['cell_stats']['speed_score']-prev_cell.stats_dict['cell_stats']['speed_score']])
    # inp_dict['hd_score'].append([prev_cell.stats_dict['cell_stats']['hd_score'],curr_cell.stats_dict['cell_stats']['hd_score'],curr_cell.stats_dict['cell_stats']['hd_score']-prev_cell.stats_dict['cell_stats']['hd_score']])

    return inp_dict

# check if used
def _read_location_from_file(path, cylinder, true_var):

    if not cylinder:
        object_location = path.split('/')[-1].split('-')[3].split('.')[0]
    else:
        items = path.split('/')[-1].split('-')
        idx = items.index(str(true_var)) + 2 # the object location is always 2 positions away from word denoting arena hape (e.g round/cylinder) defined by true_var
        # e.g. ROUND-3050-90_2.clu
        object_location = items[idx].split('.')[0].split('_')[0].lower()

    object_present = True
    if str(object_location) == 'no':
        object_present == False
        object_location = 'no'
    elif str(object_location) == 'zero':
        object_location = 0
    else:
        object_location = int(object_location)
        assert int(object_location) in [0,90,180,270], 'Failed bcs obj location is ' + str(int(object_location)) + ' and that is not in [0,90,180,270]'

    return object_location


