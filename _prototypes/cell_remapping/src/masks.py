import os, sys
import numpy as np
import re
import math
from scipy.spatial.distance import euclidean
from itertools import chain

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.maps.map_utils import _interpolate_matrix, disk_mask
from _prototypes.cell_remapping.src.utils import _downsample

# adapted from https://gis.stackexchange.com/questions/436908/selecting-n-samples-uniformly-from-a-grid-points
def _sample_grid(ids, threshold=3):
    first = True
    valid = []

    # def _sample_pt(point, first, valid):
    for point in ids:
        if first:
            # print('first')
            valid.append([point[0], point[1]])
            first = False
        else:                    
            point = [point[0], point[1]]
            if any((euclidean(point, point_ok) < threshold for point_ok in valid)):
                # print('rejected')
                pass
            else:
                valid.append(point)

    return np.array(valid)


def generate_grid(arena_height, arena_width, spacing, is_hexagonal=False, is_cylinder=False):
    if is_hexagonal:
        hexagon_size = spacing / math.sqrt(3)
        num_columns = math.ceil(arena_width / (1.5 * hexagon_size))
        num_rows = math.ceil(arena_height / ((hexagon_size * math.sqrt(3))))

    else:
        num_columns = math.ceil(arena_width / spacing)
        num_rows = math.ceil(arena_height / spacing)
   
    if is_cylinder:
        radius_x = arena_width / 2
        radius_y = arena_height / 2
        # offset_x = (arena_width - num_columns * spacing) / 2  # Adjust the x-offset
        # offset_y = (arena_height - num_rows * spacing) / 2   
        # offset_y = offset_x
    else:
        radius_x = None
        radius_y = None
    
    offset_x = offset_y = 0

    if is_hexagonal:
        print(offset_x, offset_y, num_columns, num_rows, hexagon_size)
        columns = range(num_columns)
        rows = range(num_rows)
        grid = list(
            map(
                lambda row: list(
                    map(
                        lambda col: (
                            col * (1.5 * hexagon_size) + offset_x,
                            row * (hexagon_size * math.sqrt(3)) + ((col % 2) * (hexagon_size * math.sqrt(3) / 2)) + offset_y
                        ),
                        filter(
                            lambda col: not radius_x or math.sqrt(
                                ((col * (1.5 * hexagon_size) + offset_x) - radius_x) ** 2 +
                                ((row * (hexagon_size * math.sqrt(3))) + ((col % 2) * (hexagon_size * math.sqrt(3) / 2)) - radius_y) ** 2
                            ) <= radius_x,
                            columns
                        )
                    )
                ),
                rows
            )
        )
    else:
        columns = range(num_columns)
        rows = range(num_rows)
        grid = list(
            map(
                lambda row: list(
                    map(
                        lambda col: (
                            col * spacing + offset_x,
                            row * spacing + offset_y
                        ),
                        filter(
                            lambda col: not radius_x or math.sqrt(
                                ((col * spacing + offset_x) - radius_x) ** 2 +
                                ((row * spacing + offset_y) - radius_y) ** 2
                            ) <= radius_x,
                            columns
                        )
                    )
                ),
                rows
            )
        )

    return list(chain.from_iterable(grid))

            
def make_object_ratemap(object_location, new_size=16):
    #arena_height, arena_width = rate_map_obj.arena_size
    #arena_height = arena_height[0]
    #arena_width = arena_width[0]

    # rate_map, _ = rate_map_obj.get_rate_map(new_size=new_size)

    # (64, 64)
    y = new_size 
    x = new_size
    # print(y, x)
    # convert height/width to arrayswith 64 bins
    #height = np.arange(0,arena_height, arena_height/x)
    #width = np.arange(0,arena_width, arena_width/y)

    # make zero array same shape as true ratemap == fake ratemap
    arena = np.zeros((y,x))

    # if no object, zero across all ratemap
    if object_location == 'NO':
        # even everywehre
        # cust_arena = np.ones((y,x))
        # norm_arena = cust_arena / np.sum(cust_arena)

        # obj inn middle
        norm_arena = np.zeros((y,x))
        norm_arena[int(np.floor(y/2)), int(np.floor(x/2))] = 1
        return norm_arena, {'x':int(np.floor(y/2)), 'y':int(np.floor(y/2))}

    else:
        # if object, pass into dictionary to get x/y coordinates of object location
        # object_location_dict = {
        #     0: (y-1, int(np.floor(x/2))),
        #     90: (int(np.floor(y/2)), x-1),
        #     180: (0, int(np.floor(x/2))),
        #     270: (int(np.floor(y/2)), 0)
        # }

        object_location_dict = {
            0: (0,int(np.floor(x/2))),
            90: (int(np.floor(y/2)),y-1),
            180: (y-1,int(np.floor(x/2))),
            270:  (int(np.floor(y/2)),0)
        }

        id_y, id_x = object_location_dict[object_location]

        # get x and y ids for the first bin that the object location coordinates fall into
        #id_x = np.where(height <= object_pos[0])[0][-1]
        #id_y = np.where(width <= object_pos[1])[0][-1]
        # id_x_small = np.where(height < object_pos[0])[0][0]



        # cts_x, _ = np.histogram(object_pos[0], bins=height)
        # cts_y, _ = np.histogram(object_pos[1], bins=width)

        # id_x = np.where(cts_x != 0)[0]
        # id_y = np.where(cts_y != 0)[0]
        # print(arena_height, arena_width, height, width, object_pos, id_x, id_y)

        # set that bin equal to 1

        arena[id_y, id_x] = 1

        # print(np.max(rate_map), np.max(rate_map)-np.min(rate_map), np.min(rate_map), np.sum(rate_map))
        # arena[id_x, id_y] = np.sum(rate_map)

        # print(arena)

        return arena, {'x':id_x, 'y':id_y}


def binary_mask(curr_labels, label_id, disk_ids, cylinder):
    #     # TAKE ONLY MAIN FIELD --> already sorted by size
    row, col = np.where(curr_labels == label_id)
    field_ids = np.array([row, col]).T
    if cylinder:
        # take ids that are both in disk and in field
        print('IT IS A CYLINDER, TAKING ONLY IDS IN FIELD AND IN DISK')
        field_disk_ids = np.array([x for x in field_ids if x in disk_ids])
    else:
        field_disk_ids = field_ids
    curr_masked = np.zeros((curr_labels.shape))
    curr_masked[field_disk_ids[:,0], field_disk_ids[:,1]] = 1
    return curr_masked, field_disk_ids
                                        

def flat_disk_mask(rate_map):
    masked_rate_map = disk_mask(rate_map)
    # masked_rate_map.data[masked_rate_map.mask] = 0
    # masked_rate_map.data[masked_rate_map.mask] = np.nan
    # print(np.unique(masked_rate_map.data))
    # return  masked_rate_map.data
    copy = np.copy(rate_map).astype(np.float32)
    copy[masked_rate_map.mask] = np.nan
    return copy
    # return masked_rate_map



# Apply disk mask ratemap
def apply_disk_mask(rate_map, settings, cylinder):
    if cylinder:
        curr = flat_disk_mask(rate_map)
        if settings['downsample']:
            curr_ratemap = _downsample(rate_map, settings['downsample_factor'])
            curr_ratemap = flat_disk_mask(curr_ratemap)
        else:
            curr_ratemap = curr
        row, col = np.where(~np.isnan(curr_ratemap))
        disk_ids = np.array([row, col]).T
    else:
        curr = rate_map
        if settings['downsample']:
            curr_ratemap = _downsample(rate_map, settings['downsample_factor']) 
        else:
            curr_ratemap = curr
        disk_ids = None

    return curr, curr_ratemap, disk_ids
                    
