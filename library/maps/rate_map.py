import os
import sys
from turtle import pos

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)


import numpy as np
from library.maps import occupancy_map
# from library.maps import spike_map
from core.spatial import Position2D
from core.spikes import SpikeTrain


def rate_map(spike_train: SpikeTrain, position: Position2D, smoothing_factor) -> np.ndarray:

    '''
        Computes a 2D array encoding neuron spike events in 2D space, based on where
        the subject walked during experiment.

        Params:
            pos_x, pos_y, pos_t (np.ndarray):
                Arrays of x,y coordinates and timestamps
            arena_size (tuple):
                Arena dimensions based on x,y coordinates
            spikex, spikey (np.ndarray's):
                x and y coordinates of spike occurence
            kernlen, std (int):
                kernel size and std for convolutional smoothing

        Returns:
            Tuple: (x_resize, y__resize)
                --------
                ratemap_smoothed (np.ndarray):
                    Smoothed ratemap
                ratemap_raw (np.ndarray):
                    Raw ratemap
    '''
    if spike_train.session_metadata.session_object != None:
        session_spike_objects = spike_train.session_metadata.session_object.get_spike_data()
        if 'spatial_spike_train' in session_spike_objects:
            spatial_spikes = session_spike_objects['spatial_spike_train']
        else:
            spike_train.session_metadata.session_object.make_class(SpatialSpikeTrain2D, (spike_train, position, spike_train.session_metadata))
    else:
        spatial_spikes = SpatialSpikeTrain2D(spike_train, position, spike_train.session_metadata)

    pos_x, pos_y, pos_t, arena_size = spatial_spikes.x, spatial_spikes.y, spatial_spikes.t, spatial_spikes.arena_size

    spikex, spikey = spatial_spikes.get_spike_positions()

    # smoothing_factor = 5
    # Kernel size
    kernlen = int(smoothing_factor*8)
    # Standard deviation size
    std = int(0.2*kernlen)

    # import pdb; pdb.set_trace()
    occ_map_smoothed, occ_map_raw, _ = occupancy_map(pos_x, pos_y, pos_t, arena_size, kernlen, std)

    spike_map_smoothed, spike_map_raw = spike_map(pos_x, pos_y, pos_t, arena_size, spikex, spikey, kernlen, std)

    # Compute ratemap
    ratemap_raw = np.where(occ_map_raw<0.0001, 0, spike_map_raw/occ_map_raw)
    ratemap_smoothed = np.where(occ_map_smoothed<0.0001, 0, spike_map_smoothed/occ_map_smoothed)
    ratemap_smoothed = ratemap_smoothed/max(ratemap_smoothed.flatten())

    # Smooth ratemap
    #ratemap_a_smooth = _interpolate_matrix(ratemap_raw)

    #ratemap_b_smooth = np.where(occ_map_smoothed<0.00001, 0, spikemap_smoothed/occ_map_smoothed)
    #ratemap_b_smooth = ratemap_b_smooth / max(ratemap_b_smooth.flatten())

    spatial_spikes.add_map_to_stats('rate_smooth', ratemap_smoothed)
    spatial_spikes.add_map_to_stats('ratemap_raw', ratemap_raw)
    spatial_spikes.add_map_to_stats('occ_smooth', occ_map_smoothed)
    spatial_spikes.add_map_to_stats('occ_raw', occ_map_raw)
    spatial_spikes.add_map_to_stats('spike_smooth', spike_map_smoothed)
    spatial_spikes.add_map_to_stats('spike_raw', spike_map_raw)

    return ratemap_smoothed, ratemap_raw
