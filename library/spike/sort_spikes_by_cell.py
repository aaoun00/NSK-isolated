import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
 

import numpy as np
import os, sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.batch_space import SpikeClusterBatch

def sort_spikes_by_cell(clusters: SpikeClusterBatch,matched_lbls=None):
    # spike_times, cluster_labels, waveform
    """
    Returns valid cells for session and associated waveforms
    """

    spike_times = clusters.event_times
    # print('Total spikes: ', len(spike_times))
    # print('Matching labels is None T/F: ', matched_lbls is None)
    cluster_labels = clusters.cluster_labels
    waveforms = clusters.get_all_channel_waveforms()

    assert len(spike_times) == len(waveforms[0])

    cells = []
    sorted_waveforms = []
    good_cells = []
    good_sorted_waveforms = []
    sorted_label_ids = []
    good_clusters = []

    unique_labels = np.unique(cluster_labels)
    # print('Unique labels: ', unique_labels)
    # comes in shape (channel count, spike time, nmb samples) but is nested list not numpy
    # want to rearrannge to be (spike time, channel count, nmb sample)
    # waves = np.array(waveforms).reshape((len(waveforms[0]), len(waveforms),  len(waveforms[0][0])))
    waves = np.swapaxes(waveforms, 1, 0)
    # waves = np.asarray(waveforms)
    ct = 0
    index_of_label = []
    for lbl in unique_labels:
        # print('NEW LABEL')
        idx = np.where(cluster_labels == lbl)[0]
        # print('Label: ', lbl, 'idx: ', len(idx))
        idx = idx[idx <= len(spike_times)-1]
        # print('Label: ', lbl, 'idx: ', len(idx))
        # print(np.array(spike_times).squeeze().shape, idx)
        spks = np.array(spike_times).squeeze()[idx]
        if type(spks) == float or type(spks) == np.float64:
            spks = [spks]
        # print('Label: ', lbl, 'spks: ', len(spks))
        # if len(spks) < 40000 and len(spks) > 100:
        # if len(spks) <= 30000:
        cells.append(spks)
        sorted_waveforms.append(waves[idx,:,:].squeeze())
        # sorted_waveforms.append(waves[:,idx,:].squeeze())
        sorted_label_ids.append(lbl)
        index_of_label.append(ct)
        ct += 1
        # sorted_clusters.append(indiv_clusters[idx])

    # empty_cell = 0
    # for j in range(len(sorted_waveforms)):
    #     print(j, len(sorted_waveforms[j]), empty_cell)
    #     if len(sorted_waveforms[j]) == 0 and j != 0:
    #         empty_cell = j
    #         break
    #     else:
    #         empty_cell = j + 1

    # empty_cell = sorted(set(range(unique_labels[0], unique_labels[-1] + 1)).difference(unique_labels))

    if matched_lbls is not None:
        print('Using matched cut files')
        # print(matched_lbls)
        unique_labels = matched_lbls

    if unique_labels[0] == 0:
        empty_cell = sorted(set(range(unique_labels[1], unique_labels[-1] + 1)).difference(unique_labels))
    else:
        empty_cell = sorted(set(range(unique_labels[0], unique_labels[-1] + 1)).difference(unique_labels))

    # if matched_lbls is not None: # if using matched cut files
    # if we are using matched cut files, you can have multiple gap cells if a match is missing from one file then present in the next
    # for mathced cut files, the matched_lbls are set in _read_input_dict of the Animal Class
    # what this means is we have access to all sessions for an animal and can collect all labels across matched sessionns
    # therefore when matched labels is used, we can be sure that the empty cell will be the first empty cell and will not be a result of gaps in matched sessions
    # BELOW is the code used to gather all unique labels for a sequence of matched sessions
    #    if 'matched' in self._input_dict['session_1'].session_metadata.file_paths['cut']:
    #        lbls = np.unique(np.concatenate(list(map(lambda x: np.unique(self._input_dict[x].get_spike_data()['spike_cluster'].cluster_labels), self._input_dict))))
    #        isMatchedCut = True


    if len(empty_cell) >= 1: # if there are multiple empty cells, take the first one 
        # NOTE ABOVE IF WORRIED ABOUT MULTIPLE GAP CELLS IN MATCHED CUT
        empty_cell = empty_cell[0]
    else: # otherwise take 1 label past the last cell id
        empty_cell = unique_labels[-1] + 1

    # print(empty_cell)
    sorted_label_ids = np.asarray(sorted_label_ids)
    idx = np.where((sorted_label_ids >= 1) & (sorted_label_ids < empty_cell))
    good_sorted_label_ids = sorted_label_ids[idx]
    
    # else:
    #     sorted_label_ids = np.asarray(sorted_label_ids)
    #     idx = np.where((sorted_label_ids >= 1))
    #     good_sorted_label_ids = sorted_label_ids[idx]


    # VERY IMPORTANT LINE #
    clusters.set_sorted_label_ids(good_sorted_label_ids)
    # IF SORTED LABEL IDS NOT SET, NOISE LABELS WILL BE USED TO MKAE CELL #

    indiv_clusters = clusters.get_spike_cluster_instances()
    # print(len(indiv_clusters), len(sorted_label_ids), sorted_label_ids, empty_cell)
    # good_label_ids = []
    # for j in sorted_label_ids:
    # for j in range(len(good_sorted_label_ids)):
    #     label_id = good_sorted_label_ids[j]
    #     # print(len(cells[j]))
    #     good_cells.append(cells[j])
    #     good_sorted_waveforms.append(sorted_waveforms[j])
    #     # indiv_clusters will only be made for good_label_id cells so have to adjust to make 0 index when pulling out spike cluster
    #     good_clusters.append(indiv_clusters[j])
    #     assert indiv_clusters[j].cluster_label == label_id
    # print(len(cells), len(sorted_waveforms), len(indiv_clusters), len(sorted_label_ids), good_sorted_label_ids, idx, empty_cell)
    good_cells = np.asarray(cells)[idx]
    good_sorted_waveforms = np.asarray(sorted_waveforms)[idx]
    good_clusters = np.asarray(indiv_clusters)

    # print(unique_labels, sorted_label_ids,empty_cell)
    # print(good_sorted_label_ids, len(good_cells))
    # print(good_sorted_label_ids, len(good_cells), len(good_sorted_waveforms), len(good_clusters))

    return good_cells, good_sorted_waveforms, good_clusters, good_sorted_label_ids
