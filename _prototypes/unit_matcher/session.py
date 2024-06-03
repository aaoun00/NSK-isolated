import os, sys
import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.study_space import Session
from _prototypes.unit_matcher.unit import jensen_shannon_distance, spike_level_feature_array
from _prototypes.unit_matcher.unit_mean import mean_squared_difference_of_unit_means
from core.spikes import SpikeCluster
from library.batch_space import SpikeClusterBatch
from scipy.spatial import distance as sd

def compute_distances(session1_cluster, session2_cluster, method='JSD', ses1_pca_feats=None, ses2_pca_feats=None):
    """
    Compute the distances between two sessions' units.
    Parameters
    ----------
    session1_cluster: SpikeClusterBatch
    session2_cluster: SpikeClusterBatch
    method: str
        The method of distance computation to use
    """
    if method == 'JSD':
        return compute_JSD_distances(session1_cluster, session2_cluster, ses1_pca_feats=ses1_pca_feats, ses2_pca_feats=ses2_pca_feats)
    elif method == 'MSD':
        if ses1_pca_feats is not None:
            print('Cannot use PCA with MSD, proceeding without PCA feature array')
        return compute_MSD_distances(session1_cluster, session2_cluster)
    else:
        raise ValueError('Invalid distance computation method')

def compute_MSD_distances(session1_cluster: SpikeClusterBatch, session2_cluster: SpikeClusterBatch):
    """
    Compute the mean squared difference of unit means between two sessions.
    """
    session1_unit_clusters = session1_cluster.get_spike_cluster_instances()
    session2_unit_clusters = session2_cluster.get_spike_cluster_instances()

    distances = np.zeros((len(session1_unit_clusters), len(session2_unit_clusters)))
    pairs = np.zeros((len(session1_unit_clusters), len(session2_unit_clusters), 2))

    for i in range(len(session1_unit_clusters)):
        for j in range(len(session2_unit_clusters)):

            distance = mean_squared_difference_of_unit_means(session1_unit_clusters[i].waveforms, session2_unit_clusters[j].waveforms)
            print('MSD: ' + str(distance))

            if 'MSD' not in session1_unit_clusters[i].stats_dict:
                session1_unit_clusters[i].stats_dict['MSD'] = []
            session1_unit_clusters[i].stats_dict['MSD'] = distance

            if 'MSD' not in session2_unit_clusters[j].stats_dict:
                session2_unit_clusters[j].stats_dict['MSD'] = []
            session2_unit_clusters[j].stats_dict['MSD'] = distance

            distances[i,j] = distance
            pairs[i,j] = [session1_unit_clusters[i].cluster_label, session2_unit_clusters[j].cluster_label]

    return distances, pairs, None

def compute_JSD_distances(session1_cluster: SpikeClusterBatch, session2_cluster: SpikeClusterBatch, ses1_pca_feats=None, ses2_pca_feats=None): # change to feature vector array
    """
    Iterates through all the across-session unit pairings and computing their respective Jensen-Shannon distances
    Parameters
    ----------
    session1_cluster: SpikeClusterBatch
    session2_cluster: SpikeClusterBatch
    JSD: bool
        If True, computes the Jensen-Shannon distance between the two sessions' unit feature vectors
    MSDoWFM: bool
        If True, computes the Mean Squared Difference of Waveform Means between the two sessions' unit feature vectors
    """

    session1_unit_clusters = session1_cluster.get_spike_cluster_instances()
    session2_unit_clusters = session2_cluster.get_spike_cluster_instances()

    print('New here')

    print(np.unique(session2_cluster.cluster_labels), np.unique(session2_cluster.good_label_ids))
    print(session2_cluster.session_metadata.file_paths)
    print(len(ses2_pca_feats[0]))

    distances = np.zeros((len(session1_unit_clusters), len(session2_unit_clusters)))
    pairs = np.zeros((len(session1_unit_clusters), len(session2_unit_clusters), 2))

    if ses1_pca_feats is None:
        session1_feature_arrays = []
        session2_feature_arrays = []
        # session1_feature_arrays_pca = []
        # session2_feature_arrays_pca = []
        for i in range(len(session1_unit_clusters)):
            feature_array = spike_level_feature_array(session1_unit_clusters[i], 1/session1_cluster.sample_rate)
            session1_feature_arrays.append(feature_array)
            # feature_array_pca = 
            # session1_feature_arrays_pca(feature_array_pca)
        for j in range(len(session2_unit_clusters)):
            feature_array = spike_level_feature_array(session2_unit_clusters[j], 1/session2_cluster.sample_rate)
            session2_feature_arrays.append(feature_array)
    else:
        session1_feature_arrays = ses1_pca_feats
        session2_feature_arrays = ses2_pca_feats
        print('Using post PCA feature array')

    print(np.array(session1_feature_arrays).shape, np.array(session2_feature_arrays).shape,  session1_feature_arrays[0].shape[1])
    print(len(session1_unit_clusters), len(session2_unit_clusters))
    # assert session1_unit_clusters.shape[1] == session2_unit_clusters.shape[1], 'Session 1 & 2 have different numbers of features'
    agg_distances = np.zeros((len(session1_feature_arrays), len(session2_feature_arrays), session1_feature_arrays[0].shape[1]))

    for i in range(len(session1_feature_arrays)):
        for j in range(len(session2_feature_arrays)):
           
            if ses1_pca_feats is not None:
                dists = []
                # axis=0 return dist array of size (n_feats) from feats arrays which are (n_samples, n_feats)
                for k in range(session1_feature_arrays[i].shape[1]):
                    # dist = sd.jensenshannon(session1_feature_arrays[i][:,k].reshape((-1,1)), session2_feature_arrays[j][:,k].reshape((-1,1)), axis=0)
                    dist = jensen_shannon_distance(session1_feature_arrays[i][:,k].reshape((-1,1)), session2_feature_arrays[j][:,k].reshape((-1,1)))
                    dists.append(dist)
                agg_distances[i,j] = dists
                # print('FEAT JSD HERE')
                # print(dists)
                distance = np.mean(dists)
            else:
                distance = jensen_shannon_distance(session1_feature_arrays[i], session2_feature_arrays[j])

            # distance = jensen_shannon_distance(session1_feature_arrays[i], session2_feature_arrays[j])
            
            print('JSD: ' + str(distance))    

            if 'JSD' not in session1_unit_clusters[i].stats_dict:
                session1_unit_clusters[i].stats_dict['JSD'] = []
            session1_unit_clusters[i].stats_dict['JSD'] = distance

            if 'JSD' not in session2_unit_clusters[j].stats_dict:
                session2_unit_clusters[j].stats_dict['JSD'] = []
            session2_unit_clusters[j].stats_dict['JSD'] = distance

            distances[i,j] = distance
            pairs[i,j] = [session1_unit_clusters[i].cluster_label, session2_unit_clusters[j].cluster_label]

    # for i in range(len(session1_feature_arrays)):
    #     for j in range(len(session2_feature_arrays)):

    #         distance = jensen_shannon_distance(session1_feature_arrays[i], session2_feature_arrays[j])
    #         print('JSD: ' + str(distance))

    #         if 'JSD' not in session1_unit_clusters[i].stats_dict:
    #             session1_unit_clusters[i].stats_dict['JSD'] = []
    #         session1_unit_clusters[i].stats_dict['JSD'] = distance

    #         if 'JSD' not in session2_unit_clusters[j].stats_dict:
    #             session2_unit_clusters[j].stats_dict['JSD'] = []
    #         session2_unit_clusters[j].stats_dict['JSD'] = distance

    #         distances[i,j] = distance
    #         pairs[i,j] = [session1_unit_clusters[i].cluster_label, session2_unit_clusters[j].cluster_label]

    return distances, pairs, agg_distances

def extract_full_matches(distances, pairs):
    full_matches = []
    full_match_distances = []
    row_mask = np.ones(distances.shape[0], bool)
    col_mask = np.ones(distances.shape[1], bool)
    # mask = np.ones(distances.shape, bool)


    for i in range(distances.shape[0]):
        unit1_pairs = pairs[i]
        unit1_min = np.argmin(distances[i,:])
        for j in range(distances.shape[1]):
            unit2_pairs = pairs[:,j]
            unit2_min = np.argmin(distances[:,j])
            if unit1_min == j and unit2_min == i:
                full_matches.append(unit1_pairs[unit1_min])
                full_match_distances.append(distances[i,j])
                row_mask[i] = False
                col_mask[j] = False
                # assert sorted(unit1_pairs[unit1_min]) == sorted(unit2_pairs[unit2_min])

    remaining_distances = distances[row_mask,:][:,col_mask]
    remaining_pairs = pairs[row_mask,:][:,col_mask]

    # remaining_distances = distances[row_mask,col_mask]
    # remaining_pairs = pairs[row_mask,col_mask]
    print('NEWIN')
    print(distances, pairs)
    print(row_mask, col_mask)
    print(distances[row_mask,:], distances[row_mask,:].shape)
    print(pairs[row_mask,:], pairs[row_mask,:].shape)
    print('NEWOUT')
    # distances = list(map(lambda x: x if x == False, np.arange(mask.shape[0])))
    print(full_matches, full_match_distances, remaining_distances, remaining_pairs)

    return full_matches, full_match_distances, remaining_distances, remaining_pairs


def guess_remaining_matches(distances, pairs):
    row_ind, col_ind = linear_sum_assignment(distances)
    assert len(row_ind) == len(col_ind)
    remaining_matches = []
    remaining_match_distances = []
    print(pairs.shape, pairs, row_ind, col_ind)
    for i in range(len(row_ind)):
        unit_pair = pairs[row_ind[i], col_ind[i]]
        remaining_matches.append(unit_pair)
        remaining_match_distances.append(distances[row_ind[i], col_ind[i]])

    print(distances, row_ind, col_ind, remaining_matches)
    print(pairs)

    # session1_unmatched = list(set(np.arange(len(distances))) - set(remaining_matches[0]))
    if len(distances) > 0 and len(distances[0]) > 0:
        session2_unmatched = list(set(list(np.arange(len(distances[0])))) - set(list(col_ind)))
        session1_unmatched = list(set(list(np.arange(len(distances)))) - set(list(row_ind)))
    else:
        session2_unmatched = []
        session1_unmatched = []

    print(session1_unmatched, session2_unmatched)

    unmatched_2 = []
    for i in range(len(session2_unmatched)):
        # save the actual cell label of the unmatched cell
        # get the column of that cell (list of pairings with actual cell labels of session 1 and session 2 cells)
        # take any row from that column, the second number in the pair at (row,col) is the true cell label for the cell in session 2
        # make sure any pair in that column has the same second number [session1 id, session2 id]
        unit_id = pairs[0, session2_unmatched[i]][-1]
        # assert type(unit_id) == float or type(unit_id) == np.float64 or type(unit_id) == np.int64 or type(unit_id) == int or type(unit_id) == np.int32 or type(unit_id) == np.float32
        # assert unit_id == pairs[1, session2_unmatched[j]][-1]
        # unmatched_2.append([0, unit_id])
        unmatched_2.append(unit_id)
    
    unmatched_1 = []
    for i in range(len(session1_unmatched)):
        unit_id = pairs[session1_unmatched[i], 0][0]
        # unmatched_1.append([0, unit_id])
        unmatched_1.append(unit_id)

    return remaining_matches, remaining_match_distances, unmatched_2, unmatched_1

def compare_sessions(session1: Session, session2: Session, method='JSD', ses1_pca_feats=None, ses2_pca_feats=None):
    """
    FD = feature dict
    1 & 2 = sessions 1 & 2 (session 2 follows session 1)
    """
    # compare output of extract features from session1 and session2
    # return mapping dict from session2 old label to new matched label based on session1 cell labels

    distances, pairs, agg_distances = compute_distances(session1.get_spike_data()['spike_cluster'], session2.get_spike_data()['spike_cluster'], method, ses1_pca_feats=ses1_pca_feats, ses2_pca_feats=ses2_pca_feats)
    
    print(distances, pairs, agg_distances)
    full_matches, full_match_distances, remaining_distances, remaining_pairs = extract_full_matches(distances, pairs)

    remaining_pairs = np.array(remaining_pairs)
    if len(remaining_pairs.shape) == 2:
        remaining_pairs = remaining_pairs.reshape((1,-1))
    if len(remaining_pairs.shape) == 1:
        remaining_pairs = remaining_pairs.reshape((1,1,-1))

    # if remaining_pairs.shape[0] != 2:
    #     remaining_pairs = remaining_pairs.T

    remaining_matches, remaining_match_distances, unmatched_2, unmatched_1 = guess_remaining_matches(remaining_distances, remaining_pairs)

    remaining_matches = np.asarray(remaining_matches)
    full_matches = np.asarray(full_matches)
    if full_matches.size > 0 and remaining_matches.size > 0:
        if len(remaining_matches.shape) == 1:
            remaining_matches = remaining_matches.reshape((1,-1))
        if len(full_matches.shape) == 1:
            full_matches = full_matches.reshape((1,-1))
        matches = np.vstack((full_matches, remaining_matches))
        match_distances = np.hstack((full_match_distances, remaining_match_distances))
    else:
        matches = full_matches
        match_distances = np.asarray(full_match_distances)


    # THIS IS NEW CODE, should not be needed if guess remaingand exxtract full are workingn properly
    # HERE IS PROBLEM: very first session (true session 1) has 4 cells, very next session (true session 2) has 1 cell. That should be 1 matched (works)
    # and 3 unmatched_1 (session 1), but unmatched_1 comes out empty from guess_remaining_matches.
    # Need to add code for this edge case, this was done below but new code below does not work so start there
    print("VS")
    print(unmatched_1, matches, unmatched_2, match_distances)
    print(matches.shape, match_distances.shape)
    # print(len(unmatched_1) + len(matches) + len(unmatched_2), len(match_distances))
    print(list(session1.get_cell_data()['cell_ensemble'].get_cell_label_dict().keys()))
    print(list(session2.get_cell_data()['cell_ensemble'].get_cell_label_dict().keys()))
    print(session1.get_cell_data()['cell_ensemble'].get_cell_label_dict().keys())
    print(session2.get_cell_data()['cell_ensemble'].get_cell_label_dict().keys())

    ses1_cell_ids = session1.get_cell_data()['cell_ensemble'].get_cell_label_dict().keys()
    ses2_cell_ids = session2.get_cell_data()['cell_ensemble'].get_cell_label_dict().keys()
    print(ses1_cell_ids, ses2_cell_ids)
    print(len(unmatched_1) + len(matches), len(ses1_cell_ids))
    if len(unmatched_1) + len(matches) < len(ses1_cell_ids):
        for cell_label in list(ses1_cell_ids):
            print(float(cell_label), np.array(matches,dtype=float)[:,0])
            if float(cell_label) not in unmatched_1 and float(cell_label) not in np.array(matches,dtype=float)[:,0]:
                unmatched_1.append(cell_label)
    print(len(unmatched_2) + len(matches), len(ses2_cell_ids))
    if len(unmatched_2) + len(matches) < len(ses2_cell_ids):
        for cell_label in list(ses2_cell_ids):
            print(float(cell_label), np.array(matches,dtype=float)[:,1])
            if float(cell_label) not in unmatched_2 and float(cell_label) not in np.array(matches,dtype=float)[:,1]:
                unmatched_2.append(cell_label)
                
    # THIS IS NEW CODE

    # to_stack = []
    # if np.asarray(full_matches).size > 0:
    #     to_stack.append(full_matches)
    # if np.asarray(remaining_matches).size > 0:
    #     to_stack.append(remaining_matches)
    # if np.asarray(unmatched).size > 0:
    #     to_stack.append(unmatched)

    # matches = to_stack[0]
    # for i in range(1,len(to_stack)):
    #     matches = np.vstack((matches, to_stack[i]))
    print(unmatched_1, unmatched_2)
    print(matches, full_matches, remaining_matches)

    # matches = np.vstack((full_matches, remaining_matches))
    # matches = np.vstack((matches, unmatched))

    return matches, match_distances, unmatched_2, unmatched_1, agg_distances



