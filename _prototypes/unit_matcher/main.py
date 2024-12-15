import os, sys
import numpy as np
import time
from sklearn.decomposition import PCA
import csv
import itertools
import sys
import xlsxwriter
import pandas as pd
import pickle
import tkinter as tk
from tkinter import filedialog
import traceback

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.study_space import Session, Study
# from x_io.rw.axona.batch_read import _read_cut
from _prototypes.unit_matcher.write_axona import get_current_cut_data, write_cut, format_new_cut_file_name, apply_remapping
from _prototypes.unit_matcher.read_axona import temp_read_cut
from _prototypes.unit_matcher.session import compare_sessions
from x_io.rw.axona.batch_read import make_study
from _prototypes.unit_matcher.unit import spike_level_feature_array

"""

This module reads axona cut and tetrode files. DONE (batch_read in x_io or single pair read in read_axona)
It then extracts the spike waveforms from the cut file. DONE (batch_read or single pair read in read_axona)
It then matches the spike waveforms to the units in the tetrode file. DONE (part of core class loading)
It then produces a dictionary of the spike waveforms for each unit. DONE (No dictionary --> Core class with waveforms per Spike. Collection Spike = Cluster)
It then extracts features from the spike waveforms for each unit. DONE
It then matches the spike waveforms to the units in the tetrode file. DONE
It then produces a remapping of the units in the tetrode file. DONE
It then applies the remapping to the cut file data. DONE (map dict changes cut data)
It then writes the remapped cut file data to a new cut file. DONE (new cut data writs to file)
Read, Retain, Map, Write
"""

def run_unit_matcher(paths=[], settings={}, method='JSD', dim_redux='PCA', study=None):
    print("Running Unit Matcher. Starting Time Tracker.")
    start_time = time.time()

    if study is None:
        assert len(paths) > 0 and len(settings) > 0
        # make study --> will load + sort data: SpikeClusterBatch (many units) --> SpikeCluster (a unit) --> Spike (an event)
        study = make_study(paths, settings)
        # make animals
        study.make_animals()
    elif isinstance(study, Study):
        print('Using pre-made study')
        if study.animals is None:
            study.make_animals()

    print('Loaded in animal ids: ' + str(np.unique(study.animal_ids)))

    print('Starting Unit Matching')

    animal_pca_results = {}

    # stop()
    for animal in study.animals:
        # SESSIONS INSIDE OF ANIMAL WILL BE SORTED SEQUENTIALLY AS PART OF ANIMAL(WORKSPACE) CLASS IN STUDY_SPACE.PY
        prev = None
        curr = None
        prev_map_dict = None
        isFirstSession = False

        session_mappings = {}
        animal_pca_results[animal.animal_id] = {}
        comparison_count = 1

        print('Processing animmal ' + str(animal.animal_id))

        if dim_redux == 'PCA':
            agg_session_feats = []
            agg_session_ids = []
            agg_cell_ids = []
            c = 1
            for session in animal.sessions:
                single_ses_cell_ids = []
                ses = animal.sessions[session]
                cluster_batch = ses.get_spike_data()['spike_cluster']
                clusters = cluster_batch.get_spike_cluster_instances()
                for i in range(len(clusters)):
                    feature_array = spike_level_feature_array(clusters[i], 1/cluster_batch.sample_rate)
                    if len(agg_session_feats) == 0:
                        agg_session_feats = feature_array
                        agg_session_ids = np.ones(len(feature_array)) * c
                        single_ses_cell_ids = np.ones(len(feature_array)) * i
                    else:
                        agg_session_feats = np.vstack((agg_session_feats, feature_array))
                        agg_session_ids = np.hstack((agg_session_ids, np.ones(len(feature_array)) * c))
                        single_ses_cell_ids = np.hstack((single_ses_cell_ids, np.ones(len(feature_array)) * i))
                agg_cell_ids.append(single_ses_cell_ids)
                c += 1

            agg_session_feats = np.array(agg_session_feats)
            agg_session_ids = np.array(agg_session_ids)
            agg_cell_ids = np.array(agg_cell_ids)

            # is in shape (samples, features)
            # pca = PCA(n_components=agg_session_feats.shape[1])
            pca = PCA(n_components=0.95)
            # pass in (features, samples) with components = features so output is (features, samples)
            pca.fit(agg_session_feats.T)

            components = pca.components_
            explained_variance = pca.explained_variance_
            singular_values = pca.singular_values_
            explained_variance_ratio = pca.explained_variance_ratio_

            # print('New here')
            # print(components.shape, explained_variance.shape, singular_values.shape, explained_variance_ratio.shape, agg_session_feats.shape, agg_session_ids.shape, agg_cell_ids.shape)
            # print(agg_session_ids[0].shape, agg_cell_ids[0].shape)

            # animal_pca_results[animal.animal_id]['feature_names'] = feature_names
            animal_pca_results[animal.animal_id]['explained_variance_ratio'] = explained_variance_ratio
            animal_pca_results[animal.animal_id]['singular_values'] = singular_values
            animal_pca_results[animal.animal_id]['explained_variance'] = explained_variance
            c = 1
            for comp in components:
                animal_pca_results[animal.animal_id]['component_' + str(c)] = comp
                c += 1
            c = 1
            for feat in agg_session_feats:
                animal_pca_results[animal.animal_id]['feat_' + str(c)] = feat
                c += 1
            animal_pca_results[animal.animal_id]['agg_session_ids'] = agg_session_ids
            c = 1
            for ses_cell_ids in agg_cell_ids:
                animal_pca_results[animal.animal_id]['cell_ids_ses_' + str(c)] = ses_cell_ids
                c += 1
        # stop()

            indiv_ses_feats, unique_ses_ids = split_agg_feature_array(agg_session_feats, agg_session_ids, agg_cell_ids)
        elif dim_redux is None or dim_redux == 'None' :
            indiv_ses_feats = None
            curr_pca = None
            prev_pca = None

        for session in animal.sessions:
            curr = animal.sessions[session]

            if dim_redux is not None:
                key_split = session.split('_')
                assert int(key_split[-1]) != 0

                curr_pca = indiv_ses_feats[int(key_split[-1]) - 1]
        
            # if first session of sequence there is no prev session
            if prev is not None:
                matches, match_distances, unmatched_2, unmatched_1, agg_distances = compare_sessions(prev, curr, method, ses1_pca_feats=prev_pca, ses2_pca_feats=curr_pca)
                print('COMPARED SESSION')
                print('matches: ' + str(matches))
                print('match_distances: ' + str(match_distances))
                print('unmatched_2: ' + str(unmatched_2))
                print('unmatched_1: ' + str(unmatched_1))
                print('agg_distances: ' + str(agg_distances))
                # print('Comparison ' + str(comparison_count))
                # print(matches, unmatched_1, unmatched_2)
                session_mappings[comparison_count] = {}
                session_mappings[comparison_count]['isFirstSession'] = isFirstSession
                session_mappings[comparison_count]['matches'] = matches
                session_mappings[comparison_count]['match_distances'] = match_distances
                session_mappings[comparison_count]['unmatched_2'] = unmatched_2
                session_mappings[comparison_count]['unmatched_1'] = unmatched_1
                session_mappings[comparison_count]['pair'] = (prev, curr)
                session_mappings[comparison_count]['agg_distances'] = agg_distances

                if isFirstSession:
                    isFirstSession = False

                comparison_count += 1

            else:
                isFirstSession = True

            prev = curr
            prev_pca = curr_pca

        # print(session_mappings)

        cross_session_matches, session_mappings = format_mapping_dicts(session_mappings)

        cross_session_matches = reorder_unmatched_cells(cross_session_matches)

        # print(cross_session_matches)

        # print('HERE CHECK WHAT HAPPENEING')

        remapping_dicts = apply_cross_session_remapping(session_mappings, cross_session_matches)
        print(remapping_dicts)

        # for loop or iter thru output to write cut,
        # e.g.
        for map_dict in remapping_dicts:
            new_cut_file_path, new_cut_data, header_data = format_cut(map_dict['session'], map_dict['map_dict'])
            print('Writing mapping: ' + str(map_dict['map_dict']))
            write_cut(new_cut_file_path, new_cut_data, header_data)
            print('Written to: ' + str(new_cut_file_path))

        # with xlsxwriter.Workbook(str(new_cut_file_path+'.xlsx')) as workbook:
        #     worksheet = workbook.add_worksheet(name=animal_id)
        #     for animal_id in animal_pca_results:
        #         pca_res = animal_pca_results[animal_id]
        #         for i, (key, val) in enumerate(pca_res.items()):
        #             worksheet.write(0,i,key)

        # old_path = session.session_metadata.file_paths['cut']

        file_str = new_cut_file_path.split(r'_matched.cut' )[0]

        # with open(str(file_str + '_pca.pickle'), 'wb') as handle:
        #     pickle.dump(animal_pca_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # with open(str(file_str + '_mappings.pickle'), 'wb') as handle:
        #     pickle.dump(session_mappings, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # file_name = r"testing_output.xlsx"
        # writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
        # for animal_id in animal_pca_results:
        #     df = pd.DataFrame(animal_pca_results[animal_id])
        #     df.to_excel(writer, sheet_name=animal_id)
        # writer.save()

    print("Unit Matcher Complete. Time Elapsed: " + str(time.time() - start_time))
    return study

def format_pca_stats_file_name(old_path):
    fp_split = old_path.split('.cut')
    new_fp = fp_split[0] + r'_matched_pca.xlsx'
    assert new_fp != old_path

    return new_fp

def split_agg_feature_array(agg_session_feats, agg_session_ids, agg_cell_ids):
    assert len(agg_session_feats) == len(agg_session_ids)
    unique_ses_ids = sorted(np.unique(agg_session_ids))
    indiv_ses_feats = []

    for ses_id in unique_ses_ids:
        ses_idx = np.where(agg_session_ids == ses_id)[0]
        session_cells = agg_cell_ids[int(ses_id - 1)]
        cell_ids = sorted(np.unique(session_cells))
        indiv_cell_feats = []
        for cell in cell_ids:
            # print('in')
            # print(ses_id, session_cells)
            cell_idx = np.where(session_cells == cell)[0]
            # print(cell, cell_idx)
            # print(len(agg_session_feats[ses_idx][cell_idx]))
            indiv_cell_feats.append(agg_session_feats[ses_idx][cell_idx])
            # print(agg_session_feats[ses_idx][cell_idx])
            # print('out')
        indiv_ses_feats.append(np.array(indiv_cell_feats))
    # print(unique_ses_ids)
    # print(len(indiv_ses_feats))
    # print(len(indiv_ses_feats[0]))
    # print(len(indiv_ses_feats[0][0]))
    # print(len(indiv_ses_feats[0][0][0]))
    # print(indiv_ses_feats.shape)
    return np.array(indiv_ses_feats), unique_ses_ids

def reorder_unmatched_cells(cross_session_matches):
    new_ordering = []

    for key in cross_session_matches:
        if len(cross_session_matches[key]['agg_matches']) > 0:
            new_ordering.append(cross_session_matches[key])

    for key in cross_session_matches:
        if len(cross_session_matches[key]['agg_matches']) == 0:
            new_ordering.append(cross_session_matches[key])

    new_cross_session_matches = {}
    c = 1
    for order in new_ordering:
        new_cross_session_matches[c] = order
        c += 1

    return new_cross_session_matches

def _agg_distances(cross_session_matches):
    cross_keys = list(cross_session_matches.keys())
    avg_JSD = {}
    cross_session_unmatched = []
    first_session_unmatched = []

    for cross_key in cross_keys:
        jsds = cross_session_matches[cross_key]['agg_distances']

        if len(jsds) > 0:
            avg_JSD[cross_key] = np.mean(jsds)
        else:
            avg_JSD[cross_key] = np.nan

            if 'unmatched_1' in cross_session_matches[cross_key]:
                first_session_unmatched.append(cross_key)
            if 'prev_unmatched' in cross_session_matches[cross_key]:
                cross_session_unmatched.append(cross_key)

    return avg_JSD, cross_session_unmatched, first_session_unmatched


def apply_cross_session_remapping(session_mappings, cross_session_matches):
    

    avg_JSD, cross_session_unmatched, first_session_unmatched = _agg_distances(cross_session_matches)


    JSD_vals = np.array(list(avg_JSD.values()))
    JSD_keys = np.array(list(avg_JSD.keys()))

    JSD_keys = JSD_keys[JSD_vals == JSD_vals]
    JSD_vals = JSD_vals[JSD_vals == JSD_vals]

    if len(JSD_vals) > 0:
        sorted_JSD_vals_idx = np.argsort(JSD_vals)
        # sorted_JSD_vals = np.sort(JSD_vals)
        sorted_JSD_keys = JSD_keys[sorted_JSD_vals_idx]

        for i in range(len(sorted_JSD_keys)):
            key = sorted_JSD_keys[i]
            # matches = np.array(cross_session_matches[key]['agg_matches']).reshape((-1,2))
            matches = np.array(cross_session_matches[key]['agg_matches'])

            comps = cross_session_matches[key]['comps']
            
            print(key)
            print(matches, comps)
            print(len(matches), len(comps))
            assert len(matches) == len(comps)

            print(key, matches, comps)

            prev_map_dict = None
            for j in range(len(comps)):
                pair = matches[j]
                print(pair)

                # if first comparison for this match does not contain the first session, cell is unmatched in session2 when compares to session 1
                # E.g. session1- session2, cell 5 in ses2 not matched. But session2- session3, cell 5 in ses2 is matched
                # mapping dictionary from ses2 to ses1 needs to update to account for new match label in session2-session3 comparison
                if j == 0 and not session_mappings[comps[j]]['isFirstSession']:
                    print('confusion')

                    map_to_update = session_mappings[comps[j]-1]['map_dict']

                    map_to_update[pair[0]] = i + 1

                    session_mappings[comps[j]-1]['map_dict'] = map_to_update

                    prev_map_dict = map_to_update

                if session_mappings[comps[j]]['isFirstSession']:
                    print('first ses map')

                    first_ses_map_dict = session_mappings[comps[j]]['first_ses_map_dict']
                    print(first_ses_map_dict, pair)
                    first_ses_map_dict[pair[0]] = i + 1
                    print(first_ses_map_dict)
                    session_mappings[comps[j]]['first_ses_map_dict'] = first_ses_map_dict

                    prev_map_dict = first_ses_map_dict

                map_dict = session_mappings[comps[j]]['map_dict']

                if prev_map_dict is not None:
                    map_dict[pair[1]] = prev_map_dict[pair[0]]
                else:
                    map_dict[pair[1]] = i + 1

                session_mappings[comps[j]]['map_dict'] = map_dict

                prev_map_dict = map_dict

                print(prev_map_dict, map_dict, pair)


        # add 1 for gap cell
        max_id = len(sorted_JSD_keys) + 1

        for i in range(len(cross_session_unmatched)):
            key = cross_session_unmatched[i]

            unmatched = cross_session_matches[key]['prev_unmatched']
            unmatched_comps = cross_session_matches[key]['unmatched_comps']

            for j in range(len(unmatched_comps)):
                map_dict = session_mappings[unmatched_comps[j]]['map_dict']

                max_id = max_id + i + 1
                map_dict[unmatched[j]] = max_id

                session_mappings[unmatched_comps[j]]['map_dict'] = map_dict

        for i in range(len(first_session_unmatched)):
            key = first_session_unmatched[i]

            unmatched = cross_session_matches[key]['unmatched_1']
            unmatched_comps = cross_session_matches[key]['unmatched_1_comps']

            for j in range(len(unmatched_comps)):
                first_ses_map_dict = session_mappings[unmatched_comps[j]]['first_ses_map_dict']

                max_id = max_id + i + 1
                first_ses_map_dict[unmatched[j]] = max_id

                session_mappings[unmatched_comps[j]]['first_ses_map_dict'] = first_ses_map_dict

    remapping_dicts = []
    for key in session_mappings:
        if session_mappings[key]['isFirstSession']:
            print('must be in here', session_mappings[key]['first_ses_map_dict'])
            remapping_dicts.append({'map_dict': session_mappings[key]['first_ses_map_dict'], 'session': session_mappings[key]['pair'][0]})
        remapping_dicts.append({'map_dict': session_mappings[key]['map_dict'], 'session': session_mappings[key]['pair'][1]})


    return remapping_dicts

def format_mapping_dicts(session_mappings):
    cross_session_matches = {}
    prev_map_dict = None
    prev_matched_labels = None
    prev_unmatched = []

    for comparison in session_mappings:
        unmatched_2 = session_mappings[comparison]['unmatched_2']

        print('loading1')
        print(unmatched_2, cross_session_matches)

        cross_session_matches, map_dict = make_mapping_dict(session_mappings, comparison, cross_session_matches)

        session_mappings[comparison]['map_dict'] = map_dict

        print('loading2')
        print(unmatched_2, map_dict, cross_session_matches)

        # if session_mappings[comparison]['isFirstSession']:
        #     prev_map_dict = session_mappings[comparison]['first_ses_map_dict']

        print('pre')
        print(comparison, cross_session_matches, prev_map_dict, prev_matched_labels)

        if prev_map_dict is not None:
            cross_session_matches, matched_labels, prev_matched_now_unmatched = match_cross_session_pairings(session_mappings, comparison, cross_session_matches, prev_map_dict, prev_matched_labels)
            print('mid')
            print(cross_session_matches)
            cross_session_matches = check_prev_unmatched(session_mappings, comparison, cross_session_matches, prev_unmatched, matched_labels)

        else:
            matched_labels = None

            # if len(prev_unmatched) > 0:
            #     for i in range(len(prev_unmatched)):
            #         unmatched_id = prev_unmatched[i]
            #         cross_ses_key = max(list(cross_session_matches.keys())) + 1
            #         if unmatched_id in list(map_dict.values()):
            #             idx = np.where(unmatched_id in list(map_dict.values()))[0]
            #             cross_session_matches[cross_ses_key] = {}
            #             cross_session_matches[cross_ses_key]['agg_matches'] = []
            #             cross_session_matches[cross_ses_key]['agg_distances'] = []
            #             cross_session_matches[cross_ses_key]['comps'] = []
            #             cross_session_matches[cross_ses_key]['agg_matches'].append(matches[idx])
            #             cross_session_matches[cross_ses_key]['agg_distances'].append(match_distances[idx])
            #             cross_session_matches[cross_ses_key]['comps'].append(comparison)
            #             matched_labels[list(map_dict.keys())[idx]] = cross_ses_key
            #         else:
            #             cross_session_matches[cross_ses_key] = {}
            #             cross_session_matches[cross_ses_key]['agg_matches'] = []
            #             cross_session_matches[cross_ses_key]['agg_distances'] = []
            #             cross_session_matches[cross_ses_key]['comps'] = []
            #             cross_session_matches[cross_ses_key]['prev_unmatched'] = []
            #             cross_session_matches[cross_ses_key]['unmatched_comps'] = []
            #             cross_session_matches[cross_ses_key]['prev_unmatched'].append(unmatched_id)
            #             cross_session_matches[cross_ses_key]['unmatched_comps'].append(comparison)

        prev_map_dict = map_dict
        prev_matched_labels = matched_labels
        prev_unmatched = unmatched_2

    cross_session_matches = add_last_unmatched(comparison, cross_session_matches, prev_unmatched)

    return cross_session_matches, session_mappings


def make_mapping_dict(session_mappings, comparison, cross_session_matches):
    map_dict = {}
    match_distances = np.asarray(session_mappings[comparison]['match_distances'])
    matches = np.asarray(session_mappings[comparison]['matches'])
    unmatched_1 = session_mappings[comparison]['unmatched_1']

    if session_mappings[comparison]['isFirstSession']:
        first_ses_map_dict = {}
        for i in range(len(matches)):
            cross_session_matches[int(matches[i][1])] = {}
            cross_session_matches[int(matches[i][1])]['agg_matches'] = []
            cross_session_matches[int(matches[i][1])]['agg_distances'] = []
            cross_session_matches[int(matches[i][1])]['comps'] = []
            cross_session_matches[int(matches[i][1])]['agg_matches'].append(matches[i])
            cross_session_matches[int(matches[i][1])]['agg_distances'].append(match_distances[i])
            cross_session_matches[int(matches[i][1])]['comps'].append(comparison)

            # map_dict[int(matches[i][0])] = i + 1
            first_ses_map_dict[int(matches[i][0])] = int(matches[i][0])

        if len(unmatched_1) > 0:
            for i in range(len(unmatched_1)):
                if int(unmatched_1[i]) not in cross_session_matches:
                    cross_session_matches[int(unmatched_1[i])] = {}
                    cross_session_matches[int(unmatched_1[i])]['agg_matches'] = []
                    cross_session_matches[int(unmatched_1[i])]['agg_distances'] = []
                    cross_session_matches[int(unmatched_1[i])]['comps'] = []
                cross_session_matches[int(unmatched_1[i])]['unmatched_1'] = []
                cross_session_matches[int(unmatched_1[i])]['unmatched_1_comps'] = []
                cross_session_matches[int(unmatched_1[i])]['unmatched_1'].append(int(unmatched_1[i]))
                cross_session_matches[int(unmatched_1[i])]['unmatched_1_comps'].append(comparison)

        session_mappings[comparison]['first_ses_map_dict'] = first_ses_map_dict

        print('FIRST SES MAP DICT')
        print(first_ses_map_dict)

    for pair in matches:
        map_dict[int(pair[1])] = int(pair[0])

    session_mappings[comparison]['map_dict'] = map_dict

    return cross_session_matches, map_dict


def match_cross_session_pairings(session_mappings, comparison, cross_session_matches, prev_map_dict, prev_matched_labels):
    print('IN')
    print(comparison, cross_session_matches, prev_map_dict, prev_matched_labels)
    match_distances = np.asarray(session_mappings[comparison]['match_distances'])
    matches = np.asarray(session_mappings[comparison]['matches'])
    map_dict = session_mappings[comparison]['map_dict']
    print('ses mappings')
    print(matches, matches, match_distances)
    prev_matched_now_unmatched = []

    matched_labels = {}
    for key in list(prev_map_dict.keys()):
        idx = np.where(np.array(list(map_dict.values())) == key)[0]
        if len(idx) > 0:
            idx = idx[0]

            if prev_matched_labels is None:
                cross_ses_key = key
            else:
                cross_ses_key = prev_matched_labels[key]

            cross_session_matches[cross_ses_key]['agg_matches'].append(matches[idx])
            cross_session_matches[cross_ses_key]['agg_distances'].append(match_distances[idx])
            cross_session_matches[cross_ses_key]['comps'].append(comparison)

            matched_labels[np.array(list(map_dict.keys()))[idx]] = cross_ses_key
        prev_matched_now_unmatched.append(key)
    
    print('DONE')
    print(cross_session_matches, matched_labels, prev_matched_now_unmatched)
    print('OUT')
    return cross_session_matches, matched_labels, prev_matched_now_unmatched


def check_prev_unmatched(session_mappings, comparison, cross_session_matches, prev_unmatched, matched_labels):
    match_distances = np.asarray(session_mappings[comparison]['match_distances'])
    matches = np.asarray(session_mappings[comparison]['matches'])
    map_dict = session_mappings[comparison]['map_dict']

    if len(prev_unmatched) > 0:
        for i in range(len(prev_unmatched)):
            unmatched_id = prev_unmatched[i]
            cross_ses_key = max(list(cross_session_matches.keys())) + 1
            if unmatched_id in list(map_dict.values()):
                idx = np.where(np.array(list(map_dict.values())) == unmatched_id)[0][0]
                cross_session_matches[cross_ses_key] = {}
                cross_session_matches[cross_ses_key]['agg_matches'] = []
                cross_session_matches[cross_ses_key]['agg_distances'] = []
                cross_session_matches[cross_ses_key]['comps'] = []
                cross_session_matches[cross_ses_key]['agg_matches'].append(matches[idx])
                cross_session_matches[cross_ses_key]['agg_distances'].append(match_distances[idx])
                cross_session_matches[cross_ses_key]['comps'].append(comparison)
                matched_labels[list(map_dict.keys())[idx]] = cross_ses_key
                # nowMatched = True
            else:
                cross_session_matches[cross_ses_key] = {}
                cross_session_matches[cross_ses_key]['agg_matches'] = []
                cross_session_matches[cross_ses_key]['agg_distances'] = []
                cross_session_matches[cross_ses_key]['comps'] = []
                cross_session_matches[cross_ses_key]['prev_unmatched'] = []
                cross_session_matches[cross_ses_key]['unmatched_comps'] = []
                cross_session_matches[cross_ses_key]['prev_unmatched'].append(unmatched_id)
                cross_session_matches[cross_ses_key]['unmatched_comps'].append(comparison)
                # nowMatched = False

    return cross_session_matches

def add_last_unmatched(comparison, cross_session_matches, last_unmatched):
    if len(last_unmatched) > 0:
        for i in range(len(last_unmatched)):
            unmatched_id = last_unmatched[i]
            cross_ses_key = max(list(cross_session_matches.keys())) + 1
            cross_session_matches[cross_ses_key] = {}
            cross_session_matches[cross_ses_key]['agg_matches'] = []
            cross_session_matches[cross_ses_key]['agg_distances'] = []
            cross_session_matches[cross_ses_key]['comps'] = []
            cross_session_matches[cross_ses_key]['prev_unmatched'] = []
            cross_session_matches[cross_ses_key]['unmatched_comps'] = []
            cross_session_matches[cross_ses_key]['prev_unmatched'].append(unmatched_id)
            cross_session_matches[cross_ses_key]['unmatched_comps'].append(comparison)

    return cross_session_matches






# def run_unit_matcher(paths=[], settings={}, study=None):
#     if study is None:
#         assert len(paths) > 0 and len(settings) > 0
#         # make study --> will load + sort data: SpikeClusterBatch (many units) --> SpikeCluster (a unit) --> Spike (an event)
#         study = make_study(paths, settings)
#         # make animals
#         study.make_animals()
#     elif isinstance(study, Study):
#         study.make_animals()

#     print('Starting Unit Matching')

#     for animal in study.animals:
#         # SESSIONS INSIDE OF ANIMAL WILL BE SORTED SEQUENTIALLY AS PART OF ANIMAL(WORKSPACE) CLASS IN STUDY_SPACE.PY
#         prev = None
#         curr = None
#         prev_map_dict = None
#         isFirstSession = False
#         for session in animal.sessions:
#             curr = animal.sessions[session]

#             # print(prev, curr, isFirstSession)


#             # if first session of sequence there is no prev session
#             if prev is not None:
#                 matches, match_distances, unmatched_2, unmatched_1 = compare_sessions(prev, curr)

#                 if isFirstSession:
#                     map_dict_first = map_unit_matches_first_session(matches, match_distances, unmatched_1)
#                     first_ses_cut_file_path, new_cut_data, header_data = format_cut(prev, map_dict_first)
#                     # first_ses_cut_file_path = prev.session_metadata.file_paths['cut']
#                     # cut_data, header_data = get_current_cut_data(first_ses_cut_file_path)
#                     # new_cut_data = apply_remapping(cut_data, map_dict)
#                     # new_cut_file_path = format_new_cut_file_name(first_ses_cut_file_path)
#                     print('Writing mapping: ' + str(map_dict_first))
#                     write_cut(first_ses_cut_file_path, new_cut_data, header_data)
#                     isFirstSession = False
#                     # print('NEW')
#                     # print(map_dict_first.values())
#                     # print(map_dict_first.keys())
#                     prev_map_dict = map_dict_first

#                 # prev_cut_file_path = prev.session_metadata.file_paths['cut']
#                 # prev_matched_cut_file = format_new_cut_file_name(prev_cut_file_path)
#                 # updated_cut_data, _ = get_current_cut_data(prev_matched_cut_file)
#                 # updated_labels = np.unique(updated_cut_data)

#                 map_dict = map_unit_matches_sequential_session(matches, unmatched_2)

#                 # map dict is built on non matched cut labels, remap dict based on previous mapped dictionary
#                 values = list(map_dict.values())
#                 keys = list(map_dict.keys())
#                 for i in range(len(values)):
#                     if values[i] in prev_map_dict:
#                         map_dict[keys[i]] = prev_map_dict[values[i]]

#                 new_cut_file_path, new_cut_data, header_data = format_cut(curr, map_dict)
#                 print('Writing mapping: ' + str(map_dict))
#                 write_cut(new_cut_file_path, new_cut_data, header_data)
#                 prev_map_dict = map_dict
#                 # print('NEW')
#                 # print(map_dict.values())
#                 # print(map_dict.keys())
#             else:
#                 isFirstSession = True
#             # update refernece of first session in pair
#             # prev = curr
#             # curr = session

#             prev = curr

#     return study

def format_cut(session: Session, map_dict: dict):
    cut_file_path = session.session_metadata.file_paths['cut']
    cut_data, header_data = get_current_cut_data(cut_file_path)
    new_cut_data = apply_remapping(cut_data, map_dict)
    new_cut_file_path = format_new_cut_file_name(cut_file_path)
    return new_cut_file_path, new_cut_data, header_data


def map_unit_matches_sequential_session(matches, unmatched):
    map_dict = {}

    for pair in matches:
        map_dict[int(pair[1])] = int(pair[0])

    # highest_matched_id = max(map_dict, key=map_dict.get)
    highest_matched_id = max(map_dict.values())
    # unmatched = sorted(unmatched)
    empty_cell_id = highest_matched_id + 1
    unmatched_cell_start_id = empty_cell_id + 1
    for i in range(len(unmatched)):
        map_dict[unmatched[i]] = unmatched_cell_start_id + i
    # print('Mappings :' + str(map_dict))
    return map_dict

def map_unit_matches_first_session(matches, match_distances, unmatched):
    sort_ids = np.argsort(match_distances)
    matches = np.asarray(matches)[sort_ids]

    map_dict = {}

    for i in range(len(matches)):
        map_dict[int(matches[i][0])] = i + 1

    highest_matched_id = max(map_dict.values())
    # unmatched = sorted(unmatched)
    empty_cell_id = highest_matched_id + 1
    unmatched_cell_start_id = empty_cell_id + 1

    for i in range(len(unmatched)):
        map_dict[unmatched[i]] = unmatched_cell_start_id + i
    return map_dict

# def map_unit_matches(matches, match_distances, unmatched):
#     sort_ids = np.argsort(match_distances)
#     matches = np.asarray(matches)[sort_ids]

#     map_dict = {}

#     for pair in matches:
#         map_dict[int(pair[1])] = int(pair[0])

#     for i in range(len(matches))

#     highest_matched_id = max(map_dict, key=map_dict.get)
#     unmatched = sorted(unmatched)
#     empty_cell_id = highest_matched_id + 1
#     unmatched_cell_start_id = empty_cell_id + 1

#     for i in range(len(unmatched)):
#         map_dict[unmatched[i]] = unmatched_cell_start_id + i



#     return map_dict

if __name__ == '__main__':
    """ If a setting is not used for your analysis (e.g. smoothing_factor), just pass in an arbitrary value or pass in 'None' """
    STUDY_SETTINGS = {

        'ppm': 589,  # EDIT HERE

        'smoothing_factor': 3, # EDIT HERE

        'useMatchedCut': False,  # EDIT HERE, set to False if you want to use runUnitMatcher, set to True after to load in matched.cut file

        'arena_size' : None
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
    # settings_dict['single_tet'] = 7 

    start_time = time.time()
    root = tk.Tk()
    root.withdraw()
    data_dir = filedialog.askdirectory(parent=root,title='Please select a data directory.')

    import sys
    subdirs = np.sort([ f.path for f in os.scandir(data_dir) if f.is_dir() ])
    count = 1
    # sys.stdout = open(r'C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit\_prototypes\unit_matcher\testlog.txt', 'w')
    for subdir in subdirs:
        try:
            sub_study = make_study(subdir,settings_dict=settings_dict)
            sub_study.make_animals()
            run_unit_matcher(paths=[], settings=settings_dict, study=sub_study)
            count += 1
        except Exception:
            print(traceback.format_exc())
            print('DID NOT WORK FOR DIRECTORY ' + str(subdir))
    # sys.stdout.close()
    # sys.stdout = sys.__stdout__
    print('COMPLETED UNIT MATCHING')


