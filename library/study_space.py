import os
import sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.subjects import AnimalMetadata, StudyMetadata, SessionMetadata
from library.ensemble_space import Cell, CellEnsemble, CellPopulation
from library.batch_space import SpikeClusterBatch, SpikeTrainBatch
from core.spikes import SpikeCluster, SpikeTrain, Spike, Event
from core.spatial import Position2D
from library.spike import sort_spikes_by_cell
from library.workspace import Workspace
from core.instruments import DevicesMetadata, ImplantMetadata, TrackerMetadata
from library.hafting_spatial_maps import SpatialSpikeTrain2D
from core.core_utils import make_seconds_index_from_rate


class Session(Workspace):
    def __init__(self, input_dict={}, **kwargs):
        self._input_dict = input_dict
        self.animal_id = None

        self.session_data, self.session_metadata = self._read_input_dict()

        if not isinstance(self.session_metadata, SessionMetadata) == 0:
            self.session_metadata = SessionMetadata({}, session_object=self)
            device_metadata = DevicesMetadata(input_dict={}, session_metadata=self.session_metadata)
            animal_metadata = AnimalMetadata(input_dict={}, session_metadata=self.session_metadata)
            self.session_metadata.metadata['devices'] = device_metadata
            self.session_metadata.metadata['animal'] = animal_metadata
            self.animal_id = animal_metadata.animal_id

        self.time_index = None
        self.datetime = None
        self.ses_key = None

        if 'smoothing_factor' in kwargs:
            self.smoothing_factor = kwargs['smoothing_factor']
        else:
            self.smoothing_factor = None

        dir_names = ['cluster',
                    'filters',
                    'maps',
                    'opexebo',
                    'scores',
                    'spatial',
                    'spike',
                    'tests']

        self.session_metadata.set_dir_names(dir_names)

    def set_smoothing_factor(self, smoothing_factor):
        self.smoothing_factor = smoothing_factor

    def update_time_index(self, class_object):
        if class_object.duration is None or class_object.sample_rate is None:
            time_index = None
        else:
            time_index = make_seconds_index_from_rate(class_object.duration, class_object.sample_rate)
        self.time_index = time_index

    def get_animal_id(self):
        if self.animal_id == None:
            print('Need to add animal metadata to session')
        return self.animal_id

    def set_animal_id(self):
        if 'animal' in self.session_metadata.metadata:
            self.animal_id = self.session_metadata.metadata['animal'].animal_id
            print('Session animal ID set: ' + str(self.animal_id))

    def _read_input_dict(self):
        session_data = {}
        session_metadata = {}

        if 'data' in self._input_dict:
            session_data = self._input_dict['data']
        else:
            session_data = SessionData()

        if 'metadata' in  self._input_dict:
            session_metadata = self._input_dict['metadata']
        else:
            session_metadata = SessionMetadata({}, session_object=self)

        return session_data, session_metadata

    def get_object_instances(self):
        instances = {}
        instances ['animal_metadata'] = self.get_animal_metadata()
        device_metadata = self.get_devices_metadata()
        for key in device_metadata:
            instances[key] = device_metadata[key]
        spike_data = self.get_spike_data()
        for key in spike_data:
            instances[key] = spike_data[key]
        return instances

    def get_animal_metadata(self):
        if 'animal' in self.session_metadata.metadata:
            animal_metadata = self.session_metadata.metadata['animal']
            return animal_metadata
        else:
            print('No animal metadata found')

    def get_devices_metadata(self):
        device_dict = {}
        if 'devices' in self.session_metadata.metadata:
            for key in self.session_metadata.metadata['devices'].devices_dict:
                device_dict[key] = self.session_metadata.metadata['devices'].devices_dict[key]
        return device_dict

    def get_spike_data(self):
        spike_dict = {}
        if self.session_data.data != None:
            for key in self.session_data.data:
                if 'spike' in key:
                    spike_dict[key] = self.session_data.data[key]
        return spike_dict

    def get_cell_data(self):
        cell_dict = {}
        if self.session_data.data != None:
            for key in self.session_data.data:
                if 'cell' in key:
                    cell_dict[key] = self.session_data.data[key]
        return cell_dict

    def get_position_data(self):
        pos_dict = {}
        if self.session_data.data != None:
            for key in self.session_data.data:
                if 'position' in key or 'pos' in key:
                    pos_dict[key] = self.session_data.data[key]
        return pos_dict

    def make_class(self, ClassName, input_dict: dict):
        class_object = ClassName(input_dict, session_metadata=self.session_metadata)

        if 'Metadata' in str(ClassName) or 'metadata' in str(ClassName):
            if isinstance(class_object, TrackerMetadata):
                self.session_metadata.metadata['devices']._add_device('axona_led_tracker', class_object)
            elif isinstance(class_object, ImplantMetadata):
                self.session_metadata.metadata['devices']._add_device('implant', class_object)
            elif isinstance(class_object, AnimalMetadata):
                self.session_metadata._add_metadata('animal', class_object)
                self.set_animal_id()
        elif 'cell' in str(ClassName) or 'Cell' in str(ClassName):
            if isinstance(class_object, CellEnsemble):
                self.session_data.data['cell_ensemble'] = class_object
            if isinstance(class_object, CellPopulation):
                self.session_data.data['cell_population'] = class_object
        else:
            if isinstance(class_object, SpikeClusterBatch):
                self.session_data.data['spike_cluster'] = class_object
            elif isinstance(class_object, SpikeTrain):
                self.session_data.data['spike_train'] = class_object
            elif isinstance(class_object, Position2D):
                self.session_data.data['position'] = class_object
            elif isinstance(class_object, SpatialSpikeTrain2D):
                self.session_data.data['spatial_spike_train'] = class_object

            if self.time_index is None and not isinstance(class_object, Position2D):
                self.update_time_index(class_object)
                class_object.time_index = self.time_index
                self.datetime = input_dict['datetime']

        return class_object


class Study(Workspace):
    def __init__(self, study_metadata, input_dict: dict, allowed_sessions=None):
        assert isinstance(study_metadata, StudyMetadata), 'The argument must be a Study object'
        self._input_dict = input_dict

        self.sessions = self._read_input_dict()

        self.animal_ids = self._get_animal_ids()

        self.animals = None
        self.allowed_sessions = allowed_sessions

    def _get_animal_ids(self):
        animal_ids = []
        for session in self.sessions:
            animal_ids.append(session.animal_id)
        return animal_ids

    def get_animal_by_id(self, animal_id):
        assert animal_id in self.animal_ids

        assert self.animals is not None, 'Call self.make_animals() first'

        for animal in self.animals:
            if animal.animal_id == animal_id:
                return animal

    def _read_input_dict(self):
        sessions = []
        for key in self._input_dict:
            assert isinstance(self._input_dict[key], Session), 'All arguments must be Session(Workspace) objects'
            sessions.append(self._input_dict[key])

        return sessions

    def add_session(self, session):
        assert isinstance(session, Session), 'The argument must be a Session(Workspace) object'
        self.sessions.append(session)

    def _sort_session_by_animal(self):
        sorted_animal_sessions = {}
        unsorted_animal_session = {}
        animal_session_datetime = {}

        for id in self.animal_ids:
            sorted_animal_sessions[id] = {}
            unsorted_animal_session[id] = []
            animal_session_datetime[id] = []

        for i in range(len(self.sessions)):
            animal_id = self.sessions[i].animal_id
            unsorted_animal_session[animal_id].append(self.sessions[i])
            animal_session_datetime[animal_id].append(self.sessions[i].datetime)

        for animal_id in list(unsorted_animal_session.keys()):
            sort_order = np.argsort(animal_session_datetime[animal_id])
            animal_sesions = np.asarray(unsorted_animal_session[animal_id])[sort_order]
            for j in range(len(animal_sesions)):
                sorted_animal_sessions[animal_id]['session_'+str(j+1)] = animal_sesions[j]
                animal_sesions[j].ses_key = 'session_'+str(j+1)

        return sorted_animal_sessions


    def make_animals(self):
        if self.animals is None:
            animal_sessions = self._sort_session_by_animal()
            animals = []
            for key in animal_sessions:
                if self.allowed_sessions is not None:
                    assert type(animal_sessions[key]) == dict
                    animal_instance = Animal(animal_sessions[key], allowed_sessions=self.allowed_sessions)
                    animals.append(animal_instance)
                else:
                    assert type(animal_sessions[key]) == dict
                    animal_instance = Animal(animal_sessions[key])
                    animals.append(animal_instance)
            self.animals = animals

    def get_animals(self):
        if self.animals == None:
            self.make_animals()
        return self.animals


class SessionData():
    def __init__(self, input_dict={}):
        self._input_dict = input_dict

        self.data = self._read_input_dict()

    def _read_input_dict(self):
        core_data_instances = {}

        for key in self._input_dict:
            core_data_instances[key] = self._input_dict[key]

        return core_data_instances


    def _add_session_data(self, key, data_class):
        assert 'Metadata' not in key and 'metadata' not in key, 'Cannot add metadata class to session data'
        self.data[key] = data_class


class Animal(Workspace):
    """
    Holds all sessions belonging to an animal, TO BE ADDED: ordered sequentially in time
    """
    ### Currently input is a list of dictionaries, once we save ordered sessions in x_io study class we will input nested dictionaries
    def __init__(self, input_dict: dict, allowed_sessions=None):
        self._input_dict = input_dict
        self.allowed_sessions = allowed_sessions

        # self.cell_ids = {}

        self.sessions, self.ensembles = self._read_input_dict()

        self.population = CellPopulation()

        self.animal_id = self.sessions[list(self.sessions.keys())[0]].animal_id

        self.stats_dict = {}


    def _read_input_dict(self):
        sessions = {}
        ensembles = {}
        isMatchedCut = False
        if 'matched' in self._input_dict['session_1'].session_metadata.file_paths['cut']:
            lbls = np.unique(np.concatenate(list(map(lambda x: np.unique(self._input_dict[x].get_spike_data()['spike_cluster'].cluster_labels), self._input_dict))))
            isMatchedCut = True
        for key in self._input_dict:
            if self.allowed_sessions is None or key in self.allowed_sessions:
                session = self._input_dict[key]
                assert isinstance(session, Session)
                # session is instance of SessionWorkspace, has SessionData and SessionMetadata
                # AnimalSession will hold Cells which hold SpikeTrains from SessionData
                if not isMatchedCut:
                    cell_ensemble, cell_ids = self._read_session(session, matched_lbls=None)
                else:
                    cell_ensemble, cell_ids = self._read_session(session, matched_lbls=lbls)
                sessions[key] = session
                ensembles[key] = cell_ensemble
                # self.cell_ids[key] = cell_ids
        return sessions, ensembles

    def _extract_core_classes(self, session):
        core_data = {}
        keys = list(session.session_data.data.keys())
        for i in range(len(keys)):
            core_type = session.session_data.data[keys[i]]
            core_data = self._check_core_type(keys[i], core_type, core_data)

        return core_data

    def _check_core_type(self, key, core_type, core_data):
        valid = (SpikeTrain, SpikeClusterBatch, Position2D, CellPopulation, CellEnsemble, SpatialSpikeTrain2D)
        # if isinstance(core_type, SpikeTrain):
        #     core_data[key] = core_type
        # elif isinstance(core_type, SpikeClusterBatch):
        #     core_data[key] = core_type
        # elif isinstance(core_type, Position2D):
        #     core_data[key] = core_type
        if isinstance(core_type, valid):
            core_data[key] = core_type
        else:
            print('Session data class is not valid, check inputs')
        return core_data

    def _read_session(self, session, matched_lbls=None):
        core_data = self._extract_core_classes(session)
        assert 'spike_cluster' in core_data, 'Need cluster label data to sort valid cells'
        spike_cluster = core_data['spike_cluster']
        # spike_train = core_data['spike_train']
        assert isinstance(spike_cluster, SpikeClusterBatch)
        # assert isinstance(spike_cluster, SpikeTrainBatch)
        good_sorted_cells, good_sorted_waveforms, good_clusters, good_label_ids = sort_spikes_by_cell(spike_cluster, matched_lbls=matched_lbls)
        # print(good_sorted_cells, good_label_ids)
        # spike_train.set_sorted_label_ids(good_label_ids)
        print('Session data added, spikes sorted by cell')
        ensemble = session.make_class(CellEnsemble, None)
        for i in range(len(good_label_ids)):
            cell_dict = {'event_times': good_sorted_cells[i], 'signal': good_sorted_waveforms[i], 'session_metadata': session.session_metadata, 'cluster': good_clusters[i]}
            cell = Cell(cell_dict, session_metadata=session.session_metadata)
            # cell = session.make_class(Cell, cell_dict)
            ensemble.add_cell(cell)
            # print('Cell Added')

        return ensemble, good_label_ids

    def add_session(self, session):
        cell_ensemble, _ = self._read_session(session)

        self.ensembles['session_'+str(len(self.ensembles)+1)] = cell_ensemble
        self.sessions['session_'+str(len(self.sessions)+1)] = session
        # self.cell_ids['session_'+str(len(self.sessions)+1)] = cell_ids





