import os, sys

# from prototypes.wave_form_sorter.sort_cell_spike_times import sort_cell_spike_times

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.spikes import *
from library.workspace import Workspace

class CellPopulation(Workspace): 
    def __init__(self, input_dict=None):
        self._input_dict = input_dict

        self.ensembles = []
        
        if self._input_dict != None:
            self.ensembles = self._read_input_dict()

    def _read_input_dict(self):
        ensembles = []
        for key in self._input_dict:
            ensemble = self._input_dict[key]
            assert isinstance(ensemble, CellEnsemble)
            ensembles.append(ensemble) 

        return ensembles

    def add_ensemble(self, ensemble):
        assert isinstance(ensemble, CellEnsemble)
        self.ensembles.append(ensemble)

class CellEnsemble(Workspace):
    """
    To manipulate groups of cells, flexible class, optional input is instances of Cell, can also add cells individually
    """
    def __init__(self, input_dict=None, **kwargs):
        self._input_dict = input_dict
        
        self.cells = []

        if self._input_dict != None:
            self.cells = self._read_input_dict()

        if 'sessison_metadata' in kwargs:
            self.session_metadata == kwargs['session_metadata']
            self.animal_id = self.session_metadata.metadata['animal'].animal_id
        else:
            self.session_metadata = None
            self.animal_id = None

        self.cell_label_dict = None
        self.waveforms = None
        self.event_times = None
        # self.event_ids = self.collect_cell_event_times()
        self.waveform_ids = None 
        self.event_ids = None

    def get_waveforms(self):
        if self.waveforms is None:
            self.waveforms, self.waveform_ids = self.collect_cell_signal()
        return self.waveforms, self.waveform_ids
    
    def get_event_times(self):
        if self.event_times is None:
            self.event_times, self.event_ids = self.collect_cell_event_times()
        return self.event_times, self.event_ids

    def collect_cell_signal(self):
        signals = []
        for cell_id in np.sort(self.get_label_ids()):
            # print('here', str(cell_id))
            # print(np.asarray(self.get_cell_by_id(cell_id).signal).shape)
            signals.append(self.get_cell_by_id(cell_id).signal)
        # return signals
        return list(map(lambda x: self.get_cell_by_id(x).signal, np.sort(self.get_label_ids()))), list(map(lambda x: np.ones(len(self.get_cell_by_id(x).event_times)) * x, np.sort(self.get_label_ids())))
    
    def collect_cell_event_times(self):
        # event_times = []
        # for cell_id in np.sort(self.get_label_ids()):
        #     signals.append(self.get_cell_by_id(cell_id).signal)
        return list(map(lambda x: self.get_cell_by_id(x).event_times, np.sort(self.get_label_ids()))), list(map(lambda x: np.ones(len(self.get_cell_by_id(x).event_times)) * x, np.sort(self.get_label_ids())))

    def _read_input_dict(self):
        cells = []
        for key in self._input_dict:
            cell = self._input_dict[key]
            assert isinstance(cell, Cell)
            cells.append(cell) 

        return cells

    def add_cell(self, cell):
        assert isinstance(cell, Cell)
        self.cells.append(cell)

    def get_label_ids(self):
        ids = []
        for cell in self.cells:
            ids.append(cell.cluster.cluster_label)
        return ids

    def get_cell_label_dict(self):
        if self.cell_label_dict is None:
            self.cell_label_dict = {}
            for cell in self.cells:
                lbl = cell.cluster.cluster_label
                self.cell_label_dict[lbl] = cell
        return self.cell_label_dict

    def get_cell_by_id(self, id):
        if self.cell_label_dict is None:
            self.get_cell_label_dict()
        # print(self.cell_label_dict)
        return self.cell_label_dict[int(id)]


class Cell(Workspace):
    """
    A single cell belonging to a session of an animal
    """
    def __init__(self, input_dict: dict, **kwargs):
        self._input_dict = input_dict

        self.event_times, self.signal, self.session_metadata, self.cluster = self._read_input_dict()

        if 'sessison_metadata' in kwargs and self.session_metadata is None:
            self.session_metadata == kwargs['session_metadata']
        elif self.session_metadata is not None:
            self.animal_id = self.session_metadata.metadata['animal'].animal_id
            self.time_index = self.session_metadata.session_object.time_index
        else:
            self.animal_id = None
            self.session_metadata = None
            self.time_index = None


        self.dir_names = self.session_metadata.dir_names
        self.stats_dict = self._init_stats_dict()
        self.ses_key = self.session_metadata.session_object.ses_key
        # self.stats_dict = {}


        if 'spike_cluster' in self.session_metadata.session_object.get_spike_data():
            self.spike_cluster = self.session_metadata.session_object.get_spike_data()['spike_cluster']
        else:
            self.spike_cluster = None

 

        

    def _read_input_dict(self):
        event_times = None
        waveforms = None 
        cluster = None

        if 'event_times' in self._input_dict:
            event_times = self._input_dict['event_times']
        else:
            print('No event data provided to Cell')

        if 'signal' in self._input_dict:
            signal = self._input_dict['signal']
        else:
            print('No signal data provided')

        if 'session_metadata' in self._input_dict:
            session_metadata = self._input_dict['session_metadata']
        else:
            print('No session metadata provided, cannot effectively track cells')

        if 'cluster' in self._input_dict:
            cluster = self._input_dict['cluster']
        else:
            print('No cluster reference provided, cannot use certain lib/cluster modules')

        return event_times, signal, session_metadata, cluster

    def _init_stats_dict(self):
        stats_dict = {}
        
        for dir in self.dir_names:
            if dir != 'tests' and 'cache' not in dir:
                stats_dict[dir] = {}

        return stats_dict
    
    def add_cluster_stats(self):
        for key in self.cluster.stats_dict:
            self.stats_dict[key] = self.cluster.stats_dict[key]

