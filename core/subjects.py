import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)



class AnimalMetadata():
    def __init__(self, input_dict: dict, **kwargs):
        self._input_dict = input_dict

        self.animal_id, self.species, self.sex, self.age, self.weight, self.genotype, self.animal_notes, self.session_metadata = self._read_input_dict()
        
        if 'session_metadata' in kwargs:
            if self.session_metadata != None: 
                print('Ses metadata is in the input dict and init fxn, init fnx will override')
            self.session_metadata = kwargs['session_metadata']


    def _read_input_dict(self):
        animal_id, species, sex, age, weight, genotype, animal_notes, session_metadata = None, None, None, None, None, None, None, None

        if 'animal_id' in self._input_dict:
            animal_id = self._input_dict['animal_id']
        if 'species' in self._input_dict:
            species = self._input_dict['species']
        if 'sex' in self._input_dict:
            sex = self._input_dict['sex']
        if 'age' in self._input_dict:
            age = self._input_dict['age']
        if 'weight' in self._input_dict:
            weight = self._input_dict['weight']
        if 'genotype' in self._input_dict:
            genotype = self._input_dict['genotype']
        if 'animal_notes' in self._input_dict:
            animal_notes = self._input_dict['animal_notes']
        if 'session_metadata' in self._input_dict:
            session_metadata = self._input_dict['sessionn_metadata']

        return animal_id, species, sex, age, weight, genotype, animal_notes, session_metadata


class SessionMetadata():
    def __init__(self, input_dict: dict, **kwargs):
        self._input_dict = input_dict   

        self.metadata, self.session_object = self._read_input_dict()
        
        if 'session_object' in kwargs:
            if self.session_object != None: 
                print('Ses object is in the input dict and init fxn, init fnx will override')
            self.session_object = kwargs['session_object']

        # self.dir_names = [x[1] for x in os.walk('library')][0]
        self.dir_names = None

        self.file_paths = {}

    def set_dir_names(self, names):
        self.dir_names = {}
        for key in names:
            self.dir_names[key] = {}

    def add_file_paths(self, cut_file, tet_file, pos_file, ppm):
        self.file_paths['cut'] = cut_file
        self.file_paths['tet'] = tet_file
        self.file_paths['pos'] = pos_file
        self.file_paths['ppm'] = ppm

    def _read_input_dict(self):
        core_metadata_instances = {} 
        session_object = None
        
        for key in self._input_dict:
            if key == 'session_object':
                session_object = self._input_dict[key]
            else:
                core_metadata_instances[key] = self._input_dict[key]

        return core_metadata_instances, session_object

    def _add_metadata(self, key, metadata_class):
        assert key == 'animal' or key == 'devices', 'Cann only add AnimalMetadata and DeviceMetadata objects to session'
        self.metadata[key] = metadata_class
        


class StudyMetadata():
    def __init__(self, input_dict: dict):
        self._input_dict = input_dict 
    
    def _read_input_dict(self):
        pass


    
