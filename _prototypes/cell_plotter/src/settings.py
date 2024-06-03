animal = {'animal_id': 'id', 'species': 'mouse', 'sex': 'F', 'age': 1, 'weight': 1, 'genotype': 'type', 'animal_notes': 'notes'}
devices = {'axona_led_tracker': True, 'implant': True}
implant = {'implant_id': 'id', 'implant_type': 'tetrode', 'implant_geometry': 'square', 'wire_length': 25, 'wire_length_units': 'um', 'implant_units': 'uV'}

session_settings = {'channel_count': 4, 'animal': animal, 'devices': devices, 'implant': implant}

settings_dict = {'session': session_settings, 
'ppm': None, # EDIT HERE (will read from pos file)
'smoothing_factor': 3, # EDIT HERE
'useMatchedCut': True, # EDIT HERE
'plotCellWaveforms': False, # EDIT HERE
'plotCellRatemap': True, # EDIT HERE
'outputStructure': 'nested', # EDIT HERE
# ... to add future plots
} 


###########################################################################################
                        # OUTPUT STRUCTURES TO CHOOSE FROM #
###########################################################################################

output_structures = ['nested', 'single', 'sequential']

# Nested output structure is: Animal_tetrode --> Session --> plot folders
# Single output structure is: Animal --> plot folders (tetrode1_session1_unit1)
# Sequential output structure is: Animal_tetrode --> Matched Unit --> sessions