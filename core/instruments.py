
from core.subjects import SessionMetadata


class DevicesMetadata():
    def __init__(self, input_dict={}, **kwargs):
        self._input_dict = input_dict

        self.devices_dict, self.session_metadata = self._read_input_dict()

        if 'session_metadata' in kwargs:
            if self.session_metadata != None: 
                print('Ses metadata is in the input dict and init fxn, init fnx will override')
            self.session_metadata = kwargs['session_metadata']

    def _read_input_dict(self):
        devices = {}
        session_metadata = None
        for key in self._input_dict:

            if key == 'session_metadata':
                session_metadata = self._input_dict['session_metadata']

            if key == 'implant':
                implant = self._input_dict[key]
                devices[key] = implant
            elif key == 'axona_led_tracker':
                tracker = self._input_dict[key]
                devices[key] = tracker

            # ... continue with more device types

        return devices, session_metadata

    def _add_device(self, key, device_class):
        assert isinstance(device_class, DevicesMetadata), 'Device to be added needs to be instance of device metadata'
        self.devices_dict[key] = device_class



class ImplantMetadata(DevicesMetadata):
    def __init__(self, input_dict: dict, **kwargs):
        self._input_dict = input_dict

        self.implant_id, self.implant_geometry, self.implant_type, self.implant_data, self.wire_length, self.wire_length_units, self.implant_units, self.session_metadata = self._read_input_dict()

        if 'session_metadata' in kwargs:
            if self.session_metadata != None: 
                print('Ses metadata is in the input dict and init fxn, init fnx will override')
            self.session_metadata = kwargs['session_metadata']


    def _read_input_dict(self):
        implant_id = None
        implant_geometry = None
        implant_type = None
        implant_data = None
        wire_length = None
        wire_length_units = None
        implant_units = None
        session_metadata = None

        if 'implant_id' in self._input_dict:
            implant_id = self._input_dict['implant_id']
        if 'implant_geometry' in self._input_dict:
            implant_geometry = self._input_dict['implant_geometry']
        if 'implant_type' in self._input_dict:
            implant_type = self._input_dict['implant_type']
        if 'implant_data' in self._input_dict:
            implant_data = self._input_dict['implant_data']
        if 'wire_length' in self._input_dict:
            wire_length = self._input_dict['wire_length']
        if 'wire_length_units' in self._input_dict:
            wire_length_units = self._input_dict['wire_length_units']
        if 'implant_units' in self._input_dict:
            implant_units = self._input_dict['implant_units']
        if 'session_metadata' in self._input_dict:
            session_metadata = self._input_dict['session_metadata']

        return implant_id, implant_geometry, implant_type, implant_data, wire_length, wire_length_units, implant_units, session_metadata


class TrackerMetadata(DevicesMetadata):
    def __init__(self, input_dict: dict, **kwargs):
        self._input_dict = input_dict

        self.led_tracker_id, self.led_location, self.led_position_data, self.x, self.y, self.time, self.arena_height, self.arena_width, self.session_metadata = self._read_input_dict()
        
        if 'session_metadata' in kwargs:
            if self.session_metadata != None: 
                print('Ses metadata is in the input dict and init fxn, init fnx will override')
            self.session_metadata = kwargs['session_metadata']

    def _read_input_dict(self):
        led_tracker_id = None
        led_location = None
        led_position_data = None
        x = None
        y = None
        time = None
        arena_height = None
        arena_width = None
        session_metadata = None

        if 'led_tracker_id' in self._input_dict:
            led_tracker_id = self._input_dict['led_tracker_id']
        if 'led_location' in self._input_dict:
            led_location = self._input_dict[led_location]
        if 'led_position_data' in self._input_dict:
            led_position_data = self._input_dict['led_position_data']
        if 'x' in self._input_dict:
            x = self._input_dict['x']
        if 'y' in self._input_dict:
            y = self._input_dict['y']
        if 't' in self._input_dict:
            time = self._input_dict['t']
        if 'arena_height' in self._input_dict:
            arena_height = self._input_dict['arena_height']
        if 'arena_width' in self._input_dict:
            arena_width = self._input_dict['arena_width']
        if 'session_metadata' in self._input_dict:
            session_metadata = self._input_dict['session_metadata']

        return led_tracker_id, led_location, led_position_data, x, y, time, arena_height, arena_width, session_metadata



        