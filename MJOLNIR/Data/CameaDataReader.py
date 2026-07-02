import h5py
import numpy as np
from pathlib import Path

from MJOLNIR.Data.DataReader import InstrumentCalibration
from MJOLNIR.Data.NexusDataReader import NexusDataReader 

class CameaDataReader(NexusDataReader):
    """Parse a CAMEA Nexus file"""

    n_analyser = 8
    n_detector = 104
    instrument = "CAMEA"
    entry = 'entry'

    # CAMEA-specific hdf5 keys
    nexus_keys = {
        **NexusDataReader.nexus_keys,
        'counts': f'entry/{instrument}/detector/data',
        'A4' : f'entry/{instrument}/analyzer/polar_angle',
        'A4Offset' : f'entry/{instrument}/analyzer/polar_angle_offset',
        'Ei' : f'entry/{instrument}/monochromator/energy',
        'singleDetector1': f'entry/{instrument}/segment_1/data',
        'singleDetector8': f'entry/{instrument}/segment_8/data',
        'monitor1': 'entry/control/data',
        'monitor2': 'entry/monitor_2/data'
    }

    # CAMEA-specific hdf5 metadata keys
    nexus_meta_keys = {
        **NexusDataReader.nexus_meta_keys,
    }

    def __init__(self, filepath: Path):
        super().__init__(filepath)
        self.counts = self.counts.swapaxes(1,2)
        self.monitor1 = np.array(self.getValue('monitor1'))
        self.monitor2 = np.array(self.getValue('monitor2'))
        self.get_possible_binnings()

    def get_possible_binnings(self):
        self.possible_binnings = []
        for value in self.getValue('instrument'):
            if value[:5] == 'calib':
                self.possible_binnings.append(int(value[5]))

    def get_instrument_calibration(self, binning: int) -> InstrumentCalibration:
        instrument = self.nexus_keys['instrument']
        hdf_key = f'/{instrument}/calib{binning}/'
        Ef = np.array(self.get(hdf_key + 'final_energy'))
        width = np.array(self.get(hdf_key + 'width'))
        bg = np.array(self.get(hdf_key + 'background'))
        amp = np.array(self.get(hdf_key + 'amplitude'))
        A4 = np.array(self.get(hdf_key + 'A4'))
        bound = np.array(self.get(hdf_key + 'boundaries'))
        return InstrumentCalibration(Ef, width, bg, amp, A4, bound)
