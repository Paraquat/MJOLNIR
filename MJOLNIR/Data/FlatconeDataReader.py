import numpy as np
import numpy.typing as npt
from pathlib import Path

import MJOLNIR
from MJOLNIR.Data.NexusDataReader import NexusDataReader
from MJOLNIR.Data.Sample import Sample

class FlatconeDataReader(NexusDataReader):
    """Parse a Flatcone (IN20, ILL) Nexus file"""

    n_analysers = 1
    n_detectors = 31
    instrument = "IN20"
    entry = 'entry0'
    calibration_file = MJOLNIR.__flatConeNormalization__

    # Flatcone-specific hdf5 keys
    nexus_keys = {
        **NexusDataReader.nexus_keys,
        'counts': '{entry}/data_scan/pastis_data/ii',
        'A3': '{entry}/{instrument}/A3P/value',
        'A4': '{entry}/{instrument}/A4/value',
        'Ei': '{entry}/{instrument}/Monochromator/ei',
        'monitor1': '{entry}/monitor/data',
        'temperature': '{entry}/sample/temperature',
        'sampleParameters': '{entry}/sample/param',
        'scanCommand': '{entry}/{instrument}/command_line/actual_command',
    }

    # Flatcone-specific hdf5 metadata keys
    nexus_meta_keys = {
        **NexusDataReader.nexus_meta_keys,
        'localContact': '{entry}/user/namelocalcontact',
        'proposalUser': '{entry}/user/name',
        'proposal': '{entry}/experiment_identifier',
    }

    def __init__(self, filepath: Path, calibration_file: str | None = None):
        super().__init__(filepath, calibration_file)
        self.A3 = self.A3.flatten()
        self.monitor1 = np.array(self.getValue('monitor1'))
        self.temperature = np.array(self.getValue('temperature')).flatten()
        command = self.getValue('scanCommand')[0].decode()
        self.scan_parameters, self.scan_values = self.parse_scan_command(command)
        self.scan_units = np.array(['N/A' for _ in self.scan_parameters])

    def get_counts(self) -> npt.NDArray[np.float64]:
        # Reduced (PASTIS) intensities are stored as (detectors, steps)
        counts = np.array(self.getValue('counts')).T
        return counts.reshape(-1, self.n_detectors, self.n_analysers)

    def get_sample(self) -> Sample:
        param = self.getValue('sampleParameters')
        value = lambda key: float(param[key][0])
        cell = [value(key) for key in ['AS', 'BS', 'CS', 'AA', 'BB', 'CC']]
        q1 = np.array([value(key) for key in ['AX', 'AY', 'AZ']])
        q2 = np.array([value(key) for key in ['BX', 'BY', 'BZ']])
        return self.sample_from_reflections(cell, q1, q2)
