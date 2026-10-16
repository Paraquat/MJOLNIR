import numpy as np
from pathlib import Path

from MJOLNIR.Data.NexusDataReader import NexusDataReader
from MJOLNIR.Data.Sample import Sample

class MarmotDataReader(NexusDataReader):
    """Parse a Marmot (THALES, ILL) Nexus file"""

    n_analysers = 256
    n_detectors = 30
    instrument = "THALES"
    entry = 'entry0'
    # No calibration table is shipped for Marmot: pass calibration_file to the constructor
    calibration_file = None

    # Marmot-specific hdf5 keys
    nexus_keys = {
        **NexusDataReader.nexus_keys,
        'counts': '{entry}/data_scan/detector_data/data',
        'A4': '{entry}/{instrument}/A4/value',
        'Ei': '{entry}/{instrument}/Monochromator/ei',
        'scanCommand': '{entry}/{instrument}/command_line/actual_command',
        'scannedData': '{entry}/data_scan/scanned_variables/data',
        'scannedLabels': '{entry}/data_scan/scanned_variables/variables_names/label',
    }

    # Marmot-specific hdf5 metadata keys
    nexus_meta_keys = {
        **NexusDataReader.nexus_meta_keys,
        'localContact': '{entry}/user/namelocalcontact',
        'proposalUser': '{entry}/user/name',
        'proposal': '{entry}/experiment_identifier',
    }

    def __init__(self, filepath: Path, calibration_file: str | None = None):
        super().__init__(filepath, calibration_file)
        self.A3 = self.get_scanned_variable('A3')
        self.monitor1 = self.get_scanned_variable('Monitor1')
        self.monitor2 = self.get_scanned_variable('Monitor2')
        self.time = self.get_scanned_variable('Time')
        command = self.getValue('scanCommand')[0].decode()
        self.scan_parameters, self.scan_values = self.parse_scan_command(command)
        self.scan_units = np.array(['N/A' for _ in self.scan_parameters])

    def get_scanned_variable(self, label: str) -> np.NDArray[np.float64]:
        """Values per step of a variable logged during the scan, e.g. 'A3' or 'Monitor1'"""
        labels = [x.decode() for x in self.getValue('scannedLabels')]
        return np.array(self.getValue('scannedData')[labels.index(label)])

    def get_sample(self) -> Sample:
        sample = self.getValue('sample')
        value = lambda key: float(sample[key][0])
        cell = [value(f'unit_cell_{key}') for key in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']]
        q1 = np.array([value(key) for key in ['ax', 'ay', 'az']])
        q2 = np.array([value(key) for key in ['bx', 'by', 'bz']])
        return self.sample_from_reflections(cell, q1, q2)
