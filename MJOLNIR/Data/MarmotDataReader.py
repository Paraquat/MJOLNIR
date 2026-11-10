import numpy as np
import numpy.typing as npt
from pathlib import Path

from MJOLNIR.Data.NexusDataReader import NexusDataReader
from MJOLNIR.Data.NexusLayout import MarmotLayout
from MJOLNIR.Data.Sample import Sample

class MarmotDataReader(NexusDataReader):
    """Parse a Marmot (THALES, ILL) Nexus file"""

    n_analysers = 256
    n_detectors = 30
    layout = MarmotLayout
    instrument = MarmotLayout.instrument
    file_extensions = ('nxs',)
    # No calibration table is shipped for Marmot: pass calibration_file to the constructor
    calibration_file = None

    def __init__(self, filepath: Path, calibration_file: str | None = None):
        super().__init__(filepath, calibration_file)
        self.A3 = self.get_scanned_variable('A3')
        self.monitor1 = self.get_scanned_variable('Monitor1')
        self.monitor2 = self.get_scanned_variable('Monitor2')
        self.time = self.get_scanned_variable('Time')
        command = self.getValue('scanCommand')[0].decode()
        self.scan_parameters, self.scan_values = self.parse_scan_command(command)
        self.scan_units = np.array(['N/A' for _ in self.scan_parameters])

    @classmethod
    def identify(cls, f) -> bool:
        # THALES is also run without Marmot, with the same file layout
        mode = f.get(cls.layout.path('instrumentMode'))
        return mode is not None and mode[0].strip().lower() == b'marmot'

    def default_mask(self) -> npt.NDArray[np.bool_]:
        """The general default mask, and pixels with calibrated Ef below 1 meV (as the old
        loadMarmot did)"""
        bins = self.n_analysers * self.binning
        low_Ef = self.calibration.Ef.reshape(self.n_detectors, bins) < 1
        return super().default_mask() | np.broadcast_to(low_Ef, self.intensity.shape)

    def get_scanned_variable(self, label: str) -> npt.NDArray[np.float64]:
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
