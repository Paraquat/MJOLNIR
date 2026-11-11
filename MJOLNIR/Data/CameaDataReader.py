import warnings
import numpy as np
import numpy.typing as npt
from pathlib import Path

from MJOLNIR.Data.DataReader import InstrumentCalibration
from MJOLNIR.Data.NexusDataReader import NexusDataReader
from MJOLNIR.Data.NexusLayout import CameaLayout

class CameaDataReader(NexusDataReader):
    """Parse a CAMEA Nexus file"""

    n_analysers = 8
    n_detectors = 104
    instrument = CameaLayout.instrument
    file_extensions = ('hdf',)
    default_binning = 8 # As chosen by the old DataFile conversion
    layout = CameaLayout

    def __init__(self, filepath: Path):
        super().__init__(filepath)
        self.counts = self.counts.swapaxes(1,2)
        self.monitor1 = np.array(self.getValue('monitor1'))
        self.monitor2 = np.array(self.getValue('monitor2'))
        time = self.getValue('time') # Counting time per scan step [s]
        self.time = None if time is None else np.asarray(time, dtype=float)
        self.get_possible_binnings()
        # Detector and analyser monitored during the scan (NICOS SelectDetectorAnalyser)
        for attribute, key in [('detector_selection', 'detectorSelection'),
                               ('analyser_selection', 'analyzerSelection')]:
            value = self.getValue(key)
            if value is not None:
                setattr(self, attribute, int(np.asarray(value).ravel()[0]))
        self.scan_parameters, self.scan_values, self.scan_units = self.get_scan_parameters()

    # entry/data entries that are not scan variables (old getScanParameter)
    _not_scanned = {'counts', 'summed_counts', 'en', 'h', 'intensity', 'k', 'l', 'monitor',
                    'normalization', 'qx', 'qy', 'data', 'scanvar'}
    # NICOS scan variable -> layout key and name (old getScanParameter)
    _scan_variables = {'a3': ('A3', 'A3'), 'A3': ('A3', 'A3'), 's2t': ('A4', 'A4'),
                       'a4': ('A4', 'A4'), 'A4': ('A4', 'A4'), 'ei': ('Ei', 'Ei'),
                       'Ei': ('Ei', 'Ei')}

    def get_scan_parameters(self):
        """Scanned variables, their values per step and units, as the old getScanParameter:
        the variables stored in entry/data (older files), else those named in entry/scanvars
        (NICOS), else for a manual scan the first of temperature, magnetic field, Ei, A3 and
        2theta that changes"""
        def text(value):
            return value.decode() if isinstance(value, bytes) else str(value)

        parameters, values, units = [], [], []
        data = self.getValue('data')
        if data is not None:
            for name in data:
                if name in self._not_scanned or name.endswith('zero'):
                    continue
                dataset = data[name]
                parameters.append(name)
                values.append(np.asarray(dataset, dtype=float))
                units.append(text(dataset.attrs.get('units', 'N/A')))
        if not parameters:
            scanvars = self.getValue('scanVariables')
            names = [] if scanvars is None else \
                [name for name in text(np.asarray(scanvars).ravel()[0]).split(',') if name]
            for name in names:
                if name == 'CAMEA':
                    warnings.warn(f'"{self.file.name}" is a Q scan (qcscan): its scan '
                                  'parameters are not read')
                    continue
                if name not in self._scan_variables:
                    warnings.warn(f'Scan variable "{name}" of "{self.file.name}" not known; left out')
                    continue
                key, label = self._scan_variables[name]
                dataset = self.getValue(key)
                parameters.append(label)
                values.append(np.asarray(dataset, dtype=float).ravel())
                units.append(text(dataset.attrs.get('units', 'N/A')))
            if not names: # Manual scan: guess the variable that changes
                candidates = [('T', self.environment.temperature, 'K', 0.01),
                              ('B', self.environment.magnetic_field, 'T', 0.01),
                              ('Ei', self.Ei, 'meV', 0.001), ('A3', self.A3, 'deg', 0.001),
                              ('A4', self.twotheta, 'deg', 0.001)]
                for name, value, unit, threshold in candidates:
                    value = None if value is None else np.asarray(value, dtype=float).ravel()
                    if value is not None and value.size > 1 and \
                            np.abs(np.diff(value)).mean() > threshold:
                        parameters, values, units = [name], [value], [unit]
                        break
        return np.array(parameters), np.array(values), np.array(units)

    def get_possible_binnings(self):
        """Binnings with a calibration group in the file (one digit, as the old DataFile)"""
        self.possible_binnings = [binning for binning in range(1, 10)
                                  if self.layout.calibration_group_for(binning).path in self._file]

    def get_instrument_calibration(self, binning: int) -> InstrumentCalibration:
        fields = self.layout.calibration_fields(binning)
        self.binning = binning
        return InstrumentCalibration(**{key: np.array(self.get(field.path))
                                        for key, field in fields.items()})

    def scan_monitor(self) -> npt.NDArray[np.float64]:
        """Monitor used for normalisation, chosen as the old DataFile did: files written by
        NICOS (which have entry/data/data) use entry/monitor_2/data if present, older files
        entry/control/data"""
        written_by_nicos = self.get(self.layout.group('data').path + '/data') is not None
        if written_by_nicos and self.getValue('monitor2') is not None:
            return self.monitor2
        return self.monitor1

    def default_mask(self) -> npt.NDArray[np.bool_]:
        """The general default mask, and at binning 8 the first two energy bins (as the old
        DataFile conversion did)"""
        mask = super().default_mask()
        if self.binning == 8:
            mask[:, :, :2] = True
        return mask
