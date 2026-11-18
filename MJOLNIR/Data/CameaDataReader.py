import warnings
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import numpy.typing as npt
from pathlib import Path

from MJOLNIR.Data.DataReader import InstrumentCalibration
from MJOLNIR.Data.LegacyFixes import CameaMonitor2018
from MJOLNIR.Data.NexusDataReader import NexusDataReader
from MJOLNIR.Data.NexusLayout import CameaLayout

def interval_means(times, values, starts, ends) -> npt.NDArray[np.float64]:
    """Time average over each interval [start, end] of a log (values at times), linearly
    interpolated between its points and held at its ends. An empty interval gets the value
    at its end"""
    means = np.empty(len(starts))
    for i, (start, end) in enumerate(zip(starts, ends)):
        if not end > start:
            means[i] = np.interp(end, times, values)
            continue
        inside = times[(times > start) & (times < end)]
        grid = np.concatenate([[start], inside, [end]])
        means[i] = np.trapezoid(np.interp(grid, times, values), grid)/(end - start)
    return means

class CameaDataReader(NexusDataReader):
    """Parse a CAMEA Nexus file"""

    n_analysers = 8
    n_detectors = 104
    instrument = CameaLayout.instrument
    file_extensions = ('hdf',)
    default_binning = 8 # As chosen by the old DataFile conversion
    layout = CameaLayout
    legacy_fixes = (CameaMonitor2018(),)
    # NICOS writes the start times of a file and its logs in local time at PSI, and
    # entry/control/absolute_time in Unix time
    timezone = ZoneInfo('Europe/Zurich')

    def __init__(self, filepath: Path):
        super().__init__(filepath)
        self.counts = self.counts.swapaxes(1,2)
        self.monitor1 = np.array(self.getValue('monitor1'))
        self.monitor2 = np.array(self.getValue('monitor2'))
        time = self.getValue('time') # Counting time per scan step [s]
        self.time = None if time is None else np.asarray(time, dtype=float)
        self._trim_unfinished_step()
        self._temperature_from_log()
        self.get_possible_binnings()
        # Detector and analyser monitored during the scan (NICOS SelectDetectorAnalyser)
        for attribute, key in [('detector_selection', 'detectorSelection'),
                               ('analyser_selection', 'analyzerSelection')]:
            value = self.getValue(key)
            if value is not None:
                setattr(self, attribute, int(np.asarray(value).ravel()[0]))
        self.scan_parameters, self.scan_values, self.scan_units = self.get_scan_parameters()
        self.apply_legacy_fixes()

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

    def _trim_unfinished_step(self):
        """A file opened while NICOS is still scanning holds one monitor value more than
        counts: NICOS writes the monitors and counting time of a step before its counts (as
        the old DataFile). The step without counts is dropped from the monitors and the
        counting time, and metadata['unfinished'] says whether the scan was still running.
        Any other mismatch is left for transform to refuse"""
        self._metadata['unfinished'] = any(
            values is not None and values.ndim == 1 and values.size == self.n_steps + 1
            for values in (self.monitor1, self.monitor2))
        if not self._metadata['unfinished']:
            return
        for attribute in ('monitor1', 'monitor2', 'time'):
            values = getattr(self, attribute)
            if values is not None and values.ndim == 1 and values.size == self.n_steps + 1:
                setattr(self, attribute, values[:-1])
        warnings.warn(f'"{self.file.name}" has one monitor value more than its {self.n_steps} '
                      f'scan steps: the scan is unfinished; the step without counts is dropped')

    def _temperature_from_log(self):
        """Temperature per scan step from the temperature log, for files that store none per
        step (as the old DataFile meant to, though its fallback never ran: it tested for a
        missing temperature with np.array(None) is None). absolute_time is the end of each
        step, so step i counted from absolute_time[i] - time[i] to absolute_time[i] (without
        the counting time, from the end of the step before); the log's times count from its
        start attribute. NICOS logs about every 10 s, so a step can hold no log value: a
        step's temperature is the time average of the log, linearly interpolated, over the
        step. metadata['temperature_from_log'] says whether it was used. shallow_read still
        gives the stored temperature only"""
        self._metadata['temperature_from_log'] = False
        if self.environment.temperature is not None:
            return
        log, log_time, step_end = (self.getValue(key) for key in
                                   ('temperatureLog', 'temperatureLogTime', 'absoluteTime'))
        if log is None or log_time is None or step_end is None:
            return
        start = log_time.attrs.get('start')
        try:
            start = datetime.fromisoformat(start.decode() if isinstance(start, bytes) else start)
        except (AttributeError, TypeError, ValueError):
            warnings.warn(f'"{self.file.name}": the temperature log has no start time; the '
                          'temperature is not read from it')
            return
        log_time = np.asarray(log_time, dtype=float).ravel()
        log = np.asarray(log, dtype=float).ravel()
        n_log = min(log.size, log_time.size)
        # Step ends in seconds from the start of the log; a running scan may have one more
        step_end = np.asarray(step_end, dtype=float).ravel()[:self.n_steps] \
            - start.replace(tzinfo=self.timezone).timestamp()
        if n_log == 0 or step_end.size != self.n_steps:
            return
        if self.time is not None and self.time.size == self.n_steps:
            step_start = step_end - self.time
        else:
            step_start = np.concatenate([[0.0], step_end[:-1]])
        self.environment.temperature = interval_means(log_time[:n_log], log[:n_log],
                                                      step_start, step_end)
        unit = self.layout.field('temperature').units
        if unit is not None:
            self.environment.units['temperature'] = unit
        self._metadata['temperature_from_log'] = True

    def scan_monitor_key(self) -> str:
        """Monitor used for normalisation, chosen as the old DataFile did: files written by
        NICOS (which have entry/data/data) use entry/monitor_2/data if present ('monitor2'),
        older files entry/control/data ('monitor1')"""
        written_by_nicos = self.get(self.layout.group('data').path + '/data') is not None
        if written_by_nicos and self.getValue('monitor2') is not None:
            return 'monitor2'
        return 'monitor1'

    def scan_monitor(self) -> npt.NDArray[np.float64]:
        return getattr(self, self.scan_monitor_key())

    def default_mask(self) -> npt.NDArray[np.bool_]:
        """The general default mask, and at binning 8 the first two energy bins (as the old
        DataFile conversion did)"""
        mask = super().default_mask()
        if self.binning == 8:
            mask[:, :, :2] = True
        return mask
