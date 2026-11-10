import re
import warnings
from pathlib import Path
from abc import ABC
from dataclasses import dataclass, field, fields
from typing import Any
import numpy as np
import numpy.typing as npt

from MJOLNIR import _tools
from MJOLNIR.Data.Metadata import Metadata
from MJOLNIR.Data.Sample import Sample
from MJOLNIR import TasUBlibDEG as TasUBlib

@dataclass
class InstrumentCalibration:
    """Calibration data from a CAMEA Nexus file"""
    Ef: npt.NDArray[np.float64]
    width: npt.NDArray[np.float64]
    bg: npt.NDArray[np.float64]
    amp: npt.NDArray[np.float64]
    A4: npt.NDArray[np.float64]
    bound: npt.NDArray[np.int32]

@dataclass
class SampleEnvironment:
    """Sample conditions during a scan, measured and set: one value per scan step, or None if
    the file does not record them. Units are those stored in the file (see units)"""
    temperature: npt.NDArray[np.float64] | None = None
    temperature_setpoint: npt.NDArray[np.float64] | None = None
    magnetic_field: npt.NDArray[np.float64] | None = None
    magnetic_field_setpoint: npt.NDArray[np.float64] | None = None
    electric_field: npt.NDArray[np.float64] | None = None
    electric_field_setpoint: npt.NDArray[np.float64] | None = None
    other: dict[str, npt.NDArray[np.float64]] = field(default_factory=dict) # Other quantities
    units: dict[str, str] = field(default_factory=dict) # Quantity name -> unit

def calibration_from_table(table: npt.NDArray[np.float64],
                           A4_sign: int = 1) -> InstrumentCalibration:
    """Calibration from a normalisation table, one row per pixel bin. Columns: detector,
    energy, pixel, amplitude, center (Ef), width, background, lowerBin, upperBin, A4Offset"""
    return InstrumentCalibration(Ef=table[:, 4], width=table[:, 5], bg=table[:, 6],
                                 amp=table[:, 3], A4=A4_sign * table[:, 9],
                                 bound=table[:, 7:9].astype(int))

def read_calibration_file(path: Path, A4_sign: int = 1) -> tuple[int, InstrumentCalibration]:
    """Binning and calibration of a .calib file, as written by
    Geometry.Instrument.generateCalibration: a header line naming the binning ("Normalization
    for 8 pixel(s) ..."), a date line, the column names, then the table"""
    with open(path) as f:
        header = f.readline()
    match = re.search(r'(\d+)\s+pixel', header)
    if match is None:
        raise ValueError(f'"{Path(path).name}" is not a calibration file: its first line does '
                         'not name the number of pixels')
    table = np.loadtxt(path, skiprows=3, delimiter=',', ndmin=2)
    return int(match.group(1)), calibration_from_table(table, A4_sign)

class DataReader(ABC):
    """Abstract base class for all data readers (objects that parse data files)"""

    n_analysers: int
    n_detectors: int
    n_steps: int
    binning: int
    instrument: str
    # Extensions of the instrument's data files (without the dot; '' for files without one),
    # for file dialogs. can_read decides whether a file is read, not its extension
    file_extensions: tuple[str, ...] = ()

    # Calibration: readers with one bin per analyser read it from a CSV table
    possible_binnings: list[int] = [1]
    default_binning: int = 1
    calibration_file: str | None = None
    calibration_A4_sign: int = 1

    # Input data
    counts: npt.NDArray[np.float64]
    A3: npt.NDArray[np.float64]
    A3Offset: npt.NDArray[np.float64]
    A4: npt.NDArray[np.float64]
    A4Offset: npt.NDArray[np.float64]
    Ei: npt.NDArray[np.float64]
    twotheta: npt.NDArray[np.float64]
    calibration: InstrumentCalibration
    sample: Sample
    environment: SampleEnvironment

    # Derived data
    intensity: npt.NDArray[np.int64]
    normalization: npt.NDArray[np.float64]
    monitor: npt.NDArray[np.float64]
    energy: npt.NDArray[np.float64]
    qx: npt.NDArray[np.float64]
    qy: npt.NDArray[np.float64]
    h: npt.NDArray[np.float64]
    k: npt.NDArray[np.float64]
    l: npt.NDArray[np.float64]
    # Masks, shape of intensity, True = point excluded: user_mask is set by the user,
    # _default_mask by the instrument (default_mask()); mask combines them
    user_mask: npt.NDArray[np.bool_]
    background: npt.NDArray[np.float64] | None # Background counts per point (BackgroundModel)
    # Standard deviation of intensity per point; None: counting statistics, sqrt(intensity).
    # Set for derived data, e.g. a difference of data sets
    intensity_error: npt.NDArray[np.float64] | None
    use_default_mask: bool = True
    # Multiplies the normalization at every transform: absolute normalisation (set by the
    # DataSet, see set_normalization_factor)
    normalization_factor: float = 1.0
    # Detector and analyser monitored during the scan (the old dasel), the default of
    # raw_counts; read from CAMEA files, else 0, 0
    detector_selection: int = 0
    analyser_selection: int = 0

    # Frozen readers hold derived data (e.g. a difference of two data sets): transforming them
    # again would rebuild intensity from the raw counts
    frozen: bool = False

    def __init__(self, filepath: Path, calibration_file: str | None = None):
        if not filepath.exists():
            raise AttributeError(f'File path "{filepath}" does not exist')
        if not self.can_read(filepath):
            raise ValueError(f'"{filepath.name}" is not a {self.instrument} data file that '
                             f'{type(self).__name__} can read')
        self.file = filepath
        if calibration_file is not None:
            self.calibration_file = calibration_file
        self._metadata = Metadata()
        self.calibration = None
        # Calibrations from update_calibration, used instead of the file's (by binning)
        self.calibration_updates: dict[int, InstrumentCalibration] = {}

        self.intensity = None
        self.intensity_error = None
        self.normalization = None
        self.monitor = None
        self.energy = None
        self.qx = None
        self.qy = None
        self.h = None
        self.k = None
        self.l = None
        self.user_mask = None
        self._default_mask = None
        self.masked_detectors = [] # Detectors excluded by the default mask
        self.background = None
        self.environment = SampleEnvironment()

    # Keys shallow_read gives for every reader: the file's name and folder, and 2θ (A4 - A4Offset)
    _shallow_extras = ('name', 'fileLocation', 'twoTheta')
    # Reader attributes shallow_read gives for readers without a file layout
    _shallow_attributes = ('A3', 'A4', 'A4Offset', 'Ei', 'monitor1', 'monitor2', 'time')

    @classmethod
    def shallow_keys(cls) -> list[str]:
        """Keys shallow_read accepts, sorted ignoring case (the old possibleAttributes)"""
        environment = [f.name for f in fields(SampleEnvironment) if f.name not in ('other', 'units')]
        keys = {*cls._shallow_extras, *cls._shallow_attributes, *environment,
                *cls.metadata_keys()}
        return sorted(keys, key=str.lower)

    @classmethod
    def metadata_keys(cls) -> list[str]:
        """Keys the reader's metadata can hold"""
        return []

    @classmethod
    def _check_shallow_keys(cls, keys):
        unknown = [key for key in keys if key not in cls.shallow_keys()]
        if unknown:
            raise KeyError(f'{cls.__name__} cannot read {unknown}; see shallow_keys()')

    @classmethod
    def shallow_read(cls, filepath: Path, keys: list[str]) -> dict[str, Any]:
        """A few values from a file without converting its data (the old shallowRead), e.g.
        for a log book of many files. Keys from shallow_keys(); None for a key the file does
        not have. This version reads the whole file (NICOS text files are small);
        NexusDataReader reads only the datasets asked for"""
        cls._check_shallow_keys(keys)
        filepath = Path(filepath)
        reader = cls(filepath)
        environment = {f.name for f in fields(SampleEnvironment)}
        values = {}
        for key in keys:
            if key == 'name':
                values[key] = filepath.name
            elif key == 'fileLocation':
                values[key] = str(filepath.parent)
            elif key == 'twoTheta':
                values[key] = reader.twotheta
            elif key in environment:
                values[key] = getattr(reader.environment, key)
            elif key in cls._shallow_attributes:
                values[key] = getattr(reader, key, None)
            else:
                values[key] = reader._metadata.get(key)
        return values

    @classmethod
    def can_read(cls, filepath: Path) -> bool:
        """Whether this reader can read a file, from a quick look at it (without reading the
        data). Never raises. Readers check this before reading; a GUI can use it to find the
        instrument of a file"""
        return True

    def get_instrument_calibration(self, binning: int) -> InstrumentCalibration:
        """Read the calibration table (CSV) of instruments with one bin per analyser"""
        if binning not in self.possible_binnings:
            raise AttributeError(
                f'Binning {binning} not in possible binnings {self.possible_binnings}')
        if self.calibration_file is None:
            raise AttributeError(f'No calibration file given for {self.instrument}')
        table = np.genfromtxt(self.calibration_file, skip_header=1, delimiter=',')
        self.binning = binning
        return calibration_from_table(table, self.calibration_A4_sign)

    def raw_counts(self, detector: int | None = None, analyser: int | None = None):
        """Counts per scan step of one detector and analyser, not normalised: the detector's
        pixels within the analyser's range at binning 1 summed, or its channel for instruments
        with one channel per analyser (the old cutRaw1D for one file). Defaults:
        detector_selection, analyser_selection"""
        self._check_not_frozen('read raw counts of')
        detector = self.detector_selection if detector is None else int(detector)
        analyser = self.analyser_selection if analyser is None else int(analyser)
        for name, value, limit in [('detector', detector, self.n_detectors),
                                   ('analyser', analyser, self.n_analysers)]:
            if not 0 <= value < limit:
                raise ValueError(f'{name} {value} out of range: {type(self).__name__} has '
                                 f'{limit} {name}s, numbered 0-{limit - 1}')
        if self.counts.shape[2] == self.n_analysers: # One channel per analyser
            return np.asarray(self.counts[:, detector, analyser], dtype=float)
        if 1 in self.calibration_updates:
            calibration = self.calibration_updates[1]
        else: # get_instrument_calibration sets binning: keep the current one
            binning = getattr(self, 'binning', None)
            calibration = self.get_instrument_calibration(1)
            if binning is not None:
                self.binning = binning
        low, high = calibration.bound.reshape(self.n_detectors, self.n_analysers, 2)[detector, analyser]
        return np.sum(self.counts[:, detector, int(low):int(high)], axis=1, dtype=float)

    def _check_has_data(self, data):
        """Raise for a file without measured points: a scan aborted before its first point, or a
        file opened before any data was written. data = the counts (or scan table) as read"""
        if np.ndim(data) == 0 or np.size(data) == 0:
            raise ValueError(f'"{self.file.name}" has no data (0 scan steps): the scan may have '
                             'been aborted, or not started yet')

    def _check_not_frozen(self, action: str):
        if self.frozen:
            raise ValueError(f'Cannot {action} "{self.file.name}": it holds derived data (e.g. a '
                             'difference of data sets). Change the original data instead')

    def set_instrument_calibration(self, binning: int | None = None):
        """Load the calibration for a binning (default: the reader's default binning): one
        given to update_calibration, otherwise the file's (get_instrument_calibration)"""
        self._check_not_frozen('recalibrate')
        if binning is None:
            binning = self.default_binning
        if binning in self.calibration_updates:
            self.binning = binning
            self.calibration = self.calibration_updates[binning]
        else:
            self.calibration = self.get_instrument_calibration(binning)

    def update_calibration(self, calibration_files: Path | str | list[Path | str],
                           overwrite: bool = False) -> list[int]:
        """Use calibrations from .calib files (Geometry.Instrument.generateCalibration)
        instead of the file's, e.g. a newer vanadium normalisation. A binning the file already
        has is replaced only with overwrite (as the old updateCalibration); new binnings are
        added to possible_binnings. Takes effect at the next set_instrument_calibration (a
        DataSet transforms its files again); nothing is saved (see CameaDataWriter). Returns
        the binnings updated"""
        self._check_not_frozen('recalibrate')
        if isinstance(calibration_files, (str, Path)):
            calibration_files = [calibration_files]
        updated = []
        for path in calibration_files:
            binning, calibration = read_calibration_file(path, self.calibration_A4_sign)
            if binning in self.possible_binnings and not overwrite:
                warnings.warn(f'Binning {binning} from "{Path(path).name}" is skipped as '
                              'overwrite is False')
                continue
            n_bins = self.n_detectors * self.n_analysers * binning
            if len(calibration.Ef) != n_bins:
                raise ValueError(f'"{Path(path).name}" has {len(calibration.Ef)} rows; '
                                 f'{type(self).__name__} with binning {binning} needs {n_bins}')
            self.calibration_updates[binning] = calibration
            if binning not in self.possible_binnings:
                self.possible_binnings = sorted([*self.possible_binnings, binning])
            updated.append(binning)
        return updated

    def rebin(self, binning: int, shape: tuple, bounds: npt.NDArray[np.int32]):
        """Sum the detector pixels between the calibration bounds of each energy bin"""
        intensity = np.zeros(shape, dtype=np.result_type(self.counts.dtype, int))
        for i in range(self.n_detectors):
            for j in range(self.n_analysers):
                for k in range(binning):
                    intensity[:, i, j*binning+k] = \
                        np.sum(self.counts[:, i, bounds[i, j, k, 0]:bounds[i, j, k, 1]],
                               axis=1)
        return intensity

    @staticmethod
    def sample_from_reflections(cell, q1, q2, A3Offset: float = 0.0,
                                name: str = 'Unknown') -> Sample:
        """Build a sample whose UB matrix is defined by two reflections in the scattering
        plane, for files which only store the lattice and two orientation vectors"""
        Ei = 10.0 # TODO: Dummy incoming energy needed to calculate UB (from DataFile)
        k = np.sqrt(Ei) * TasUBlib.factorsqrtEK
        B = TasUBlib.calculateBMatrix(TasUBlib.calcCell(cell))

        A41 = TasUBlib.calTwoTheta(B, [*q1, Ei, Ei], -1)
        A31 = TasUBlib.calcTheta(k, k, A41) + A3Offset
        A42 = TasUBlib.calTwoTheta(B, [*q2, Ei, Ei], -1)
        A32 = TasUBlib.calcTheta(k, k, A42)

        # h, k, l, A3, A4, sgu, sgl, Ei, Ef
        plane_vector1 = np.array([*q1, A31, A41, 0.0, 0.0, Ei, Ei])
        plane_vector2 = np.array([*q2, A32, A42, 0.0, 0.0, Ei, Ei])
        plane_vector2[3] += TasUBlib.tasAngleBetweenReflections(
            B, plane_vector1, plane_vector2)

        a, b, c, alpha, beta, gamma = cell
        sample = Sample(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                        projectionVector1=plane_vector1, projectionVector2=plane_vector2,
                        name=name)
        sample.A3Off = np.array([0.0])
        return sample


    def transform(self):
        self._check_not_frozen('transform')
        A4 = np.deg2rad(self.calibration.A4)
        A4 = A4.reshape(self.n_detectors, self.binning * self.n_analysers, order='C')
        pixelEdge = self.calibration.bound.reshape(
            self.n_detectors, self.n_analysers, self.binning, 2)
        A4File = self.A4.reshape((-1,1,1))
        A4Mean = (
            A4.reshape((1, self.n_detectors, self.binning*self.n_analysers)) +
            np.deg2rad(A4File-self.A4Offset))
        shape = (
            self.counts.shape[0], self.counts.shape[1], self.n_analysers*self.binning)
        self.intensity = self.rebin(self.binning, shape, pixelEdge)
        self.intensity_error = None # Counts: counting statistics

        EfMean = self.calibration.Ef.reshape(
            1, A4.shape[0], self.n_analysers*self.binning)
        # Normalization and monitor per point, like the data: (steps, detectors, energy bins)
        self.normalization = self._base_normalization()
        if self.normalization_factor != 1.0:
            self.normalization = self.normalization*self.normalization_factor
        monitor = np.asarray(self.scan_monitor(), dtype=float)
        if monitor.shape != (self.n_steps,):
            raise ValueError(f'"{self.file.name}" has {monitor.size} monitor values for '
                             f'{self.n_steps} scan steps')
        self.monitor = monitor.reshape(-1, 1, 1) * np.ones((1, *self.intensity.shape[1:]))
        A3 = np.deg2rad(self.A3) + np.deg2rad(self.A3Offset)
        if A3.shape[0]==1:
            A3 = A3*np.ones((self.n_steps))
        A3 = A3.reshape((self.n_steps, 1, 1))
        Ei = self.Ei.copy().reshape(-1, 1, 1)
        self.energy = (Ei - EfMean) * np.ones((self.n_steps, 1, 1))
        UBINV = np.linalg.inv(self.sample.orientationMatrix)
        HKL, self.qx, self.qy = TasUBlib.calcTasQH(
            UBINV,[np.rad2deg(A3), np.rad2deg(A4Mean)], Ei, EfMean)
        self.h, self.k, self.l = np.swapaxes(np.swapaxes(HKL, 1, 2), 0, 3)
        self.sample.B = TasUBlib.calculateBMatrix(self.sample.cell)
        # Masks follow the binning, so a new transform rebuilds them
        self.user_mask = np.zeros(self.intensity.shape, dtype=bool)
        self.background = None # Its shape follows the binning: the DataSet computes it again
        self._default_mask = self.default_mask()
        self._report_masked_detectors()

    def intensity_variance(self) -> npt.NDArray[np.float64]:
        """Variance of intensity per point: intensity_error squared, or the counts themselves
        (counting statistics)"""
        if self.intensity_error is not None:
            return np.square(self.intensity_error)
        return np.asarray(self.intensity, dtype=float)

    def _base_normalization(self) -> npt.NDArray[np.float64]:
        """Normalization per point from the calibration, without normalization_factor"""
        bins = self.n_analysers*self.binning
        return self.calibration.amp.reshape(1, self.n_detectors, bins) * np.ones((self.n_steps, 1, 1))

    def set_normalization_factor(self, factor: float):
        """Multiply the normalization from the calibration by factor (absolute
        normalisation), now and at every later transform. 1 undoes it. Also for frozen
        readers (e.g. a difference of data sets): only the normalization changes"""
        self.normalization_factor = float(factor)
        if self.calibration is not None and self.normalization is not None:
            self.normalization = self._base_normalization()
            if self.normalization_factor != 1.0:
                self.normalization = self.normalization*self.normalization_factor

    @property
    def mask(self) -> npt.NDArray[np.bool_] | None:
        """Points to exclude: the user's mask, and the default mask unless use_default_mask
        is False"""
        if self.user_mask is None:
            return None
        if self.use_default_mask and self._default_mask is not None:
            return self.user_mask | self._default_mask
        return self.user_mask

    @mask.setter
    def mask(self, mask):
        # Sets the user's mask; the default mask always comes from default_mask()
        self.user_mask = np.asarray(mask, dtype=bool)

    def per_step(self, values, name: str) -> npt.NDArray[np.float64] | None:
        """Values of a quantity as one per scan step: a single value is repeated. Values that
        do not fit the scan are left out (None), with a warning"""
        try:
            values = np.asarray(values, dtype=float).ravel()
        except (TypeError, ValueError):
            return None
        if values.size == 1:
            return np.full(self.n_steps, values[0])
        if values.size == self.n_steps:
            return values
        warnings.warn(f'"{self.file.name}" has {values.size} values of {name} for '
                      f'{self.n_steps} scan steps; {name} is left out')
        return None

    def scan_monitor(self) -> npt.NDArray[np.float64]:
        """Monitor counts per scan step, used to normalise the data"""
        return self.monitor1

    def default_mask(self) -> npt.NDArray[np.bool_]:
        """Points that cannot be used, for every instrument: pixels with a NaN in their
        calibration (A4, Ef or amplitude). Subclasses add instrument-specific rules"""
        bins = self.n_analysers * self.binning
        invalid = (np.isnan(self.calibration.A4) | np.isnan(self.calibration.Ef)
                   | np.isnan(self.calibration.amp)).reshape(self.n_detectors, bins)
        return np.broadcast_to(invalid, self.intensity.shape).copy()

    def _report_masked_detectors(self):
        """Warn once about detectors the default mask excludes completely"""
        excluded = self._default_mask.all(axis=(0, 2))
        detectors = [int(d) for d in np.flatnonzero(excluded)]
        if detectors and detectors != self.masked_detectors:
            warnings.warn(f'Detectors {", ".join(map(str, detectors))} of "{self.file.name}" are '
                          'masked: their calibration is missing (NaN)')
        self.masked_detectors = detectors
