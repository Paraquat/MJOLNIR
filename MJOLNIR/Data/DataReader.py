import warnings
from pathlib import Path
from abc import ABC
from dataclasses import dataclass
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

class DataReader(ABC):
    """Abstract base class for all data readers (objects that parse data files)"""

    n_analysers: int
    n_detectors: int
    n_steps: int
    binning: int
    instrument: str

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
    use_default_mask: bool = True

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

        self.intensity = None
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
        # Columns: detector, energy, pixel, amplitude, center, width, background,
        # lowerBin, upperBin, A4Offset
        table = np.genfromtxt(self.calibration_file, skip_header=1, delimiter=',')
        Ef = table[:, 4]
        width = table[:, 5]
        bg = table[:, 6]
        amp = table[:, 3]
        A4 = self.calibration_A4_sign * table[:, 9]
        bound = table[:, 7:9].astype(int)
        self.binning = binning
        return InstrumentCalibration(Ef, width, bg, amp, A4, bound)

    def set_instrument_calibration(self, binning: int | None = None):
        """Load the calibration for a binning (default: the reader's default binning)"""
        if binning is None:
            binning = self.default_binning
        self.calibration = self.get_instrument_calibration(binning)

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

        EfMean = self.calibration.Ef.reshape(
            1, A4.shape[0], self.n_analysers*self.binning)
        # Normalization and monitor per point, like the data: (steps, detectors, energy bins)
        self.normalization = self.calibration.amp.reshape(
            1, A4.shape[0], self.n_analysers*self.binning) * np.ones((self.n_steps, 1, 1))
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
        self._default_mask = self.default_mask()
        self._report_masked_detectors()

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
