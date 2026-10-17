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
    energy: npt.NDArray[np.float64]
    qx: npt.NDArray[np.float64]
    qy: npt.NDArray[np.float64]
    h: npt.NDArray[np.float64]
    k: npt.NDArray[np.float64]
    l: npt.NDArray[np.float64]
    mask: npt.NDArray[np.bool_] # True = point excluded, shape of intensity

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
        self.energy = None
        self.qx = None
        self.qy = None
        self.h = None
        self.k = None
        self.l = None
        self.mask = None

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
        self.normalization = self.calibration.amp.reshape(
            1, A4.shape[0], self.n_analysers*self.binning)
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
        # Nothing masked: the shape follows the binning, so a new transform resets it
        self.mask = np.zeros(self.intensity.shape, dtype=bool)
