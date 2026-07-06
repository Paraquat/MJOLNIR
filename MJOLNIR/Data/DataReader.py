from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from MJOLNIR import _tools
from MJOLNIR.Data.Metadata import Metadata
from MJOLNIR.Data.Sample import Sample
from MJOLNIR import TasUBlibDEG as TasUBlib

@dataclass
class InstrumentCalibration:
    """Calibration data from a CAMEA Nexus file"""
    Ef: np.NDArray[np.float64]
    width: np.NDArray[np.float64]
    bg: np.NDArray[np.float64]
    amp: np.NDArray[np.float64]
    A4: np.NDArray[np.float64]
    bound: np.NDArray[np.int32]

class DataReader(ABC):
    """Abstract base class for all data readers (objects that parse data files)"""

    n_analysers: int
    n_detectors: int
    n_steps: int
    binning: int

    # Input data
    counts: np.NDArray[np.float64]
    A3: np.NDArray[np.float64]
    A3Offset: np.NDArray[np.float64]
    A4: np.NDArray[np.float64]
    A4Offset: np.NDArray[np.float64]
    Ei: np.NDArray[np.float64]
    twotheta: np.NDArray[np.float64]
    calibration: InstrumentCalibration
    sample: Sample

    # Derived data
    Qx: np.NDArray[np.float64]
    Qx: np.NDArray[np.float64]
    h: np.NDArray[np.float64]
    k: np.NDArray[np.float64]
    l: np.NDArray[np.float64]

    def __init__(self, filepath: Path):
        if not filepath.exists():
            raise AttributeError(f'File path "{filepath}" does not exist')
        self.file = filepath
        self._metadata = Metadata()
        self.calibration = None

        self.Qx = None
        self.Qy = None
        self.h = None
        self.k = None
        self.l = None

    @abstractmethod
    def rebin(self, binning: int, shape: tuple, bounds: np.NDArray[np.int32]):
        pass


    def transform(self):
        A4 = np.deg2rad(self.calibration.A4)
        A4 = A4.reshape(self.n_detectors, self.binning * self.n_analysers, order='C')
        pixelEdge = self.calibration.bound.reshape(
            self.n_detectors, self.n_analysers, self.binning, 2)
        A3Offset_rad = np.deg2rad(self.A4Offset)
        A4File = self.A4.reshape((-1,1,1))
        A4Mean = (
            A4.reshape((1, self.n_detectors, self.binning*self.n_analysers)) +
            np.deg2rad(A4File-self.A4Offset))
        shape = (
            self.counts.shape[0], self.counts.shape[1], self.n_analysers*self.binning)
        intensity = self.rebin(self.binning, shape, pixelEdge)

        EfMean = self.calibration.Ef.reshape(
            1, A4.shape[0], self.n_analysers*self.binning)
        EfNormalization = self.calibration.amp
        EfNormalization.shape = (1, A4.shape[0], self.n_analysers*self.binning)
        A3 = np.deg2rad(self.A3)+self.A3Offset
        if A3.shape[0]==1:
            A3 = A3*np.ones((self.n_steps))
        A3.resize((self.n_steps, 1, 1))
        Ei = self.Ei.copy().reshape(-1, 1, 1)
        A4Mean = (A4.reshape((1, self.n_detectors, self.binning*self.n_analysers)) + \
                  np.deg2rad(A4File - self.A4Offset))
        UBINV = np.linalg.inv(self.sample.orientationMatrix)
        HKL, self.Qx, self.Qy = TasUBlib.calcTasQH(
            UBINV,[np.rad2deg(A3), np.rad2deg(A4Mean)], Ei, EfMean)
        self.h, self.k, self.l = np.swapaxes(np.swapaxes(HKL, 1, 2), 0, 3)
        self.sample.B = TasUBlib.calculateBMatrix(self.sample.cell)
