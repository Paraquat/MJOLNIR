from pathlib import Path
from abc import ABC
from dataclasses import dataclass
import numpy as np

from MJOLNIR import _tools
from MJOLNIR.Data.Metadata import Metadata
from MJOLNIR.Data.Sample import Sample

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

    EPrDetector = 1
    n_analysers = 1
    n_detectors = 1

    counts: np.NDArray[np.float64]
    A3: np.NDArray[np.float64]
    A3Offset: np.NDArray[np.float64]
    A4: np.NDArray[np.float64]
    A4Offset: np.NDArray[np.float64]
    Ei: np.NDArray[np.float64]
    twotheta: np.NDArray[np.float64]
    calibration: InstrumentCalibration
    sample: Sample

    def __init__(self, filepath: Path):
        if not filepath.exists():
            raise AttributeError(f'File path "{filepath}" does not exist')
        self._file = filepath
        self._metadata = Metadata()


    def transform(self):
        pass
