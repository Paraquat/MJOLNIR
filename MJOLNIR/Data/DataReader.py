from pathlib import Path
from abc import ABC
from dataclasses import dataclass
import numpy as np

from MJOLNIR import _tools

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
    analyser_lim = 1
    detector_lim = 1

    A3 = np.array([], dtype=float)
    A4 = np.array([], dtype=float)
    A4Offset = np.array([], dtype=float)
    Ei = np.array([], dtype=float)
    twotheta = np.array([], dtype=float)

    def __init__(self, filepath: Path):
        if not filepath.exists():
            raise AttributeError(f'File path "{filepath}" does not exist')
        self._file = filepath

