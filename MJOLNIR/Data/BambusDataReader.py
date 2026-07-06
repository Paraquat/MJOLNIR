from pathlib import Path
import numpy as np

from MJOLNIR.Data import NICOSDataReader

class BAMBUSDataReader(NICOSDataReader):
    """Parser a BAMBUS ASCII file"""

    n_analyser = 5
    n_detector = 20

    def __init__(self, filepath: Path):
        super().__init__(filepath)

    def rebin(self, binning: int, shape: tuple, bounds: np.NDArray[np.int32]):
        pass
