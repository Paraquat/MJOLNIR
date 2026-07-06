from pathlib import Path
import numpy as np

from MJOLNIR.Data import NICOSDataReader

class BAMBUSDataReader(NICOSDataReader):
    """Parser a BAMBUS ASCII file"""

    n_analysers = 5
    n_detectors = 20
    instrument = "BAMBUS"

    def __init__(self, filepath: Path):
        super().__init__(filepath)

    def rebin(self, binning: int, shape: tuple, bounds: np.NDArray[np.int32]):
        pass
