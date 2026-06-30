from pathlib import Path

from MJOLNIR.Data import NICOSDataReader

class BAMBUSDataReader(NICOSDataReader):
    """Parser a BAMBUS ASCII file"""

    analyser_lim = 4
    detector_lim = 19


    def __init__(self, filepath: Path):
        super().__init__(filepath)
