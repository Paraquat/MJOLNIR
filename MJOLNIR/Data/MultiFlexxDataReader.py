from pathlib import Path

import MJOLNIR
from MJOLNIR.Data.NicosDataReader import NICOSDataReader

class MultiFLEXXDataReader(NICOSDataReader):
    """Parse a MultiFLEXX ASCII file"""

    n_analysers = 5
    n_detectors = 31
    instrument = "MultiFLEXX"
    calibration_file = MJOLNIR.__multiFLEXXNormalization__

    def __init__(self, filepath: Path, calibration_file: str | None = None):
        super().__init__(filepath, calibration_file)

    def scan_columns(self, titles: list[str]) -> list[int]:
        # As in DataFile, scan parameters end one column before the first channel
        return list(range(titles.index('ch0') - 1))

    @classmethod
    def channel_columns(cls, titles: list[str]) -> list[int]:
        # Channels are named ch0, ..., ch154 (raises ValueError if one is missing)
        return [titles.index(f'ch{i}') for i in range(cls.n_detectors * cls.n_analysers)]
