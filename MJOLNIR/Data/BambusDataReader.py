import re
from pathlib import Path

import MJOLNIR
from MJOLNIR.Data.NicosDataReader import NICOSDataReader

class BAMBUSDataReader(NICOSDataReader):
    """Parse a BAMBUS ASCII file"""

    n_analysers = 5
    n_detectors = 20
    instrument = "BAMBUS"
    calibration_file = MJOLNIR.__bambusNormalization__

    def __init__(self, filepath: Path, calibration_file: str | None = None):
        super().__init__(filepath, calibration_file)

    def scan_columns(self, titles: list[str]) -> list[int]:
        # Scan parameters precede the ';' separator, which starts with the timer
        return list(range(titles.index('timer')))

    def channel_columns(self, titles: list[str]) -> list[int]:
        # Channels are named by wedge and energy: 1A, 1B, ..., 20E
        return [i for i, title in enumerate(titles) if re.fullmatch(r'\d+[A-E]', title)]
