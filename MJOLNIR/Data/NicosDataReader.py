from pathlib import Path

from MJOLNIR.Data import DataReader

class NICOSDataReader(DataReader):
    """Parse a NICOS ASCII output file"""

    def __init__(self, filepath: Path):
        super().__init__(filepath)
