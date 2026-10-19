from abc import ABC, abstractmethod
from pathlib import Path

from MJOLNIR.Data.DataReader import DataReader

class DataWriter(ABC):
    """Abstract base class for writers: save what a reader holds back into a data file of the
    reader's format. A writer never writes over the file the reader read"""

    def __init__(self, reader: DataReader):
        self.reader = reader

    @abstractmethod
    def save(self, path: Path, overwrite: bool = False) -> Path:
        """Write the file to path and return it. An existing file at path is replaced only if
        overwrite is True; the reader's own file never is"""

    def _check_target(self, path: Path, overwrite: bool) -> Path:
        path = Path(path)
        source = Path(self.reader.file)
        if path.resolve() == source.resolve() or (path.exists() and path.samefile(source)):
            raise ValueError(f'"{path.name}" is the file the data was read from; save to a '
                             'new file instead')
        if path.exists() and not overwrite:
            raise FileExistsError(f'"{path}" exists; pass overwrite=True to replace it')
        return path
