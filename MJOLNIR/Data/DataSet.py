from pathlib import Path
from operator import attrgetter
from collections.abc import Callable, Iterable, Iterator
import numpy as np

from MJOLNIR.Data.ArrayType import ArrayType
from MJOLNIR.Data.AxisView import AxisView
from MJOLNIR.Data.DataReader import DataReader
from MJOLNIR.Data.DataReaderFactory import DataReaderFactory
from MJOLNIR.Data.InstrumentName import InstrumentName

# Anything that can be added to a DataSet: a reader, or the path of a file to read
DataFileLike = DataReader | str | Path

class DataSet:
    """A container of DataReader objects for different scans, measured on the same sample and
    all read by the same type of data reader. Data members of different scans can be returned
    as an AxisView object, which treats them as a single array contiguous along one axis, for
    the purposes of indexing and slicing."""
    instrument: InstrumentName | None
    reader_type: type[DataReader] | None
    _binning: int | None # None: the default binning of the reader type
    _axis: int # The contiguous axis for AxisView
    _readers: list[DataReader]

    def __init__(self, instrument: InstrumentName | None = None,
                 files: DataFileLike | Iterable[DataFileLike] | None = None):
        """Currently we only allow for a data set containing data from one instrument. Without
        an instrument, it is taken from the first data file added"""
        self._reader_factory = DataReaderFactory()
        self.instrument = instrument
        self.reader_type = None
        self._binning = None
        if instrument is not None:
            self.reader_type = self._reader_factory.get_reader_type(instrument)
            self._binning = self.reader_type.default_binning
        self._axis = 0
        self._readers = []
        if files is not None:
            self.extend(files)

    def set_binning(self, binning: int):
        """Recalibrate and transform all data files with a new binning"""
        self._binning = binning
        for reader in self._readers:
            reader.set_instrument_calibration(binning)
            reader.transform()

    # Adding data files

    def append(self, item: DataFileLike):
        """Add a data reader, or read a file with the instrument of the DataSet. Data files are
        calibrated with the binning of the DataSet and transformed when added"""
        reader = self._to_reader(item)
        if self.reader_type is not None and type(reader) is not self.reader_type:
            raise TypeError(f'Cannot add {type(reader).__name__} "{reader.file.name}" to a '
                            f'DataSet of {self.reader_type.__name__}')
        if len(self) > 0 and not reader.sample == self.sample:
            raise ValueError(f'Sample of "{reader.file.name}" differs from the sample of '
                             f'"{self._readers[0].file.name}"')
        if self.reader_type is None:
            self.reader_type = type(reader)
            self.instrument = self._reader_factory.get_instrument(self.reader_type)
        if self._binning is None:
            self._binning = self.reader_type.default_binning
        if reader.calibration is None or reader.binning != self._binning:
            reader.set_instrument_calibration(self._binning)
            reader.transform()
        self._readers.append(reader)

    def extend(self, items: DataFileLike | Iterable[DataFileLike]):
        if isinstance(items, (DataReader, str, Path)):
            items = [items]
        for item in items:
            self.append(item)

    def add_file(self, path: Path):
        self.append(path)

    def add_files(self, paths: Iterable[Path]):
        self.extend(paths)

    def _to_reader(self, item: DataFileLike) -> DataReader:
        if isinstance(item, DataReader):
            return item
        if isinstance(item, (str, Path)):
            if self.instrument is None:
                raise ValueError(
                    f'An instrument is needed to read "{item}": pass instrument to DataSet')
            return self._reader_factory.create(self.instrument, Path(item))
        raise TypeError(f'Cannot add {type(item).__name__} to DataSet: expected a DataReader '
                        'or a file path')

    # Container

    def number_of_files(self) -> int:
        return len(self._readers)

    def __len__(self) -> int:
        return len(self._readers)

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            subset = DataSet(instrument=self.instrument)
            subset.reader_type = self.reader_type
            subset._binning = self._binning
            subset.extend(self._readers[index])
            return subset
        try:
            return self._readers[index]
        except IndexError:
            raise IndexError(
                f'Index {index} is out of bounds for DataSet with length {len(self)}') from None

    def __delitem__(self, index: int | slice):
        if isinstance(index, int) and not -len(self) <= index < len(self):
            raise IndexError(
                f'Index {index} is out of bounds for DataSet with length {len(self)}')
        del self._readers[index]

    def __iter__(self) -> Iterator[DataReader]:
        return iter(self._readers)

    def __reversed__(self) -> Iterator[DataReader]:
        return reversed(self._readers)

    def __repr__(self) -> str:
        names = ', '.join(reader.file.name for reader in self)
        return f'DataSet([{names}])'

    @property
    def sample(self):
        """The sample shared by all data files"""
        if len(self) == 0:
            return None
        return self._readers[0].sample

    # Contiguous arrays

    def get_view(self, arr_type: ArrayType | str) -> AxisView:
        """One array of all data files, contiguous along the axis of the DataSet"""
        arr_type = ArrayType(arr_type)
        if len(self) == 0:
            raise ValueError('DataSet contains no data files')
        get = attrgetter(arr_type.value)
        arrays = [get(reader) for reader in self._readers]
        if any(array is None for array in arrays):
            raise AttributeError(f'{arr_type.value} is not available in every data file')
        return AxisView([np.atleast_1d(array) for array in arrays], axis = self._axis)

    # Sorting

    @staticmethod
    def default_sort_key(reader: DataReader) -> tuple:
        """Sort by lowest Ei, then |2theta|, then A3 scan direction, then A3 start"""
        direction = -np.sign(reader.A3[1] - reader.A3[0]) if len(reader.A3) > 1 else 0.0
        return (np.round(reader.Ei[0], 1), np.abs(np.round(reader.twotheta[0], 1)),
                direction, np.round(reader.A3[0], 2))

    def arg_auto_sort(self, key: Callable[[DataReader], tuple] | None = None) -> np.ndarray:
        """Indices that sort the data files by key (default: default_sort_key)"""
        if key is None:
            key = self.default_sort_key
        return np.array(sorted(range(len(self)), key=lambda i: key(self._readers[i])),
                        dtype=int)

    def auto_sort(self, key: Callable[[DataReader], tuple] | None = None):
        """Sort the data files by key (default: default_sort_key)"""
        self.apply_sorting(self.arg_auto_sort(key))

    def arg_sort_to(self, other: 'DataSet',
                    key: Callable[[DataReader], tuple] | None = None) -> np.ndarray:
        """Indices that order these data files like the matching files of another data set"""
        return self.arg_auto_sort(key)[np.argsort(other.arg_auto_sort(key))]

    def apply_sorting(self, indices: Iterable[int]):
        indices = list(indices)
        if sorted(indices) != list(range(len(self))):
            raise ValueError(f'Sorting indices {indices} are not a permutation of the '
                             f'{len(self)} data files')
        self._readers = [self._readers[i] for i in indices]
