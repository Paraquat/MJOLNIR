from pathlib import Path
from enum import Enum

import MJOLNIR.Data.DataReaderFactory
from MJOLNIR.Data.AxisView import AxisView

class ArrayType(Enum):
    # Raw data
    COUNTS = "counts"
    A3OFF = "A3Off"
    A4OFF = "A4Off"
    EI = "Ei"

    # Instrument calibration
    EF_CAL = "calibration.Ef"
    WIDTH_CAL = "calibration.width"
    BG_CAL = "calibration.bg"
    AMP_CAL = "calibration.amp"
    A4_CAL = "calibration.A4"
    BOUND_CAL = "calibration.bound"

    # Derived data
    QX = "Qx"
    QY = "Qy"
    ENERGY = "energy"
    H = "h"
    K = "k"
    L = "l"

class DataSet:
    """A container of DataReader objects for different scans. Data members of
    different scans can be returned as an AxisView object, which treats them as
    a single array contiguous along one axis, for the purposes of indexing and
    slicing."""
    instrument: InstrumentName
    _binning: int
    _axis: int # The contiguous axis for AxisView
    _readers: list[DataReader]

    def __init__(self, instrument: InstrumentName):
        """Currently we only allow for a data set containing data from one instrument"""
        self.instrument = instrument
        self._binning = 1
        self._axis = 0
        self._reader_factory = MJOLNIR.Data.DataReaderFactory.DataReaderFactory()
        self._readers = []

    def set_binning(self, binning: int):
        self._binning = binning
        for reader in self._readers:
            reader.set_instrument_calibration(binning)

    def add_file(self, path: Path):
        reader = self._reader_factory.create(self.instrument, path)
        reader.set_instrument_calibration(self._binning)
        reader.transform()
        self._readers.append(reader)

    def add_files(self, paths: list[Path]):
        for path in paths:
            self.add_file(path)

    def number_of_files(self) -> int:
        return len(self._readers)

    def get_view(self, arr_type: ArrayType) -> AxisView:
        arrays = []
        for reader in self._readers:
            arrays.append(getattr(reader, arr_type.value))
        return AxisView(arrays, axis = self._axis)
