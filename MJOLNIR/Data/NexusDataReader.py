import h5py
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Any

from MJOLNIR.Data.DataReader import DataReader, SampleEnvironment
from MJOLNIR.Data.NexusLayout import NexusLayout
from MJOLNIR.Data.Sample import Sample

class NexusDataReader(DataReader):
    """Parse a Nexus HDF5 file. Where the data is stored comes from layout (a NexusLayout
    subclass), set by each instrument's reader"""

    layout: type[NexusLayout]

    def __init__(self, filepath: Path, calibration_file: str | None = None):
        super().__init__(filepath, calibration_file)
        self._file = h5py.File(filepath, mode = 'r')
        self.sample = self.get_sample()
        self.counts = self.get_counts()
        self.n_steps = self.counts.shape[0]
        self.Ei = np.array(self.getValue('Ei'))
        self.A3 = np.array(self.getValue('A3'))
        self.A3Offset = self.sample.A3Off
        self.A4 = np.array(self.getValue('A4'))
        A4Offset = self.getValue('A4Offset')
        self.A4Offset = np.array([0.0]) if A4Offset is None else np.array(A4Offset)
        self.twotheta = self.A4 - self.A4Offset
        self.environment = self.get_environment()
        self.get_metadata()

    @classmethod
    def can_read(cls, filepath: Path) -> bool:
        """An HDF5 file with this instrument's group and counts"""
        try:
            if not h5py.is_hdf5(filepath):
                return False
            with h5py.File(filepath, mode = 'r') as f:
                return (cls.layout.group('instrument').path in f
                        and any(path in f for path in cls.layout.field('counts').paths)
                        and cls.identify(f))
        except (OSError, KeyError):
            return False

    @classmethod
    def shallow_keys(cls) -> list[str]:
        """Keys shallow_read accepts: every field of the layout, the file name and folder,
        and 2θ, sorted ignoring case"""
        return sorted({*cls._shallow_extras, *cls.layout.all_fields()}, key=str.lower)

    @classmethod
    def metadata_keys(cls) -> list[str]:
        return list(cls.layout.metadata)

    @classmethod
    def shallow_read(cls, filepath: Path, keys: list[str]) -> dict[str, Any]:
        """A few values from a file, reading only the datasets asked for (the old
        shallowRead). Values are as stored in the file, not reshaped as the reader does
        (e.g. CAMEA counts, one temperature for a whole scan); metadata and single strings
        are decoded as the reader's metadata. None for a key the file does not have"""
        cls._check_shallow_keys(keys)
        filepath = Path(filepath)
        if not cls.can_read(filepath):
            raise ValueError(f'"{filepath.name}" is not a {cls.instrument} data file that '
                             f'{cls.__name__} can read')
        with h5py.File(filepath, mode = 'r') as f:
            def dataset(key: str):
                return next((f[path] for path in cls.layout.field(key).paths
                             if isinstance(f.get(path), h5py.Dataset)), None)

            values = {}
            for key in keys:
                if key == 'name':
                    values[key] = filepath.name
                elif key == 'fileLocation':
                    values[key] = str(filepath.parent)
                elif key == 'twoTheta': # As the reader: A4 - A4Offset, offset 0 if not stored
                    A4 = dataset('A4') if 'A4' in cls.layout.all_fields() else None
                    offset = dataset('A4Offset') if 'A4Offset' in cls.layout.all_fields() \
                        else None
                    values[key] = None if A4 is None else \
                        np.array(A4) - (np.array([0.0]) if offset is None else np.array(offset))
                else:
                    values[key] = cls._shallow_value(key, dataset(key))
        return values

    @classmethod
    def _shallow_value(cls, key: str, dataset):
        if dataset is None:
            return None
        if key in cls.layout.metadata: # As get_metadata
            value = dataset[0]
            return value.decode() if isinstance(value, bytes) else value
        value = np.array(dataset)
        if value.dtype.kind in 'SO' and value.size == 1:
            value = value.ravel()[0]
            return value.decode() if isinstance(value, bytes) else value
        return value

    @classmethod
    def identify(cls, f: h5py.File) -> bool:
        """Extra checks for instruments that share a file layout with others"""
        return True

    def getValue(self, key: str):
        """Dataset (or group) for a key, or None if the key is not defined for this instrument
        or not in the file: the first of the field's paths in the file"""
        if key in self.layout.groups:
            return self._file.get(self.layout.group(key).path)
        try:
            paths = self.layout.field(key).paths
        except KeyError:
            return None
        for path in paths:
            value = self._file.get(path)
            if value is not None:
                return value
        return None

    def environment_paths(self) -> dict[str, tuple[str, ...]]:
        """Candidate paths per SampleEnvironment field, in the order they are tried"""
        return {key: self.layout.field(key).paths for key in self.layout.environment}

    def metadata_paths(self) -> dict[str, tuple[str, ...]]:
        """Candidate paths per metadata key, in the order they are tried"""
        return {key: self.layout.field(key).paths for key in self.layout.metadata}

    def get_environment(self) -> SampleEnvironment:
        """Sample conditions from the first existing path of each quantity"""
        environment = SampleEnvironment()
        for quantity, paths in self.environment_paths().items():
            for path in paths:
                dataset = self._file.get(path)
                if not isinstance(dataset, h5py.Dataset): # Missing, or a group
                    continue
                values = self.per_step(dataset[()], quantity)
                if values is None:
                    continue
                setattr(environment, quantity, values)
                unit = dataset.attrs.get('units')
                if unit is not None:
                    environment.units[quantity] = unit.decode() if isinstance(unit, bytes) else str(unit)
                break
        return environment

    def get_sample(self) -> Sample:
        return Sample(sample=self.getValue('sample'), recalculateUB=True)

    def get_counts(self) -> npt.NDArray[np.float64]:
        return np.array(self.getValue('counts'))

    def get(self, key: str):
        return self._file.get(key)

    def get_metadata(self):
        for key, paths in self.metadata_paths().items():
            value = next((self._file.get(path) for path in paths if path in self._file), None)
            if value is not None:
                value = value[0]
                self._metadata[key] = value.decode() if isinstance(value, bytes) else value

    @staticmethod
    def parse_scan_command(command: str):
        """Scan parameters and their values from an ILL (MAD) scan command, e.g.
        'bs a3p 37.5 da3p 2.5 np 6' (start, step) or 'sc a3 45 da3 .5 np 361 ti 15'
        (centre, step)"""
        tokens = command.split()
        scan_type = tokens[0]
        n_points = int(tokens[tokens.index('np') + 1])
        params = np.array(tokens[1:tokens.index('np')]).reshape(-1, 4)
        steps = np.arange(n_points)
        if scan_type == 'sc':
            steps = steps - (n_points - 1) / 2
        elif scan_type != 'bs':
            raise AttributeError(f'Scan type not recognised: "{scan_type}"')
        scan_values = np.array([float(p[1]) + steps * float(p[3]) for p in params])
        return params[:, 0], scan_values
