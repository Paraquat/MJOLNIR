import h5py
import numpy as np
import numpy.typing as npt
from pathlib import Path

from MJOLNIR.Data.DataReader import DataReader, SampleEnvironment
from MJOLNIR.Data.NexusLayout import NexusLayout
from MJOLNIR.Data.Sample import Sample

class NexusDataReader(DataReader):
    """Parse a Nexus HDF5 file.

    Where the data is comes from layout (a NexusLayout) if the subclass sets one, otherwise
    from the key tables below (readers not yet moved to a layout)"""

    layout: type[NexusLayout] | None = None

    # These keys are for critical data
    nexus_keys = {
        'instrument': '{entry}/{instrument}',
        'sample': '{entry}/sample',
        'unitCell': '{entry}/sample/unit_cell',
        'intensity': '{entry}/data/intensity',
        'energy': '{entry}/data/en',
        'normalization': '{entry}/data/normalization',
        'preset': '{entry}/control/preset',
        'hdfMonitor': '{entry}/control/data',
        'monitor': '{entry}/monitor_2/data',
        'monitor_1': '{entry}/control/data',
        'time': '{entry}/control/time',
        'A3': '{entry}/sample/rotation_angle',
        'temperature': '{entry}/sample/temperature',
        'magneticField': '{entry}/sample/magnetic_field',
        'electricField': '{entry}/sample/electric_field',
        'scanCommand': '{entry}/scancommand',
        'absoluteTime': '{entry}/control/absolute_time',
        'protonBeam': '{entry}/proton_beam/data',
    }

    # These keys are for metadata fields
    nexus_meta_keys = {
        'title': '{entry}/title',
        'startTime': '{entry}/start_time',
        'endTime': '{entry}/end_time',
        'comment': '{entry}/comment',
        'proposal': '{entry}/proposal_id',
        'proposalTitle': '{entry}/proposal_title',
        'experimentalIdentifier': '{entry}/experiment_identifier',
        'sampleName': '{entry}/sample/name',
        'localContact': '{entry}/local_contact/name',
        'proposalUser': '{entry}/proposal_user/name',
        'proposalEmail': '{entry}/proposal_user/email',
        'user': '{entry}/user/name',
        'email': '{entry}/user/email',
        'address': '{entry}/user/address',
        'affiliation': '{entry}/user/affiliation',
        'mode': '{entry}/control/mode',
    }

    # Sample environment: candidate paths per SampleEnvironment field, the first that exists
    # is used
    environment_keys = {
        'temperature': ['{entry}/sample/temperature'],
        'temperature_setpoint': ['{entry}/sample/setpoint_temperature'],
        'magnetic_field': ['{entry}/sample/magnetic_field'],
        'magnetic_field_setpoint': ['{entry}/sample/setpoint_magnetic_field'],
        'electric_field': ['{entry}/sample/electric_field'],
        'electric_field_setpoint': ['{entry}/sample/setpoint_electric_field'],
    }

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

    def __init_subclass__(cls):
        if cls.layout is not None:
            return
        context = {}
        for c in reversed(cls.__mro__):
            context.update(vars(c))

        cls.nexus_keys = {
            key: value.format_map(context)
            for key, value in cls.nexus_keys.items()
        }

        cls.nexus_meta_keys = {
            key: value.format_map(context)
            for key, value in cls.nexus_meta_keys.items()
        }

        cls.environment_keys = {
            key: [path.format_map(context) for path in paths]
            for key, paths in cls.environment_keys.items()
        }

    @classmethod
    def can_read(cls, filepath: Path) -> bool:
        """An HDF5 file with this instrument's group and counts"""
        try:
            if not h5py.is_hdf5(filepath):
                return False
            with h5py.File(filepath, mode = 'r') as f:
                if cls.layout is not None:
                    return (cls.layout.group('instrument').path in f
                            and any(path in f for path in cls.layout.field('counts').paths)
                            and cls.identify(f))
                return (cls.nexus_keys['instrument'] in f and cls.nexus_keys['counts'] in f
                        and cls.identify(f))
        except (OSError, KeyError):
            return False

    @classmethod
    def identify(cls, f: h5py.File) -> bool:
        """Extra checks for instruments that share a file layout with others"""
        return True

    def getValue(self, key: str):
        """Dataset (or group) for a key, or None if the key is not defined for this instrument
        or not in the file. With a layout, the first of the field's paths in the file"""
        if self.layout is not None:
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
        if key not in self.nexus_keys:
            return None
        return self._file.get(self.nexus_keys[key])

    def environment_paths(self) -> dict[str, tuple[str, ...]]:
        """Candidate paths per SampleEnvironment field, in the order they are tried"""
        if self.layout is not None:
            return {key: self.layout.field(key).paths for key in self.layout.environment}
        return {key: tuple(paths) for key, paths in self.environment_keys.items()}

    def metadata_paths(self) -> dict[str, tuple[str, ...]]:
        """Candidate paths per metadata key, in the order they are tried"""
        if self.layout is not None:
            return {key: self.layout.field(key).paths for key in self.layout.metadata}
        return {key: (path,) for key, path in self.nexus_meta_keys.items()}

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
