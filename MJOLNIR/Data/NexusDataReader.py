import h5py
import numpy as np
from pathlib import Path

from MJOLNIR.Data.DataReader import DataReader
from MJOLNIR.Data.Sample import Sample

class NexusDataReader(DataReader):
    """Parse a Nexus HDF5 file"""

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
        self.get_metadata()

    def __init_subclass__(cls):
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

    def getValue(self, key: str):
        """Dataset for a key, or None if the key is not defined for this instrument"""
        if key not in self.nexus_keys:
            return None
        return self._file.get(self.nexus_keys[key])

    def get_sample(self) -> Sample:
        return Sample(sample=self.getValue('sample'), recalculateUB=True)

    def get_counts(self) -> np.NDArray[np.float64]:
        return np.array(self.getValue('counts'))

    def get(self, key: str):
        return self._file.get(key)

    def get_metadata(self):
        for key in self.nexus_meta_keys:
            value = self._file.get(self.nexus_meta_keys[key])
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
