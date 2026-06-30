import h5py
import numpy as np
from pathlib import Path

from MJOLNIR.Data.DataReader import DataReader
from MJOLNIR.Data.Sample import Sample

class NexusDataReader(DataReader):
    """Parse a Nexus HDF5 file"""

    # These keys are for critical data
    nexus_keys = {
        'instrument': 'entry/{instrument}',
        'sample':'/entry/sample',
        'unitCell':'/entry/sample/unit_cell',
        'intensity':'entry/data/intensity',
        'energy':'entry/data/en',
        'normalization':'entry/data/normalization',
        'preset':'entry/control/preset',
        'hdfMonitor':'entry/control/data',
        'monitor':'entry/monitor_2/data',
        'monitor_1':'entry/control/data',
        'time':'entry/control/time',
        'A3':'entry/sample/rotation_angle',
        'temperature':'entry/sample/temperature',
        'magneticField':'entry/sample/magnetic_field',
        'electricField':'entry/sample/electric_field',
        'scanCommand':'entry/scancommand',
        'absoluteTime':'entry/control/absolute_time',
        'protonBeam':'entry/proton_beam/data',
    }

    # These keys are for metadata fields
    nexus_meta_keys = {
        'title':'entry/title',
        'startTime':'entry/start_time',
        'endTime':'entry/end_time',
        'comment':'entry/comment',
        'proposal':'entry/proposal_id',
        'proposalTitle':'entry/proposal_title',
        'experimentalIdentifier':'entry/experiment_identifier',
        'sampleName':'/entry/sample/name',
        'localContact':'entry/local_contact/name',
        'proposalUser':'entry/proposal_user/name',
        'proposalEmail':'entry/proposal_user/email',
        'user':'entry/user/name',
        'email':'entry/user/email',
        'address':'entry/user/address',
        'affiliation':'entry/user/affiliation',
        'mode':'entry/control/mode',
    }

    def __init__(self, filepath: Path):
        super().__init__(filepath)
        self._file = h5py.File(filepath, mode = 'r')
        self.counts = np.array(self.getValue('counts'))
        self.Ei = np.array(self.getValue('Ei'))
        self.A3 = np.array(self.getValue('A3'))
        self.A4 = np.array(self.getValue('A4'))
        self.A4Offset = np.array(self.getValue('A4Offset'))
        self.twotheta = self.A4 - self.A4Offset
        self.sample = Sample(sample=self.getValue('sample'))

    def __init_subclass__(cls):
        context = {}
        for c in reversed(cls.__mro__):
            context.update(vars(c))

        cls.nexus_keys = {
            key: value.format_map(context)
            for key, value in cls.nexus_keys.items()
        }

    def getValue(self, key: str):
        return self._file.get(self.nexus_keys[key])

    def get(self, key: str):
        return self._file.get(key)
