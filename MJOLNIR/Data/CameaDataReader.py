import numpy as np
import numpy.typing as npt
from pathlib import Path

from MJOLNIR.Data.DataReader import InstrumentCalibration
from MJOLNIR.Data.NexusDataReader import NexusDataReader
from MJOLNIR.Data.NexusLayout import CameaLayout

class CameaDataReader(NexusDataReader):
    """Parse a CAMEA Nexus file"""

    n_analysers = 8
    n_detectors = 104
    instrument = CameaLayout.instrument
    default_binning = 8 # As chosen by the old DataFile conversion
    layout = CameaLayout

    def __init__(self, filepath: Path):
        super().__init__(filepath)
        self.counts = self.counts.swapaxes(1,2)
        self.monitor1 = np.array(self.getValue('monitor1'))
        self.monitor2 = np.array(self.getValue('monitor2'))
        self.get_possible_binnings()

    def get_possible_binnings(self):
        """Binnings with a calibration group in the file (one digit, as the old DataFile)"""
        self.possible_binnings = [binning for binning in range(1, 10)
                                  if self.layout.calibration_group_for(binning).path in self._file]

    def get_instrument_calibration(self, binning: int) -> InstrumentCalibration:
        fields = self.layout.calibration_fields(binning)
        self.binning = binning
        return InstrumentCalibration(**{key: np.array(self.get(field.path))
                                        for key, field in fields.items()})

    def scan_monitor(self) -> npt.NDArray[np.float64]:
        """Monitor used for normalisation, chosen as the old DataFile did: files written by
        NICOS (which have entry/data/data) use entry/monitor_2/data if present, older files
        entry/control/data"""
        written_by_nicos = self.get(self.layout.group('data').path + '/data') is not None
        if written_by_nicos and self.getValue('monitor2') is not None:
            return self.monitor2
        return self.monitor1

    def default_mask(self) -> npt.NDArray[np.bool_]:
        """The general default mask, and at binning 8 the first two energy bins (as the old
        DataFile conversion did)"""
        mask = super().default_mask()
        if self.binning == 8:
            mask[:, :, :2] = True
        return mask
