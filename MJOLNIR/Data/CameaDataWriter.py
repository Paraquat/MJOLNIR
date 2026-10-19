import h5py

from MJOLNIR.Data.CameaDataReader import CameaDataReader
from MJOLNIR.Data.DataReader import InstrumentCalibration
from MJOLNIR.Data.NexusLayout import CameaLayout
from MJOLNIR.Data.NexusDataWriter import NexusDataWriter

class CameaDataWriter(NexusDataWriter):
    """Save a CAMEA file with new instrument calibrations; replaces the old
    DataFile.saveHDF. Without calibrations, the reader's updated calibrations
    (update_calibration) are written, or if there are none its current calibration (for its
    binning). Calibrations for binnings the file does not have are added"""

    layout = CameaLayout

    def __init__(self, reader: CameaDataReader,
                 calibrations: dict[int, InstrumentCalibration] | None = None):
        super().__init__(reader)
        if calibrations is None and reader.calibration_updates:
            calibrations = reader.calibration_updates
        elif calibrations is None:
            calibrations = {} if reader.calibration is None \
                else {reader.binning: reader.calibration}
        self.calibrations = dict(calibrations)

    def set_calibration(self, binning: int, calibration: InstrumentCalibration):
        self.calibrations[binning] = calibration

    def update(self, f: h5py.File):
        super().update(f)
        for binning, calibration in self.calibrations.items():
            self._check_calibration(binning, calibration)
            for key, field in self.layout.calibration_fields(binning).items():
                self.write_field(f, field, getattr(calibration, key))

    def groups(self):
        groups = dict(super().groups())
        for binning in self.calibrations:
            groups[f'calib{binning}'] = self.layout.calibration_group_for(binning)
        return groups

    def _check_calibration(self, binning: int, calibration: InstrumentCalibration):
        """A calibration has one value per pixel bin: detectors x analysers x binning"""
        n_bins = self.reader.n_detectors * self.reader.n_analysers * binning
        for key in self.layout.calibration_fields(binning):
            shape = getattr(calibration, key).shape
            expected = (n_bins, 2) if key == 'bound' else (n_bins,)
            if shape != expected:
                raise ValueError(f'Calibration for binning {binning}: {key} has shape {shape}, '
                                 f'expected {expected}')
