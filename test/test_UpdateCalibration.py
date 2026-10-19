import pytest

import warnings
import numpy as np
from pathlib import Path

from MJOLNIR.Data.CameaDataReader import CameaDataReader
from MJOLNIR.Data.CameaDataWriter import CameaDataWriter
from MJOLNIR.Data.DataReader import read_calibration_file
from MJOLNIR.Data.DataSet import DataSet
from MJOLNIR.Data.InstrumentName import InstrumentName

ROOT = Path(__file__).parent.parent
CALIBRATION = {binning: ROOT / 'Data' / f'Normalization_{binning}.calib' for binning in (1, 3, 5, 8)}
SAMPLE_DATA = sorted((ROOT / 'sampledata').glob('camea2024*.hdf'))
needs_data = pytest.mark.skipif(not SAMPLE_DATA, reason='sampledata/camea2024*.hdf not available')

def with_header(tmp_path, source: Path, header: str, repeat: int = 1) -> Path:
    """A copy of a .calib file with another first line, its rows repeated"""
    lines = source.read_text().splitlines()
    path = tmp_path / 'edited.calib'
    path.write_text('\n'.join([header, *lines[1:3], *lines[3:] * repeat]) + '\n')
    return path

def test_read_calibration_file(tmp_path):
    binning, calibration = read_calibration_file(CALIBRATION[3])
    assert binning == 3
    assert calibration.Ef.shape == (2496,) and calibration.bound.shape == (2496, 2)
    assert calibration.bound.dtype.kind == 'i'
    table = np.loadtxt(CALIBRATION[3], skiprows=3, delimiter=',')
    assert np.array_equal(calibration.amp, table[:, 3]) # IntegratedIntensity
    assert np.array_equal(calibration.Ef, table[:, 4]) # Center
    assert np.array_equal(calibration.A4, table[:, 9])
    assert np.array_equal(read_calibration_file(CALIBRATION[3], A4_sign=-1)[1].A4, -table[:, 9])
    with pytest.raises(ValueError, match='not a calibration file'):
        read_calibration_file(with_header(tmp_path, CALIBRATION[1], 'No binning here'))

@needs_data
def test_reader_update_calibration(tmp_path):
    reader = CameaDataReader(SAMPLE_DATA[0])
    with pytest.warns(UserWarning, match='overwrite is False'):
        assert reader.update_calibration(CALIBRATION[8]) == [] # The file has binning 8
    assert reader.calibration_updates == {}
    assert reader.update_calibration(str(CALIBRATION[8]), overwrite=True) == [8] # One path
    reader.set_instrument_calibration(8)
    assert reader.binning == 8
    assert np.array_equal(reader.calibration.Ef, read_calibration_file(CALIBRATION[8])[1].Ef)
    assert not np.array_equal(reader.get_instrument_calibration(8).Ef, reader.calibration.Ef)

@needs_data
def test_reader_update_calibration_checks(tmp_path):
    reader = CameaDataReader(SAMPLE_DATA[0])
    wrong_rows = with_header(tmp_path, CALIBRATION[1], 'Normalization for 3 pixel(s)')
    with pytest.raises(ValueError, match='needs 2496'):
        reader.update_calibration(wrong_rows, overwrite=True)
    new_binning = with_header(tmp_path, CALIBRATION[1], 'Normalization for 2 pixel(s)', 2)
    assert reader.update_calibration(new_binning) == [2] # New binnings need no overwrite
    assert reader.possible_binnings == [1, 2, 3, 5, 8]
    assert CameaDataReader(SAMPLE_DATA[0]).possible_binnings == [1, 3, 5, 8]

@needs_data
def test_dataset_update_calibration():
    ds = DataSet(InstrumentName.CAMEA, files=SAMPLE_DATA[:1])
    normalization = ds._readers[0].normalization.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert ds.update_calibration(list(CALIBRATION.values()), overwrite=True) == [1, 3, 5, 8]
    new = read_calibration_file(CALIBRATION[8])[1]
    reader = ds._readers[0]
    assert np.array_equal(reader.calibration.amp, new.amp) # Transformed again
    assert not np.array_equal(reader.normalization, normalization)
    ds.append(SAMPLE_DATA[1]) # Files added later get the same calibration
    assert np.array_equal(ds._readers[1].calibration.amp, new.amp)
    ds.set_binning(3)
    assert np.array_equal(ds._readers[1].calibration.Ef, read_calibration_file(CALIBRATION[3])[1].Ef)

@needs_data
def test_dataset_update_other_binning_keeps_data():
    ds = DataSet(InstrumentName.CAMEA, files=SAMPLE_DATA[:1])
    intensity = ds._readers[0].intensity
    assert ds.update_calibration(CALIBRATION[3], overwrite=True) == [3]
    assert ds._readers[0].intensity is intensity # Binning 8 unchanged: not transformed again

@needs_data
def test_frozen_dataset_cannot_be_recalibrated():
    ds = DataSet(InstrumentName.CAMEA, files=SAMPLE_DATA[:1])
    difference = ds - ds
    with pytest.raises(ValueError, match='recalibrate'):
        difference.update_calibration(CALIBRATION[8], overwrite=True)

@needs_data
def test_writer_saves_updated_calibrations(tmp_path):
    reader = CameaDataReader(SAMPLE_DATA[0])
    reader.update_calibration([CALIBRATION[3], CALIBRATION[8]], overwrite=True)
    writer = CameaDataWriter(reader)
    assert sorted(writer.calibrations) == [3, 8]
    saved = CameaDataReader(writer.save(tmp_path / 'recalibrated.hdf'))
    for binning in (3, 8):
        new = read_calibration_file(CALIBRATION[binning])[1]
        assert np.array_equal(saved.get_instrument_calibration(binning).Ef,
                              new.Ef.astype(np.float32))
        assert np.array_equal(saved.get_instrument_calibration(binning).bound, new.bound)
    assert np.array_equal(saved.get_instrument_calibration(1).Ef,
                          reader.get_instrument_calibration(1).Ef) # Not updated: unchanged
