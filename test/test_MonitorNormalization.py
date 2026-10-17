import pytest

import numpy as np
from pathlib import Path

from MJOLNIR.Data.ArrayType import ArrayType
from MJOLNIR.Data.CameaDataReader import CameaDataReader
from MJOLNIR.Data.BambusDataReader import BAMBUSDataReader
from MJOLNIR.Data.MultiFlexxDataReader import MultiFLEXXDataReader
from MJOLNIR.Data.FlatconeDataReader import FlatconeDataReader
from MJOLNIR.Data.MarmotDataReader import MarmotDataReader
from MJOLNIR.Data.DataSet import DataSet
from MJOLNIR.Data.InstrumentName import InstrumentName

camea_files = sorted((Path(__file__).parent.parent / 'sampledata').glob('camea2024*.hdf'))
bambus_file = Path.home() / 'projects' / 'CoRhO4' / 'corh2o4_test.dat'
flatcone_file = Path.home() / 'projects' / 'marmot+flatcone' / '043555_pastis.nxs'
marmot_file = Path.home() / 'projects' / 'marmot+flatcone' / '054724.nxs'

def transformed(reader):
    reader.set_instrument_calibration()
    reader.transform()
    return reader

@pytest.fixture(scope="module")
def marmot_calibration(tmp_path_factory):
    rows = [[det, pix, 0, 1.0, 1.0 + 4.0*pix/255, 0.1, 0.0, pix, pix + 1, 2.0*det]
            for det in range(30) for pix in range(256)]
    path = tmp_path_factory.mktemp('marmot') / 'calibration.csv'
    np.savetxt(path, rows, delimiter=',',
               header='Detector,Energy,Pixel,Amplitude,Center,Width,Background,lowerBin,upperBin,A4Offset')
    return path

@pytest.fixture(scope="module")
def multiflexx_file(tmp_path_factory):
    # No MultiFLEXX NICOS file is available: the BAMBUS header with channels ch0, ..., ch154
    lines = bambus_file.read_text().splitlines()
    header = lines[:next(i for i, line in enumerate(lines) if '### Scan data' in line) + 1]
    columns = ['sth', 'stt', 'mon1'] + [f'ch{i}' for i in range(155)] + ['events']
    rows = [' '.join([f'{sth:.3f}', '60.500', '7'] + ['1'] * 155 + ['0'])
            for sth in np.arange(-20, 21, 2.0)]
    path = tmp_path_factory.mktemp('multiflexx') / 'multiflexx_synthetic'
    path.write_text('\n'.join(header + ['#  ' + ' '.join(columns),
                                        '#  ' + ' '.join(['deg', 'deg'] + ['cts'] * 157)]
                              + rows + ['### End of NICOS data file']) + '\n')
    return path

def check_spread(reader, monitor_per_step):
    """Monitor and normalization have the data's shape: the monitor repeats one value per
    step, the normalization repeats the calibrated amplitude over the steps"""
    shape = reader.intensity.shape
    assert reader.monitor.shape == reader.normalization.shape == shape
    assert np.array_equal(reader.monitor, np.broadcast_to(
        np.asarray(monitor_per_step, dtype=float).reshape(-1, 1, 1), shape))
    amplitude = reader.calibration.amp.reshape(1, *shape[1:])
    assert np.array_equal(reader.normalization, np.broadcast_to(amplitude, shape), equal_nan=True)

def test_camea():
    # Files written by NICOS are normalised by entry/monitor_2/data, as by the old DataFile
    reader = transformed(CameaDataReader(camea_files[0]))
    assert reader.monitor.shape == (135, 104, 64)
    assert reader.monitor[0, 0, 0] == 62500
    check_spread(reader, reader.monitor2)

def test_bambus():
    reader = transformed(BAMBUSDataReader(bambus_file))
    check_spread(reader, reader.monitor1)

def test_multiflexx(multiflexx_file):
    # As the old DataFile: MultiFLEXX data are not normalised by monitor
    reader = transformed(MultiFLEXXDataReader(multiflexx_file))
    assert np.all(reader.monitor1 == 7)
    check_spread(reader, np.ones(reader.n_steps))

def test_flatcone():
    reader = transformed(FlatconeDataReader(flatcone_file))
    assert reader.monitor[0, 0, 0] == 5000000
    check_spread(reader, reader.monitor1)

def test_marmot(marmot_calibration):
    reader = transformed(MarmotDataReader(marmot_file, calibration_file=marmot_calibration))
    assert reader.monitor[0, 0, 0] == 143373
    check_spread(reader, reader.monitor1)

def test_follows_binning():
    reader = transformed(CameaDataReader(camea_files[0]))
    reader.set_instrument_calibration(3)
    reader.transform()
    assert reader.monitor.shape == reader.normalization.shape == (135, 104, 24)

def test_wrong_monitor_length():
    reader = CameaDataReader(camea_files[0])
    reader.set_instrument_calibration()
    reader.monitor2 = reader.monitor2[:-1]
    with pytest.raises(ValueError, match='134 monitor values for 135 scan steps'):
        reader.transform()

def test_views():
    dataset = DataSet(InstrumentName.CAMEA, files=camea_files[:2])
    assert dataset.get_view(ArrayType.MONITOR).shape == (270, 104, 64)
    assert dataset.get_view(ArrayType.NORMALIZATION).shape == (270, 104, 64)
