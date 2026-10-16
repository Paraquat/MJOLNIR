import pytest

import numpy as np
from pathlib import Path

from MJOLNIR.Data.BambusDataReader import BAMBUSDataReader

@pytest.fixture(scope="module") # Read the file: common across all test functions here
def datareader():
    data_file = Path.home() / 'projects' / 'CoRhO4' / 'corh2o4_test.dat'
    return BAMBUSDataReader(data_file)

def test_read(datareader):
    bambus_file = datareader
    assert bambus_file.file.name == 'corh2o4_test.dat'
    assert bambus_file.counts.shape == (101, 20, 5)
    assert list(bambus_file.counts[0, 0]) == [0, 0, 112, 0, 0]
    assert bambus_file.n_steps == 101
    assert bambus_file.A3.shape == (101,)
    assert bambus_file.A3[1] == pytest.approx(-49.0, abs=1e-8)
    assert bambus_file.A4[0] == pytest.approx(60.5, abs=1e-8)
    assert bambus_file.twotheta[0] == pytest.approx(60.5, abs=1e-8)
    assert bambus_file.Ei == pytest.approx(6.3, abs=1e-8)
    assert bambus_file.monitor1.shape == (101,)
    assert bambus_file.time[0] == pytest.approx(2.0, abs=1e-8)
    assert list(bambus_file.scan_parameters) == ['stt', 'sth']
    assert bambus_file.sample.a == pytest.approx(8.5031, abs=1e-5)
    bambus_file.set_instrument_calibration(1)
    assert bambus_file.calibration.Ef.shape == (100,)
    assert bambus_file.calibration.bound.shape == (100, 2)
    assert bambus_file.calibration.Ef[3] == pytest.approx(4.5, abs=1e-8)
    assert bambus_file.calibration.A4[5] == pytest.approx(-17.0, abs=1e-8)
    with pytest.raises(AttributeError):
        bambus_file.set_instrument_calibration(3)
    metadata = bambus_file._metadata
    assert metadata['startTime'] == '2026-09-09 15:33:22'
    assert metadata['sampleName'] == 'CoRh2O4'

def test_transform(datareader):
    bambus_file = datareader
    bambus_file.set_instrument_calibration(1)
    bambus_file.transform()
    assert bambus_file.intensity.shape == (101, 20, 5)
    assert np.array_equal(bambus_file.intensity, bambus_file.counts)
    assert bambus_file.Qx.shape == (101, 20, 5)
    assert bambus_file.h.shape == (101, 20, 5)
    assert bambus_file.Qx[50, 7, 2] == pytest.approx(1.145028, abs=1e-6)
    assert bambus_file.Qy[50, 7, 2] == pytest.approx(0.956707, abs=1e-6)
    assert bambus_file.h[50, 7, 2] == pytest.approx(0.915507, abs=1e-6)
    assert bambus_file.k[50, 7, 2] == pytest.approx(0.915507, abs=1e-6)
    assert bambus_file.l[50, 7, 2] == pytest.approx(-1.549578, abs=1e-6)
    assert bambus_file.energy[50, 7, 2] == pytest.approx(2.3, abs=1e-6)
