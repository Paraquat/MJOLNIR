import pytest

import numpy as np
from pathlib import Path

from MJOLNIR.Data.FlatconeDataReader import FlatconeDataReader

@pytest.fixture(scope="module") # Read the file: common across all test functions here
def datareader():
    data_file = Path.home() / 'projects' / 'marmot+flatcone' / '043555_pastis.nxs'
    return FlatconeDataReader(data_file)

def test_read(datareader):
    flatcone_file = datareader
    assert flatcone_file.file.name == '043555_pastis.nxs'
    assert flatcone_file.nexus_keys['instrument'] == 'entry0/IN20'
    assert flatcone_file.counts.shape == (6, 31, 1)
    assert flatcone_file.counts[0, 1, 0] == pytest.approx(1.556019, abs=1e-6)
    assert flatcone_file.n_steps == 6
    assert flatcone_file.A3.shape == (6,)
    assert flatcone_file.A3[1] == pytest.approx(40.0, abs=1e-8)
    assert flatcone_file.A4[0] == pytest.approx(44.49, abs=1e-2)
    assert flatcone_file.A4Offset == pytest.approx(0.0, abs=1e-8)
    assert flatcone_file.Ei == pytest.approx(35.12867, abs=1e-5)
    assert flatcone_file.monitor1.shape == (6,)
    assert flatcone_file.monitor1[0] == 5000000
    assert flatcone_file.temperature[0] == pytest.approx(1.557, abs=1e-6)
    assert list(flatcone_file.scan_parameters) == ['a3p']
    assert flatcone_file.scan_values[0] == pytest.approx([37.5, 40, 42.5, 45, 47.5, 50])
    assert flatcone_file.sample.a == pytest.approx(7.952, abs=1e-5)
    assert flatcone_file.sample.c == pytest.approx(4.44, abs=1e-5)
    flatcone_file.set_instrument_calibration()
    assert flatcone_file.binning == 1
    assert flatcone_file.calibration.Ef.shape == (31,)
    assert flatcone_file.calibration.bound.shape == (31, 2)
    assert flatcone_file.calibration.Ef[0] == pytest.approx(18.64910, abs=1e-5)
    assert flatcone_file.calibration.A4[1] == pytest.approx(35.0, abs=1e-8)
    metadata = flatcone_file._metadata
    assert metadata['startTime'] == '02-Sep-25 14:20:58'
    assert metadata['proposalUser'] == 'internal use'

def test_transform(datareader):
    flatcone_file = datareader
    flatcone_file.set_instrument_calibration()
    flatcone_file.transform()
    assert flatcone_file.intensity.shape == (6, 31, 1)
    assert np.array_equal(flatcone_file.intensity, flatcone_file.counts)
    assert flatcone_file.Qx.shape == (6, 31, 1)
    assert flatcone_file.h.shape == (6, 31, 1)
    assert flatcone_file.Qx[3, 10, 0] == pytest.approx(3.534665, abs=1e-6)
    assert flatcone_file.Qy[3, 10, 0] == pytest.approx(-0.023110, abs=1e-6)
    assert flatcone_file.h[3, 10, 0] == pytest.approx(-2.236736, abs=1e-6)
    assert flatcone_file.k[3, 10, 0] == pytest.approx(-2.236736, abs=1e-6)
    assert flatcone_file.l[3, 10, 0] == pytest.approx(-0.016330, abs=1e-6)
    assert flatcone_file.energy[3, 10, 0] == pytest.approx(16.479564, abs=1e-6)
