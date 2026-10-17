import pytest

import numpy as np
from pathlib import Path

from MJOLNIR.Data.MarmotDataReader import MarmotDataReader

data_file = Path.home() / 'projects' / 'marmot+flatcone' / '054724.nxs'

@pytest.fixture(scope="module")
def calibration_file(tmp_path_factory):
    # No Marmot calibration is shipped, so write a placeholder table: one energy bin per
    # pixel (Ef from 1 to 5 meV), detectors 2 degrees apart. Not physically meaningful.
    rows = [[det, pix, 0, 1.0, 1.0 + 4.0*pix/255, 0.1, 0.0, pix, pix + 1, 2.0*det]
            for det in range(30) for pix in range(256)]
    path = tmp_path_factory.mktemp('marmot') / 'calibration.csv'
    np.savetxt(path, rows, delimiter=',',
               header='Detector,Energy,Pixel,Amplitude,Center,Width,Background,lowerBin,upperBin,A4Offset')
    return path

@pytest.fixture(scope="module") # Read the file: common across all test functions here
def datareader(calibration_file):
    return MarmotDataReader(data_file, calibration_file=calibration_file)

def test_read(datareader):
    marmot_file = datareader
    assert marmot_file.file.name == '054724.nxs'
    assert marmot_file.nexus_keys['instrument'] == 'entry0/THALES'
    assert marmot_file.counts.shape == (361, 30, 256)
    assert marmot_file.counts[208, 2, 182] == 3803
    assert marmot_file.n_steps == 361
    assert marmot_file.A3.shape == (361,)
    assert marmot_file.A3[1] == pytest.approx(-44.5, abs=1e-8)
    assert marmot_file.A4[0] == pytest.approx(72.5, abs=1e-8)
    assert marmot_file.A4Offset == pytest.approx(0.0, abs=1e-8)
    assert marmot_file.Ei == pytest.approx(6.000828, abs=1e-6)
    assert marmot_file.monitor1.shape == (361,)
    assert marmot_file.monitor1[0] == 143373
    assert marmot_file.time[0] == pytest.approx(15.0, abs=1e-8)
    assert list(marmot_file.scan_parameters) == ['a3']
    assert marmot_file.scan_values[0, [0, -1]] == pytest.approx([-45.0, 135.0])
    assert marmot_file.sample.a == pytest.approx(4.87, abs=1e-5)
    assert marmot_file.sample.c == pytest.approx(3.31, abs=1e-5)
    metadata = marmot_file._metadata
    assert metadata['startTime'] == '13-Jun-26 23:32:11'
    assert metadata['proposalUser'] == 'steffens'

def test_no_calibration():
    marmot_file = MarmotDataReader(data_file)
    with pytest.raises(AttributeError):
        marmot_file.set_instrument_calibration()

def test_transform(datareader):
    marmot_file = datareader
    marmot_file.set_instrument_calibration()
    assert marmot_file.calibration.Ef.shape == (7680,)
    assert marmot_file.calibration.bound.shape == (7680, 2)
    marmot_file.transform()
    assert marmot_file.intensity.shape == (361, 30, 256)
    assert np.array_equal(marmot_file.intensity, marmot_file.counts)
    assert marmot_file.qx.shape == (361, 30, 256)
    assert marmot_file.h.shape == (361, 30, 256)
    assert marmot_file.intensity[208, 2, 182] == 3803
    assert marmot_file.qx[208, 2, 182] == pytest.approx(1.868840, abs=1e-6)
    assert marmot_file.qy[208, 2, 182] == pytest.approx(-0.424353, abs=1e-6)
    assert marmot_file.h[208, 2, 182] == pytest.approx(-1.024250, abs=1e-6)
    assert marmot_file.k[208, 2, 182] == pytest.approx(-1.024250, abs=1e-6)
    assert marmot_file.l[208, 2, 182] == pytest.approx(-0.223551, abs=1e-6)
    assert marmot_file.energy[208, 2, 182] == pytest.approx(2.145926, abs=1e-6)
