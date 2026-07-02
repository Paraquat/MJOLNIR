import pytest

import numpy as np
from pathlib import Path
from pdb import set_trace

from MJOLNIR.Data.CameaDataReader import CameaDataReader

data_path = Path('/home/zamaan/Projects/bambus')
files = data_path.glob("camea2024*.hdf")

def test_camea_data_reader():
    data_file = sorted(list(files))[0]
    assert data_file.name == 'camea2024n001486.hdf'
    camea_file = CameaDataReader(data_file)
    assert camea_file.nexus_keys['instrument'] == 'entry/CAMEA'
    assert camea_file.counts.shape == (135, 104, 1024)
    assert camea_file.A4[0] == pytest.approx(-43.99, abs=1e-2)
    assert camea_file.A4Offset == pytest.approx(-4, abs=1e-2)
    assert camea_file.twotheta == pytest.approx(-39.99, abs=1e-2)
    assert camea_file.A3.shape == (135,)
    assert camea_file.Ei == pytest.approx(4.00048, abs=1e-5)
    assert camea_file.monitor1.shape == (135,)
    assert camea_file.monitor1[0] == 31181
    assert camea_file.monitor2[0] == 62500
    assert camea_file.sample.a == pytest.approx(5.34993, abs=1e-5)
    assert camea_file.sample.plane_vector1[9] == pytest.approx(4.99012, abs=1e-5)
    assert camea_file.sample.A3Off.shape == (1,)
    assert camea_file.sample.A3Off[0] == pytest.approx(0, abs=1e-8)
    instrument_calibration = camea_file.get_instrument_calibration(3)
    assert instrument_calibration.Ef.shape == (2496,)
    assert instrument_calibration.bound.shape == (2496, 2)
    assert instrument_calibration.bound[2, 0] == 152
    assert instrument_calibration.bound[2, 1] == 182
    assert instrument_calibration.Ef[862] == pytest.approx(4.98141, abs=1e-5)
    metadata = camea_file._metadata
    assert metadata['proposal'] == '20240261'
    assert metadata['sampleName'] == 'TbFeO3'
