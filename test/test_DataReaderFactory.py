import pytest

import numpy as np
from pathlib import Path
from pdb import set_trace

from MJOLNIR.Data.InstrumentName import InstrumentName
from MJOLNIR.Data.DataReaderFactory import DataReaderFactory
from MJOLNIR.Data.CameaDataReader import CameaDataReader

def test_factory():
    data_path = Path('/home/zamaan/Projects/bambus')
    files = data_path.glob("camea2024*.hdf")
    data_file = sorted(list(files))[0]

    factory = DataReaderFactory()
    reader = factory.create(InstrumentName.CAMEA, data_file)
    assert isinstance(factory._registry[InstrumentName.CAMEA], type(CameaDataReader))

    assert reader.file.name == 'camea2024n001486.hdf'
    assert reader.counts.shape == (135, 104, 1024)
    assert reader.A4[0] == pytest.approx(-43.99, abs=1e-2)
    assert reader.Ei == pytest.approx(4.00048, abs=1e-5)
    assert reader._metadata['proposal'] == '20240261'
    assert reader._metadata['sampleName'] == 'TbFeO3'
