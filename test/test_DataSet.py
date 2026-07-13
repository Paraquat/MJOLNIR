import pytest

import numpy as np
from pathlib import Path

from MJOLNIR.Data.DataSet import DataSet
from MJOLNIR.Data.DataSet import ArrayType
from MJOLNIR.Data.InstrumentName import InstrumentName

def test_dataset():
    data_path = Path('/home/zamaan/Projects/bambus')
    files = data_path.glob("camea2024*.hdf")
    data_files = sorted(list(files))

    dataset = DataSet(InstrumentName.CAMEA)
    dataset.set_binning(3)
    dataset.add_files(data_files)
    assert dataset.number_of_files() == 8
    first_reader = dataset._readers[0]
    second_reader = dataset._readers[1]
    assert first_reader.file.name == 'camea2024n001486.hdf'
    assert second_reader.file.name == 'camea2024n001487.hdf'
    assert first_reader.counts.shape == (135, 104, 1024)
    assert first_reader.A4[0] == pytest.approx(-43.99, abs=1e-2)
    assert first_reader.Ei == pytest.approx(4.00048, abs=1e-5)
    assert first_reader._metadata['proposal'] == '20240261'
    assert first_reader._metadata['sampleName'] == 'TbFeO3'

    ref_h_1 = dataset._readers[0].h[134, 0, 0]
    ref_h_2 = dataset._readers[1].h[0, 0, 0]
    h_view = dataset.get_view(ArrayType.H)

    # array shape
    assert len(h_view) == 1080
    assert h_view.shape[0] == 1080
    assert h_view.shape[1] == 104
    assert h_view.shape[2] == 24

    # contiguity
    assert h_view[134, 0, 0] == pytest.approx(ref_h_1)
    assert h_view[135, 0, 0] == pytest.approx(ref_h_2)
